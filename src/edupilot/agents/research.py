"""
Research branch
===============
The subgraph each sub-question runs through — one instance per domain, in
parallel, launched by `Send` from the parent graph.

    START -> retrieve -> grade --relevant--------------------> answer <-> tools
                ^          |                                     |
                |          +--nothing relevant, retries left     +--> done -> END
                |          v
                +------ rewrite

**Corrective retrieval.** The reranker's floor already drops obviously
irrelevant chunks, but a chunk can clear it by sharing vocabulary with the
question without addressing it. `grade` asks the model which excerpts
actually do, and keeps only those. When none do, the question is usually in
the student's words rather than the course's ("self attention block" vs
"scaled dot-product attention"), so `rewrite` rephrases the search in the
course's terminology and the branch searches again, keeping anything already
judged relevant. Only when the retries are spent does it refuse. Partial
evidence is answered as partial rather than searched again: the answerer
names the gap, and every retry costs the student a round-trip.

**Tool loop.** With tutor tools enabled the answerer can call `calculate`.
`tools` runs each call and appends the result as an extra numbered source, so
a computed number is cited like an excerpt and checked by the same
guardrails. The loop is capped; the last round is offered no tools, so the
model has to answer in text.

The branch keeps no checkpoints of its own (`checkpointer=False`): it runs to
completion within one turn and only its `sub_answers` reach the parent.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.constants import TAG_NOSTREAM
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from edupilot.core.config import LLM_MAX_TOKENS_CLASSIFY
from edupilot.llm import message_text
from edupilot.retrieval import EduPilotRetriever, HybridRetriever

from . import prompts
from .context import PipelineConfig
from .contracts import citation_label, is_refusal
from .nodes import Answerer, EvidenceGrader, ModelFactory, QueryRewriter
from .state import ResearchOutput, ResearchState, SubAnswer
from .tools import TUTOR_TOOLS, CalculatorError, evaluate

logger = logging.getLogger(__name__)

#: Model turns that may call a tool before the answerer must reply in text.
MAX_TOOL_ROUNDS = 3


class ResearchGraph:
    """Builds the research subgraph around the agents it needs."""

    def __init__(
        self,
        *,
        retriever_factory: Callable[[str], HybridRetriever],
        models: ModelFactory,
        answerer: Answerer,
        grader: EvidenceGrader,
        rewriter: QueryRewriter,
    ) -> None:
        self.retriever_factory = retriever_factory
        self.models = models
        self.answerer = answerer
        self.grader = grader
        self.rewriter = rewriter

    def compile(self):
        g = StateGraph(ResearchState, context_schema=PipelineConfig, output_schema=ResearchOutput)
        g.add_node("retrieve", self._retrieve)
        g.add_node("grade", self._grade)
        g.add_node("rewrite", self._rewrite)
        g.add_node("answer", self._answer)
        g.add_node("tools", self._tools)
        g.add_node("done", self._done)

        g.add_edge(START, "retrieve")
        g.add_edge("retrieve", "grade")
        g.add_conditional_edges("grade", _next_step, ["rewrite", "answer", "done"])
        g.add_conditional_edges("rewrite", _next_step, ["retrieve", "answer", "done"])
        g.add_conditional_edges("answer", _next_step, ["tools", "done"])
        g.add_edge("tools", "answer")
        g.add_edge("done", END)
        return g.compile(checkpointer=False)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    async def _retrieve(self, state: ResearchState, runtime: Runtime[PipelineConfig]) -> dict:
        cfg = runtime.context
        query = state.get("query") or state["question"]
        attempt = (state.get("attempts") or 0) + 1
        runtime.stream_writer({"stage": "retrieving", "domain": state["domain"], "attempt": attempt})

        retriever = EduPilotRetriever(
            retriever=self.retriever_factory(state["retriever_key"]),
            config=cfg.retrieval,
            filenames=state.get("filenames"),
            query_llm=self._query_llm(cfg),
        )
        outcome = await retriever.asearch(query)

        # A retry keeps what an earlier attempt found relevant and adds to it.
        kept = list(state.get("documents") or [])
        seen = {d.metadata.get("chunk_id") for d in kept}
        fresh = [d for d in outcome.documents if d.metadata.get("chunk_id") not in seen]
        return {
            "query": query,
            "attempts": attempt,
            "documents": (kept + fresh)[: cfg.retrieval.top_k * 2],
            "retrieval": outcome.diagnostics,
        }

    async def _grade(self, state: ResearchState, runtime: Runtime[PipelineConfig]) -> dict:
        cfg = runtime.context
        documents = list(state.get("documents") or [])
        proceed = "answer" if state.get("task", "answer") == "answer" else "done"
        if not cfg.corrective_retrieval:
            return {"next_step": proceed}

        grade = {
            "attempt": state.get("attempts", 1),
            "query": state.get("query") or state["question"],
            "total": len(documents),
        }
        kept = documents
        if documents:
            verdict = await self.grader.grade(state["question"], documents, model=cfg.model)
            if verdict is None:
                grade.update(relevant=len(documents), sufficient=True, graded=False)
            else:
                picked = sorted({i for i in verdict.relevant_sources if 1 <= i <= len(documents)})
                kept = [documents[i - 1] for i in picked]
                grade.update(
                    relevant=len(kept),
                    sufficient=verdict.sufficient and bool(kept),
                    missing=verdict.missing,
                    graded=True,
                )
        else:
            grade.update(relevant=0, sufficient=False, missing="nothing cleared the relevance floor")

        runtime.stream_writer(
            {"stage": "graded", "domain": state["domain"], "relevant": grade["relevant"], "total": len(documents)}
        )
        retries_left = grade["attempt"] <= cfg.max_query_rewrites
        return {
            "documents": kept,
            "grade": grade,
            "grades": [grade],
            "next_step": "rewrite" if not kept and retries_left else proceed,
        }

    async def _rewrite(self, state: ResearchState, runtime: Runtime[PipelineConfig]) -> dict:
        cfg = runtime.context
        proceed = "answer" if state.get("task", "answer") == "answer" else "done"
        tried = [state["question"], *(state.get("rewrites") or [])]
        query = await self.rewriter.rewrite(
            state["question"],
            state["domain"],
            tried=tried,
            missing=(state.get("grade") or {}).get("missing", ""),
            model=cfg.model,
        )
        if not query:
            return {"next_step": proceed}
        runtime.stream_writer({"stage": "rewriting", "domain": state["domain"], "query": query})
        return {"query": query, "rewrites": [query], "next_step": "retrieve"}

    def _query_llm(self, cfg: PipelineConfig) -> Callable[[str, str], str] | None:
        """`(system, user) -> text` for HyDE / multi-query probes, when enabled."""
        if not (cfg.retrieval.use_hyde or cfg.retrieval.use_multi_query):
            return None
        try:
            llm = self.models(cfg.model, max_tokens=LLM_MAX_TOKENS_CLASSIFY).with_config(
                tags=[TAG_NOSTREAM]
            )
        except Exception:
            logger.warning("no model for query transforms — searching the raw query only")
            return None

        def ask(system: str, user: str) -> str:
            # Runs inside the retriever's worker thread, hence sync `invoke`.
            return message_text(llm.invoke([SystemMessage(content=system), HumanMessage(content=user)]))

        return ask

    # ------------------------------------------------------------------
    # Answering
    # ------------------------------------------------------------------

    async def _answer(self, state: ResearchState, runtime: Runtime[PipelineConfig]) -> dict:
        cfg = runtime.context
        documents = list(state.get("documents") or [])
        # No evidence means refuse. There is deliberately no path that answers
        # from the model's own knowledge.
        if not documents:
            logger.info("no evidence for %r in %s — refusing", state["question"][:60], state["domain"])
            return {
                "answer_messages": [AIMessage(content=prompts.NO_EVIDENCE_RESPONSE)],
                "answer_started": time.perf_counter(),
                "next_step": "done",
            }

        update: dict = {}
        messages = list(state.get("answer_messages") or [])
        if not messages:
            runtime.stream_writer(
                {"stage": "answering", "domain": state["domain"], "chunks": len(documents)}
            )
            messages = self.answerer.prepare(
                state["question"],
                state["domain"],
                documents,
                history=state.get("history"),
                memory=state.get("summary") or "",
                profile_block=state.get("profile_block") or "",
                self_study=bool(state.get("self_study")),
                tools=cfg.enable_tools,
            )
            update = {"answer_messages": messages, "answer_started": time.perf_counter()}

        use_tools = cfg.enable_tools and (state.get("tool_rounds") or 0) < MAX_TOOL_ROUNDS
        reply = await self.answerer.respond(
            messages,
            model=cfg.model,
            max_tokens=cfg.max_tokens_answer,
            tools=TUTOR_TOOLS if use_tools else None,
        )
        update["answer_messages"] = [*update.get("answer_messages", []), reply]
        update["next_step"] = "tools" if use_tools and reply.tool_calls else "done"
        return update

    async def _tools(self, state: ResearchState, runtime: Runtime[PipelineConfig]) -> dict:
        """Run the answerer's tool calls; each result becomes a citable source."""
        call_message = state["answer_messages"][-1]
        numbered = len(state.get("documents") or []) + len(state.get("calculations") or [])
        results: list[ToolMessage] = []
        sources: list[Document] = []

        for call in getattr(call_message, "tool_calls", None) or []:
            expression = str((call.get("args") or {}).get("expression", ""))
            if call.get("name") != "calculate":
                results.append(
                    ToolMessage(content=f"Error: no tool named {call.get('name')!r}", tool_call_id=call["id"], status="error")
                )
                continue
            try:
                value = evaluate(expression)
            except CalculatorError as exc:
                results.append(ToolMessage(content=f"Error: {exc}", tool_call_id=call["id"], status="error"))
                continue

            number = numbered + len(sources) + 1
            sources.append(
                Document(
                    page_content=f"{expression} = {value}",
                    metadata={
                        "doc_title": "Calculator",
                        "kind": "calculation",
                        "chunk_id": f"calc-{number}",
                        "relevance": 1.0,
                    },
                )
            )
            results.append(
                ToolMessage(
                    content=f"{expression} = {value}\nCite this result as [Source {number}].",
                    tool_call_id=call["id"],
                )
            )
            runtime.stream_writer(
                {"stage": "calculating", "domain": state["domain"], "expression": expression, "result": value}
            )

        return {
            "answer_messages": results,
            "calculations": sources,
            "tool_rounds": (state.get("tool_rounds") or 0) + 1,
        }

    async def _done(self, state: ResearchState) -> dict:
        retrieved = list(state.get("documents") or [])
        calculations = list(state.get("calculations") or [])
        documents = retrieved + calculations

        if state.get("task", "answer") == "answer":
            reply = next(
                (m for m in reversed(state.get("answer_messages") or []) if isinstance(m, AIMessage)),
                None,
            )
            text = message_text(reply).strip() if reply else prompts.NO_EVIDENCE_RESPONSE
            refused = not retrieved or is_refusal(text)
            started = state.get("answer_started")
            latency_ms = int((time.perf_counter() - started) * 1000) if started else 0
        else:
            text, refused, latency_ms = "", not retrieved, 0

        return {
            "sub_answers": [
                SubAnswer(
                    domain=state["domain"],
                    question=state["question"],
                    answer=text,
                    documents=documents,
                    labels=[citation_label(d.metadata) for d in documents],
                    refused=refused,
                    latency_ms=latency_ms,
                    retrieval={
                        **(state.get("retrieval") or {}),
                        "attempts": state.get("attempts", 1),
                        "rewrites": state.get("rewrites") or [],
                        "grades": state.get("grades") or [],
                        "calculations": [d.page_content for d in calculations],
                    },
                )
            ]
        }


def _next_step(state: ResearchState) -> str:
    return state.get("next_step") or "done"
