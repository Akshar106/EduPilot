r"""
Agent graph
===========
The LangGraph `StateGraph` that runs one turn of course chat, and the smaller
one that runs Self Study.

    START -> memory -> route --out of scope----------------------------> finalize
                         |                                                  ^
                         +--unclear--> clarify --interrupt(): wait for the student
                         |                |  resumed with their reply
                         |                +--> route
                         +--> plan --Send x N--> research (subgraph, parallel)
                                                     |
                                                  collect
                                     question  /         \  quiz, flashcards
                                  synthesize             tutor
                                       |                   |
                                     verify <-> revise     |
                                       \                  /
                                        guardrails -> learn

Each `research` branch is a subgraph (`agents/research.py`): retrieve, grade
the excerpts, rephrase the search and retry when none address the question,
then answer, calling the calculator tool as needed.

What the graph does:

  * **Corrective retrieval** in every branch, before anything is refused.
  * **Verify -> revise.** A failed audit sends the answer back with the
    verifier's findings; the better-scoring draft is kept.
  * **Clarification via `interrupt()`.** An unanswerable message ("explain
    this") pauses the turn at `clarify`. The checkpointer holds it; the
    student's next message resumes it with `Command(resume=...)`, and routing
    runs again on the clarified question.
  * **Tutor tasks.** "Quiz me on X" and "flashcards on Y" retrieve like a
    question, then the tutor writes cited practice items. "Quiz me" with no
    topic draws on the student's profile.
  * **A learning profile** per student (LangGraph store), updated by `learn`
    and read by `memory`, adapts explanations across conversations.
  * **Streaming** of stage progress, branch progress and answer tokens.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.store.base import BaseStore
from langgraph.types import Command, Send, interrupt

from edupilot.core import tracing
from edupilot.core.config import DEFAULT_QUIZ_SIZE, EMBEDDING_MODEL, MAX_QUIZ_SIZE, RERANKER_MODEL
from edupilot.guardrails import scan_chunks, shift_citations
from edupilot.guardrails.grounding import ClaimVerdict, check_grounding
from edupilot.guardrails.output import ANNOTATE_BELOW_GROUNDING, apply_output_guardrails
from edupilot.llm import build_llm, message_text
from edupilot.retrieval import HybridRetriever

from . import prompts
from .context import PipelineConfig
from .memory import SUMMARY_MAX_TOKENS, fold_memory, seed_from_db, summarize_turns
from .nodes import (
    ANSWER_STREAM_TAG,
    TASKS,
    Answerer,
    EvidenceGrader,
    ModelFactory,
    Planner,
    QueryRewriter,
    Reviser,
    Router,
    RouterDecision,
    Synthesizer,
    Tutor,
    Verifier,
    detect_task,
    render_practice,
    topic_of,
)
from .profile import (
    empty_profile,
    format_profile,
    load_profile,
    save_profile,
    update_profile,
    weak_topics,
)
from .research import ResearchGraph
from .state import TURN_RESET, AgentState, ResearchTask, SubAnswer

logger = logging.getLogger(__name__)

#: Domain label the answerer and citations use for a student's own uploads.
SELF_STUDY_DOMAIN = "Self Study"

#: Clarifying questions per turn. After one, an answer that is still unclear
#: gets the fixed reply instead of another question.
MAX_CLARIFICATIONS = 1

TUTOR_FAILED_RESPONSE = (
    "I found course material on this, but could not turn it into practice items "
    "just now. Try again, or ask a question about the topic instead."
)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass
class PipelineResult:
    """Everything the API layer and the evaluation suite read from one turn."""

    final_answer: str
    intent_type: str = "single"
    domains: list[str] = field(default_factory=list)
    is_course_related: bool = True
    needs_clarification: bool = False
    refused: bool = False
    grounding_score: float | None = None
    """Measured grounding, or None when it was not measured. Never a default."""
    guardrail_action: str | None = None
    sources: list[dict] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    task: str = "answer"
    awaiting_clarification: bool = False
    """The turn is paused on a clarifying question; the next message resumes it."""
    verification_revised: bool = False
    """The answer shown is a revision made after a failed verification."""
    run_id: str | None = None
    """The turn's root run: its LangSmith trace id, and what feedback attaches to."""
    practice: dict | None = None
    """Structured quiz or flashcards behind a tutor answer; see nodes.practice_items."""

    @classmethod
    def from_result(cls, result: dict[str, Any]) -> PipelineResult:
        return cls(**{k: result[k] for k in cls.__dataclass_fields__ if k in result})


def build_sources(answers: Sequence[SubAnswer]) -> list[dict]:
    """
    Flat, de-duplicated citation list for the UI.

    `source_num` is the global [Source N] the merged answer uses: each
    branch's sources are numbered after the previous branch's.
    """
    out: list[dict] = []
    seen: set[str] = set()
    offset = 0
    for sub in answers:
        for i, (label, doc) in enumerate(zip(sub["labels"], sub["documents"]), start=1):
            key = f"{sub['domain']}:{label}"
            if key in seen:
                continue
            seen.add(key)
            meta = doc.metadata
            out.append(
                {
                    "source_num": offset + i,
                    "domain": sub["domain"],
                    "citation_label": label,
                    "filename": meta.get("filename"),
                    "page_number": meta.get("page_number"),
                    "section_path": meta.get("section_path"),
                    # "book" marks a textbook excerpt; the UI badges it.
                    "source_type": meta.get("source_type") or "lecture",
                    "relevance": round(float(meta.get("relevance", 0.0)), 4),
                    "kind": meta.get("kind", "excerpt"),
                    "text": doc.page_content[:1200],
                }
            )
        offset += len(sub["documents"])
    return out


def _retrieval_diagnostics(answers: Sequence[SubAnswer]) -> list[dict]:
    """Per-branch retrieval figures for the debug panel."""
    return [
        {
            "domain": s["domain"],
            "question": s["question"],
            "chunks": len(s["documents"]),
            "refused": s["refused"],
            "latency_ms": s["latency_ms"],
            **s["retrieval"],
        }
        for s in answers
    ]


def _stage_detail(node: str, update: dict) -> dict:
    """A small, JSON-safe summary of a node's update for the progress stream."""
    if node == "route":
        d = update.get("decision") or {}
        return {
            "intent_type": d.get("intent_type"),
            "domains": d.get("domains"),
            "is_course_related": d.get("is_course_related"),
            "task": d.get("task"),
        }
    if node == "plan":
        return {"sub_questions": update.get("sub_questions") or []}
    if node == "research":
        sub = (update.get("sub_answers") or [{}])[0]
        return {
            "domain": sub.get("domain"),
            "chunks": len(sub.get("documents") or []),
            "refused": sub.get("refused"),
            "rewrites": (sub.get("retrieval") or {}).get("rewrites") or [],
        }
    if node == "verify":
        v = (update.get("diagnostics") or {}).get("verifier") or {}
        return {
            "skipped": v.get("skipped", True),
            "grounding_score": v.get("grounding_score"),
            "needs_revision": bool(update.get("needs_revision")),
        }
    if node == "revise":
        return {"revisions": update.get("revisions")}
    if node == "guardrails":
        return {"action": (update.get("verdict") or {}).get("action")}
    return {}


def _task(state: AgentState) -> str:
    return (state.get("decision") or {}).get("task") or state.get("task") or "answer"


# ---------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------


def _miscitation(verdict: ClaimVerdict, n_sources: int) -> str:
    """A miscited claim as the reviser reads it: what it cites, and what supports it."""
    cited = ", ".join(
        f"[Source {n}]" if 1 <= n <= n_sources else f"[Source {n}] (never supplied)"
        for n in verdict.cited_sources
    )
    return f"{verdict.claim} (cites {cited}; [Source {verdict.best_source}] supports it)"


class EduPilotAgent:
    """
    Builds the agent graphs and runs turns through them.

    Args:
        domains: Domain registry from config.
        retriever_factory: Maps a retriever key — a domain code, or a study
            session id for Self Study — to a `HybridRetriever`.
        models: Builds a model runnable; `edupilot.llm.build_llm` by default.
            Tests substitute fakes here.
        checkpointer: Persists conversation state between turns. Can be
            attached later with `attach_checkpointer`, once the app's event
            loop is running.
        store: Holds student learning profiles across conversations.
    """

    def __init__(
        self,
        domains: dict[str, dict],
        retriever_factory: Callable[[str], HybridRetriever],
        *,
        models: ModelFactory = build_llm,
        checkpointer: BaseCheckpointSaver | None = None,
        store: BaseStore | None = None,
    ) -> None:
        self.domains = domains
        self.retriever_factory = retriever_factory
        self.models = models

        self.router = Router(models, domains)
        self.planner = Planner(models, domains)
        self.answerer = Answerer(models, domains)
        self.synthesizer = Synthesizer(models, domains)
        self.verifier = Verifier(models)
        self.reviser = Reviser(models)
        self.tutor = Tutor(models)
        self.research = ResearchGraph(
            # Late-bound, so replacing `retriever_factory` takes effect.
            retriever_factory=lambda key: self.retriever_factory(key),
            models=models,
            answerer=self.answerer,
            grader=EvidenceGrader(models),
            rewriter=QueryRewriter(models, domains),
        ).compile()

        self.store = store
        self._course = self._course_graph()
        self.graph = self._course.compile(store=store)
        """Stateless course graph: evaluation, and anything without a thread."""
        self.study_graph = self._study_graph().compile()

        self.checkpointer: BaseCheckpointSaver | None = None
        self.chat_graph = None
        """Course graph compiled with the checkpointer: course chat."""
        if checkpointer is not None:
            self.attach_checkpointer(checkpointer)

    def attach_checkpointer(
        self, checkpointer: BaseCheckpointSaver, store: BaseStore | None = None
    ) -> None:
        if store is not None:
            self.store = store
        self.checkpointer = checkpointer
        self.chat_graph = self._course.compile(checkpointer=checkpointer, store=self.store)

    # ------------------------------------------------------------------
    # Graph wiring
    # ------------------------------------------------------------------

    def _course_graph(self) -> StateGraph:
        g = StateGraph(AgentState, context_schema=PipelineConfig)
        g.add_node("memory", self._memory)
        g.add_node("route", self._route)
        g.add_node("clarify", self._clarify, destinations=("route", "finalize"))
        g.add_node("plan", self._plan)
        g.add_node("research", self.research)
        g.add_node("collect", self._collect)
        g.add_node("synthesize", self._synthesize)
        g.add_node("verify", self._verify)
        g.add_node("revise", self._revise, destinations=("verify", "guardrails"))
        g.add_node("tutor", self._tutor)
        g.add_node("guardrails", self._guardrails)
        g.add_node("learn", self._learn)
        g.add_node("finalize", self._finalize)

        g.add_edge(START, "memory")
        g.add_edge("memory", "route")
        g.add_conditional_edges("route", self._after_route, ["plan", "clarify", "finalize"])
        g.add_conditional_edges("plan", self._fan_out, ["research"])
        g.add_edge("research", "collect")
        g.add_conditional_edges(
            "collect", lambda s: "synthesize" if _task(s) == "answer" else "tutor",
            ["synthesize", "tutor"],
        )
        g.add_edge("synthesize", "verify")
        g.add_conditional_edges(
            "verify", lambda s: "revise" if s.get("needs_revision") else "guardrails",
            ["revise", "guardrails"],
        )
        g.add_edge("tutor", "guardrails")
        g.add_edge("guardrails", "learn")
        g.add_edge("learn", "finalize")
        g.add_edge("finalize", END)
        return g

    def _study_graph(self) -> StateGraph:
        # Nothing to route or split across domains: straight to the research
        # branch, then the same guardrails (or tutor) as course chat.
        g = StateGraph(AgentState, context_schema=PipelineConfig)
        g.add_node("research", self.research)
        g.add_node("collect", self._collect)
        g.add_node("tutor", self._tutor)
        g.add_node("guardrails", self._guardrails)
        g.add_node("finalize", self._finalize)

        g.add_conditional_edges(START, self._study_fan_out, ["research"])
        g.add_edge("research", "collect")
        g.add_conditional_edges(
            "collect", lambda s: "guardrails" if _task(s) == "answer" else "tutor",
            ["guardrails", "tutor"],
        )
        g.add_edge("tutor", "guardrails")
        g.add_edge("guardrails", "finalize")
        g.add_edge("finalize", END)
        return g

    @staticmethod
    def _after_route(state: AgentState) -> str:
        d = state.get("decision") or {}
        if not d.get("is_course_related", True):
            return "finalize"
        if d.get("needs_clarification"):
            return "clarify"
        # "Quiz me" with no topic and no history to draw on.
        if d.get("task") != "answer" and not d.get("domains") and not weak_topics(state.get("profile") or {}):
            return "clarify"
        return "plan"

    @staticmethod
    def _fan_out(state: AgentState) -> list[Send]:
        # Sub-questions have no ordering dependency, so each gets its own
        # branch; a three-domain question costs one round-trip, not three.
        profile_block = format_profile(state.get("profile") or {})
        return [
            Send(
                "research",
                ResearchTask(
                    question=sq["question"],
                    domain=sq["domain"],
                    retriever_key=sq["domain"],
                    history=state.get("history") or [],
                    summary=state.get("summary") or "",
                    profile_block=profile_block,
                    filenames=state.get("filenames"),
                    self_study=False,
                    task=_task(state),
                ),
            )
            for sq in state.get("sub_questions") or []
        ]

    @staticmethod
    def _study_fan_out(state: AgentState) -> list[Send]:
        task = _task(state)
        return [
            Send(
                "research",
                ResearchTask(
                    question=state["query"] if task == "answer" else topic_of(state["query"]),
                    domain=SELF_STUDY_DOMAIN,
                    retriever_key=state["retriever_key"] or "",
                    history=state.get("history") or [],
                    summary="",
                    profile_block="",
                    filenames=state.get("filenames"),
                    self_study=True,
                    task=task,
                ),
            )
        ]

    @staticmethod
    def _ordered(state: AgentState) -> list[SubAnswer]:
        """Sub-answers in sub-question order, whatever order the branches finished in."""
        subs = list(state.get("sub_answers") or [])
        order = {
            (q["domain"], q["question"]): i
            for i, q in enumerate(state.get("sub_questions") or [])
        }
        return sorted(subs, key=lambda s: order.get((s["domain"], s["question"]), len(order)))

    # ------------------------------------------------------------------
    # Nodes: understanding the message
    # ------------------------------------------------------------------

    async def _memory(self, state: AgentState, runtime: Runtime[PipelineConfig]) -> dict:
        cfg = runtime.context
        messages = list(state.get("messages") or [])
        summary = state.get("summary") or ""
        update: dict[str, Any] = {"profile": await self._load_profile(runtime)}
        seeded: list = []

        # A thread with no checkpoint: a conversation from before checkpoints
        # existed, or one reset after the student edited an earlier message.
        if not messages and state.get("session_id"):
            digest, seeded = seed_from_db(
                state["session_id"],
                exclude_message_ids=set(state.get("exclude_message_ids") or []),
            )
            if seeded or digest:
                messages, summary = seeded, digest or summary
                update["summary"] = summary

        if not messages:
            # Stateless runs carry whatever history they have in the input.
            return {**update, "history": list(state.get("history") or [])}

        async def summarize(existing: str, turns: list[dict]) -> str:
            try:
                llm = self.models(cfg.model, max_tokens=SUMMARY_MAX_TOKENS)
            except Exception:
                logger.warning("no model available for summarization", exc_info=True)
                return ""
            return await summarize_turns(llm, existing, turns)

        mem = await fold_memory(messages, summary, summarize)
        removals = [RemoveMessage(id=i) for i in mem.remove_ids]
        if seeded or removals:
            update["messages"] = [*seeded, *removals]
        return {
            **update,
            "history": mem.recent,
            "summary": mem.summary,
            "diagnostics": {
                "memory": {
                    "messages": len(messages),
                    "folded": len(mem.remove_ids),
                    "seeded_from_db": bool(seeded),
                    "summary_failed": mem.failed,
                }
            },
        }

    async def _load_profile(self, runtime: Runtime[PipelineConfig]) -> dict:
        try:
            return await load_profile(runtime.store, runtime.context.user_id)
        except Exception:
            logger.warning("could not load the student profile", exc_info=True)
            return empty_profile()

    async def _route(self, state: AgentState, runtime: Runtime[PipelineConfig]) -> dict:
        decision = await self.router.classify(
            state["query"],
            model=runtime.context.model,
            history=state.get("history"),
            memory=state.get("summary") or "",
        )
        diagnostics = {
            "intent_type": decision.intent_type,
            "domains": decision.domains,
            "is_course_related": decision.is_course_related,
            "reasoning": decision.reasoning,
            "used_fallback": decision.used_fallback,
            "task": decision.task,
            "topics": decision.topics,
            "standalone_question": decision.standalone_question,
        }

        if state.get("task") in TASKS:
            # The caller asked for a task explicitly (e.g. a "Quiz me" button).
            decision.task = state["task"]
            if decision.task != "answer":
                decision.needs_clarification = False
        manual = state.get("manual_domains")
        if manual:
            decision.domains = [d for d in manual if d in self.domains]
            decision.is_course_related = True
            decision.needs_clarification = False

        return {"decision": asdict(decision), "diagnostics": {"router": diagnostics}}

    async def _clarify(self, state: AgentState, runtime: Runtime[PipelineConfig]) -> Command:
        """
        Ask the student what they mean, and wait for the answer.

        `interrupt()` stops the run here; the checkpointer keeps the paused
        turn. When the student replies, the run resumes on this line with their
        reply, and routing runs again on the clarified question. Everything
        before `interrupt()` re-runs on resume, so it must stay side-effect free.
        """
        decision = state.get("decision") or {}
        message = (
            prompts.QUIZ_TOPIC_QUESTION
            if decision.get("task", "answer") != "answer"
            else prompts.CLARIFICATION_RESPONSE
        )
        asked = state.get("clarifications") or 0
        if not runtime.context.interactive or asked >= MAX_CLARIFICATIONS:
            return Command(goto="finalize", update={"clarify_message": message})

        reply = str(interrupt({"type": "clarification", "message": message})).strip()

        original = state.get("original_query") or state["query"]
        return Command(
            goto="route",
            update={
                "original_query": original,
                "query": f"{original} ({reply})" if reply else original,
                "clarification": {"asked": message, "reply": reply},
                "clarifications": asked + 1,
                "started_at": time.perf_counter(),
            },
        )

    async def _plan(self, state: AgentState, runtime: Runtime[PipelineConfig]) -> dict:
        decision = RouterDecision(**state["decision"])
        if decision.task != "answer" and decision.topics:
            # Practice items: retrieve each named concept, not "quiz me on ...".
            subs = [{"question": t["name"], "domain": t["domain"]} for t in decision.topics]
        elif decision.task != "answer" and not decision.domains:
            subs = [
                {"question": t["name"], "domain": t["domain"]}
                for t in weak_topics(state.get("profile") or {})
                if t["domain"] in self.domains
            ]
        else:
            # A follow-up ("it has an answer, please explain") is searched as the
            # question it refers to; its own words would match nothing.
            question = decision.standalone_question or state["query"]
            subs = await self.planner.plan(question, decision, model=runtime.context.model)
            if decision.task != "answer":
                subs = [{**s, "question": topic_of(s["question"])} for s in subs]
        return {"sub_questions": subs, "diagnostics": {"sub_questions": subs}}

    # ------------------------------------------------------------------
    # Nodes: answering
    # ------------------------------------------------------------------

    async def _collect(self, state: AgentState) -> dict:
        """
        Join the branches: order them, and number every source globally.

        Each branch cites its own sources from [Source 1]. Downstream the
        verifier, the guardrails and the UI see one combined list, so each
        branch's citations are shifted past the sources of the branches before
        it.
        """
        answers: list[SubAnswer] = []
        offset = 0
        for sub in self._ordered(state):
            answers.append(SubAnswer(**{**sub, "answer": shift_citations(sub["answer"], offset)}))
            offset += len(sub["documents"])

        diagnostics: dict[str, Any] = {"retrieval": _retrieval_diagnostics(answers)}
        injection = [r for s in answers for r in scan_chunks(s["documents"])[0]]
        if injection:
            diagnostics["injection_flags"] = injection
        return {"answers": answers, "diagnostics": diagnostics}

    async def _synthesize(self, state: AgentState, runtime: Runtime[PipelineConfig]) -> dict:
        cfg = runtime.context
        merged = await self.synthesizer.synthesize(
            state["query"], state.get("answers") or [], model=cfg.model, max_tokens=cfg.max_tokens_synth
        )
        return {"merged": merged}

    async def _verify(self, state: AgentState, runtime: Runtime[PipelineConfig]) -> dict:
        cfg = runtime.context
        if not cfg.enable_verification:
            return {"needs_revision": False}

        answers = state.get("answers") or []
        merged = state.get("merged") or ""
        revision = state.get("revisions") or 0
        verdict = await self._claim_audit(answers, merged, cfg)
        if verdict is None:
            verdict = await self.verifier.verify(
                state["query"], answers, merged, model=cfg.verify_model or cfg.model
            )
        update: dict[str, Any] = {
            "diagnostics": {"verifier": verdict},
            "audits": [
                {
                    "revision": revision,
                    "auditor": verdict.get("auditor", "model"),
                    "skipped": verdict.get("skipped", False),
                    "grounding_score": verdict.get("grounding_score"),
                    "coverage_score": verdict.get("coverage_score"),
                    "unsupported_claims": len(verdict.get("unsupported_claims") or []),
                    "miscited_claims": len(verdict.get("miscited_claims") or []),
                }
            ],
            "needs_revision": False,
        }
        if verdict.get("skipped"):
            return update

        # Keep whichever draft verified best, so a revision that makes things
        # worse is never what the student sees.
        score = (verdict.get("grounding_score") or 0.0, verdict.get("coverage_score") or 0.0)
        best = state.get("best") or {}
        if not best or score > (best["grounding"], best["coverage"]):
            update["best"] = {
                "answer": merged,
                "grounding": score[0],
                "coverage": score[1],
                "revision": revision,
            }

        update["needs_revision"] = (
            not self.verifier.satisfied(verdict)
            and revision < cfg.max_revisions
            and bool(self.reviser.findings(verdict))
            and any(not s["refused"] for s in answers)
        )
        return update

    @staticmethod
    async def _claim_audit(answers: Sequence[SubAnswer], merged: str, cfg: PipelineConfig) -> dict | None:
        """
        Audit a draft with the claim check instead of the model verifier;
        None when only the model verifier can.

        The model verifier was two of a turn's model calls (before and after
        a revision): on a free tier, a third of the turn's tokens and the
        rate-limit waits that come with them. For a question in one part it
        checks what the claim check checks locally in about a second: whether
        each claim is supported, and by the excerpt it cites. The verdict has
        the verifier's shape, so revision works from either; the findings
        are the claims themselves, and a citation of a source never supplied
        counts as miscited. The output guardrails then reuse the score.

        The model verifier still audits a question in several parts, since
        only it judges whether each part was covered, and any draft when the
        claim check cannot run (switched off, or no local scorer).
        """
        if all(s["refused"] for s in answers):
            return {"skipped": True, "reason": "a refusal makes no claims to audit"}
        if not cfg.enable_grounding_check or len(answers) != 1:
            return None
        evidence = [d.page_content for d in answers[0]["documents"]]
        if not evidence:
            return None
        report = await asyncio.to_thread(check_grounding, merged.strip(), evidence)
        if not report.checked:
            return None
        score = report.grounding_score
        return {
            "auditor": "claim check",
            "grounding_score": round(score, 3),
            "coverage_score": None,  # not measured; `satisfied` falls back to is_satisfactory
            "is_satisfactory": score >= ANNOTATE_BELOW_GROUNDING and not report.miscited,
            "unsupported_claims": [v.claim for v in report.unsupported],
            "miscited_claims": [_miscitation(v, len(evidence)) for v in report.miscited if v.is_supported],
            "missing_topics": [],
            "claims_checked": len(report.verdicts),
        }

    async def _revise(self, state: AgentState, runtime: Runtime[PipelineConfig]) -> Command:
        cfg = runtime.context
        verdict = (state.get("diagnostics") or {}).get("verifier") or {}
        documents = [d for s in state.get("answers") or [] for d in s["documents"]]
        runtime.stream_writer({"stage": "revising", "issues": verdict.get("unsupported_claims", [])[:3]})

        revised = await self.reviser.revise(
            state["query"],
            state.get("merged") or "",
            documents,
            verdict,
            model=cfg.model,
            max_tokens=cfg.max_tokens_synth,
        )
        revisions = (state.get("revisions") or 0) + 1
        if not revised:
            return Command(goto="guardrails", update={"revisions": revisions})
        return Command(goto="verify", update={"merged": revised, "revisions": revisions})

    async def _tutor(self, state: AgentState, runtime: Runtime[PipelineConfig]) -> dict:
        """Write a quiz or flashcards from everything the branches retrieved."""
        cfg = runtime.context
        decision = state.get("decision") or {}
        answers = state.get("answers") or []
        documents = [d for s in answers for d in s["documents"]]
        if not documents:
            return {"merged": prompts.NO_EVIDENCE_RESPONSE}

        task = _task(state)
        topic = (
            ", ".join(t["name"] for t in decision.get("topics") or [])
            or ", ".join(s["question"] for s in answers)
        )
        count = max(1, min(int(decision.get("item_count") or DEFAULT_QUIZ_SIZE), MAX_QUIZ_SIZE))
        runtime.stream_writer({"stage": "writing", "task": task, "count": count})

        practice = await self.tutor.write(
            task,
            topic,
            documents,
            count=count,
            focus=[t["name"] for t in weak_topics(state.get("profile") or {})],
            model=cfg.model,
            max_tokens=cfg.max_tokens_answer,
        )
        if practice is None:
            return {"merged": TUTOR_FAILED_RESPONSE, "practice": None}
        return {"merged": render_practice(practice), "practice": practice}

    async def _guardrails(self, state: AgentState, runtime: Runtime[PipelineConfig]) -> dict:
        answers = state.get("answers") or []
        best = state.get("best") or {}
        # The best-verified draft when verification ran; otherwise the merged
        # answer. Self Study has no synthesize step, so its single branch's
        # answer is checked as written.
        answer = (
            best.get("answer")
            or state.get("merged")
            or (answers[0]["answer"] if answers else prompts.NO_EVIDENCE_RESPONSE)
        )
        documents = [d for s in answers for d in s["documents"]]
        labels = [label for s in answers for label in s["labels"]]

        # Practice questions are not claims, so claim-level grounding is for
        # answers only; citations are validated either way. The check runs the
        # cross-encoder: CPU work, off the loop.
        check_claims = (
            runtime.context.enable_grounding_check and bool(documents) and _task(state) == "answer"
        )
        verdict = await asyncio.to_thread(
            apply_output_guardrails,
            answer,
            [d.page_content for d in documents],
            labels,
            check_claims=check_claims,
        )
        diagnostics: dict[str, Any] = {"guardrails": verdict.as_dict()}
        if state.get("audits"):
            diagnostics["audits"] = state["audits"]
        return {
            "verdict": {
                "answer": verdict.answer,
                "action": verdict.action,
                "is_refusal": verdict.is_refusal,
                "grounding_score": verdict.grounding_score,
                "revised": bool(best.get("revision")),
            },
            "diagnostics": diagnostics,
        }

    async def _learn(self, state: AgentState, runtime: Runtime[PipelineConfig]) -> dict:
        """Fold this turn into the student's profile. No model call: router fields only."""
        store, user_id = runtime.store, runtime.context.user_id
        decision = state.get("decision") or {}
        if store is None or not user_id or not decision.get("is_course_related", True):
            return {}
        profile = update_profile(
            state.get("profile") or empty_profile(),
            topics=decision.get("topics") or [],
            confusion=bool(decision.get("confusion")),
            depth=decision.get("requested_depth"),
        )
        try:
            await save_profile(store, user_id, profile)
        except Exception:
            logger.warning("could not save the student profile", exc_info=True)
            return {}
        return {
            "diagnostics": {
                "profile": {
                    "topics_tracked": len(profile["topics"]),
                    "weak_topics": [t["name"] for t in weak_topics(profile)],
                    "depth": profile.get("depth"),
                }
            }
        }

    async def _finalize(self, state: AgentState) -> dict:
        """Assemble the turn's result, record the exchange, drop the turn's evidence."""
        decision = state.get("decision") or {}
        verdict = state.get("verdict") or {}
        answers = state.get("answers") or []
        diagnostics = dict(state.get("diagnostics") or {})

        outcome: dict[str, Any] = {}
        if not decision.get("is_course_related", True):
            answer = prompts.OUT_OF_DOMAIN_RESPONSE
            outcome = {"is_course_related": False, "refused": True}
        elif decision.get("needs_clarification") or state.get("clarify_message"):
            answer = state.get("clarify_message") or prompts.CLARIFICATION_RESPONSE
            outcome = {"needs_clarification": True}
        else:
            answer = verdict.get("answer") or ""
            outcome = {
                "refused": bool(verdict.get("is_refusal")) or all(s["refused"] for s in answers),
                "verification_revised": bool(verdict.get("revised")),
            }
            if started := state.get("started_at"):
                diagnostics["total_latency_ms"] = int((time.perf_counter() - started) * 1000)

        result = {
            "final_answer": answer,
            "intent_type": decision.get("intent_type", "single"),
            "domains": [s["domain"] for s in answers],
            "is_course_related": True,
            "needs_clarification": False,
            "refused": False,
            "grounding_score": verdict.get("grounding_score"),
            "guardrail_action": verdict.get("action"),
            "sources": build_sources(answers),
            "diagnostics": diagnostics,
            "task": _task(state),
            "practice": state.get("practice"),
            **outcome,
        }

        clarification = state.get("clarification") or {}
        if clarification:
            # Mirror what the student saw: question, clarifying question, reply.
            exchange = [
                HumanMessage(content=state.get("original_query") or state["query"]),
                AIMessage(content=clarification["asked"]),
                HumanMessage(content=clarification["reply"]),
            ]
        else:
            exchange = [HumanMessage(content=state["query"])]
        return {
            "result": result,
            "messages": [*exchange, AIMessage(content=answer)],
            # Evidence is per-turn; the checkpoint keeps the conversation.
            "sub_answers": None,
            "answers": [],
        }

    # ------------------------------------------------------------------
    # Running turns
    # ------------------------------------------------------------------

    @staticmethod
    def _turn_input(query: str, **fields: Any) -> dict[str, Any]:
        return {
            **TURN_RESET,
            "query": query,
            "started_at": time.perf_counter(),
            "session_id": fields.get("session_id"),
            "exclude_message_ids": list(fields.get("exclude_message_ids") or ()),
            "manual_domains": fields.get("manual_domains"),
            "filenames": fields.get("filenames"),
            "retriever_key": fields.get("retriever_key"),
            "history": list(fields.get("history") or ()),
            "task": fields.get("task"),
        }

    @staticmethod
    def _run_config(
        cfg: PipelineConfig,
        *,
        thread_id: str | None,
        callbacks: Sequence[BaseCallbackHandler],
        run_id: str,
        run_name: str,
        trace_thread: str | None,
    ) -> RunnableConfig:
        """
        The turn's runnable config, including what its trace is labelled with.

        The run id is chosen here rather than by the tracer so the API can
        hand it back with the answer: a rating or a "view trace" link names
        the turn by it. Metadata records what produced the answer — models and
        feature switches — so LangSmith can filter and compare by them.
        """
        config: RunnableConfig = {
            "max_concurrency": cfg.max_parallel_domains,
            "callbacks": list(callbacks),
            "run_id": run_id,
            "run_name": run_name,
            "tags": ["edupilot", run_name, *cfg.trace_tags],
            "metadata": {
                # LangSmith groups runs sharing a thread_id into one thread,
                # so a conversation's turns read as a conversation.
                "thread_id": trace_thread,
                "user": tracing.pseudonym(cfg.user_id),
                "model": cfg.model,
                "verify_model": cfg.verify_model or cfg.model,
                "embedding_model": EMBEDDING_MODEL,
                "reranker_model": RERANKER_MODEL,
                "corrective_retrieval": cfg.corrective_retrieval,
                "max_revisions": cfg.max_revisions,
                "tools": cfg.enable_tools,
                "verification": cfg.enable_verification,
                **cfg.trace_metadata,
            },
        }
        if thread_id:
            config["configurable"] = {"thread_id": thread_id}
        return config

    def _graph_for(self, thread_id: str | None):
        if not thread_id:
            return self.graph
        if self.chat_graph is None:
            logger.warning("no checkpointer attached — conversation memory is in-process only")
            self.attach_checkpointer(InMemorySaver())
        return self.chat_graph

    async def pending_clarification(self, thread_id: str) -> str | None:
        """The clarifying question a paused turn is waiting on, if any."""
        if self.chat_graph is None:
            return None
        snapshot = await self.chat_graph.aget_state({"configurable": {"thread_id": thread_id}})
        for task in snapshot.tasks:
            for paused in task.interrupts:
                return str((paused.value or {}).get("message", ""))
        return None

    async def _course_input(self, query: str, thread_id: str | None, **fields: Any):
        """A fresh turn — or, when this thread is paused on a question, the reply to it."""
        if thread_id and await self.pending_clarification(thread_id) is not None:
            return Command(resume=query)
        return self._turn_input(query, **fields)

    async def arun(
        self,
        query: str,
        cfg: PipelineConfig,
        *,
        thread_id: str | None = None,
        callbacks: Sequence[BaseCallbackHandler] = (),
        **fields: Any,
    ) -> PipelineResult:
        """
        Run one course-chat turn and return its result.

        With `thread_id`, the turn reads and extends that conversation's
        checkpointed memory (pass the chat session id as both `thread_id` and
        `session_id`), and resumes the thread instead if it is paused on a
        clarifying question. Without it the run is stateless and `history` is
        used as given.

        Keyword fields: `session_id`, `exclude_message_ids`, `manual_domains`,
        `filenames`, `history`, `task`.
        """
        result: PipelineResult | None = None
        async for event in self.astream(query, cfg, thread_id=thread_id, callbacks=callbacks, **fields):
            if event["type"] == "result":
                result = event["result"]
        assert result is not None
        return result

    async def astream(
        self,
        query: str,
        cfg: PipelineConfig,
        *,
        thread_id: str | None = None,
        callbacks: Sequence[BaseCallbackHandler] = (),
        **fields: Any,
    ) -> AsyncIterator[dict]:
        """`arun`, yielding progress events as the turn runs. See `_stream`."""
        graph_input = await self._course_input(query, thread_id, **fields)
        async for event in self._stream(
            self._graph_for(thread_id),
            graph_input,
            cfg,
            thread_id=thread_id,
            callbacks=callbacks,
            run_name="chat_turn",
            trace_thread=thread_id,
        ):
            yield event

    async def arun_study(
        self,
        query: str,
        cfg: PipelineConfig,
        *,
        ss_session_id: str,
        callbacks: Sequence[BaseCallbackHandler] = (),
        **fields: Any,
    ) -> PipelineResult:
        """One Self Study turn over the study session's own uploads."""
        result: PipelineResult | None = None
        async for event in self.astream_study(
            query, cfg, ss_session_id=ss_session_id, callbacks=callbacks, **fields
        ):
            if event["type"] == "result":
                result = event["result"]
        assert result is not None
        return result

    async def astream_study(
        self,
        query: str,
        cfg: PipelineConfig,
        *,
        ss_session_id: str,
        callbacks: Sequence[BaseCallbackHandler] = (),
        **fields: Any,
    ) -> AsyncIterator[dict]:
        """`arun_study`, yielding progress events as the turn runs."""
        fields["task"] = fields.get("task") or detect_task(query)
        graph_input = self._turn_input(query, retriever_key=ss_session_id, **fields)
        async for event in self._stream(
            self.study_graph,
            graph_input,
            cfg,
            thread_id=None,
            callbacks=callbacks,
            run_name="self_study_turn",
            trace_thread=ss_session_id,
        ):
            yield event

    async def _stream(
        self,
        graph,
        graph_input: Any,
        cfg: PipelineConfig,
        *,
        thread_id: str | None,
        callbacks: Sequence[BaseCallbackHandler],
        run_name: str,
        trace_thread: str | None,
    ) -> AsyncIterator[dict]:
        """
        Run a turn, yielding events as they happen:

            {"type": "stage", "stage": <node>, ...}       a parent node finished
            {"type": "progress", "stage": ..., ...}       progress inside a node or branch
            {"type": "token", "text": ...}                a piece of the answer
            {"type": "reset"}                             discard the tokens so far
            {"type": "result", "result": PipelineResult}  always last

        Tokens come only from model calls tagged as the answer. When a
        question splits across domains, the per-domain answers run in parallel
        and would interleave, so only the synthesizer's merge is streamed.
        `reset` precedes the tokens of each answer-writing call after the
        first — a revision replacing a draft, or the answer after a tool call.
        """
        planned = 0
        run_id = tracing.new_run_id()
        result: dict[str, Any] | None = None
        paused: dict | None = None
        decision: dict[str, Any] = {}
        diagnostics: dict[str, Any] = {}
        current_run, streamed = None, False

        # A checkpointed turn is saved once, when it ends or pauses, rather
        # than after every step: nothing resumes mid-turn except a pause.
        persistence = {"durability": "exit"} if thread_id else {}
        async for namespace, mode, chunk in graph.astream(
            graph_input,
            self._run_config(
                cfg,
                thread_id=thread_id,
                callbacks=callbacks,
                run_id=run_id,
                run_name=run_name,
                trace_thread=trace_thread,
            ),
            context=cfg,
            stream_mode=["updates", "messages", "custom"],
            subgraphs=True,
            **persistence,
        ):
            in_branch = bool(namespace) and namespace[0].startswith("research")
            if mode == "custom":
                yield {"type": "progress", **chunk}
                continue

            if mode == "messages":
                message, meta = chunk
                if ANSWER_STREAM_TAG not in (meta.get("tags") or ()):
                    continue
                if in_branch and planned > 1:
                    continue
                text = message_text(message)
                if not text:
                    continue
                if message.id != current_run:
                    if streamed:
                        yield {"type": "reset"}
                    current_run = message.id
                streamed = True
                yield {"type": "token", "text": text}
                continue

            if namespace:
                continue  # a branch's own node updates; its progress came as custom events
            for node, update in (chunk or {}).items():
                if node == "__interrupt__":
                    paused = update[0].value if update else {}
                    continue
                update = update if isinstance(update, dict) else {}
                diagnostics.update(update.get("diagnostics") or {})
                if node == "route":
                    decision = update.get("decision") or {}
                if node == "plan":
                    planned = len(update.get("sub_questions") or [])
                if node == "finalize":
                    result = update.get("result") or {}
                    continue
                yield {"type": "stage", "stage": node, **_stage_detail(node, update)}

        await self._prune(thread_id)
        if result is None and paused is not None:
            yield {
                "type": "result",
                "result": PipelineResult(
                    final_answer=str(paused.get("message", "")),
                    intent_type=decision.get("intent_type", "single"),
                    needs_clarification=True,
                    awaiting_clarification=True,
                    task=decision.get("task", "answer"),
                    diagnostics=diagnostics,
                    run_id=run_id,
                ),
            }
            return
        yield {"type": "result", "result": PipelineResult.from_result({**(result or {}), "run_id": run_id})}

    def run(
        self,
        query: str,
        cfg: PipelineConfig,
        *,
        history: Sequence[dict] | None = None,
        manual_domains: Sequence[str] | None = None,
        filenames: Sequence[str] | None = None,
        callbacks: Sequence[BaseCallbackHandler] = (),
    ) -> PipelineResult:
        """
        Synchronous, stateless run — for the evaluation suite and the CLI.

        Starts its own event loop, so it must be called from a thread that is
        not already running one (the API calls it through the worker pool).
        """
        return asyncio.run(
            self.arun(
                query,
                cfg,
                history=history,
                manual_domains=manual_domains,
                filenames=filenames,
                callbacks=callbacks,
            )
        )

    # ------------------------------------------------------------------
    # Thread maintenance
    # ------------------------------------------------------------------

    async def reset_thread(self, thread_id: str) -> None:
        """
        Forget a conversation's checkpointed memory, including a paused turn.

        Called when its messages are edited or deleted, so the next turn
        reseeds from the database instead of remembering turns that are gone.
        """
        if self.checkpointer is not None:
            await self.checkpointer.adelete_thread(thread_id)

    async def _prune(self, thread_id: str | None) -> None:
        if not thread_id or self.checkpointer is None:
            return
        try:
            await self.checkpointer.aprune([thread_id], strategy="keep_latest")
        except NotImplementedError:
            pass  # the in-memory saver; nothing persists anyway
        except Exception:
            logger.warning("checkpoint pruning failed for %s", thread_id, exc_info=True)


__all__ = [
    "MAX_CLARIFICATIONS",
    "SELF_STUDY_DOMAIN",
    "EduPilotAgent",
    "PipelineConfig",
    "PipelineResult",
    "build_sources",
]
