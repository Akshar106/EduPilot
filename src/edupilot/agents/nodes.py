"""
Agents
======
The agents the graph's nodes delegate to. Each owns one prompt and one output
contract; none of them knows it is running inside a graph.

    Router -> Planner -> [retrieve -> EvidenceGrader -> QueryRewriter? -> Answerer] x N
           -> Synthesizer -> Verifier -> Reviser?          (questions)
           -> Tutor                                        (quizzes, flashcards)

Each agent is a LangChain chain (`ChatPromptTemplate | model`) awaited from an
async node. Agents that return JSON bind a Pydantic schema with structured
output instead of regex-parsing the reply.

Behavioural guarantees:

  * **No answer without evidence.** Empty retrieval produces a refusal; there
    is no path that answers from the model's own knowledge. The rewriter
    widens the search, never the evidence standard.
  * **The verifier's verdict is acted on**: the reviser repairs what it found,
    and the output guardrails annotate or replace what is still unsupported.
  * **Quality scores are measured or absent.** Never floored or defaulted.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import Runnable

from edupilot.llm import message_text

from . import prompts
from .contracts import (
    EvidenceGrade,
    FlashcardsOutput,
    PlannerOutput,
    QueryRewrite,
    QuizOutput,
    RouterOutput,
    VerifierOutput,
    build_evidence,
    build_evidence_summary,
    format_history,
    make_fence,
)
from .memory import format_memory

logger = logging.getLogger(__name__)

ModelFactory = Callable[..., Runnable]
"""`(model, *, max_tokens, schema=None, tools=None) -> Runnable` — `edupilot.llm.build_llm`."""

#: Tag on the model calls whose tokens are the answer the student reads. The
#: streaming layer forwards tokens only from calls carrying it, so routing,
#: grading and summarization never leak into the answer stream.
ANSWER_STREAM_TAG = "edupilot:answer"

TASKS = ("answer", "quiz", "flashcards")

_QUIZ_WORDS = re.compile(
    r"\b(quiz|test me|mcqs?|practice (?:questions?|problems?)|multiple[- ]choice)\b", re.I
)
_CARD_WORDS = re.compile(r"\bflash ?cards?\b", re.I)


def detect_task(text: str) -> str:
    """Keyword task detection, for when no router call is made or it fails."""
    if _CARD_WORDS.search(text or ""):
        return "flashcards"
    if _QUIZ_WORDS.search(text or ""):
        return "quiz"
    return "answer"


_REQUEST_WORDS = re.compile(
    r"\b(?:please|can you|could you|quiz me|test me|give me|make(?: me)?|create|generate|"
    r"write|some|a few|\d+|practice|questions?|problems?|mcqs?|multiple[- ]choice|"
    r"flash ?cards?|a quiz|quiz|my (?:notes|documents|uploads|files|slides)|on|about|"
    r"for|covering)\b",
    re.I,
)

#: What a practice request over "my notes" (no topic at all) searches for.
BROAD_TOPIC = "main ideas, key definitions and important results"


def topic_of(request: str) -> str:
    """
    The subject of a quiz or flashcard request, for searching.

    "quiz me on hypothesis testing" -> "hypothesis testing". Used when the
    router named no topics (it was unavailable, or this is Self Study, which
    has no router): searching the request verbatim spends half the query on
    "quiz me on".
    """
    topic = _REQUEST_WORDS.sub(" ", request or "")
    topic = re.sub(r"[()\[\]?!.,:;\"']", " ", topic)
    topic = " ".join(topic.split())
    return topic or BROAD_TOPIC


def context_block(memory: str, profile_block: str, history: Sequence[dict] | None) -> str:
    """Conversation digest, student profile and recent turns, as one prompt block."""
    return format_memory(memory) + profile_block + format_history(history)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass
class RouterDecision:
    intent_type: str = "single"
    domains: list[str] = field(default_factory=list)
    is_course_related: bool = True
    needs_clarification: bool = False
    clarification_hint: str | None = None
    reasoning: str = ""
    used_fallback: bool = False
    task: str = "answer"
    topics: list[dict] = field(default_factory=list)
    item_count: int | None = None
    confusion: bool = False
    requested_depth: str | None = None
    standalone_question: str = ""
    """The question with what it refers to spelled out, for retrieval; "" when unknown."""


# ---------------------------------------------------------------------------
# Routing and planning
# ---------------------------------------------------------------------------


#: A rewritten question longer than this is the model answering, not rewriting.
MAX_STANDALONE_CHARS = 400


class Router:
    """Classifies intent, domains and task. Never answers."""

    def __init__(self, models: ModelFactory, domains: dict[str, dict]) -> None:
        self.models = models
        self.domains = domains

    async def classify(
        self,
        query: str,
        *,
        model: str,
        history: Sequence[dict] | None = None,
        memory: str = "",
    ) -> RouterDecision:
        # The digest matters here as much as in the answerer: routing a
        # follow-up like "explain that in more detail" requires knowing what
        # "that" was, and the verbatim window may no longer contain it.
        try:
            chain = prompts.ROUTER_PROMPT | self.models(
                model, max_tokens=700, schema=RouterOutput
            )
            out: RouterOutput | None = await chain.ainvoke(
                {"query": query, "history_block": format_memory(memory) + format_history(history)}
            )
        except Exception as exc:
            logger.warning("router LLM failed (%s) — using keyword fallback", exc)
            return self._keyword_fallback(query)

        if out is None:
            logger.warning("router returned no classification — using keyword fallback")
            return self._keyword_fallback(query)

        task = out.task.strip().lower()
        task = task if task in TASKS else detect_task(query)

        domains = [d.strip().upper() for d in out.domains]
        domains = [d for d in domains if d in self.domains]
        if not domains and out.is_course_related and task == "answer":
            domains = self._keyword_fallback(query).domains

        topics = [
            {"name": t.name.strip(), "domain": t.domain.strip().upper()}
            for t in out.topics
            if t.name.strip() and t.domain.strip().upper() in self.domains
        ][:3]
        depth = (out.requested_depth or "").strip().lower() or None

        intent = out.intent_type.strip().lower()
        return RouterDecision(
            intent_type=intent if intent in ("single", "multi") else "single",
            domains=domains,
            is_course_related=out.is_course_related,
            # A topic-less quiz request is answered from the student's profile
            # (or asked about), not treated as an unanswerable question.
            needs_clarification=out.needs_clarification and not domains and task == "answer",
            clarification_hint=out.clarification_hint,
            reasoning=out.reasoning,
            task=task,
            topics=topics,
            item_count=out.item_count,
            confusion=out.confusion,
            requested_depth=depth if depth in ("concise", "detailed") else None,
            standalone_question=out.standalone_question.strip()[:MAX_STANDALONE_CHARS],
        )

    def _keyword_fallback(self, query: str) -> RouterDecision:
        """Keyword classification for when the LLM is unavailable."""
        lowered = query.lower()
        hits = [
            domain
            for domain, cfg in self.domains.items()
            if any(kw.lower() in lowered for kw in cfg.get("keywords", []))
        ]
        task = detect_task(query)
        return RouterDecision(
            intent_type="multi" if len(hits) > 1 else "single",
            domains=hits,
            is_course_related=bool(hits) or task != "answer",
            needs_clarification=not hits and len(query.split()) < 4 and task == "answer",
            reasoning="keyword fallback (router unavailable)",
            used_fallback=True,
            task=task,
        )


class Planner:
    """Decomposes a multi-topic question into per-domain sub-questions."""

    def __init__(self, models: ModelFactory, domains: dict[str, dict]) -> None:
        self.models = models
        self.domains = domains

    async def plan(
        self, query: str, decision: RouterDecision, *, model: str
    ) -> list[dict[str, str]]:
        if decision.intent_type == "single" or len(decision.domains) <= 1:
            domain = decision.domains[0] if decision.domains else next(iter(self.domains))
            return [{"question": query, "domain": domain}]

        try:
            chain = prompts.PLANNER_PROMPT | self.models(
                model, max_tokens=700, schema=PlannerOutput
            )
            out: PlannerOutput | None = await chain.ainvoke(
                {"query": query, "domains": ", ".join(decision.domains)}
            )
        except Exception as exc:
            logger.warning("planner failed (%s) — one sub-question per domain", exc)
            out = None

        subs = [
            {"question": sq.question.strip(), "domain": sq.domain.strip().upper()}
            for sq in (out.sub_questions if out else [])
            if sq.question.strip()
        ]
        subs = [s for s in subs if s["domain"] in self.domains]

        if not subs:
            # Fall back to asking the original question of each detected domain
            # rather than inventing a split the model did not produce.
            return [{"question": query, "domain": d} for d in decision.domains]
        return subs


# ---------------------------------------------------------------------------
# Corrective retrieval
# ---------------------------------------------------------------------------


class EvidenceGrader:
    """Decides which retrieved excerpts actually address a question."""

    def __init__(self, models: ModelFactory) -> None:
        self.models = models

    async def grade(
        self, question: str, documents: Sequence[Document], *, model: str
    ) -> EvidenceGrade | None:
        """The grade, or None when grading failed (the caller keeps everything)."""
        fence = make_fence()
        try:
            chain = prompts.GRADER_PROMPT | self.models(
                model, max_tokens=300, schema=EvidenceGrade
            )
            return await chain.ainvoke(
                {
                    "question": question,
                    "fence_open": fence.open,
                    "fence_close": fence.close,
                    "evidence_summary": fence.scrub(
                        build_evidence_summary(documents, max_chars=1000, matched=True)
                    ),
                }
            )
        except Exception as exc:
            logger.warning("evidence grading failed (%s) — keeping every excerpt", exc)
            return None


class QueryRewriter:
    """Rephrases a question in the course's terminology for another search."""

    def __init__(self, models: ModelFactory, domains: dict[str, dict]) -> None:
        self.models = models
        self.domains = domains

    async def rewrite(
        self,
        question: str,
        domain: str,
        *,
        tried: Sequence[str],
        missing: str,
        model: str,
    ) -> str | None:
        cfg = self.domains.get(domain)
        if cfg:
            name = f"the {cfg['name']} course materials"
            description = cfg.get("description", "")
        else:
            name = "the student's own uploaded documents"
            description = "Whatever the student uploaded: notes, slides, textbook chapters, papers."
        try:
            chain = prompts.REWRITER_PROMPT | self.models(
                model, max_tokens=200, schema=QueryRewrite
            )
            out: QueryRewrite | None = await chain.ainvoke(
                {
                    "source_name": name,
                    "source_description": description,
                    "question": question,
                    "tried": "; ".join(f'"{q}"' for q in tried) or "none",
                    "missing": missing or "not stated",
                }
            )
        except Exception as exc:
            logger.warning("query rewrite failed (%s)", exc)
            return None
        query = (out.query if out else "").strip()
        if not query or query.lower() in {q.lower() for q in tried}:
            return None
        return query


# ---------------------------------------------------------------------------
# Answering
# ---------------------------------------------------------------------------


class Answerer:
    """Generates one grounded answer from retrieved evidence, or refuses."""

    def __init__(self, models: ModelFactory, domains: dict[str, dict]) -> None:
        self.models = models
        self.domains = domains

    def prepare(
        self,
        question: str,
        domain: str,
        documents: Sequence[Document],
        *,
        history: Sequence[dict] | None = None,
        memory: str = "",
        profile_block: str = "",
        self_study: bool = False,
        tools: bool = False,
    ) -> list[BaseMessage]:
        """The opening messages of the answering conversation, evidence fenced."""
        fence = make_fence()
        evidence = build_evidence(documents, fence)
        variables = {
            "question": question,
            "history_block": context_block(memory, profile_block, history),
            "fence_open": fence.open,
            "fence_close": fence.close,
            "retrieved_chunks": evidence.text,
            "tools_block": prompts.TOOLS_BLOCK if tools else "",
        }
        if self_study:
            return prompts.SELF_STUDY_PROMPT.format_messages(**variables)
        cfg = self.domains.get(domain, {})
        return prompts.ANSWERER_PROMPT.format_messages(
            **variables,
            domain_name=cfg.get("name", domain),
            domain_abbr=cfg.get("abbr", domain),
        )

    async def respond(
        self,
        messages: Sequence[BaseMessage],
        *,
        model: str,
        max_tokens: int,
        tools: Sequence | None = None,
    ) -> AIMessage:
        """
        One model turn. With `tools`, the reply may be a tool call instead of
        text; the caller runs it and calls `respond` again.
        """
        try:
            llm = self.models(model, max_tokens=max_tokens, tools=list(tools) if tools else None)
            reply = await llm.with_config(tags=[ANSWER_STREAM_TAG]).ainvoke(list(messages))
            return reply if isinstance(reply, AIMessage) else AIMessage(content=message_text(reply))
        except Exception as exc:
            from edupilot.security.errors import classify_upstream

            error = classify_upstream(exc)
            error.log()
            return AIMessage(content=f"⚠️ {error.client_message}")


class Synthesizer:
    """Merges per-domain answers. Reorganizes only; never adds facts."""

    def __init__(self, models: ModelFactory, domains: dict[str, dict]) -> None:
        self.models = models
        self.domains = domains

    async def synthesize(
        self, query: str, sub_answers: Sequence[dict], *, model: str, max_tokens: int
    ) -> str:
        answered = [s for s in sub_answers if not s["refused"]]

        if not answered:
            return prompts.NO_EVIDENCE_RESPONSE
        if len(answered) == 1:
            return answered[0]["answer"]

        blocks = "\n\n".join(
            f"### [{s['domain']}] {s['question']}\n\n{s['answer']}" for s in answered
        )
        # A domain that refused is reported as a gap rather than dropped, so
        # the student knows the question was only partly covered.
        refused = [s for s in sub_answers if s["refused"]]
        if refused:
            gaps = "\n".join(f"- {s['domain']}: {s['question']}" for s in refused)
            blocks += f"\n\n### Not covered by the course materials\n{gaps}"

        try:
            llm = self.models(model, max_tokens=max_tokens).with_config(tags=[ANSWER_STREAM_TAG])
            reply = await (prompts.SYNTHESIZER_PROMPT | llm).ainvoke(
                {
                    "original_query": query,
                    "num_parts": len(answered),
                    "num_domains": len({s["domain"] for s in answered}),
                    "sub_answers": blocks,
                }
            )
            return message_text(reply).strip()
        except Exception as exc:
            logger.warning("synthesis failed (%s) — concatenating sub-answers", exc)
            # Concatenation preserves grounding; a failed merge must not lose
            # answers that were already produced and cited.
            return "\n\n".join(
                f"## {self.domains.get(s['domain'], {}).get('name', s['domain'])}\n\n{s['answer']}"
                for s in answered
            )


# ---------------------------------------------------------------------------
# Verification and revision
# ---------------------------------------------------------------------------


class Verifier:
    """LLM audit of an answer against its evidence. Scores only; never rewrites."""

    #: Thresholds the verifier prompt itself states for `is_satisfactory`.
    MIN_GROUNDING = 0.85
    MIN_COVERAGE = 0.60

    def __init__(self, models: ModelFactory) -> None:
        self.models = models

    @classmethod
    def satisfied(cls, verdict: dict) -> bool:
        """Whether a verdict passes, judged from its scores when it has them."""
        grounding, coverage = verdict.get("grounding_score"), verdict.get("coverage_score")
        if grounding is None or coverage is None:
            return bool(verdict.get("is_satisfactory"))
        return grounding >= cls.MIN_GROUNDING and coverage >= cls.MIN_COVERAGE

    async def verify(
        self,
        query: str,
        sub_answers: Sequence[dict],
        answer: str,
        *,
        model: str,
    ) -> dict:
        documents = [d for s in sub_answers for d in s["documents"]]
        if not documents:
            return {"skipped": True, "reason": "no evidence to verify against"}

        fence = make_fence()
        try:
            chain = prompts.VERIFIER_PROMPT | self.models(
                model, max_tokens=1000, schema=VerifierOutput
            )
            out: VerifierOutput | None = await chain.ainvoke(
                {
                    "original_query": query,
                    "sub_questions": "\n".join(
                        f"- [{s['domain']}] {s['question']}" for s in sub_answers
                    ),
                    "fence_open": fence.open,
                    "fence_close": fence.close,
                    "evidence_summary": fence.scrub(build_evidence_summary(documents)),
                    "answer": answer,
                }
            )
        except Exception as exc:
            logger.warning("verifier unavailable: %s", exc)
            return {"skipped": True, "reason": f"verifier unavailable: {exc}"}

        if out is None:
            return {"skipped": True, "reason": "verifier returned no verdict"}

        def clamp(value: float | None) -> float | None:
            if value is None:
                return None
            return max(0.0, min(1.0, float(value)))

        return {
            "skipped": False,
            "grounding_score": clamp(out.grounding_score),
            "coverage_score": clamp(out.coverage_score),
            "quality_score": clamp(out.quality_score),
            "is_satisfactory": out.is_satisfactory,
            "unsupported_claims": out.unsupported_claims[:10],
            "miscited_claims": out.miscited_claims[:10],
            "missing_topics": out.missing_topics[:10],
            "issues": out.issues[:10],
        }


class Reviser:
    """Repairs exactly what the verifier reported, from the same evidence."""

    def __init__(self, models: ModelFactory) -> None:
        self.models = models

    @staticmethod
    def findings(verdict: dict) -> str:
        """The verifier's actionable findings as a prompt block, "" if none."""
        sections = [
            ("Unsupported claims (delete, or restate only what an excerpt says)", "unsupported_claims"),
            ("Miscited claims (fix the [Source N], or delete)", "miscited_claims"),
            ("Parts of the question left unaddressed (cover only if an excerpt does)", "missing_topics"),
        ]  # fmt: skip
        out = []
        for title, key in sections:
            items = verdict.get(key) or []
            if items:
                out.append(f"{title}:\n" + "\n".join(f"- {item}" for item in items))
        return "\n\n".join(out)

    async def revise(
        self,
        question: str,
        answer: str,
        documents: Sequence[Document],
        verdict: dict,
        *,
        model: str,
        max_tokens: int,
    ) -> str | None:
        findings = self.findings(verdict)
        if not findings or not documents:
            return None
        fence = make_fence()
        try:
            llm = self.models(model, max_tokens=max_tokens).with_config(tags=[ANSWER_STREAM_TAG])
            reply = await (prompts.REVISER_PROMPT | llm).ainvoke(
                {
                    "question": question,
                    "fence_open": fence.open,
                    "fence_close": fence.close,
                    "retrieved_chunks": build_evidence(documents, fence).text,
                    "answer": answer,
                    "findings": findings,
                }
            )
        except Exception as exc:
            logger.warning("revision failed (%s) — keeping the draft", exc)
            return None
        return message_text(reply).strip() or None


# ---------------------------------------------------------------------------
# Tutor
# ---------------------------------------------------------------------------


class Tutor:
    """Writes practice quizzes and flashcards from retrieved evidence."""

    def __init__(self, models: ModelFactory) -> None:
        self.models = models

    async def write(
        self,
        task: str,
        topic: str,
        documents: Sequence[Document],
        *,
        count: int,
        focus: Sequence[str] = (),
        model: str,
        max_tokens: int,
    ) -> dict | None:
        """The quiz or deck as practice data (see `practice_items`); None on failure."""
        fence = make_fence()
        schema, prompt, kind = (
            (FlashcardsOutput, prompts.FLASHCARDS_PROMPT, "flashcards")
            if task == "flashcards"
            else (QuizOutput, prompts.QUIZ_PROMPT, "multiple-choice questions")
        )
        focus_block = (
            f"The student has found these topics difficult: {', '.join(focus)}.\n" if focus else ""
        )
        try:
            chain = prompt | self.models(model, max_tokens=max_tokens, schema=schema)
            out = await chain.ainvoke(
                {
                    "count": count,
                    "kind": kind,
                    "topic": topic,
                    "focus_block": focus_block,
                    "fence_open": fence.open,
                    "fence_close": fence.close,
                    "retrieved_chunks": build_evidence(documents, fence).text,
                }
            )
        except Exception as exc:
            logger.warning("tutor generation failed (%s)", exc)
            return None
        if out is None:
            return None
        return practice_items(out, task, topic, len(documents))


#: Most answer options a quiz question keeps.
MAX_OPTIONS = 6


def practice_items(
    out: QuizOutput | FlashcardsOutput, task: str, topic: str, n_sources: int
) -> dict | None:
    """
    Validated quiz or flashcard data, or None when nothing usable came back.

    The one place model output is checked: questions with fewer than two
    options or an answer index out of range are dropped, and a source number
    that names no excerpt becomes None. The frontend renders this
    interactively; `render_practice` turns the same data into markdown for
    the stored conversation.

        {"type": "quiz", "title": ..., "items": [
            {"question", "options", "answer_index", "explanation", "source"}]}
        {"type": "flashcards", "title": ..., "items": [{"front", "back", "source"}]}
    """

    def source(n: int) -> int | None:
        return n if 1 <= n <= n_sources else None

    if task == "flashcards":
        items = [
            {"front": c.front.strip(), "back": c.back.strip(), "source": source(c.source)}
            for c in out.cards
            if c.front.strip() and c.back.strip()
        ]
        title = out.title.strip() or f"Flashcards: {topic}"
        kind = "flashcards"
    else:
        items = []
        for q in out.questions:
            options = [o.strip() for o in q.options if o.strip()][:MAX_OPTIONS]
            if len(options) < 2 or not 0 <= q.answer_index < len(options):
                continue
            items.append(
                {
                    "question": q.question.strip(),
                    "options": options,
                    "answer_index": q.answer_index,
                    "explanation": q.explanation.strip(),
                    "source": source(q.source),
                }
            )
        title = out.title.strip() or f"Practice quiz: {topic}"
        kind = "quiz"
    if not items:
        return None
    return {"type": kind, "title": title, "items": items}


def _cite(source: int | None) -> str:
    return f" [Source {source}]" if source else ""


def render_practice(practice: dict) -> str:
    """
    Practice data as markdown, answers folded into <details> so the student
    can try first. This is what the conversation stores, and what a client
    without an interactive view shows.
    """
    if practice["type"] == "flashcards":
        blocks = [
            f"<details><summary><b>{i}.</b> {card['front']}</summary>\n\n"
            f"{card['back']}{_cite(card['source'])}\n\n</details>"
            for i, card in enumerate(practice["items"], start=1)
        ]
    else:
        letters = "ABCDEF"
        blocks = []
        for i, q in enumerate(practice["items"], start=1):
            lines = [f"**{i}. {q['question']}**", ""]
            lines += [f"- **{letters[j]}.** {opt}" for j, opt in enumerate(q["options"])]
            lines += [
                "",
                "<details><summary>Show answer</summary>",
                "",
                f"**{letters[q['answer_index']]}** — {q['explanation']}{_cite(q['source'])}",
                "",
                "</details>",
            ]
            blocks.append("\n".join(lines))
    return f"## {practice['title']}\n\n" + "\n\n".join(blocks)


__all__ = [
    "ANSWER_STREAM_TAG",
    "TASKS",
    "Answerer",
    "EvidenceGrader",
    "ModelFactory",
    "Planner",
    "QueryRewriter",
    "Reviser",
    "Router",
    "RouterDecision",
    "Synthesizer",
    "Tutor",
    "Verifier",
    "context_block",
    "detect_task",
    "practice_items",
    "render_practice",
    "topic_of",
]
