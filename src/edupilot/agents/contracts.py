"""
Prompt contracts
================
How untrusted document text is packaged into a prompt, and how agent output
is parsed back out.

The central mechanism is the **nonce fence**. Retrieved chunks are wrapped in
a delimiter containing a random token generated per request:

    <<<SOURCE-MATERIAL a3f9c1e2>>>
    [Source 1] ...
    <<<END-SOURCE-MATERIAL a3f9c1e2>>>

A static delimiter like the previous `--- COURSE SOURCE MATERIAL ---` can be
forged: a PDF containing that exact line followed by "--- END ---" and then
its own instructions appears to the model to be outside the data region. The
attacker cannot predict a per-request nonce, so the fence cannot be closed
early. Any occurrence of the nonce inside the content itself is stripped
before assembly, which closes the reflection case.

This is defence in depth, not a complete solution — it pairs with the explicit
instruction hierarchy in the system prompt and the injection scanner in
`guardrails/injection.py`.

The router, planner and verifier return JSON. Their shapes are declared below
as Pydantic models and bound to the model with `with_structured_output`, so
the provider enforces the shape instead of `parse_json_response` repairing it
after the fact.
"""

from __future__ import annotations

import re
import secrets
from collections.abc import Sequence
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from edupilot.core.config import DOMAINS

# Deliberate re-export, not dead weight: `agents/__init__.py` pulls these from
# here so agent code has a single import surface for the contract types and the
# refusal protocol together. Unused *in this module*, which is why they need the
# noqa — an automated "unused import" cleanup will otherwise delete them and
# break `from edupilot.agents import ...` at runtime.
from edupilot.guardrails.refusal import (  # noqa: F401
    REFUSAL_MARKER,
    is_refusal,
    strip_refusal_marker,
)
from edupilot.ingestion.titles import title_from_filename

#: Maximum characters of a single chunk placed in a prompt. Chunks are already
#: token-bounded by the chunker; this is a defensive cap for anything that
#: reaches the prompt by another path.
MAX_CHUNK_CHARS = 6000

#: Budget across every chunk in one prompt, ~4k tokens.
#:
#: One request runs several LLM calls (router, planner, answerer, verifier,
#: and occasionally the memory summarizer), and providers meter tokens per
#: minute across all of them. Sized so a full request stays inside a modest
#: free-tier allowance; raise it if your provider limit is higher.
TOTAL_EVIDENCE_CHARS = 16000

#: Floor on each chunk's share, so a large `top_k` still yields usable
#: excerpts rather than unreadable slivers.
MIN_CHUNK_SHARE = 900


@dataclass
class SourceFence:
    """A per-request delimiter pair for untrusted content."""

    nonce: str

    @property
    def open(self) -> str:
        return f"<<<SOURCE-MATERIAL {self.nonce}>>>"

    @property
    def close(self) -> str:
        return f"<<<END-SOURCE-MATERIAL {self.nonce}>>>"

    def scrub(self, text: str) -> str:
        """Remove any occurrence of this fence's nonce from untrusted text."""
        if self.nonce in text:
            return text.replace(self.nonce, "[redacted]")
        return text


def make_fence() -> SourceFence:
    """Generate a fresh fence. Call once per request, never reuse."""
    return SourceFence(nonce=secrets.token_hex(4))


# ---------------------------------------------------------------------------
# Citation labels
# ---------------------------------------------------------------------------

_ACRONYM = re.compile(r"^[A-Z][A-Z0-9]*s?$")


def citation_label(metadata: dict) -> str:
    """
    Human-readable source label, e.g. "LLMs 04 Attention Transformers, p.41".

    Preserves acronyms (ML, LLM, BCNF, SP26) that naive title-casing would
    mangle into "Ml" and "Bcnf".
    """
    title = str(metadata.get("doc_title") or "").strip()
    if not title:
        title = title_from_filename(str(metadata.get("filename") or "source"))

    if metadata.get("source_type") == "book":
        # Told apart from the lecture slides in the citation itself, so the
        # model and the student both know which kind of source this is.
        title = f"{title} (book)"

    page = metadata.get("page_number")
    start, end = metadata.get("page_start"), metadata.get("page_end")

    if start and end and start != end:
        return f"{title}, pp.{start}-{end}"
    if page:
        return f"{title}, p.{page}"
    return title


# ---------------------------------------------------------------------------
# Evidence formatting
# ---------------------------------------------------------------------------


@dataclass
class EvidenceBlock:
    """The formatted evidence for one request, plus the mapping back to sources."""

    text: str
    fence: SourceFence
    labels: list[str] = field(default_factory=list)
    """`labels[i]` is the citation label for [Source i+1]."""
    chunk_ids: list[str] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.labels)


def chunk_text(chunk) -> str:
    """Text of a `RerankedChunk` (`.text`) or a LangChain `Document` (`.page_content`)."""
    text = getattr(chunk, "text", None)
    if text is None:
        text = getattr(chunk, "page_content", "")
    return str(text)


def build_evidence(
    chunks: Sequence,
    fence: SourceFence | None = None,
    *,
    max_chars: int = MAX_CHUNK_CHARS,
    total_max_chars: int = TOTAL_EVIDENCE_CHARS,
) -> EvidenceBlock:
    """
    Format retrieved chunks as numbered, fenced source blocks.

    Args:
        chunks: LangChain `Document`s, or objects exposing `.text`,
            `.metadata`, and optionally `.chunk_id` (a `RerankedChunk`).
        fence: Fence to use. A fresh one is generated when omitted.
        total_max_chars: Budget across *all* chunks. Each chunk gets an equal
            share, so prompt size is bounded by this rather than by how large
            the retrieved passages happen to be.

    Returns:
        An `EvidenceBlock` whose `labels` list maps [Source N] back to a
        citation string — the mapping the citation validator checks against.
    """
    fence = fence or make_fence()
    if not chunks:
        return EvidenceBlock(text="(no source material retrieved)", fence=fence)

    # Budget per chunk rather than per chunk *alone*. A per-chunk cap does not
    # bound the prompt: eight parent-expanded passages at the 6000-char cap is
    # ~12k tokens of evidence, and with the router, planner, answerer and
    # verifier all firing inside one minute that alone trips a provider's
    # tokens-per-minute ceiling — which surfaces as a 413 and an empty answer.
    # An equal share keeps every source citable while making size predictable.
    per_chunk = min(max_chars, max(MIN_CHUNK_SHARE, total_max_chars // len(chunks)))

    parts: list[str] = []
    labels: list[str] = []
    ids: list[str] = []

    for i, chunk in enumerate(chunks, start=1):
        meta = dict(getattr(chunk, "metadata", {}) or {})
        label = citation_label(meta)
        raw = chunk_text(chunk)
        clipped = raw[:per_chunk]
        if len(raw) > per_chunk:
            clipped += "\n…(excerpt truncated)"
        body = fence.scrub(clipped)

        labels.append(label)
        ids.append(str(getattr(chunk, "chunk_id", None) or meta.get("chunk_id") or f"chunk-{i}"))

        header = f"[Source {i}] {label}"
        section = str(meta.get("section_path") or "")
        if section:
            header += f" — section: {section}"
        # Flag OCR-derived text so the model can hedge on garbled passages.
        if float(meta.get("confidence", 1.0)) < 1.0:
            header += " — (recovered by OCR; text may contain errors)"

        parts.append(f"{header}\n{body}")

    return EvidenceBlock(
        text="\n\n".join(parts), fence=fence, labels=labels, chunk_ids=ids
    )


def build_evidence_summary(chunks: Sequence, *, max_chars: int = 700, matched: bool = False) -> str:
    """
    Compact evidence rendering for the verifier, which needs breadth over depth.

    With `matched`, each excerpt shows the passage the search matched rather
    than the start of its section. The grader needs that: an excerpt is a
    whole section, often thousands of characters, and its opening is rarely
    the part that addresses the question. Shown the first 700 characters of a
    section the reranker had scored 0.87, the grader judged it irrelevant.
    """
    if not chunks:
        return "(no evidence retrieved)"
    lines = []
    for i, chunk in enumerate(chunks, start=1):
        meta = dict(getattr(chunk, "metadata", {}) or {})
        text = (matched and meta.get("matched_text")) or chunk_text(chunk)
        clipped = text[:max_chars] + ("…" if len(text) > max_chars else "")
        lines.append(f"[Source {i}] {citation_label(meta)}\n{clipped}")
    return "\n\n".join(lines)


def format_history(history: Sequence[dict] | None, *, max_turns: int = 6) -> str:
    """
    Render recent conversation turns as a prompt block.

    History is prior model output and student text — not retrieved documents —
    so it sits outside the source fence, but it is still untrusted and is
    labelled as context rather than instruction.
    """
    if not history:
        return ""
    recent = list(history)[-max_turns:]
    lines = ["<conversation_context>"]
    for msg in recent:
        role = "Student" if msg.get("role") == "user" else "Assistant"
        content = str(msg.get("content", "")).strip()
        if content:
            lines.append(f"{role}: {content[:1500]}")
    lines.append("</conversation_context>")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Structured outputs
# ---------------------------------------------------------------------------
# Field types are deliberately lenient (plain `str` rather than a Literal of
# domain codes): a model that writes "Algo" instead of "ALGO" should be
# normalized by the agent, not rejected outright and pushed down the fallback
# chain for a cosmetic difference.

#: "AML, ALGO or LLM" — the configured domain codes, for field descriptions.
_CODES = ", ".join(list(DOMAINS)[:-1]) + f" or {list(DOMAINS)[-1]}"


class TopicRef(BaseModel):
    name: str = Field(description="A short concept name, e.g. 'merge sort' or 'L2 regularization'")
    domain: str = Field(description=f"Domain code: {_CODES}")


class RouterOutput(BaseModel):
    """Classification of a student message before retrieval."""

    intent_type: str = Field(default="single", description='"single" or "multi"')
    domains: list[str] = Field(
        default_factory=list, description=f"Domain codes ({', '.join(DOMAINS)}), most relevant first"
    )
    is_course_related: bool = True
    needs_clarification: bool = False
    clarification_hint: str | None = None
    reasoning: str = Field(default="", description="One sentence explaining the classification")
    task: str = Field(default="answer", description='"answer", "quiz" or "flashcards"')
    topics: list[TopicRef] = Field(
        default_factory=list, description="The 1-3 concepts the message is about"
    )
    item_count: int | None = Field(
        default=None, description="How many quiz questions or flashcards the student asked for"
    )
    confusion: bool = Field(
        default=False,
        description="The student says an earlier explanation did not make sense, or asks for it again",
    )
    requested_depth: str | None = Field(
        default=None, description='"concise" or "detailed" when the student states a preference'
    )
    standalone_question: str = Field(
        default="",
        description="The question rewritten to make sense without the conversation; unchanged when it already does",
    )


class EvidenceGrade(BaseModel):
    """Which retrieved excerpts actually address a question."""

    relevant_sources: list[int] = Field(
        default_factory=list, description="Numbers of the excerpts that address the question"
    )
    sufficient: bool = Field(
        default=False, description="The relevant excerpts answer the whole question"
    )
    missing: str = Field(default="", description="What the excerpts do not cover, if anything")


class QueryRewrite(BaseModel):
    """A search query rephrased in the course's own terminology."""

    query: str
    reasoning: str = ""


class QuizQuestion(BaseModel):
    question: str
    options: list[str] = Field(description="Two to five answer options")
    answer_index: int = Field(description="Zero-based index of the correct option")
    explanation: str = Field(description="Why that option is correct, from the excerpt")
    source: int = Field(description="Number of the excerpt that establishes the answer")


class QuizOutput(BaseModel):
    """Multiple-choice practice questions written from the excerpts."""

    title: str = ""
    questions: list[QuizQuestion] = Field(default_factory=list)


class Flashcard(BaseModel):
    front: str = Field(description="A term or a short question")
    back: str = Field(description="The definition or answer, from the excerpt")
    source: int = Field(description="Number of the excerpt it comes from")


class FlashcardsOutput(BaseModel):
    """Study flashcards written from the excerpts."""

    title: str = ""
    cards: list[Flashcard] = Field(default_factory=list)


class SubQuestion(BaseModel):
    question: str = Field(description="A self-contained question")
    domain: str = Field(description=f"Exactly one domain code: {_CODES}")
    reasoning: str = ""


class PlannerOutput(BaseModel):
    """A multi-topic question decomposed into per-domain sub-questions."""

    sub_questions: list[SubQuestion] = Field(default_factory=list)


class VerifierOutput(BaseModel):
    """An audit of an answer against the evidence it was built from."""

    grounding_score: float | None = Field(default=None, description="0.0-1.0")
    coverage_score: float | None = Field(default=None, description="0.0-1.0")
    quality_score: float | None = Field(default=None, description="0.0-1.0")
    is_satisfactory: bool = False
    unsupported_claims: list[str] = Field(default_factory=list)
    miscited_claims: list[str] = Field(default_factory=list)
    missing_topics: list[str] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

# `is_refusal` and `strip_refusal_marker` are re-exported from
# guardrails.refusal so callers can import everything prompt-related from one
# place without agents and guardrails importing each other.


_JSON_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.MULTILINE)


def parse_json_response(raw: str) -> dict:
    """
    Parse a JSON object from a model response.

    Handles markdown fences and leading prose. Returns {} on failure so
    callers can fall back rather than raise.
    """
    import json

    if not raw:
        return {}

    cleaned = _JSON_FENCE.sub("", raw).strip()
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass

    # Fall back to the outermost brace-balanced span.
    start = cleaned.find("{")
    if start == -1:
        return {}
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(cleaned)):
        ch = cleaned[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    parsed = json.loads(cleaned[start : i + 1])
                    return parsed if isinstance(parsed, dict) else {}
                except json.JSONDecodeError:
                    return {}
    return {}
