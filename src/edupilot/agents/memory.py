"""
Rolling conversation memory
===========================
A conversation that runs long cannot be sent to the model verbatim: prompt cost
grows with every turn, and the oldest turns get silently truncated by whatever
window is in force — so the model forgets the beginning of a conversation
precisely when the student starts referring back to it.

This keeps the last `KEEP_RECENT_TURNS` messages verbatim and folds everything
older into a running digest. Folding is incremental: each pass summarizes only
the messages that have fallen out of the verbatim window since the last one,
so summarization cost stays constant no matter how long the conversation gets.

Where the state lives changed with the move to LangGraph. The messages and the
digest are now graph state, persisted per conversation by the checkpointer
(`thread_id = session_id`), and `fold_memory` runs as the graph's first node.
Folded messages leave the state through `RemoveMessage`. The `messages` table
is still the record the UI reads and edits; `seed_from_db` rebuilds a thread
from it when the thread has no checkpoint — a conversation from before this
change, or one whose checkpoint was dropped because the student edited an
earlier message.

The digest is *conversation state*, not retrieved evidence. It never licenses a
factual claim — every claim in an answer must still cite a source excerpt. Its
job is to resolve references ("explain that in more detail", "how does it
compare to the one you mentioned?") that would otherwise be unanswerable.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from edupilot import db
from edupilot.llm import message_text

logger = logging.getLogger(__name__)

#: Messages kept verbatim. Below this, no summarization happens at all — short
#: conversations pay nothing.
KEEP_RECENT_TURNS = 6

#: Summarize only once this many un-folded messages have accumulated beyond the
#: verbatim window. Batching keeps it to roughly one extra LLM call per eight
#: turns rather than one per turn.
SUMMARIZE_BATCH = 6

#: Hard cap on digest length. A digest that grows every pass would recreate the
#: unbounded-prompt problem it exists to solve.
MAX_DIGEST_CHARS = 1200

#: Output budget for one summarization call.
SUMMARY_MAX_TOKENS = 400

_SUMMARY_SYSTEM = """\
You maintain a running summary of a tutoring conversation so the tutor can \
follow back-references in later questions.

Write a compact digest capturing:
- what topics the student has asked about, in order
- specific concepts, algorithms, or terms that were explained
- anything the student stated about themselves (their course, what they are \
studying for, what they said they already understand)
- questions that were refused or left unresolved

Rules:
- Under 150 words. Prose or terse bullets, no preamble.
- Record what was discussed, not the explanations themselves. This is an index \
to the conversation, not a replacement for it.
- Never invent. If the transcript does not say it, it does not go in.
- If an existing digest is supplied, merge the new turns into it and return the \
merged result — do not append a second summary.\
"""

_SUMMARY_USER = """\
{existing_block}<new_turns>
{turns}
</new_turns>

Return only the merged digest.\
"""

SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [("system", _SUMMARY_SYSTEM), ("human", _SUMMARY_USER)]
)

Summarize = Callable[[str, list[dict]], Awaitable[str]]
"""`(existing_digest, turns) -> merged_digest`, "" on failure."""


# ---------------------------------------------------------------------------
# Conversions
# ---------------------------------------------------------------------------


def to_turns(messages: Sequence[BaseMessage]) -> list[dict]:
    """LangChain messages as `{role, content}` dicts, for `format_history`."""
    return [
        {"role": "user" if m.type == "human" else "assistant", "content": message_text(m)}
        for m in messages
    ]


def from_db_rows(rows: Sequence[dict]) -> list[BaseMessage]:
    """Database message rows as LangChain messages, ids tied to their row."""
    return [
        (HumanMessage if r["role"] == "user" else AIMessage)(
            content=str(r["content"]), id=f"db-{r['id']}"
        )
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Folding
# ---------------------------------------------------------------------------


@dataclass
class MemoryUpdate:
    """What one memory pass decided."""

    summary: str
    recent: list[dict]
    """History for this turn's prompts, `{role, content}`, oldest first."""
    remove_ids: list[str] = field(default_factory=list)
    """Ids of messages folded into `summary`, to drop from graph state."""
    failed: bool = False


async def fold_memory(
    messages: Sequence[BaseMessage], summary: str, summarize: Summarize
) -> MemoryUpdate:
    """
    Fold messages that have left the verbatim window into the digest.

    `messages` are the conversation's prior turns — the question being answered
    now is not among them.
    """
    if len(messages) <= KEEP_RECENT_TURNS + SUMMARIZE_BATCH:
        return MemoryUpdate(summary=summary, recent=to_turns(messages))

    to_fold = list(messages[:-KEEP_RECENT_TURNS])
    recent = to_turns(messages[-KEEP_RECENT_TURNS:])

    merged = await summarize(summary, to_turns(to_fold))
    if not merged:
        # Degrade to verbatim recent turns rather than dropping context
        # silently. Nothing is removed, so the next turn retries the fold.
        logger.warning("digest update failed; using recent turns only")
        return MemoryUpdate(summary=summary, recent=recent, failed=True)

    return MemoryUpdate(
        summary=merged, recent=recent, remove_ids=[m.id for m in to_fold if m.id]
    )


async def summarize_turns(llm: Runnable, existing: str, turns: list[dict]) -> str:
    """One LLM call folding `turns` into `existing`. Returns "" on failure."""
    transcript = "\n".join(
        f"{'Student' if t['role'] == 'user' else 'Assistant'}: "
        f"{str(t['content']).strip()[:800]}"
        for t in turns
        if str(t.get("content", "")).strip()
    )
    if not transcript:
        return existing

    existing_block = (
        f"<existing_digest>\n{existing}\n</existing_digest>\n\n" if existing else ""
    )
    try:
        out = await (SUMMARY_PROMPT | llm).ainvoke(
            {"existing_block": existing_block, "turns": transcript}
        )
    except Exception:
        logger.warning("conversation summarization failed", exc_info=True)
        return ""

    return message_text(out).strip()[:MAX_DIGEST_CHARS]


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------


def seed_from_db(
    session_id: str, *, exclude_message_ids: set[int] | None = None
) -> tuple[str, list[BaseMessage]]:
    """
    `(digest, messages)` to start a thread that has no checkpoint.

    Uses the digest stored on the session by the pre-LangGraph memory, when
    there is one, plus every message after the point it covers.
    """
    if not session_id:
        return "", []
    digest, through_id = db.get_session_memory(session_id)
    rows = db.get_messages_after(session_id, through_id)
    if exclude_message_ids:
        rows = [r for r in rows if r["id"] not in exclude_message_ids]
    return digest, from_db_rows(rows)


def format_memory(digest: str) -> str:
    """
    Render the digest as a prompt block.

    Labelled as conversation state and explicitly denied evidentiary weight —
    otherwise a model that sees "we discussed bagging" may treat that as
    licence to assert facts about bagging without a source excerpt.
    """
    if not digest.strip():
        return ""
    return (
        "<conversation_memory>\n"
        "Summary of earlier turns in this conversation. Use it only to "
        "understand what the student is referring back to. It is NOT evidence "
        "and can never support a factual claim — only source excerpts can.\n"
        f"{digest.strip()}\n"
        "</conversation_memory>\n"
    )


__all__ = [
    "KEEP_RECENT_TURNS",
    "MAX_DIGEST_CHARS",
    "SUMMARIZE_BATCH",
    "SUMMARY_PROMPT",
    "MemoryUpdate",
    "fold_memory",
    "format_memory",
    "from_db_rows",
    "seed_from_db",
    "summarize_turns",
    "to_turns",
]
