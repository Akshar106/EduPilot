"""
Rolling conversation memory.

The folding logic is tested directly with stub summarizers; seeding runs
against the throwaway database conftest pins DATA_DIR to. How the graph uses
both across turns is covered in test_agent_graph.py.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

from edupilot import db
from edupilot.agents.memory import (
    KEEP_RECENT_TURNS,
    MAX_DIGEST_CHARS,
    SUMMARIZE_BATCH,
    fold_memory,
    format_memory,
    seed_from_db,
    summarize_turns,
)
from fakes import ScriptedChatModel


class StubSummarizer:
    """Records its calls and returns a canned digest."""

    def __init__(self, reply: str = "Student asked about bias and variance."):
        self.reply = reply
        self.calls: list[tuple[str, list[dict]]] = []

    async def __call__(self, existing: str, turns: list[dict]) -> str:
        self.calls.append((existing, turns))
        return self.reply


@pytest.fixture(autouse=True)
def _schema():
    db.init_db()


def _conversation(turns: int) -> list:
    messages = []
    for i in range(turns):
        messages.append(HumanMessage(content=f"question {i}", id=f"h{i}"))
        messages.append(AIMessage(content=f"answer {i}", id=f"a{i}"))
    return messages


def _session(sid: str, turns: int) -> str:
    db.ensure_session(sid, "test-user")
    for i in range(turns):
        db.save_message(sid, "user", f"question {i}")
        db.save_message(sid, "assistant", f"answer {i}")
    return sid


# ---------------------------------------------------------------------------
# Short conversations pay nothing
# ---------------------------------------------------------------------------


async def test_a_short_conversation_is_never_summarized():
    summarize = StubSummarizer()
    update = await fold_memory(_conversation(2), "", summarize)

    assert update.summary == ""
    assert summarize.calls == [], "summarization should not run on a short conversation"
    assert len(update.recent) == 4
    assert update.remove_ids == []


async def test_an_empty_conversation_returns_empty_context():
    update = await fold_memory([], "", StubSummarizer())
    assert update.summary == "" and update.recent == []


async def test_a_conversation_at_the_threshold_is_kept_verbatim():
    summarize = StubSummarizer()
    turns = (KEEP_RECENT_TURNS + SUMMARIZE_BATCH) // 2
    update = await fold_memory(_conversation(turns), "", summarize)
    assert summarize.calls == []
    assert len(update.recent) == KEEP_RECENT_TURNS + SUMMARIZE_BATCH


# ---------------------------------------------------------------------------
# Rollover
# ---------------------------------------------------------------------------


async def test_a_long_conversation_gets_a_digest_and_a_short_tail():
    summarize = StubSummarizer()
    messages = _conversation(12)  # 24 messages

    update = await fold_memory(messages, "", summarize)

    assert update.summary == summarize.reply
    assert len(summarize.calls) == 1
    assert len(update.recent) == KEEP_RECENT_TURNS, "tail must be exactly the verbatim window"
    # The tail is the newest messages, and everything older leaves the state.
    assert update.recent[-1]["content"] == "answer 11"
    assert update.remove_ids == [m.id for m in messages[:-KEEP_RECENT_TURNS]]


async def test_the_existing_digest_is_carried_into_the_next_fold():
    summarize = StubSummarizer()
    await fold_memory(_conversation(12), "earlier digest", summarize)

    existing, turns = summarize.calls[0]
    assert existing == "earlier digest"
    assert turns[0] == {"role": "user", "content": "question 0"}


# ---------------------------------------------------------------------------
# Failure and truncation behaviour
# ---------------------------------------------------------------------------


async def test_a_failing_summarizer_degrades_to_recent_turns():
    update = await fold_memory(_conversation(12), "", StubSummarizer(reply=""))

    assert update.summary == "", "no digest is better than a wrong one"
    assert update.failed
    assert len(update.recent) == KEEP_RECENT_TURNS, "the conversation must still have context"
    assert update.remove_ids == [], "a failed fold must not drop messages, so it is retried"


async def test_summarize_turns_merges_into_the_existing_digest():
    seen: list[str] = []

    def reply(messages):
        seen.append(str(messages[-1].content))
        return "merged digest"

    out = await summarize_turns(
        ScriptedChatModel(reply=reply), "old digest", [{"role": "user", "content": "q"}]
    )
    assert out == "merged digest"
    assert "<existing_digest>" in seen[0] and "old digest" in seen[0]


async def test_summarize_turns_returns_empty_on_failure():
    def boom(_):
        raise RuntimeError("provider down")

    out = await summarize_turns(RunnableLambda(boom), "", [{"role": "user", "content": "q"}])
    assert out == ""


async def test_the_digest_is_length_capped():
    llm = ScriptedChatModel(reply=lambda _: "x" * (MAX_DIGEST_CHARS * 3))
    out = await summarize_turns(llm, "", [{"role": "user", "content": "q"}])
    assert len(out) <= MAX_DIGEST_CHARS


async def test_nothing_to_summarize_keeps_the_existing_digest():
    out = await summarize_turns(ScriptedChatModel(reply=lambda _: "new"), "kept", [])
    assert out == "kept"


# ---------------------------------------------------------------------------
# Seeding a thread from the database
# ---------------------------------------------------------------------------


def test_seeding_reads_the_whole_conversation():
    sid = _session("seed-all", turns=3)
    digest, messages = seed_from_db(sid)
    assert digest == ""
    assert [m.content for m in messages][:2] == ["question 0", "answer 0"]
    assert [m.type for m in messages][:2] == ["human", "ai"]
    assert all(m.id.startswith("db-") for m in messages)


def test_seeding_starts_after_a_legacy_digest():
    sid = _session("seed-digest", turns=3)
    rows = db.get_session_messages(sid)
    db.set_session_memory(sid, "legacy digest", rows[1]["id"])

    digest, messages = seed_from_db(sid)
    assert digest == "legacy digest"
    assert len(messages) == 4, "only messages after the digest's pointer are loaded"


def test_seeding_excludes_the_question_being_answered():
    sid = _session("seed-exclude", turns=1)
    mid = db.save_message(sid, "user", "the question being answered right now")
    _, messages = seed_from_db(sid, exclude_message_ids={mid})
    assert all(m.content != "the question being answered right now" for m in messages)


def test_seeding_with_no_session_is_empty():
    assert seed_from_db("") == ("", [])


# ---------------------------------------------------------------------------
# Truncation must invalidate a legacy digest that covered deleted turns
# ---------------------------------------------------------------------------


def test_editing_an_early_message_drops_a_digest_that_covered_it():
    sid = _session("mem-truncate", turns=12)
    through = db.get_session_messages(sid)[-7]["id"]
    db.set_session_memory(sid, "digest", through)

    # Student edits a message the digest already folded in.
    db.delete_messages_from(sid, through - 1)

    stored_after, through_after = db.get_session_memory(sid)
    assert stored_after == "" and through_after == 0


def test_truncating_a_recent_message_keeps_the_digest():
    """Only a digest that actually covered the deleted turns is dropped."""
    sid = _session("mem-truncate-late", turns=12)
    through = db.get_session_messages(sid)[-7]["id"]
    db.set_session_memory(sid, "digest", through)

    db.delete_messages_from(sid, through + 5)

    stored_after, _ = db.get_session_memory(sid)
    assert stored_after == "digest"


# ---------------------------------------------------------------------------
# Prompt block
# ---------------------------------------------------------------------------


def test_empty_digest_renders_nothing():
    assert format_memory("") == ""
    assert format_memory("   ") == ""


def test_the_memory_block_denies_itself_evidentiary_weight():
    block = format_memory("We discussed bagging.")
    assert "<conversation_memory>" in block
    assert "NOT evidence" in block
    assert "We discussed bagging." in block
