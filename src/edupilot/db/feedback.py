"""
Answer feedback
===============
A student's 👍/👎 on an answer, from course chat or Self Study.

Stored here whether or not LangSmith tracing is on, so ratings are never lost
to a missing key. One row per (answer, student): rating again replaces the
earlier vote rather than adding one.
"""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from .engine import get_engine, transaction
from .tables import (
    answer_feedback,
    chat_messages,
    chat_sessions,
    self_study_messages,
    self_study_sessions,
)

#: kind -> (messages table, sessions table, session id column)
_SOURCES = {
    "chat": (chat_messages, chat_sessions, "session_id"),
    "self_study": (self_study_messages, self_study_sessions, "ss_session_id"),
}


def get_rated_message(kind: str, message_id: int) -> dict | None:
    """
    An assistant message with its owner and trace, or None if there is none.

    Returns `{"owner": user_id, "run_id": str | None}`. The run id comes from
    what the server stored with the answer, never from the client.
    """
    messages, sessions, key = _SOURCES[kind]
    query = (
        select(messages.c.pipeline_meta, sessions.c.user_id)
        .join(sessions, sessions.c[key] == messages.c[key])
        .where(messages.c.id == message_id, messages.c.role == "assistant")
    )
    with get_engine().connect() as conn:
        row = conn.execute(query).first()
    if row is None:
        return None
    trace = (row.pipeline_meta or {}).get("trace") or {}
    return {"owner": row.user_id, "run_id": trace.get("run_id")}


def save_feedback(
    *,
    kind: str,
    message_id: int,
    user_id: str,
    rating: int,
    comment: str | None,
    run_id: str | None,
) -> bool:
    """Record a rating. Returns True when it replaced an earlier one."""
    statement = pg_insert(answer_feedback).values(
        kind=kind,
        message_id=message_id,
        user_id=user_id,
        rating=rating,
        comment=comment,
        run_id=run_id,
    )
    statement = statement.on_conflict_do_update(
        index_elements=["kind", "message_id", "user_id"],
        set_={
            "rating": statement.excluded.rating,
            "comment": statement.excluded.comment,
            "run_id": statement.excluded.run_id,
            "created_at": func.now(),
        },
    )
    with transaction() as conn:
        existed = conn.execute(
            select(answer_feedback.c.id).where(
                answer_feedback.c.kind == kind,
                answer_feedback.c.message_id == message_id,
                answer_feedback.c.user_id == user_id,
            )
        ).first()
        conn.execute(statement)
    return existed is not None
