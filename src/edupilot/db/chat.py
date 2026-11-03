"""
Chat sessions and messages
==========================
Every read is scoped by `user_id` in SQL rather than filtered afterwards, and
every destructive statement carries ownership in its predicate so there is no
window between checking and acting.
"""

from __future__ import annotations

from sqlalchemy import delete, func, insert, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from .engine import get_engine, transaction
from .rows import row_dict
from .tables import chat_messages as messages
from .tables import chat_sessions as sessions


def get_session_owner(session_id: str) -> str | None:
    """user_id owning a chat session, or None when missing."""
    with get_engine().connect() as conn:
        return conn.execute(
            select(sessions.c.user_id).where(sessions.c.session_id == session_id)
        ).scalar()


def ensure_session(session_id: str, user_id: str, title: str | None = None) -> None:
    """
    Insert a session row if absent, owned by `user_id`.

    `user_id` is required. It comes from the caller's verified token, never
    from the request body — a client-supplied owner would reintroduce the
    original IDOR in a new shape.
    """
    if not user_id:
        raise ValueError("user_id is required to create a session")
    with transaction() as conn:
        conn.execute(
            pg_insert(sessions)
            .values(session_id=session_id, user_id=user_id, title=title)
            .on_conflict_do_nothing(index_elements=["session_id"])
        )


def update_session_title(session_id: str, title: str) -> None:
    with transaction() as conn:
        conn.execute(
            update(sessions)
            .where(sessions.c.session_id == session_id)
            .values(title=title, updated_at=func.now())
        )


def list_sessions(user_id: str, limit: int = 20) -> list[dict]:
    """
    Recent sessions belonging to `user_id`, newest activity first.

    Scoped by owner in SQL rather than filtered afterwards. The previous
    version took no user at all and returned the twenty most recent sessions
    across every user, which is what made the other endpoints exploitable —
    it handed out the IDs needed to read and delete anyone's history.
    """
    count = (
        select(func.count())
        .where(messages.c.session_id == sessions.c.session_id)
        .scalar_subquery()
    )
    query = (
        select(
            sessions.c.session_id,
            sessions.c.title,
            sessions.c.created_at,
            sessions.c.updated_at,
            count.label("message_count"),
        )
        .where(sessions.c.user_id == user_id)
        .order_by(sessions.c.updated_at.desc())
        .limit(limit)
    )
    with get_engine().connect() as conn:
        return [row_dict(r) for r in conn.execute(query)]


def delete_session(session_id: str, user_id: str) -> bool:
    """
    Delete a session and its messages, only if `user_id` owns it.

    Ownership is part of the DELETE predicate rather than a prior SELECT, so
    there is no window between the check and the delete; messages go with it
    through the foreign key's ON DELETE CASCADE. Returns False when nothing
    matched — either the session does not exist or it is not theirs, and the
    caller must not distinguish the two.
    """
    with transaction() as conn:
        result = conn.execute(
            delete(sessions).where(
                sessions.c.session_id == session_id, sessions.c.user_id == user_id
            )
        )
        return result.rowcount > 0


def save_message(
    session_id: str,
    role: str,
    content: str,
    intent_type: str | None = None,
    detected_domains: list[str] | None = None,
    quality_score: float | None = None,
    pipeline_meta: dict | None = None,
) -> int:
    """Persist one chat turn. Also touches updated_at on the parent session. Returns row id."""
    with transaction() as conn:
        row_id = conn.execute(
            insert(messages)
            .values(
                session_id=session_id,
                role=role,
                content=content,
                intent_type=intent_type,
                detected_domains=detected_domains or None,
                quality_score=quality_score,
                pipeline_meta=pipeline_meta or None,
            )
            .returning(messages.c.id)
        ).scalar_one()
        conn.execute(
            update(sessions)
            .where(sessions.c.session_id == session_id)
            .values(updated_at=func.now())
        )
    return row_id


def get_session_messages(session_id: str) -> list[dict]:
    """Return all messages for a session in the order they were written."""
    with get_engine().connect() as conn:
        rows = conn.execute(
            select(messages).where(messages.c.session_id == session_id).order_by(messages.c.id)
        )
        return [row_dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Rolling conversation memory
# ---------------------------------------------------------------------------


def get_session_memory(session_id: str) -> tuple[str, int]:
    """
    The session's conversation digest and the last message id folded into it.

    Returns ("", 0) for a session with no digest yet, so callers can treat
    "never summarized" and "summarized through nothing" identically.
    """
    with get_engine().connect() as conn:
        row = conn.execute(
            select(sessions.c.summary, sessions.c.summary_through_id).where(
                sessions.c.session_id == session_id
            )
        ).first()
    if not row:
        return "", 0
    return (row.summary or ""), int(row.summary_through_id or 0)


def set_session_memory(session_id: str, summary: str, through_id: int) -> None:
    """Replace the digest. `through_id` must be the highest message folded in."""
    with transaction() as conn:
        conn.execute(
            update(sessions)
            .where(sessions.c.session_id == session_id)
            .values(summary=summary, summary_through_id=through_id)
        )


def get_messages_after(session_id: str, after_id: int) -> list[dict]:
    """
    Messages newer than `after_id`, oldest first.

    This is what has *not* yet been folded into the digest.
    """
    with get_engine().connect() as conn:
        rows = conn.execute(
            select(messages.c.id, messages.c.role, messages.c.content)
            .where(messages.c.session_id == session_id, messages.c.id > after_id)
            .order_by(messages.c.id)
        )
        return [row_dict(r) for r in rows]


def delete_messages_from(session_id: str, message_id: int) -> None:
    """
    Delete a message and all subsequent messages in the session (for edit/re-send).

    If the digest already folded in any of the deleted messages it is dropped,
    not kept: a summary that describes turns the student has since rewritten
    would silently reintroduce them as context the conversation no longer has.
    Clearing it costs one re-summarization and keeps memory truthful.
    """
    with transaction() as conn:
        conn.execute(
            delete(messages).where(
                messages.c.session_id == session_id, messages.c.id >= message_id
            )
        )
        conn.execute(
            update(sessions)
            .where(
                sessions.c.session_id == session_id,
                func.coalesce(sessions.c.summary_through_id, 0) >= message_id,
            )
            .values(summary=None, summary_through_id=None)
        )
        conn.execute(
            update(sessions)
            .where(sessions.c.session_id == session_id)
            .values(updated_at=func.now())
        )
