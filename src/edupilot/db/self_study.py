"""
Self Study — private per-student sessions
=========================================
The student's own uploads, kept apart from the shared course corpus. Exposure
here leaks personal files rather than lecture text, so every session read goes
through an ownership check at the route layer.
"""

from __future__ import annotations

from sqlalchemy import delete, func, insert, select, update

from .engine import get_engine, transaction
from .rows import row_dict
from .tables import self_study_documents as documents
from .tables import self_study_messages as messages
from .tables import self_study_sessions as sessions


def get_ss_session_owner(ss_session_id: str) -> str | None:
    """user_id owning a study session, or None when missing."""
    with get_engine().connect() as conn:
        return conn.execute(
            select(sessions.c.user_id).where(sessions.c.ss_session_id == ss_session_id)
        ).scalar()


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------


def create_ss_session(
    ss_session_id: str, user_id: str, name: str, description: str | None = None
) -> None:
    """Create a study session owned by `user_id`."""
    if not user_id:
        raise ValueError("user_id is required to create a study session")
    with transaction() as conn:
        conn.execute(
            insert(sessions).values(
                ss_session_id=ss_session_id, user_id=user_id, name=name, description=description
            )
        )


def list_ss_sessions(user_id: str) -> list[dict]:
    """
    Study sessions belonging to `user_id`.

    These carry the student's own uploaded documents, so cross-user exposure
    here leaks personal files, not just chat text.
    """
    per_session = (
        select(
            documents.c.ss_session_id,
            func.count().label("doc_count"),
            func.sum(documents.c.chunk_count).label("total_chunks"),
        )
        .group_by(documents.c.ss_session_id)
        .subquery()
    )
    query = (
        select(
            sessions.c.ss_session_id,
            sessions.c.name,
            sessions.c.description,
            sessions.c.created_at,
            sessions.c.updated_at,
            func.coalesce(per_session.c.doc_count, 0).label("doc_count"),
            func.coalesce(per_session.c.total_chunks, 0).label("total_chunks"),
        )
        .select_from(sessions.outerjoin(per_session, per_session.c.ss_session_id == sessions.c.ss_session_id))
        .where(sessions.c.user_id == user_id)
        .order_by(sessions.c.updated_at.desc())
    )
    with get_engine().connect() as conn:
        return [row_dict(r) for r in conn.execute(query)]


def get_ss_session(ss_session_id: str) -> dict | None:
    with get_engine().connect() as conn:
        row = conn.execute(
            select(sessions).where(sessions.c.ss_session_id == ss_session_id)
        ).first()
    return row_dict(row) if row else None


def delete_ss_session(ss_session_id: str) -> None:
    """Delete a study session; its documents and messages go with it (ON DELETE CASCADE)."""
    with transaction() as conn:
        conn.execute(delete(sessions).where(sessions.c.ss_session_id == ss_session_id))


def touch_ss_session(ss_session_id: str) -> None:
    with transaction() as conn:
        conn.execute(
            update(sessions)
            .where(sessions.c.ss_session_id == ss_session_id)
            .values(updated_at=func.now())
        )


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------


def save_ss_document(
    ss_session_id: str,
    filename: str,
    file_type: str,
    file_size_bytes: int,
    chunk_count: int,
) -> int:
    with transaction() as conn:
        return conn.execute(
            insert(documents)
            .values(
                ss_session_id=ss_session_id,
                filename=filename,
                file_type=file_type,
                file_size_bytes=file_size_bytes,
                chunk_count=chunk_count,
            )
            .returning(documents.c.id)
        ).scalar_one()


def list_ss_documents(ss_session_id: str) -> list[dict]:
    with get_engine().connect() as conn:
        rows = conn.execute(
            select(documents)
            .where(documents.c.ss_session_id == ss_session_id)
            .order_by(documents.c.id)
        )
        return [row_dict(r) for r in rows]


def get_ss_document(doc_id: int) -> dict | None:
    with get_engine().connect() as conn:
        row = conn.execute(select(documents).where(documents.c.id == doc_id)).first()
    return row_dict(row) if row else None


def delete_ss_document_record(doc_id: int) -> None:
    with transaction() as conn:
        conn.execute(delete(documents).where(documents.c.id == doc_id))


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------


def save_ss_message(
    ss_session_id: str,
    role: str,
    content: str,
    quality_score: float | None = None,
    pipeline_meta: dict | None = None,
) -> int:
    with transaction() as conn:
        row_id = conn.execute(
            insert(messages)
            .values(
                ss_session_id=ss_session_id,
                role=role,
                content=content,
                quality_score=quality_score,
                pipeline_meta=pipeline_meta or None,
            )
            .returning(messages.c.id)
        ).scalar_one()
        conn.execute(
            update(sessions)
            .where(sessions.c.ss_session_id == ss_session_id)
            .values(updated_at=func.now())
        )
        return row_id


def get_ss_messages(ss_session_id: str) -> list[dict]:
    with get_engine().connect() as conn:
        rows = conn.execute(
            select(messages)
            .where(messages.c.ss_session_id == ss_session_id)
            .order_by(messages.c.id)
        )
        return [row_dict(r) for r in rows]
