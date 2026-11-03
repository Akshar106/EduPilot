"""Chat session listing, retrieval, deletion, and message truncation."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends

from edupilot import db
from edupilot.api.deps import CurrentUser, run_blocking
from edupilot.api.schemas import NewSessionRequest
from edupilot.core.services import services
from edupilot.security import AppError, ErrorCode, owns_or_admin, rate_limited

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


@router.get("", dependencies=[Depends(rate_limited("read"))])
async def list_sessions(user: CurrentUser):
    return {"sessions": await run_blocking(db.list_sessions, user.user_id, limit=50)}


@router.post("")
async def create_session(req: NewSessionRequest, user: CurrentUser):
    session_id = str(uuid.uuid4())
    await run_blocking(db.ensure_session, session_id, user.user_id, title=req.title)
    return {"session_id": session_id}


@router.get("/{session_id}", dependencies=[Depends(rate_limited("read"))])
async def get_session(session_id: str, user: CurrentUser):
    owner = await run_blocking(db.get_session_owner, session_id)
    if owner is None:
        raise AppError(code=ErrorCode.NOT_FOUND, internal=f"no session {session_id}")
    owns_or_admin(user, owner, what="chat session")
    return {"session_id": session_id, "messages": await run_blocking(db.get_session_messages, session_id)}


@router.delete("/{session_id}")
async def delete_session(session_id: str, user: CurrentUser):
    # Ownership is inside the DELETE predicate, so a miss here means either
    # "no such session" or "not yours" — and the caller learns neither.
    if not await run_blocking(db.delete_session, session_id, user.user_id):
        raise AppError(
            code=ErrorCode.NOT_FOUND,
            internal=f"user {user.user_id} could not delete session {session_id}",
        )
    await services.agent.reset_thread(session_id)
    return {"deleted": session_id}


@router.delete("/{session_id}/messages/{message_id}")
async def truncate_from_message(session_id: str, message_id: int, user: CurrentUser):
    """Drop a message and everything after it, backing an edit-and-resend."""
    owner = await run_blocking(db.get_session_owner, session_id)
    if owner is None:
        raise AppError(code=ErrorCode.NOT_FOUND, internal=f"no session {session_id}")
    owns_or_admin(user, owner, what="chat session")
    await run_blocking(db.delete_messages_from, session_id, message_id)
    # The checkpoint still remembers the deleted turns. Drop it; the next turn
    # reseeds the conversation's memory from what is left in the database.
    await services.agent.reset_thread(session_id)
    return {"truncated": True, "from_message_id": message_id}
