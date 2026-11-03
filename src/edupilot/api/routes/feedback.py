"""
Answer feedback and trace links.

`POST /api/feedback` records a student's 👍/👎 on one of their answers. The
rating is always stored locally; when LangSmith tracing is on it is also
attached to the answer's trace, so a thumbs-down opens next to every step that
produced the answer.

`GET /api/trace/{run_id}` redirects to a turn's trace in LangSmith, for the
debug panel. Admin-only: traces hold students' questions, and the lookup costs
a LangSmith API call.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends
from fastapi.responses import RedirectResponse

from edupilot import db
from edupilot.api.deps import AdminUser, CurrentUser, run_blocking
from edupilot.api.schemas import FeedbackRequest
from edupilot.core import tracing
from edupilot.security import AppError, ErrorCode, rate_limited

router = APIRouter(prefix="/api", tags=["feedback"])


@router.post("/feedback", dependencies=[Depends(rate_limited("read"))])
async def rate_answer(req: FeedbackRequest, user: CurrentUser):
    rated = await run_blocking(db.get_rated_message, req.kind, req.message_id)
    # Only the student who received an answer can rate it. NOT_FOUND rather
    # than FORBIDDEN, so the endpoint does not confirm other users' messages.
    if rated is None or rated["owner"] != user.user_id:
        raise AppError(code=ErrorCode.NOT_FOUND, internal=f"no {req.kind} answer {req.message_id}")

    comment = (req.comment or "").strip() or None
    replaced = await run_blocking(
        db.save_feedback,
        kind=req.kind,
        message_id=req.message_id,
        user_id=user.user_id,
        rating=req.rating,
        comment=comment,
        run_id=rated["run_id"],
    )

    sent = False
    if rated["run_id"]:
        sent = await run_blocking(
            tracing.send_feedback,
            rated["run_id"],
            rating=req.rating,
            comment=comment,
            feedback_key=tracing.feedback_id(req.kind, req.message_id, user.user_id),
            update=replaced,
        )
    return {"recorded": True, "updated": replaced, "sent_to_langsmith": sent}


@router.get("/trace/{run_id}", dependencies=[Depends(rate_limited("read"))])
async def open_trace(run_id: str, admin: AdminUser):
    try:
        run_id = str(uuid.UUID(run_id))
    except ValueError:
        raise AppError(code=ErrorCode.NOT_FOUND, internal=f"bad run id {run_id!r}") from None
    if not tracing.enabled():
        raise AppError(code=ErrorCode.NOT_FOUND, internal="tracing is off")

    url = await tracing.trace_url(run_id)
    if url is None:
        raise AppError(
            code=ErrorCode.UPSTREAM_UNAVAILABLE, internal=f"no LangSmith URL for run {run_id}"
        )
    return RedirectResponse(url, status_code=307)
