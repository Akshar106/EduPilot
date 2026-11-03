"""
Self Study
==========
A student's private documents and the chat over them.

Each session gets its own vector namespace derived from its id, so one
student's uploads are physically separated from another's rather than filtered
apart at query time. Every route re-checks ownership.
"""

from __future__ import annotations

import time
import uuid

from fastapi import APIRouter, Depends, File, UploadFile

from edupilot import db
from edupilot.api.deps import CurrentUser, retrieval_config, run_blocking, validate_model
from edupilot.api.routes.chat import finished, overloaded
from edupilot.api.schemas import CreateStudySessionRequest, StudyChatRequest
from edupilot.core import metrics, tracing
from edupilot.core.admission import Overloaded, turns
from edupilot.core.services import services
from edupilot.security import (
    AppError,
    ErrorCode,
    User,
    owns_or_admin,
    rate_limited,
    validate_batch,
)
from edupilot.security.uploads import UploadRejected

router = APIRouter(prefix="/api/self-study", tags=["self-study"])


def _namespace(ss_session_id: str) -> str:
    return f"ss_{ss_session_id.replace('-', '')}"


async def _require_study_session(ss_session_id: str, user: User) -> dict:
    session = await run_blocking(db.get_ss_session, ss_session_id)
    if not session:
        raise AppError(code=ErrorCode.NOT_FOUND, internal=f"no study session {ss_session_id}")
    owns_or_admin(user, session["user_id"] or "", what="study session")
    return session


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------


@router.post("/sessions")
async def create_study_session(req: CreateStudySessionRequest, user: CurrentUser):
    ss_session_id = str(uuid.uuid4())
    await run_blocking(db.create_ss_session, ss_session_id, user.user_id, req.name.strip(), req.description)
    return {"ss_session_id": ss_session_id, "name": req.name.strip()}


@router.get("/sessions", dependencies=[Depends(rate_limited("read"))])
async def list_study_sessions(user: CurrentUser):
    return {"sessions": await run_blocking(db.list_ss_sessions, user.user_id)}


@router.get("/sessions/{ss_session_id}", dependencies=[Depends(rate_limited("read"))])
async def get_study_session(ss_session_id: str, user: CurrentUser):
    session = await _require_study_session(ss_session_id, user)
    return {
        "session": session,
        "documents": await run_blocking(db.list_ss_documents, ss_session_id),
        "messages": await run_blocking(db.get_ss_messages, ss_session_id),
    }


@router.delete("/sessions/{ss_session_id}")
async def delete_study_session(ss_session_id: str, user: CurrentUser):
    await _require_study_session(ss_session_id, user)

    await run_blocking(services.store.delete, _namespace(ss_session_id), delete_all=True)
    # Documents, messages and queued jobs go with it (ON DELETE CASCADE).
    await run_blocking(db.delete_ss_session, ss_session_id)
    return {"deleted": ss_session_id}


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------


@router.post(
    "/sessions/{ss_session_id}/upload",
    status_code=202,
    dependencies=[Depends(rate_limited("upload"))],
)
async def upload_study_documents(
    ss_session_id: str, user: CurrentUser, files: list[UploadFile] = File(...)
):
    """
    Queue files for indexing. Answers 202 at once; each file's job reports
    progress at `GET /api/jobs/{job_id}`.
    """
    from edupilot.ingestion import jobs

    await _require_study_session(ss_session_id, user)

    raw = [(f.filename or "", await f.read()) for f in files]
    try:
        validated = validate_batch(raw)
    except UploadRejected as exc:
        raise AppError(code=ErrorCode.UPLOAD_REJECTED, message=str(exc)) from exc

    queued = []
    for item in validated:
        ids = await run_blocking(
            jobs.enqueue,
            kind="study",
            user_id=user.user_id,
            namespace=_namespace(ss_session_id),
            domain="SELF_STUDY",
            filename=item.safe_name,
            content=item.content,
            file_type=item.extension,
            ss_session_id=ss_session_id,
        )
        queued.append({
            "filename": item.safe_name,
            "job_id": ids["job_id"],
            "document_id": ids["document_id"],
            "status": "queued",
            "warnings": item.warnings,
        })
    return {"queued": queued}


@router.delete("/sessions/{ss_session_id}/documents/{doc_id}")
async def delete_study_document(ss_session_id: str, doc_id: int, user: CurrentUser):
    await _require_study_session(ss_session_id, user)
    doc = await run_blocking(db.get_ss_document, doc_id)
    if not doc or doc["ss_session_id"] != ss_session_id:
        raise AppError(code=ErrorCode.NOT_FOUND, internal=f"no document {doc_id}")

    await run_blocking(services.indexer.remove_document, _namespace(ss_session_id), doc["filename"])
    # A job still indexing it notices the row is gone and removes its vectors.
    await run_blocking(db.delete_ss_document_record, doc_id)
    await run_blocking(db.touch_ss_session, ss_session_id)
    return {"deleted": doc_id, "filename": doc["filename"]}


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------


async def _run_study_chat(req: StudyChatRequest) -> dict:
    """
    Single-domain answer over the student's own uploads.

    Runs the agent's Self Study graph rather than the course graph: there is
    nothing to route or split across domains, so it goes straight from
    retrieval to answering, then through the same output guardrails as course
    chat.
    """
    from edupilot.agents import PipelineConfig
    from edupilot.llm import UsageTracker

    usage = UsageTracker()
    result = await services.agent.arun_study(
        req.query,
        PipelineConfig(
            model=req.model,
            max_tokens_answer=2500,
            retrieval=retrieval_config(req.top_k, req.rerank_top_k),
        ),
        ss_session_id=req.ss_session_id,
        history=req.chat_history or [],
        filenames=req.source_filter,
        task=req.task,
        callbacks=[usage],
    )

    retrieval = (result.diagnostics.get("retrieval") or [{}])[0]
    return {
        "final_answer": result.final_answer,
        "refused": result.refused,
        "task": result.task,
        "practice": result.practice,
        "grounding_score": result.grounding_score,
        "guardrail_action": result.guardrail_action,
        "sources": [
            {
                "citation_label": s["citation_label"],
                "filename": s["filename"],
                "page_number": s["page_number"],
                "relevance": s["relevance"],
            }
            for s in result.sources
        ],
        "debug": {
            "retrieval": retrieval,
            "guardrails": result.diagnostics.get("guardrails", {}),
            "usage": usage.as_dict(),
            "trace": {"run_id": result.run_id, "traced": tracing.enabled()},
        },
    }


@router.post("/chat", dependencies=[Depends(rate_limited("chat"))])
async def study_chat(req: StudyChatRequest, user: CurrentUser):
    validate_model(req.model)
    await _require_study_session(req.ss_session_id, user)

    started = time.perf_counter()
    try:
        ticket = await turns.admit()
    except Overloaded as exc:
        raise overloaded(exc) from None
    try:
        await run_blocking(db.save_ss_message, req.ss_session_id, "user", req.query)
        result = await _run_study_chat(req)
    except Exception:
        metrics.TURN_DURATION.labels("study", "error").observe(time.perf_counter() - started)
        raise
    finally:
        await ticket.release()
    finished("study", result, started)
    result["assistant_message_id"] = await run_blocking(
        db.save_ss_message,
        ss_session_id=req.ss_session_id,
        role="assistant",
        content=result["final_answer"],
        quality_score=result.get("grounding_score"),
        # The trace, so a rating can find it; practice data, so a reload can
        # show the quiz interactively.
        pipeline_meta={"trace": result["debug"]["trace"], "practice": result["practice"]},
    )
    return result
