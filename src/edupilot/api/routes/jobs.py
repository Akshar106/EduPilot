"""Ingestion job status: how a queued upload is getting on."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from edupilot.api.deps import CurrentUser, run_blocking
from edupilot.security import AppError, ErrorCode, owns_or_admin, rate_limited

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


@router.get("/{job_id}", dependencies=[Depends(rate_limited("read"))])
async def get_job(job_id: int, user: CurrentUser):
    """
    `queued`, `running`, `done` or `failed`, with the chunk count or error.

    Owner-only, NOT_FOUND otherwise: a job id is a sequence number, and the
    file name it reveals is the student's own.
    """
    from edupilot.ingestion import jobs

    job = await run_blocking(jobs.get, job_id)
    if job is None:
        raise AppError(code=ErrorCode.NOT_FOUND, internal=f"no job {job_id}")
    owns_or_admin(user, job["user_id"], what="ingestion job")
    result = job.get("result") or {}
    return {
        "job_id": job["id"],
        "kind": job["kind"],
        "status": job["status"],
        "filename": job["filename"],
        "document_id": job["document_id"],
        "chunks_indexed": result.get("chunks_indexed"),
        "error": job["error"],
        "created_at": job["created_at"],
        "finished_at": job["finished_at"],
    }
