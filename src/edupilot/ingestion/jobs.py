"""
Ingestion jobs
==============
Uploads are indexed by a worker, not inside the HTTP request.

Extracting, chunking and embedding a 40-page PDF takes tens of seconds of CPU.
Done in the request, it held a connection open that long (past many proxies'
timeouts), competed with chat turns for the same process's CPU, and could not
be retried if the process restarted halfway. Now the request validates the
file, queues a job and returns 202; a worker indexes it; the frontend polls
`GET /api/jobs/{id}` and shows progress.

The queue is a Postgres table rather than a broker, since Postgres is already
the system of record: a job is enqueued in the same database as the document
row it fills in, and workers claim jobs with `FOR UPDATE SKIP LOCKED`, so any
number of them can poll one table without ever taking the same job twice.

Workers run embedded in each API process by default (INGEST_WORKER=embedded),
or as separate `edupilot-worker` processes, which is what the compose stack
does so indexing never shares a CPU with answering.

A job whose worker died mid-run is put back on the queue once it has been
running for STALE_AFTER_SECONDS, up to MAX_ATTEMPTS times.
"""

from __future__ import annotations

import logging
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from sqlalchemy import func, insert, literal_column, select, text, update

from edupilot.core import metrics
from edupilot.db.engine import get_engine
from edupilot.db.rows import row_dict
from edupilot.db.tables import ingest_jobs as jobs
from edupilot.db.tables import self_study_documents as study_documents
from edupilot.db.tables import self_study_sessions as study_sessions
from edupilot.db.tables import uploaded_documents

logger = logging.getLogger(__name__)

#: A running job is presumed abandoned after this long.
STALE_AFTER_SECONDS = 600
#: Attempts before a job is failed for good.
MAX_ATTEMPTS = 3
#: How often an idle worker looks for work.
POLL_SECONDS = 0.5

_PUBLIC = [c for c in jobs.c if c.name != "content"]


# ---------------------------------------------------------------------------
# Queue
# ---------------------------------------------------------------------------


def enqueue(
    *,
    kind: str,
    user_id: str,
    namespace: str,
    domain: str,
    filename: str,
    content: bytes,
    file_type: str | None = None,
    ss_session_id: str | None = None,
) -> dict[str, Any]:
    """
    Queue one file for indexing. Returns `{"job_id", "document_id"}`.

    A Self Study upload also gets its document row now, marked queued, in the
    same transaction — the student sees the file in their list straight away
    and the worker fills in its chunk count.
    """
    with get_engine().begin() as conn:
        document_id = None
        if kind == "study":
            document_id = conn.execute(
                insert(study_documents)
                .values(
                    ss_session_id=ss_session_id,
                    filename=filename,
                    file_type=file_type,
                    file_size_bytes=len(content),
                    chunk_count=0,
                    status="queued",
                )
                .returning(study_documents.c.id)
            ).scalar_one()
            conn.execute(
                update(study_sessions)
                .where(study_sessions.c.ss_session_id == ss_session_id)
                .values(updated_at=func.now())
            )
        job_id = conn.execute(
            insert(jobs)
            .values(
                kind=kind,
                user_id=user_id,
                namespace=namespace,
                domain=domain,
                filename=filename,
                file_type=file_type,
                size_bytes=len(content),
                ss_session_id=ss_session_id,
                document_id=document_id,
                content=content,
            )
            .returning(jobs.c.id)
        ).scalar_one()
    return {"job_id": job_id, "document_id": document_id}


def get(job_id: int) -> dict[str, Any] | None:
    """A job's status and result, without its file."""
    with get_engine().connect() as conn:
        row = conn.execute(select(*_PUBLIC).where(jobs.c.id == job_id)).first()
    return row_dict(row) if row else None


def queue_depth() -> dict[str, int]:
    """Jobs waiting and running. Finished jobs are history, not depth."""
    with get_engine().connect() as conn:
        rows = conn.execute(
            select(jobs.c.status, func.count())
            .where(jobs.c.status.in_(("queued", "running")))
            .group_by(jobs.c.status)
        ).all()
    depth = {"queued": 0, "running": 0}
    depth.update({status: int(n) for status, n in rows})
    return depth


def claim() -> dict[str, Any] | None:
    """
    Take the oldest queued job, or None.

    SKIP LOCKED makes a job another worker is claiming invisible rather than
    something to wait for, so workers never block each other or double-claim.
    """
    with get_engine().begin() as conn:
        row = conn.execute(
            text(
                """
                UPDATE ingest_jobs
                   SET status = 'running', started_at = now(), attempts = attempts + 1
                 WHERE id = (
                       SELECT id FROM ingest_jobs
                        WHERE status = 'queued'
                        ORDER BY id
                        FOR UPDATE SKIP LOCKED
                        LIMIT 1)
                RETURNING *
                """
            )
        ).first()
        if row is None:
            return None
        job = dict(row._mapping)
        if job["document_id"]:
            conn.execute(
                update(study_documents)
                .where(study_documents.c.id == job["document_id"])
                .values(status="indexing")
            )
    return job


def requeue_stale() -> int:
    """Put back jobs whose worker vanished; fail those out of attempts. Returns rows touched."""
    cutoff = literal_column(f"now() - interval '{int(STALE_AFTER_SECONDS)} seconds'")
    with get_engine().begin() as conn:
        failed = conn.execute(
            update(jobs)
            .where(jobs.c.status == "running", jobs.c.started_at < cutoff, jobs.c.attempts >= MAX_ATTEMPTS)
            .values(status="failed", error="the worker stopped responding", finished_at=func.now(), content=None)
        ).rowcount
        requeued = conn.execute(
            update(jobs)
            .where(jobs.c.status == "running", jobs.c.started_at < cutoff)
            .values(status="queued")
        ).rowcount
    if failed or requeued:
        logger.warning("ingestion: requeued %d stale job(s), failed %d", requeued, failed)
    return failed + requeued


def _finish(job: dict, *, status: str, result: dict | None, error: str | None) -> None:
    with get_engine().begin() as conn:
        conn.execute(
            update(jobs)
            .where(jobs.c.id == job["id"])
            .values(status=status, result=result, error=error, finished_at=func.now(), content=None)
        )
        if job["document_id"]:
            conn.execute(
                update(study_documents)
                .where(study_documents.c.id == job["document_id"])
                .values(
                    status="ready" if status == "done" else "failed",
                    chunk_count=(result or {}).get("chunks_indexed", 0),
                    error=error,
                )
            )
        if job["kind"] == "course" and status == "done":
            conn.execute(
                insert(uploaded_documents).values(
                    filename=job["filename"],
                    domain=job["domain"],
                    file_type=job["file_type"],
                    chunk_count=(result or {}).get("chunks_indexed", 0),
                    file_size_bytes=job["size_bytes"],
                )
            )


def _document_exists(document_id: int) -> bool:
    with get_engine().connect() as conn:
        return conn.execute(
            select(study_documents.c.id).where(study_documents.c.id == document_id)
        ).first() is not None


# ---------------------------------------------------------------------------
# Work
# ---------------------------------------------------------------------------


def process(job: dict, indexer) -> None:
    """Index one claimed job's file and record the outcome."""
    started = time.perf_counter()
    try:
        with tempfile.TemporaryDirectory(prefix="edupilot-ingest-") as tmp:
            # The registry identifies a document by file name, so the file
            # keeps its own.
            path = Path(tmp) / job["filename"]
            path.write_bytes(job["content"] or b"")
            outcome = indexer.index_document(path, namespace=job["namespace"], domain=job["domain"])
        if job["document_id"] and not _document_exists(job["document_id"]):
            # The student deleted the file while it was being indexed; do not
            # leave its vectors behind.
            indexer.remove_document(job["namespace"], job["filename"])
            _finish(job, status="failed", result=None, error="deleted while indexing")
            metrics.INGEST_JOBS.labels("cancelled").inc()
            return
        result = {
            "chunks_indexed": outcome.chunks_indexed,
            "skipped": outcome.skipped,
            "replaced_version": outcome.replaced_version,
            "duration_ms": outcome.duration_ms,
        }
        if outcome.error:
            _finish(job, status="failed", result=result, error=outcome.error)
            metrics.INGEST_JOBS.labels("failed").inc()
        else:
            _finish(job, status="done", result=result, error=None)
            metrics.INGEST_JOBS.labels("done").inc()
    except Exception as exc:
        logger.exception("ingestion job %s failed", job["id"])
        _finish(job, status="failed", result=None, error=f"{type(exc).__name__}: {exc}")
        metrics.INGEST_JOBS.labels("failed").inc()
    finally:
        metrics.INGEST_DURATION.observe(time.perf_counter() - started)


def run_once(indexer) -> bool:
    """Claim and process one job. Returns False when the queue was empty."""
    job = claim()
    if job is None:
        return False
    logger.info("ingestion: job %s (%s) %s", job["id"], job["kind"], job["filename"])
    process(job, indexer)
    return True


class Worker:
    """
    A polling worker loop, run in its own thread or process.

    Args:
        indexer: Called as `indexer()` for the current Indexer — resolved per
            job, so a promoted index version is picked up between jobs.
    """

    def __init__(self, indexer, *, poll_seconds: float = POLL_SECONDS) -> None:
        self._indexer = indexer
        self._poll = poll_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def run(self) -> None:
        logger.info("ingestion worker started")
        last_sweep = 0.0
        while not self._stop.is_set():
            try:
                if time.monotonic() - last_sweep > 60:
                    requeue_stale()
                    last_sweep = time.monotonic()
                if run_once(self._indexer()):
                    continue  # more may be waiting; poll again at once
            except Exception:
                logger.exception("ingestion worker loop error")
            self._stop.wait(self._poll)
        logger.info("ingestion worker stopped")

    def start(self) -> None:
        """Run in a daemon thread (the embedded mode)."""
        self._thread = threading.Thread(target=self.run, name="edupilot-ingest", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 10.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
