"""
Scale-out pieces: the turn limiter, shared rate limits, the ingestion queue,
metrics, and the simulated model used for load tests.

The Redis tests run against a real Redis (conftest's `redis_url`), because
what is being tested is the Lua scripts' atomicity, which a fake cannot vouch
for.
"""

from __future__ import annotations

import asyncio
import gc
import threading

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete

from edupilot import db
from edupilot.core import metrics
from edupilot.core.admission import Overloaded, TurnLimiter, guard
from edupilot.db.tables import ingest_jobs
from edupilot.ingestion import jobs
from edupilot.retrieval.indexer import IndexResult
from edupilot.security.ratelimit import RateLimit, RateLimiter, RateLimitExceeded

# ---------------------------------------------------------------------------
# Admission
# ---------------------------------------------------------------------------


async def test_turns_beyond_the_limit_wait_for_a_slot():
    limiter = TurnLimiter(local_limit=2, global_limit=0, timeout=2)
    running, peak = 0, 0

    async def turn():
        nonlocal running, peak
        async with limiter.slot():
            running += 1
            peak = max(peak, running)
            await asyncio.sleep(0.05)
            running -= 1

    await asyncio.gather(*(turn() for _ in range(6)))
    assert peak == 2


async def test_a_turn_that_cannot_get_a_slot_in_time_is_refused():
    limiter = TurnLimiter(local_limit=1, global_limit=0, timeout=0.1)
    ticket = await limiter.admit()
    with pytest.raises(Overloaded) as caught:
        await limiter.admit()
    assert caught.value.retry_after > 0
    await ticket.release()
    await ticket.release()  # idempotent
    async with limiter.slot():
        pass


async def test_a_stream_dropped_before_it_ran_gives_its_slot_back():
    limiter = TurnLimiter(local_limit=1, global_limit=0, timeout=0.1)
    ticket = await limiter.admit()

    async def never_started():
        yield "x"

    stream = guard(never_started(), ticket)
    del stream
    gc.collect()
    async with limiter.slot():  # would time out if the slot had leaked
        pass


async def test_the_global_limit_holds_across_limiters(shared_redis):
    """Two processes' limiters share one Redis lease set."""
    a = TurnLimiter(local_limit=5, global_limit=2, timeout=0.3)
    b = TurnLimiter(local_limit=5, global_limit=2, timeout=0.3)
    first, second = await a.admit(), await b.admit()
    with pytest.raises(Overloaded):
        await a.admit()
    await first.release()
    third = await b.admit()
    await second.release()
    await third.release()


async def test_a_turn_cancelled_mid_answer_still_frees_its_global_slot(shared_redis):
    """
    A student closing the tab cancels the stream; the lease must not leak.

    Starlette cancels through an anyio scope, which keeps cancelling every
    await inside it — including the one that would release the lease. Plain
    asyncio cancellation fires once and would not catch this.
    """
    import anyio

    limiter = TurnLimiter(local_limit=5, global_limit=1, timeout=0.3)
    with anyio.CancelScope() as scope:
        ticket = await limiter.admit()
        try:
            scope.cancel()
            await anyio.sleep(10)
        finally:
            await ticket.release()
    await asyncio.sleep(0.05)  # the release task finishes in the background
    async with limiter.slot():  # raises Overloaded if the lease leaked
        pass


async def test_a_turn_cancelled_while_waiting_gives_every_slot_back(shared_redis):
    """The client leaves while its turn waits for a global slot."""
    limiter = TurnLimiter(local_limit=1, global_limit=1, timeout=5)
    other = TurnLimiter(local_limit=1, global_limit=1, timeout=5)
    holder = await other.admit()  # the one global slot, held by "another process"

    waiting = asyncio.create_task(limiter.admit())
    await asyncio.sleep(0.2)  # it now holds the local slot and polls Redis
    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting
    await holder.release()
    await asyncio.sleep(0.05)

    # Both the local slot and the global lease are free again.
    ticket = await asyncio.wait_for(limiter.admit(), timeout=1)
    await ticket.release()


# ---------------------------------------------------------------------------
# Rate limits
# ---------------------------------------------------------------------------


async def test_shared_rate_limits_hold_across_processes(shared_redis):
    limits = {"chat": RateLimit(requests=3, per_seconds=60), "default": RateLimit(10, 60)}
    one, two = RateLimiter(limits), RateLimiter(limits)
    await one.acheck("user:a", "chat")
    await two.acheck("user:a", "chat")
    await one.acheck("user:a", "chat")
    with pytest.raises(RateLimitExceeded) as caught:
        await two.acheck("user:a", "chat")
    assert caught.value.retry_after >= 1
    await two.acheck("user:b", "chat")  # another caller has their own bucket


async def test_without_redis_the_limits_are_per_process():
    limiter = RateLimiter({"chat": RateLimit(1, 60), "default": RateLimit(10, 60)})
    await limiter.acheck("user:a", "chat")
    with pytest.raises(RateLimitExceeded):
        await limiter.acheck("user:a", "chat")


async def test_an_unreachable_redis_lets_requests_through(monkeypatch):
    from edupilot.core import redis as shared

    monkeypatch.setattr(shared, "REDIS_URL", "redis://127.0.0.1:1/0")
    shared._clients.clear()
    try:
        limiter = RateLimiter({"chat": RateLimit(1, 60), "default": RateLimit(10, 60)})
        await limiter.acheck("user:a", "chat")
        await limiter.acheck("user:a", "chat")
    finally:
        shared._clients.clear()


# ---------------------------------------------------------------------------
# Ingestion jobs
# ---------------------------------------------------------------------------


class RecordingIndexer:
    """Stands in for Indexer: records what it was asked to index."""

    def __init__(self, delay: float = 0.0) -> None:
        self.indexed: list[str] = []
        self.removed: list[str] = []
        self._lock = threading.Lock()
        self.delay = delay

    def index_document(self, path, *, namespace, domain):
        import time

        time.sleep(self.delay)
        with self._lock:
            self.indexed.append(path.name)
        assert path.read_bytes().startswith(b"%PDF")
        return IndexResult(filename=path.name, content_hash="h", chunks_indexed=7)

    def remove_document(self, namespace, filename):
        self.removed.append(filename)
        return True


@pytest.fixture
def empty_queue():
    with db.transaction() as conn:
        conn.execute(delete(ingest_jobs))
    yield


def _study_session(sid: str) -> None:
    if db.get_ss_session(sid) is None:
        db.create_ss_session(sid, "student-jobs", "Jobs")


def test_a_queued_upload_is_indexed_and_its_document_filled_in(empty_queue):
    _study_session("ss-jobs-1")
    ids = jobs.enqueue(
        kind="study", user_id="student-jobs", namespace="ss_jobs1", domain="SELF_STUDY",
        filename="notes.pdf", content=b"%PDF-1.4 notes", file_type=".pdf", ss_session_id="ss-jobs-1",
    )
    doc = db.get_ss_document(ids["document_id"])
    assert doc["status"] == "queued" and doc["chunk_count"] == 0

    indexer = RecordingIndexer()
    assert jobs.run_once(indexer) is True
    assert jobs.run_once(indexer) is False  # queue drained

    job = jobs.get(ids["job_id"])
    assert job["status"] == "done" and job["result"]["chunks_indexed"] == 7
    doc = db.get_ss_document(ids["document_id"])
    assert doc["status"] == "ready" and doc["chunk_count"] == 7
    assert indexer.indexed == ["notes.pdf"]


def test_concurrent_workers_never_take_the_same_job(empty_queue):
    _study_session("ss-jobs-2")
    for i in range(12):
        jobs.enqueue(
            kind="study", user_id="student-jobs", namespace="ss_jobs2", domain="SELF_STUDY",
            filename=f"f{i}.pdf", content=b"%PDF-1.4", ss_session_id="ss-jobs-2",
        )
    indexer = RecordingIndexer(delay=0.01)

    def drain():
        while jobs.run_once(indexer):
            pass

    threads = [threading.Thread(target=drain) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(indexer.indexed) == sorted(f"f{i}.pdf" for i in range(12))
    assert jobs.queue_depth() == {"queued": 0, "running": 0}


def test_a_file_deleted_while_indexing_leaves_no_vectors(empty_queue):
    _study_session("ss-jobs-3")
    ids = jobs.enqueue(
        kind="study", user_id="student-jobs", namespace="ss_jobs3", domain="SELF_STUDY",
        filename="gone.pdf", content=b"%PDF-1.4", ss_session_id="ss-jobs-3",
    )
    job = jobs.claim()
    db.delete_ss_document_record(ids["document_id"])  # cascades the job row away
    indexer = RecordingIndexer()
    jobs.process(job, indexer)
    assert indexer.removed == ["gone.pdf"]


def test_a_job_whose_worker_vanished_is_requeued(empty_queue, monkeypatch):
    _study_session("ss-jobs-4")
    ids = jobs.enqueue(
        kind="study", user_id="student-jobs", namespace="ss_jobs4", domain="SELF_STUDY",
        filename="stuck.pdf", content=b"%PDF-1.4", ss_session_id="ss-jobs-4",
    )
    jobs.claim()
    monkeypatch.setattr(jobs, "STALE_AFTER_SECONDS", 0)
    assert jobs.requeue_stale() == 1
    assert jobs.get(ids["job_id"])["status"] == "queued"


# ---------------------------------------------------------------------------
# HTTP: uploads, job status, overload, metrics
# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    from edupilot.api.app import create_app

    with TestClient(create_app()) as c:
        yield c


def test_an_upload_is_accepted_at_once_and_reports_through_its_job(client, empty_queue, monkeypatch):
    from edupilot.api.routes import self_study

    monkeypatch.setattr(self_study, "validate_batch", _accept_anything)
    sid = client.post("/api/self-study/sessions", json={"name": "Jobs"}).json()["ss_session_id"]
    r = client.post(
        f"/api/self-study/sessions/{sid}/upload",
        files=[("files", ("notes.pdf", b"%PDF-1.4 hello", "application/pdf"))],
    )
    assert r.status_code == 202
    queued = r.json()["queued"][0]
    status = client.get(f"/api/jobs/{queued['job_id']}").json()
    assert status["filename"] == "notes.pdf" and status["status"] in {"queued", "running", "done", "failed"}

    documents = client.get(f"/api/self-study/sessions/{sid}").json()["documents"]
    assert [(d["filename"], d["status"]) for d in documents] == [("notes.pdf", "queued")]
    assert client.get("/api/jobs/999999").status_code == 404


def _accept_anything(raw):
    from edupilot.security.uploads import ValidatedUpload

    return [
        ValidatedUpload(original_name=name, safe_name=name, extension=".pdf", content=content,
                        size_bytes=len(content), warnings=[])
        for name, content in raw
    ]


def test_a_saturated_server_answers_503_with_retry_after(client, monkeypatch):
    from edupilot.api.routes import chat

    async def full():
        raise Overloaded(retry_after=4)

    monkeypatch.setattr(chat.turns, "admit", full)
    monkeypatch.setattr(chat, "_cached_reply", lambda req, turn: None)
    r = client.post("/api/chat/stream", json={"query": "what is merge sort"})
    assert r.status_code == 503
    assert r.headers["Retry-After"] == "4"
    assert r.json()["error"]["code"] == "overloaded"


def test_metrics_are_exposed_with_route_templates(client):
    client.get("/api/sessions/does-not-exist")
    body = client.get("/metrics").text
    assert "edupilot_http_requests_total" in body
    assert 'route="/api/sessions/{session_id}"' in body
    assert "does-not-exist" not in body
    assert "edupilot_ingest_queue" in body


def test_metrics_can_require_a_token(client, monkeypatch):
    from edupilot.api.routes import system

    monkeypatch.setattr(system, "METRICS_TOKEN", "s3cret")
    assert client.get("/metrics").status_code == 401
    assert client.get("/metrics", headers={"Authorization": "Bearer s3cret"}).status_code == 200


# ---------------------------------------------------------------------------
# Simulated model
# ---------------------------------------------------------------------------


async def test_a_turn_runs_end_to_end_on_the_simulated_model(monkeypatch):
    from langgraph.checkpoint.memory import InMemorySaver

    from edupilot.agents import EduPilotAgent, PipelineConfig
    from edupilot.core.config import DOMAINS
    from edupilot.llm import simulated
    from fakes import chunk, retriever_factory

    monkeypatch.setattr(simulated, "SIMULATED_LLM_LATENCY_MS", 1)
    monkeypatch.setattr(simulated, "SIMULATED_LLM_TOKENS_PER_SECOND", 100000)
    agent = EduPilotAgent(
        DOMAINS,
        retriever_factory({"ALGO": [chunk("Merge sort splits the array in half.", "algo.pdf")]}),
        models=simulated.simulated_llm,
        checkpointer=InMemorySaver(),
    )
    cfg = PipelineConfig(model="simulated", enable_grounding_check=False)
    tokens = 0
    result = None
    async for event in agent.astream("why is merge sort n log n", cfg, thread_id="sim-1"):
        if event["type"] == "token":
            tokens += 1
        elif event["type"] == "result":
            result = event["result"]
    assert result is not None and result.domains == ["ALGO"]
    assert "[Source 1]" in result.final_answer and not result.refused
    assert tokens > 3, "the answer is streamed, not delivered whole"
    assert metrics.LLM_CALLS.labels("simulated", "ok")._value.get() >= 0
