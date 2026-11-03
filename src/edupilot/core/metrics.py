"""
Metrics
=======
Prometheus metrics for latency, throughput and saturation, served at
`/metrics`.

Each API process exposes its own counters; Prometheus scrapes every instance
and aggregates (the compose stack's prometheus.yml discovers the replicas by
DNS). Histograms rather than averages, because what students feel is the
tail: p95 time to first token, not the mean.

What is measured, and where:

  HTTP            every request: count, latency, in flight      api/app.py
  turns           chat and Self Study: duration by outcome,
                  time to the first answer token                api/routes
  retrieval       search and rerank time                         retrieval/hybrid.py
  LLM calls       count, latency, tokens, by model              llm/usage.py
  answer cache    hits and misses                                api/routes/chat.py
  backpressure    rate-limit rejections, turns waiting and
                  running, admission waits and rejections       security, core/admission.py
  ingestion       jobs by outcome and duration; queue depth      ingestion/jobs.py
  database        pooled connections in use                      collected at scrape

Labels are bounded: routes are path templates (`/api/sessions/{session_id}`),
never raw paths, so a scan of random URLs cannot grow the series count.
"""

from __future__ import annotations

import logging

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    REGISTRY,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from prometheus_client.core import GaugeMetricFamily
from prometheus_client.registry import Collector

logger = logging.getLogger(__name__)

#: Request latency: static files and reads are milliseconds, a non-streamed
#: chat turn is tens of seconds.
_HTTP_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 20, 40, 60)
_TURN_BUCKETS = (0.25, 0.5, 1, 2, 4, 8, 15, 30, 60, 120)
_FAST_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10)

# -- HTTP --------------------------------------------------------------------

HTTP_REQUESTS = Counter(
    "edupilot_http_requests_total", "HTTP requests served", ["method", "route", "status"]
)
HTTP_LATENCY = Histogram(
    "edupilot_http_request_duration_seconds",
    "Time to response headers. A streamed turn is measured by the turn metrics.",
    ["method", "route"],
    buckets=_HTTP_BUCKETS,
)
HTTP_IN_FLIGHT = Gauge("edupilot_http_requests_in_flight", "Requests being handled")

# -- Turns -------------------------------------------------------------------

TURN_DURATION = Histogram(
    "edupilot_turn_duration_seconds",
    "One question answered end to end",
    ["kind", "outcome"],
    buckets=_TURN_BUCKETS,
)
TIME_TO_FIRST_TOKEN = Histogram(
    "edupilot_time_to_first_token_seconds",
    "From the question arriving to the first answer token streamed",
    ["kind"],
    buckets=_TURN_BUCKETS,
)
ANSWER_CACHE = Counter(
    "edupilot_answer_cache_lookups_total", "Answer cache lookups", ["result"]
)

# -- Retrieval and models ------------------------------------------------------

RETRIEVAL_STAGE = Histogram(
    "edupilot_retrieval_stage_seconds",
    "Retrieval time per stage",
    ["stage"],
    buckets=_FAST_BUCKETS,
)
INFERENCE_WAIT = Histogram(
    "edupilot_inference_wait_seconds",
    "Time a cross-encoder call waited for the CPU inference gate",
    ["kind"],
    buckets=_FAST_BUCKETS + (20, 40),
)
LLM_CALLS = Counter("edupilot_llm_calls_total", "Model calls", ["model", "outcome"])
LLM_LATENCY = Histogram(
    "edupilot_llm_call_seconds", "One model call", ["model"], buckets=_TURN_BUCKETS
)
LLM_TOKENS = Counter("edupilot_llm_tokens_total", "Model tokens", ["model", "direction"])

# -- Backpressure ----------------------------------------------------------------

RATE_LIMITED = Counter(
    "edupilot_rate_limited_total", "Requests refused by a per-caller rate limit", ["scope"]
)
TURNS_RUNNING = Gauge("edupilot_turns_running", "Turns admitted and running in this process")
TURNS_WAITING = Gauge("edupilot_turns_waiting", "Turns queued for admission in this process")
ADMISSION_WAIT = Histogram(
    "edupilot_admission_wait_seconds",
    "Time a turn waited for a slot",
    buckets=(0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 30),
)
ADMISSION_REJECTED = Counter(
    "edupilot_admission_rejected_total", "Turns refused because the system was saturated"
)

# -- Ingestion -------------------------------------------------------------------

INGEST_JOBS = Counter("edupilot_ingest_jobs_total", "Ingestion jobs finished", ["outcome"])
INGEST_DURATION = Histogram(
    "edupilot_ingest_job_seconds", "One upload extracted, embedded and indexed",
    buckets=_TURN_BUCKETS,
)


class _StateCollector(Collector):
    """
    Gauges read at scrape time: pooled database connections and the ingestion
    queue. Reading them on demand costs one query per scrape and cannot drift
    the way a counter maintained across processes would.
    """

    def collect(self):
        from edupilot.db.engine import _engine

        pool = GaugeMetricFamily(
            "edupilot_db_pool_connections", "Pooled database connections", labels=["state"]
        )
        if _engine is not None:
            p = _engine.pool
            checked_out = p.checkedout() if hasattr(p, "checkedout") else 0
            pool.add_metric(["in_use"], checked_out)
            pool.add_metric(["idle"], p.checkedin() if hasattr(p, "checkedin") else 0)
        yield pool

        queue = GaugeMetricFamily(
            "edupilot_ingest_queue", "Ingestion jobs by status", labels=["status"]
        )
        if _engine is not None:
            try:
                from edupilot.ingestion.jobs import queue_depth

                for status, count in queue_depth().items():
                    queue.add_metric([status], count)
            except Exception:  # a scrape must never fail on a database hiccup
                logger.debug("could not read the ingestion queue for metrics", exc_info=True)
        yield queue


REGISTRY.register(_StateCollector())


def render() -> tuple[bytes, str]:
    """The current metrics in Prometheus text format, and its content type."""
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST
