"""
Ingestion worker
================
Indexes uploaded files from the job queue (ingestion/jobs.py).

    edupilot-worker

Run as many as indexing needs; they share the queue without double-claiming.
With INGEST_WORKER=external the API processes leave all indexing to these.
SIGTERM finishes the current job, then exits.
"""

from __future__ import annotations

import logging
import os
import signal
import sys

from edupilot.core.observability import configure_logging

configure_logging()
logger = logging.getLogger("edupilot.worker")


def main() -> int:
    from edupilot import db
    from edupilot.core.services import services
    from edupilot.ingestion.jobs import Worker

    db.init_db()
    port = os.getenv("WORKER_METRICS_PORT")
    if port:
        # Jobs processed, their duration and the process's own CPU and memory,
        # for Prometheus; the API replicas expose theirs at /metrics.
        from prometheus_client import start_http_server

        start_http_server(int(port))
        logger.info("worker metrics on :%s/metrics", port)
    worker = Worker(lambda: services.indexer)

    def stop(signum, frame):
        logger.info("signal %d — finishing the current job, then stopping", signum)
        worker.stop(timeout=0)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    # Load the embedding model before the first job rather than during it.
    services.indexer  # noqa: B018 - property access with an intentional side effect
    worker.run()
    db.dispose_engine()
    return 0


if __name__ == "__main__":
    sys.exit(main())
