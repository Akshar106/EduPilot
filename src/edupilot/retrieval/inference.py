"""
CPU inference gate
==================
Runs in-process cross-encoder inference a bounded number of calls at a time.

Reranking and the claim-level grounding check score (query, passage) pairs
with a ~280M-parameter cross-encoder on the CPU, and torch spreads each call
over every core. Several turns reranking at once therefore do not share the
CPU; they fight over it — each call's threads preempt the others', and a load
test with five concurrent students took 90 seconds to the first token where
one student took ten. Measured, serializing the calls and letting each use
the whole machine finishes every one of them sooner.

Model calls to the LLM provider are not gated: they wait on the network, and
any number can overlap with the CPU work here.

  INFERENCE_CONCURRENCY  calls at once per process (default 1)
  INFERENCE_THREADS      torch threads per call; 0 keeps torch's default of
                         one per core. With several API processes on one
                         machine, give each cores / processes.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager

from edupilot.core import metrics
from edupilot.core.config import INFERENCE_CONCURRENCY, INFERENCE_THREADS

logger = logging.getLogger(__name__)

_gate = threading.BoundedSemaphore(max(1, INFERENCE_CONCURRENCY))
_configured = False
_configure_lock = threading.Lock()


def configure_threads() -> None:
    """Apply INFERENCE_THREADS to torch, once, before the first model loads."""
    global _configured
    if _configured:
        return
    with _configure_lock:
        if _configured:
            return
        _configured = True
        if INFERENCE_THREADS > 0:
            import torch

            torch.set_num_threads(INFERENCE_THREADS)
            logger.info("torch inference threads set to %d", INFERENCE_THREADS)


@contextmanager
def model_slot(kind: str) -> Iterator[None]:
    """Hold the inference gate for one model call; `kind` labels the metrics."""
    started = time.perf_counter()
    with _gate:
        metrics.INFERENCE_WAIT.labels(kind).observe(time.perf_counter() - started)
        yield
