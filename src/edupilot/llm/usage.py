"""
Token accounting
================
A LangChain callback handler that totals every model call made while
answering one request.

Replaces the thread-local `Usage` in the old client. Thread-locals stopped
being a fit once the pipeline went async: concurrent branches of one request
run on one thread, and concurrent requests interleave on it too. A handler
passed in the run's config is scoped to exactly that run, and LangGraph hands
it to every model call in every node.

    usage = UsageTracker()
    await graph.ainvoke(state, config={"callbacks": [usage]})
    usage.as_dict()

It also feeds the process-wide Prometheus metrics (core/metrics.py): every
call's latency, outcome and tokens, by model.
"""

from __future__ import annotations

import threading
import time
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from edupilot.core import metrics


class UsageTracker(BaseCallbackHandler):
    """Token and call counts for one request, aggregated across stages."""

    #: Run on the event loop rather than in an executor: the work is a few
    #: integer additions, and the lock covers the sync callers in threads.
    run_inline = True

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.fallbacks = 0
        self.by_model: dict[str, int] = {}
        self._started: dict[UUID, tuple[float, str]] = {}

    def _start(self, run_id: UUID, kwargs: dict) -> None:
        params = kwargs.get("invocation_params") or {}
        model = str(params.get("model") or params.get("model_name") or "unknown")
        with self._lock:
            self._started[run_id] = (time.perf_counter(), model)

    def _finish(self, run_id: UUID | None, outcome: str, model: str | None = None) -> None:
        with self._lock:
            started, requested = self._started.pop(run_id, (None, "unknown")) if run_id else (None, "unknown")
        name = model or requested
        metrics.LLM_CALLS.labels(name, outcome).inc()
        if started is not None:
            metrics.LLM_LATENCY.labels(name).observe(time.perf_counter() - started)

    def on_chat_model_start(self, serialized, messages, *, run_id: UUID, **kwargs: Any) -> None:
        self._start(run_id, kwargs)

    def on_llm_start(self, serialized, prompts, *, run_id: UUID, **kwargs: Any) -> None:
        self._start(run_id, kwargs)

    def on_llm_end(self, response: LLMResult, *, run_id: UUID | None = None, **kwargs: Any) -> None:
        llm_output = response.llm_output or {}
        model = None
        for generations in response.generations:
            for gen in generations:
                message = getattr(gen, "message", None)
                usage = getattr(message, "usage_metadata", None) or {}
                meta = getattr(message, "response_metadata", None) or {}
                model = (
                    meta.get("model_name")
                    or meta.get("model")
                    or llm_output.get("model_name")
                    or "unknown"
                )
                self._record(model, usage.get("input_tokens", 0), usage.get("output_tokens", 0))
        self._finish(run_id, "ok", model)

    def on_llm_error(self, error: BaseException, *, run_id: UUID | None = None, **kwargs: Any) -> None:
        # Inside a fallback chain a failed call is followed by the next model,
        # so each one is a hop down the chain (or the final failure).
        with self._lock:
            self.fallbacks += 1
        self._finish(run_id, "error")

    def _record(self, model: str, prompt_tokens: int, completion_tokens: int) -> None:
        with self._lock:
            self.calls += 1
            self.input_tokens += prompt_tokens or 0
            self.output_tokens += completion_tokens or 0
            self.by_model[model] = self.by_model.get(model, 0) + 1
        metrics.LLM_TOKENS.labels(model, "input").inc(prompt_tokens or 0)
        metrics.LLM_TOKENS.labels(model, "output").inc(completion_tokens or 0)

    def as_dict(self) -> dict:
        with self._lock:
            return {
                "calls": self.calls,
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "fallbacks": self.fallbacks,
                "by_model": dict(self.by_model),
            }
