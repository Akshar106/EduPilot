"""
Server-sent events
==================
Framing for the streaming endpoints.

A stream is a sequence of named events, each a JSON object:

    event: stage      a graph node finished          {"stage": "route", ...}
    event: progress   retrieval progress in a branch {"stage": "retrieving", "domain": "ALGO"}
    event: token      a piece of the answer          {"text": "..."}
    event: result     the final payload — the same body the JSON endpoint returns
    event: error      the error envelope's "error" object

`result` or `error` is always the last event.

Once the first byte is sent the HTTP status is fixed at 200, so a failure
mid-turn cannot become a 500. It is reported as an `error` event carrying the
same client-safe envelope the exception handlers produce, and logged the same
way. Errors before streaming starts — validation, ownership, rate limiting —
still take the normal JSON error path, because they are raised before the
response object exists.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi.responses import StreamingResponse

from edupilot.core.observability import request_id_var
from edupilot.security import AppError, classify_upstream


def sse_event(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


def error_event(exc: Exception) -> str:
    """Log `exc` like the exception handlers do, and frame it as an error event."""
    error = exc if isinstance(exc, AppError) else classify_upstream(exc)
    error.request_id = request_id_var.get() or error.request_id
    if error.__cause__ is None and error is not exc:
        error.__cause__ = exc
    error.log()
    return sse_event("error", error.to_response()["error"])


def event_stream(events: AsyncIterator[str]) -> StreamingResponse:
    return StreamingResponse(
        events,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            # Stop reverse proxies (nginx, Fly's edge) buffering the stream,
            # which would deliver every token at once at the end.
            "X-Accel-Buffering": "no",
        },
    )
