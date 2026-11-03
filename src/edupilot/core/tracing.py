"""
Tracing
=======
LangSmith tracing for agent turns, plus the two things the app does with a
trace besides record it: link to it from the debug panel, and attach a
student's rating to it.

Recording needs no code. LangChain and LangGraph send every graph step, model
call and tool call to LangSmith when `LANGSMITH_TRACING=true` and
`LANGSMITH_API_KEY` are set. What the app adds is what makes those traces
usable: a run id chosen up front for each turn, so feedback and links can find
the trace later; metadata to filter and compare by (endpoint, model, feature
switches); and a flush at shutdown so the last turns are not dropped.

Every function here is a no-op when tracing is off, and none of them can fail
a request — a LangSmith outage costs a trace, never an answer.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import threading
import uuid

logger = logging.getLogger(__name__)

#: Feedback key the student's 👍/👎 is recorded under in LangSmith.
FEEDBACK_KEY = "student_rating"


def enabled() -> bool:
    """True when runs are being sent to LangSmith."""
    if not (os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")):
        return False
    from langsmith.utils import tracing_is_enabled

    return tracing_is_enabled() is True


def project_name() -> str:
    return os.getenv("LANGSMITH_PROJECT") or os.getenv("LANGCHAIN_PROJECT") or "edupilot"


def new_run_id() -> str:
    return str(uuid.uuid4())


def pseudonym(user_id: str | None) -> str | None:
    """
    A stable stand-in for a user id, safe to put in trace metadata.

    Traces are filterable by it — "every turn this student had" — without the
    trace store holding the account id itself.
    """
    if not user_id:
        return None
    return hashlib.sha256(f"edupilot-user:{user_id}".encode()).hexdigest()[:16]


def feedback_id(kind: str, message_id: int, user_id: str | None) -> str:
    """
    Deterministic id for one student's rating of one answer.

    Rating the same answer again updates that feedback instead of adding a
    second vote, so LangSmith's aggregates count each answer once.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"edupilot:{kind}:{message_id}:{user_id or ''}"))


# ---------------------------------------------------------------------------
# LangSmith client
# ---------------------------------------------------------------------------

_client = None
_client_lock = threading.Lock()
_project_id: str | None = None


def client():
    """The shared LangSmith client, built on first use."""
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                from langsmith import Client

                _client = Client()
    return _client


def send_feedback(
    run_id: str, *, rating: int, comment: str | None, feedback_key: str, update: bool
) -> bool:
    """
    Attach a student's rating to a traced turn. Blocking; run it off the loop.

    The score is 1 for 👍 and 0 for 👎, so LangSmith's average reads as the
    share of positive ratings.
    """
    if not enabled():
        return False
    score = 1 if rating > 0 else 0
    try:
        if update:
            client().update_feedback(feedback_key, score=score, comment=comment)
        else:
            client().create_feedback(
                run_id,
                FEEDBACK_KEY,
                score=score,
                comment=comment,
                trace_id=run_id,
                feedback_id=feedback_key,
            )
        return True
    except Exception as exc:
        logger.warning("could not send feedback for run %s to LangSmith: %s", run_id, exc)
        return False


async def trace_url(run_id: str) -> str | None:
    """The LangSmith page for a turn's trace, or None if it cannot be resolved."""
    global _project_id
    if not enabled():
        return None
    try:
        ls = client()
        if _project_id is None:
            project = await asyncio.to_thread(ls.read_project, project_name=project_name())
            _project_id = str(project.id)
        # A turn's run id is its root run, so it is also the trace id.
        response = await ls.runs.get_url(run_id, project_id=_project_id, trace_id=run_id)
        return response.url
    except Exception as exc:
        logger.warning("could not resolve the LangSmith URL for run %s: %s", run_id, exc)
        return None


def flush() -> None:
    """Send traces still queued in the background. Called at shutdown."""
    if not enabled():
        return
    try:
        from langchain_core.tracers.langchain import wait_for_all_tracers

        wait_for_all_tracers()
    except Exception as exc:
        logger.warning("could not flush pending LangSmith traces: %s", exc)
