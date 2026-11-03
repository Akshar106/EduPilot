"""
The course-chat endpoints: one turn of the multi-agent graph.

`POST /api/chat` returns the finished turn as JSON. `POST /api/chat/stream`
runs the same turn and streams it as server-sent events — stage progress and
answer tokens while it runs, then the identical JSON payload as the final
`result` event (see `api/sse.py`).

Both are async end to end. The graph awaits the model providers on the event
loop and moves only the blocking work — vector search, reranking, the
grounding check — onto worker threads, so a waiting turn no longer occupies
one of a handful of pool threads for its whole 20-45 seconds. Database calls
go to the worker pool too: against a managed Postgres each is a network round
trip, and one on the event loop stalls every other request in the process.

A turn runs only once the admission limiter (core/admission.py) gives it a
slot; past the wait limit the request is refused with 503 and Retry-After
instead of joining an unbounded backlog.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from fastapi import APIRouter, Depends

from edupilot import db
from edupilot.api.deps import CurrentUser, retrieval_config, run_blocking, validate_model
from edupilot.api.schemas import ChatRequest
from edupilot.api.sse import error_event, event_stream, sse_event
from edupilot.core import metrics, tracing
from edupilot.core.admission import Overloaded, Ticket, guard, turns
from edupilot.core.config import VERIFY_MODEL
from edupilot.core.services import services
from edupilot.security import AppError, ErrorCode, User, owns_or_admin, rate_limited

router = APIRouter(prefix="/api", tags=["chat"])


@dataclass
class Turn:
    session_id: str
    user_msg_id: int
    standalone: bool
    """No earlier messages in the conversation — the only case the cache serves."""
    ask_count: int = 0
    started: float = field(default_factory=time.perf_counter)


def overloaded(exc: Overloaded) -> AppError:
    return AppError(
        code=ErrorCode.OVERLOADED,
        internal="turn refused: no admission slot within the timeout",
        details={"retry_after": exc.retry_after},
    )


def finished(kind: str, payload: dict, started: float) -> None:
    """Record one turn's duration, by how it ended."""
    if (payload.get("debug") or {}).get("cache", {}).get("hit"):
        outcome = "cached"
    elif payload.get("refused"):
        outcome = "refused"
    else:
        outcome = "answered"
    metrics.TURN_DURATION.labels(kind, outcome).observe(time.perf_counter() - started)


def _open_turn(req: ChatRequest, user: User) -> Turn:
    """Resolve or create the session, check ownership, and persist the question."""
    validate_model(req.model)

    if req.session_id:
        owner = db.get_session_owner(req.session_id)
        if owner is None:
            # Unknown id — create it for this caller rather than 404ing, so a
            # client that generated an id locally still works.
            db.ensure_session(req.session_id, user.user_id)
        else:
            owns_or_admin(user, owner, what="chat session")
        session_id = req.session_id
    else:
        session_id = str(uuid.uuid4())
        db.ensure_session(session_id, user.user_id)

    user_msg_id = db.save_message(session_id, "user", req.query)
    standalone = len(db.get_session_messages(session_id)) == 1
    if standalone:
        db.update_session_title(session_id, req.query[:60])
    return Turn(session_id=session_id, user_msg_id=user_msg_id, standalone=standalone)


def _cached_reply(req: ChatRequest, turn: Turn) -> dict | None:
    """
    A cached payload for this question, or None. Runs in the worker pool.

    The cache is only consulted for *standalone* questions — those asked with
    no prior turns in the conversation. "Explain that in more detail"
    normalizes to the same key no matter what "that" was, so caching by
    question text alone would serve one student the answer to another
    student's follow-up. A fresh conversation has no such ambiguity.
    """
    if not turn.standalone:
        return None
    cache = services.answer_cache
    turn.ask_count = cache.record_ask(req.query, domains=req.manual_domains, model=req.model)
    hit = cache.lookup(req.query, domains=req.manual_domains, model=req.model)
    metrics.ANSWER_CACHE.labels("miss" if hit is None else "hit").inc()
    if hit is None:
        return None

    cached = dict(hit.payload)
    cached["session_id"] = turn.session_id
    cached["debug"] = {
        "cache": {
            "hit": True,
            "similarity": round(hit.similarity, 4),
            "ask_count": hit.ask_count,
            "age_seconds": int(hit.age_seconds),
        },
    }
    return cached


def _payload(req: ChatRequest, turn: Turn, result, usage) -> dict:
    """The response body for a turn the graph answered. Runs in the worker pool."""
    diagnostics = dict(result.diagnostics)
    diagnostics["usage"] = usage.as_dict()
    # Persisted with the message, so a later rating can find the trace
    # without trusting a run id sent by the client.
    diagnostics["trace"] = {"run_id": result.run_id, "traced": tracing.enabled()}

    payload = {
        "session_id": turn.session_id,
        "final_answer": result.final_answer,
        "intent_type": result.intent_type,
        "detected_domains": result.domains,
        "is_course_related": result.is_course_related,
        "needs_clarification": result.needs_clarification,
        # The turn is paused on a clarifying question; the student's next
        # message in this session answers it.
        "awaiting_clarification": result.awaiting_clarification,
        "refused": result.refused,
        "task": result.task,
        # None when grounding was not measured. Never defaulted, and never
        # floored.
        "grounding_score": result.grounding_score,
        "guardrail_action": result.guardrail_action,
        "verification_revised": result.verification_revised,
        "sources": result.sources,
        # Quiz questions or flashcards as data, for an interactive view.
        "practice": result.practice,
    }

    # Promote only once the question has been asked PROMOTE_AFTER times, and
    # only when the answer is one worth freezing — see should_store.
    if turn.standalone:
        cache = services.answer_cache
        if cache.should_store(turn.ask_count, payload):
            cache.store(req.query, payload, domains=req.manual_domains, model=req.model)

    diagnostics["cache"] = {"hit": False, "ask_count": turn.ask_count}
    payload["debug"] = diagnostics
    return payload


def _record_reply(turn: Turn, payload: dict) -> dict:
    payload["assistant_message_id"] = db.save_message(
        session_id=turn.session_id,
        role="assistant",
        content=payload["final_answer"],
        intent_type=payload.get("intent_type"),
        detected_domains=payload.get("detected_domains"),
        quality_score=payload.get("grounding_score"),
        # Practice data is kept with the message so a reloaded conversation
        # can still show the quiz interactively.
        pipeline_meta={**(payload.get("debug") or {}), "practice": payload.get("practice")},
    )
    payload["user_message_id"] = turn.user_msg_id
    return payload


def _turn_args(req: ChatRequest, turn: Turn, user: User, usage) -> tuple[tuple, dict]:
    """Arguments for `EduPilotAgent.arun` / `astream`."""
    from edupilot.agents import PipelineConfig

    cfg = PipelineConfig(
        model=req.model,
        verify_model=VERIFY_MODEL,
        retrieval=retrieval_config(req.top_k, req.rerank_top_k),
        enable_verification=req.enable_verification,
        # A chat session can answer a clarifying question on its next turn.
        interactive=True,
        user_id=user.user_id,
    )
    # History comes from the conversation's checkpoint (seeded from the
    # database), not `req.chat_history`. The client's copy is a truncated view
    # it happens to hold, and a client could send anything at all.
    kwargs = {
        "thread_id": turn.session_id,
        "session_id": turn.session_id,
        # The question for this turn is already persisted; it must not appear
        # in its own history block.
        "exclude_message_ids": [turn.user_msg_id],
        "manual_domains": req.manual_domains,
        "filenames": req.attached_filenames,
        "task": req.task,
        "callbacks": [usage],
    }
    return (req.query, cfg), kwargs


@router.post("/chat", dependencies=[Depends(rate_limited("chat"))])
async def chat(req: ChatRequest, user: CurrentUser):
    from edupilot.llm import UsageTracker

    # Capacity first: a turn refused for lack of a slot must not have saved
    # the question (an unanswered message left in the conversation) or spent
    # CPU embedding it for the cache.
    started = time.perf_counter()
    try:
        ticket = await turns.admit()
    except Overloaded as exc:
        raise overloaded(exc) from None
    try:
        turn = await run_blocking(_open_turn, req, user)
        cached = await run_blocking(_cached_reply, req, turn)
        if cached is not None:
            await ticket.release()
            reply = await run_blocking(_record_reply, turn, cached)
            finished("chat", reply, turn.started)
            return reply
        usage = UsageTracker()
        args, kwargs = _turn_args(req, turn, user, usage)
        result = await services.agent.arun(*args, **kwargs)
    except AppError:
        raise
    except Exception:
        metrics.TURN_DURATION.labels("chat", "error").observe(time.perf_counter() - started)
        raise
    finally:
        await ticket.release()
    payload = await run_blocking(_payload, req, turn, result, usage)
    reply = await run_blocking(_record_reply, turn, payload)
    finished("chat", reply, turn.started)
    return reply


@router.post("/chat/stream", dependencies=[Depends(rate_limited("chat"))])
async def chat_stream(req: ChatRequest, user: CurrentUser):
    # Capacity, ownership and validation are settled here, before streaming
    # starts, so each can still answer with a proper status code. Capacity
    # comes first: a refused turn saves nothing and computes nothing.
    try:
        ticket = await turns.admit()
    except Overloaded as exc:
        raise overloaded(exc) from None
    try:
        turn = await run_blocking(_open_turn, req, user)
        cached = await run_blocking(_cached_reply, req, turn)
    except BaseException:
        await ticket.release()
        raise
    if cached is not None:
        await ticket.release()
        return event_stream(_cached_events(turn, cached))
    return event_stream(guard(_chat_events(req, turn, user, ticket), ticket))


async def _cached_events(turn: Turn, cached: dict) -> AsyncIterator[str]:
    try:
        reply = await run_blocking(_record_reply, turn, cached)
        finished("chat", reply, turn.started)
        yield sse_event("result", reply)
    except Exception as exc:
        yield error_event(exc)


async def _chat_events(req: ChatRequest, turn: Turn, user: User, ticket: Ticket) -> AsyncIterator[str]:
    from edupilot.llm import UsageTracker

    first_token = True
    try:
        usage = UsageTracker()
        args, kwargs = _turn_args(req, turn, user, usage)
        async for event in services.agent.astream(*args, **kwargs):
            kind = event.pop("type")
            if kind == "result":
                await ticket.release()
                payload = await run_blocking(_payload, req, turn, event["result"], usage)
                reply = await run_blocking(_record_reply, turn, payload)
                finished("chat", reply, turn.started)
                yield sse_event("result", reply)
            else:
                if kind == "token" and first_token:
                    first_token = False
                    metrics.TIME_TO_FIRST_TOKEN.labels("chat").observe(time.perf_counter() - turn.started)
                yield sse_event(kind, event)
    except Exception as exc:
        metrics.TURN_DURATION.labels("chat", "error").observe(time.perf_counter() - turn.started)
        yield error_event(exc)
    finally:
        await ticket.release()
