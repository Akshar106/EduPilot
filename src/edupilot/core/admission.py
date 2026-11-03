"""
Admission control
=================
A bound on how many turns run at once, so overload sheds requests instead of
slowing every one of them down.

A turn — one question answered — is several model calls plus CPU-bound
embedding and reranking. Admit them all during a burst and every turn slows
together, the provider starts rate-limiting the whole batch, and timeouts
arrive before any student gets an answer. With a bound, up to N turns run at
full speed, the rest wait their turn briefly, and once a wait would exceed
ADMISSION_TIMEOUT_SECONDS the API answers 503 with Retry-After, which the
frontend shows and retries.

Two bounds stack:

  per process    MAX_CONCURRENT_TURNS — CPU and memory are per process
  global         GLOBAL_MAX_TURNS, across every process sharing REDIS_URL —
                 the model provider's rate limit is per API key, not per
                 process, so it has to be counted in one place

The global bound is a lease set in Redis: each admitted turn adds an entry
that expires on its own after LEASE_SECONDS, so a process that crashes
mid-turn cannot leak a slot forever. If Redis is unreachable the global bound
is skipped and the per-process one still holds.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
import weakref
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from edupilot.core import metrics
from edupilot.core import redis as shared
from edupilot.core.config import (
    ADMISSION_TIMEOUT_SECONDS,
    GLOBAL_MAX_TURNS,
    MAX_CONCURRENT_TURNS,
)

logger = logging.getLogger(__name__)

#: A turn that has not finished after this long releases its global slot.
LEASE_SECONDS = 180

#: What Retry-After says when a turn is refused: long enough for a turn or two
#: to finish, short enough that the student is not left waiting idle.
RETRY_AFTER_SECONDS = 5

_GLOBAL_KEY = "edupilot:turns"

#: KEYS[1] lease set; ARGV limit, lease seconds, token. Returns 1 if admitted.
_ACQUIRE = """
local t = redis.call('TIME')
local now = tonumber(t[1]) + tonumber(t[2]) / 1000000
redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', now)
if redis.call('ZCARD', KEYS[1]) < tonumber(ARGV[1]) then
  redis.call('ZADD', KEYS[1], now + tonumber(ARGV[2]), ARGV[3])
  redis.call('EXPIRE', KEYS[1], tonumber(ARGV[2]))
  return 1
end
return 0
"""


class Overloaded(Exception):
    """No slot became free in time."""

    def __init__(self, retry_after: int = RETRY_AFTER_SECONDS) -> None:
        self.retry_after = retry_after
        super().__init__(f"no capacity; retry in {retry_after}s")


class TurnLimiter:
    """
    Per-process and global bounds on concurrent turns.

    Args:
        local_limit: Turns this process runs at once.
        global_limit: Turns all processes run at once, via Redis. 0 disables.
        timeout: Seconds a turn may wait for both slots.
    """

    def __init__(
        self,
        local_limit: int = MAX_CONCURRENT_TURNS,
        global_limit: int = GLOBAL_MAX_TURNS,
        timeout: float = ADMISSION_TIMEOUT_SECONDS,
    ) -> None:
        self.local_limit = max(1, local_limit)
        self.global_limit = global_limit
        self.timeout = timeout
        # A semaphore belongs to the loop it is first awaited on.
        self._semaphores: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()

    def _semaphore(self) -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        semaphore = self._semaphores.get(loop)
        if semaphore is None:
            semaphore = self._semaphores[loop] = asyncio.Semaphore(self.local_limit)
        return semaphore

    async def _acquire_global(self, token: str, deadline: float) -> bool:
        """Take a global lease before `deadline`. False when Redis is unusable."""
        redis = shared.client()
        if redis is None or self.global_limit <= 0:
            return False
        delay = 0.05
        while True:
            try:
                if await redis.eval(_ACQUIRE, 1, _GLOBAL_KEY, self.global_limit, LEASE_SECONDS, token):
                    return True
            except Exception as exc:
                logger.warning("global turn limit skipped, Redis unavailable: %s", exc)
                return False
            if time.monotonic() + delay > deadline:
                raise Overloaded()
            await asyncio.sleep(delay)
            delay = min(delay * 2, 0.5)

    def release_in_background(self, token: str) -> asyncio.Task:
        """
        Drop a global lease in a task of its own.

        Releases often happen while the request is being cancelled, and every
        await in a cancelled scope is cancelled too; a separate task runs to
        completion regardless. The set keeps a reference until it does.
        """
        task = asyncio.get_running_loop().create_task(self._release_global(token))
        _releasing.add(task)
        task.add_done_callback(_releasing.discard)
        return task

    async def _release_global(self, token: str) -> None:
        redis = shared.client()
        if redis is None:
            return
        try:
            await redis.zrem(_GLOBAL_KEY, token)
        except Exception as exc:  # the lease expires on its own
            logger.warning("could not release a global turn slot: %s", exc)

    async def admit(self) -> Ticket:
        """
        Wait for a slot and return the ticket that holds it.

        Raises:
            Overloaded: no slot became free within the timeout.
        """
        started = time.monotonic()
        deadline = started + self.timeout
        semaphore = self._semaphore()

        metrics.TURNS_WAITING.inc()
        try:
            await asyncio.wait_for(semaphore.acquire(), timeout=self.timeout)
        except TimeoutError:
            metrics.ADMISSION_REJECTED.inc()
            raise Overloaded() from None
        finally:
            metrics.TURNS_WAITING.dec()

        token = uuid.uuid4().hex
        try:
            leased = await self._acquire_global(token, deadline)
        except Overloaded:
            semaphore.release()
            metrics.ADMISSION_REJECTED.inc()
            raise
        except BaseException:
            # Cancelled while waiting — the client went away. Give the local
            # slot back, and drop the lease too: Redis may have granted it in
            # the instant before the cancellation arrived.
            semaphore.release()
            self.release_in_background(token)
            raise

        metrics.ADMISSION_WAIT.observe(time.monotonic() - started)
        metrics.TURNS_RUNNING.inc()
        return Ticket(self, semaphore, token if leased else None)

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        """Hold a slot for the duration of one turn. Raises Overloaded."""
        ticket = await self.admit()
        try:
            yield
        finally:
            await ticket.release()


#: Global releases in flight; a reference keeps each task alive until it ends.
_releasing: set[asyncio.Task] = set()


class Ticket:
    """
    One admitted turn's slot. Release it exactly once.

    A streamed turn takes its slot before the response starts, so overload can
    still be a 503, and releases it when the stream ends. Should the stream
    never run at all, `release_now` from a finalizer returns the local slot;
    the global lease then expires on its own.
    """

    def __init__(self, limiter: TurnLimiter, semaphore: asyncio.Semaphore, token: str | None) -> None:
        self._limiter = limiter
        self._semaphore = semaphore
        self._token = token
        self._released = False

    def release_now(self) -> None:
        """Release the local slot synchronously; the global lease is left to expire."""
        if self._released:
            return
        self._released = True
        metrics.TURNS_RUNNING.dec()
        self._semaphore.release()

    async def release(self) -> None:
        if self._released:
            return
        token = self._token
        self.release_now()
        if not token:
            return
        # Often called while the request is being cancelled — the student
        # closed the tab mid-answer — so the Redis call runs as its own task;
        # shield() lets this caller stop waiting without stopping it.
        try:
            await asyncio.shield(self._limiter.release_in_background(token))
        except asyncio.CancelledError:
            pass


def guard(stream, ticket: Ticket):
    """Release `ticket` if `stream` is discarded without ever being run."""
    weakref.finalize(stream, ticket.release_now)
    return stream


#: Process-wide limiter.
turns = TurnLimiter()
