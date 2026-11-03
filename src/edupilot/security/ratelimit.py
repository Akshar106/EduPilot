"""
Rate limiting
=============
Token-bucket limiter keyed by user (or IP for unauthenticated routes).

Nothing throttled the previous API. Two endpoints made that expensive rather
than merely rude:

  POST /api/chat      every call runs router + planner + N generations +
                      verifier against a metered LLM API
  POST /api/evaluate  runs the entire 50-case suite — hundreds of LLM calls —
                      unauthenticated

A single scripted client could exhaust the daily quota in under a minute,
denying service to everyone and, on a paid key, running up the bill.

With REDIS_URL set, buckets live in Redis and every API process draws from
the same ones, so a student's limit is the same with one instance or ten.
Without it they are per-process memory — right for a single process, and the
effective limit multiplies by the process count otherwise.

If Redis stops answering, requests are let through (and counted in
`edupilot_rate_limited_total{scope="redis_unavailable"}`) rather than all
refused: the limit protects the service, and refusing everyone would be the
outage it exists to prevent.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

from edupilot.core import metrics
from edupilot.core import redis as shared

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RateLimit:
    """A bucket's capacity and refill rate."""

    requests: int
    """Burst capacity."""
    per_seconds: float
    """Window over which the bucket fully refills."""

    @property
    def refill_per_second(self) -> float:
        return self.requests / self.per_seconds

    def __str__(self) -> str:
        return f"{self.requests}/{self.per_seconds:g}s"


#: Per-endpoint-class limits. Chat and evaluation are throttled hardest
#: because each request fans out into many upstream LLM calls.
LIMITS: dict[str, RateLimit] = {
    "chat": RateLimit(requests=20, per_seconds=60),
    "upload": RateLimit(requests=10, per_seconds=300),
    "evaluate": RateLimit(requests=2, per_seconds=3600),
    "auth": RateLimit(requests=8, per_seconds=300),
    "read": RateLimit(requests=120, per_seconds=60),
    "default": RateLimit(requests=60, per_seconds=60),
}


class RateLimitExceeded(Exception):
    """Caller exceeded their allowance."""

    def __init__(self, scope: str, retry_after: float) -> None:
        self.scope = scope
        self.retry_after = max(1, int(retry_after + 0.999))
        super().__init__(
            f"Rate limit exceeded for '{scope}'. Retry in {self.retry_after}s."
        )


@dataclass
class _Bucket:
    tokens: float
    last_refill: float


class RateLimiter:
    """
    In-memory token-bucket limiter.

    Token buckets are used rather than fixed windows because they permit a
    short burst — a student sending three questions quickly — while still
    bounding the sustained rate. A fixed window would reject the third
    question and also allow a 2x burst across a window boundary.

    `acheck` is what request handlers call. It uses the Redis buckets when
    REDIS_URL is set and these in-memory ones otherwise.
    """

    def __init__(self, limits: dict[str, RateLimit] | None = None) -> None:
        self._limits = limits or LIMITS
        self._buckets: dict[tuple[str, str], _Bucket] = {}
        self._lock = threading.Lock()
        self._last_sweep = time.monotonic()

    def limit_for(self, scope: str) -> RateLimit:
        return self._limits.get(scope, self._limits["default"])

    def _sweep(self, now: float) -> None:
        """Drop buckets that have been full and idle. Caller holds the lock."""
        if now - self._last_sweep < 300:
            return
        stale = [
            key for key, bucket in self._buckets.items()
            if now - bucket.last_refill > 3600
        ]
        for key in stale:
            del self._buckets[key]
        self._last_sweep = now
        if stale:
            logger.debug("swept %d idle rate-limit buckets", len(stale))

    def check(self, identity: str, scope: str = "default", *, cost: float = 1.0) -> None:
        """
        Consume `cost` tokens for `identity` in `scope`.

        Raises:
            RateLimitExceeded: not enough tokens are available.
        """
        limit = self.limit_for(scope)
        key = (identity, scope)
        now = time.monotonic()

        with self._lock:
            self._sweep(now)
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = _Bucket(tokens=float(limit.requests), last_refill=now)
                self._buckets[key] = bucket

            elapsed = now - bucket.last_refill
            bucket.tokens = min(
                float(limit.requests), bucket.tokens + elapsed * limit.refill_per_second
            )
            bucket.last_refill = now

            if bucket.tokens < cost:
                deficit = cost - bucket.tokens
                retry_after = deficit / limit.refill_per_second
                logger.info(
                    "rate limit hit: identity=%s scope=%s limit=%s retry_after=%.1fs",
                    identity[:24], scope, limit, retry_after,
                )
                raise RateLimitExceeded(scope, retry_after)

            bucket.tokens -= cost

    async def acheck(self, identity: str, scope: str = "default", *, cost: float = 1.0) -> None:
        """`check`, against the shared Redis buckets when Redis is configured."""
        redis = shared.client()
        if redis is None:
            self.check(identity, scope, cost=cost)
            return

        limit = self.limit_for(scope)
        try:
            allowed, retry_after = await redis.eval(
                _TOKEN_BUCKET,
                1,
                f"edupilot:ratelimit:{scope}:{identity}",
                limit.requests,
                limit.refill_per_second,
                cost,
                int(limit.per_seconds * 2) + 60,
            )
        except Exception as exc:
            metrics.RATE_LIMITED.labels("redis_unavailable").inc()
            logger.warning("rate limit check skipped, Redis unavailable: %s", exc)
            return
        if not int(allowed):
            logger.info(
                "rate limit hit: identity=%s scope=%s limit=%s", identity[:24], scope, limit
            )
            raise RateLimitExceeded(scope, float(retry_after))

    def remaining(self, identity: str, scope: str = "default") -> int:
        """Tokens currently available. For the X-RateLimit-Remaining header."""
        limit = self.limit_for(scope)
        with self._lock:
            bucket = self._buckets.get((identity, scope))
            if bucket is None:
                return limit.requests
            elapsed = time.monotonic() - bucket.last_refill
            return int(
                min(float(limit.requests), bucket.tokens + elapsed * limit.refill_per_second)
            )

    def reset(self, identity: str | None = None) -> None:
        """Clear buckets. Test support, and for lifting a limit manually."""
        with self._lock:
            if identity is None:
                self._buckets.clear()
            else:
                for key in [k for k in self._buckets if k[0] == identity]:
                    del self._buckets[key]


#: A token bucket in one atomic step. The clock is Redis's own, so API
#: processes whose clocks disagree still refill the bucket at one rate.
#: KEYS[1] bucket; ARGV capacity, refill per second, cost, expiry seconds.
#: Returns {allowed, seconds until enough tokens}.
_TOKEN_BUCKET = """
local capacity = tonumber(ARGV[1])
local rate = tonumber(ARGV[2])
local cost = tonumber(ARGV[3])
local t = redis.call('TIME')
local now = tonumber(t[1]) + tonumber(t[2]) / 1000000
local state = redis.call('HMGET', KEYS[1], 'tokens', 'ts')
local tokens = tonumber(state[1]) or capacity
local ts = tonumber(state[2]) or now
tokens = math.min(capacity, tokens + math.max(0, now - ts) * rate)
local allowed = 0
local wait = 0
if tokens >= cost then
  tokens = tokens - cost
  allowed = 1
else
  wait = (cost - tokens) / rate
end
redis.call('HSET', KEYS[1], 'tokens', tostring(tokens), 'ts', tostring(now))
redis.call('EXPIRE', KEYS[1], tonumber(ARGV[4]))
return {allowed, tostring(wait)}
"""


#: Process-wide limiter.
limiter = RateLimiter()
