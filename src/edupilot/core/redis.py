"""
Redis
=====
The async Redis client for state every API process must share: rate-limit
buckets and the global turn limit. Optional — with REDIS_URL unset, those
fall back to per-process memory.

Clients are cached per event loop. An asyncio connection belongs to the loop
that opened it, and a process can run more than one loop over its life (the
test suite starts an app per test client).
"""

from __future__ import annotations

import asyncio
import logging
import threading
import weakref

from edupilot.core.config import REDIS_URL

logger = logging.getLogger(__name__)

_clients: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_lock = threading.Lock()


def enabled() -> bool:
    return bool(REDIS_URL)


def client():
    """The current loop's client, or None when Redis is not configured."""
    if not REDIS_URL:
        return None
    loop = asyncio.get_running_loop()
    with _lock:
        existing = _clients.get(loop)
        if existing is None:
            from redis.asyncio import Redis

            # Short timeouts: these calls sit in front of every request, and
            # a slow Redis must degrade to the in-process fallback, not stall
            # traffic.
            existing = Redis.from_url(
                REDIS_URL, socket_timeout=0.5, socket_connect_timeout=0.5, health_check_interval=30
            )
            _clients[loop] = existing
        return existing


async def ping() -> bool:
    redis = client()
    if redis is None:
        return False
    try:
        return bool(await redis.ping())
    except Exception:
        return False


async def close() -> None:
    """Close the current loop's client. Called at shutdown."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    with _lock:
        existing = _clients.pop(loop, None)
    if existing is not None:
        await existing.aclose()
