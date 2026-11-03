"""
Database engine
===============
The one SQLAlchemy engine every synchronous EduPilot component shares: the
app tables, the user store, the document registry, the index pointer, the
caches and the pgvector store. One pool per process, so the number of
connections the app opens is set in one place (DB_POOL_SIZE, DB_MAX_OVERFLOW)
rather than once per component.

The async LangGraph checkpointer and store open their own psycopg pool
(agents/checkpoint.py), since they run on the event loop.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from sqlalchemy import Connection, Engine, create_engine, text

from edupilot.core.config import (
    DATABASE_URL,
    DB_MAX_OVERFLOW,
    DB_POOL_SIZE,
    DB_POOL_TIMEOUT,
)

_engine: Engine | None = None
_lock = threading.Lock()

_SCHEMES = ("postgresql+psycopg://", "postgresql://", "postgres://")


def _split(url: str) -> str:
    for scheme in _SCHEMES:
        if url.startswith(scheme):
            return url[len(scheme):]
    raise ValueError(
        "DATABASE_URL must be a Postgres URL (postgresql://user:password@host:5432/db)"
    )


def sqlalchemy_url(url: str = DATABASE_URL) -> str:
    """`url` with the driver SQLAlchemy needs (psycopg 3)."""
    return "postgresql+psycopg://" + _split(url)


def libpq_url(url: str = DATABASE_URL) -> str:
    """`url` as a plain libpq connection string, for psycopg itself."""
    return "postgresql://" + _split(url)


def _without_nul(value: Any) -> Any:
    """
    Strip NUL characters, which Postgres refuses in text and JSONB.

    Extracted text is already cleaned of control characters, but message
    metadata carries model output and tool results too, and one stray NUL
    would fail the whole insert.
    """
    if isinstance(value, str):
        return value.replace("\x00", "")
    if isinstance(value, dict):
        return {k: _without_nul(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_without_nul(v) for v in value]
    return value


def json_dumps(value: Any) -> str:
    """JSON for JSONB columns. Anything not JSON-native is stored as its str()."""
    return json.dumps(_without_nul(value), default=str)


def create(url: str = DATABASE_URL, **overrides: Any) -> Engine:
    """A new engine for `url`. Most code wants the shared `get_engine()`."""
    options: dict[str, Any] = {
        "pool_size": DB_POOL_SIZE,
        "max_overflow": DB_MAX_OVERFLOW,
        "pool_timeout": DB_POOL_TIMEOUT,
        # A pooled connection can be dropped by the server or a proxy while
        # idle; checking on checkout costs one round trip and turns that into
        # a reconnect instead of a failed request.
        "pool_pre_ping": True,
        # Managed Postgres and PgBouncer close long-lived connections.
        "pool_recycle": 1800,
        "json_serializer": json_dumps,
        "connect_args": {"application_name": "edupilot"},
    }
    options.update(overrides)
    return create_engine(sqlalchemy_url(url), **options)


def get_engine() -> Engine:
    """The process-wide engine, created on first use."""
    global _engine
    if _engine is None:
        with _lock:
            if _engine is None:
                _engine = create()
    return _engine


def dispose_engine() -> None:
    """Close every pooled connection. Called at shutdown and between tests."""
    global _engine
    with _lock:
        if _engine is not None:
            _engine.dispose()
            _engine = None


@contextmanager
def transaction() -> Iterator[Connection]:
    """A connection in a transaction: committed on success, rolled back on error."""
    with get_engine().begin() as conn:
        yield conn


def ping() -> bool:
    """True when the database answers."""
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


@contextmanager
def advisory_lock(conn: Connection, key: int, *, wait: bool = True) -> Iterator[bool]:
    """
    Hold a session-level Postgres advisory lock for the block.

    Serializes work that must happen once across every process sharing the
    database — schema migrations at startup, an index rebuild. Yields whether
    the lock was taken; with `wait=False` it may not be.
    """
    if wait:
        conn.execute(text("SELECT pg_advisory_lock(:k)"), {"k": key})
        acquired = True
    else:
        acquired = bool(conn.execute(text("SELECT pg_try_advisory_lock(:k)"), {"k": key}).scalar())
    conn.commit()
    try:
        yield acquired
    finally:
        if acquired:
            conn.execute(text("SELECT pg_advisory_unlock(:k)"), {"k": key})
            conn.commit()


#: Advisory lock keys. Arbitrary, but fixed: every process must agree.
LOCK_MIGRATIONS = 7_210_001
LOCK_REBUILD = 7_210_002
LOCK_LANGGRAPH_SETUP = 7_210_003
