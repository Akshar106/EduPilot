"""
Conversation checkpoints and student profiles
=============================================
Where the agent graph keeps each conversation's working memory between turns
(LangGraph's Postgres checkpointer) and what it remembers about each student
across conversations (LangGraph's Postgres store), both in the app database.

They run on the event loop, so they share one async psycopg pool instead of
the synchronous SQLAlchemy engine the rest of the app uses. The pool's size is
CHECKPOINT_POOL_SIZE per process.

A turn only ever resumes from the latest checkpoint, so the agent prunes the
rest after each turn with the saver's own `aprune(..., strategy="keep_latest")`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from edupilot.core.config import CHECKPOINT_POOL_SIZE, DATABASE_URL
from edupilot.db.engine import LOCK_LANGGRAPH_SETUP, libpq_url


@asynccontextmanager
async def open_persistence(
    url: str = DATABASE_URL, *, max_size: int = CHECKPOINT_POOL_SIZE
) -> AsyncIterator[tuple[AsyncPostgresSaver, AsyncPostgresStore]]:
    """
    Open the checkpointer and the profile store for the app's lifetime.

    Both create their tables on first use. Several processes starting at once
    would race those migrations, so setup runs under an advisory lock.
    """
    pool = AsyncConnectionPool(
        libpq_url(url),
        # One connection holds the setup lock while setup itself uses another.
        min_size=1,
        max_size=max(2, max_size),
        # What LangGraph's savers require: they manage their own transactions,
        # read rows as dicts, and must work through PgBouncer, which does not
        # support prepared statements.
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
        open=False,
    )
    async with pool:
        saver = AsyncPostgresSaver(pool)
        store = AsyncPostgresStore(pool)
        async with pool.connection() as conn:
            await conn.execute("SELECT pg_advisory_lock(%s)", (LOCK_LANGGRAPH_SETUP,))
            try:
                await saver.setup()
                await store.setup()
            finally:
                await conn.execute("SELECT pg_advisory_unlock(%s)", (LOCK_LANGGRAPH_SETUP,))
        yield saver, store
