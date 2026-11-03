"""
Schema migrations
=================
Alembic owns the schema. `init_db()` runs on every startup and brings the
database to the latest revision; with nothing to do it costs one query.

Several API processes may start at once against the same database, so the
upgrade runs under a Postgres advisory lock: the first process migrates, the
rest wait for it and then find nothing left to do. Two processes running the
same `CREATE TABLE` concurrently would otherwise fail one of them.

New migration: change db/tables.py, then
`edupilot-db revision -m "what changed"` and review the generated file.
"""

from __future__ import annotations

import logging
from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory

from .engine import LOCK_MIGRATIONS, advisory_lock, get_engine

logger = logging.getLogger(__name__)

MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"


def alembic_config() -> Config:
    """Alembic configuration for the packaged migrations; no alembic.ini needed."""
    cfg = Config()
    cfg.set_main_option("script_location", str(MIGRATIONS_DIR))
    return cfg


def head_revision() -> str | None:
    return ScriptDirectory.from_config(alembic_config()).get_current_head()


def current_revision() -> str | None:
    with get_engine().connect() as conn:
        return MigrationContext.configure(conn).get_current_revision()


def init_db() -> None:
    """Upgrade the database to the latest schema revision."""
    cfg = alembic_config()
    head = head_revision()
    with get_engine().connect() as conn, advisory_lock(conn, LOCK_MIGRATIONS):
        before = MigrationContext.configure(conn).get_current_revision()
        conn.commit()
        if before == head:
            return
        cfg.attributes["connection"] = conn
        command.upgrade(cfg, "head")
        conn.commit()
    logger.info("database schema migrated %s -> %s", before or "(empty)", head)
