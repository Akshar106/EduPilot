"""
Alembic environment
===================
Runs migrations against DATABASE_URL, or against the connection the caller
passes in `config.attributes["connection"]` — which is how `db.init_db()`
migrates while holding an advisory lock, so several app processes starting
at once cannot run the same migration twice.
"""

from __future__ import annotations

from alembic import context

from edupilot.db.engine import create
from edupilot.db.tables import metadata

config = context.config


def _include_object(obj, name, type_, reflected, compare_to):
    """
    Leave tables this metadata does not own out of autogenerate.

    The database also holds one vector table per index version and
    LangGraph's checkpoint and store tables, all created at runtime. Without
    this, `--autogenerate` would propose dropping every one of them.
    """
    if type_ == "table" and reflected and compare_to is None:
        return False
    return True


def _run(connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=metadata,
        include_object=_include_object,
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connection = config.attributes.get("connection")
    if connection is not None:
        _run(connection)
        return
    engine = create(pool_size=1, max_overflow=0)
    try:
        with engine.connect() as connection:
            _run(connection)
            connection.commit()
    finally:
        engine.dispose()


if context.is_offline_mode():
    raise SystemExit("offline (--sql) migrations are not supported; run against a database")
run_migrations_online()
