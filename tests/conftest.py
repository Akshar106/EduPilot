"""
Test configuration
==================
Pins the run to throwaway state *before* any edupilot module is imported, so
a test run can never read or write the developer's real database, corpus, or
uploads:

  * EDUPILOT_DATA_DIR points at a temporary directory.
  * DATABASE_URL points at a Postgres database that exists only for this run:
    a fresh database on the server named by EDUPILOT_TEST_DATABASE_URL (what
    CI does, with a service container), or otherwise a pgvector container
    started here with testcontainers, which needs Docker.

This has to happen before collection — `edupilot.core.config` resolves both
once, at import, and every other module reads them from there.
"""

from __future__ import annotations

import os
import tempfile
import uuid

_TMP_DATA = tempfile.mkdtemp(prefix="edupilot-test-data-")
os.environ["EDUPILOT_DATA_DIR"] = _TMP_DATA
os.environ.setdefault("JWT_SECRET_KEY", "test-only-key-not-used-in-production-0123456789")
os.environ.setdefault("EDUPILOT_ENV", "development")
# Never trace a test run, even when the developer's .env turns LangSmith on:
# python-dotenv does not override variables that are already set.
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
# Tests run ingestion jobs themselves (jobs.run_once), with stand-in indexers;
# an in-process worker would pick them up first, with the real embedder.
os.environ["INGEST_WORKER"] = "external"
# Small pools: tests are sequential, and the server is shared with CI jobs.
os.environ.setdefault("DB_POOL_SIZE", "3")
os.environ.setdefault("CHECKPOINT_POOL_SIZE", "3")

import pytest  # noqa: E402

PGVECTOR_IMAGE = "pgvector/pgvector:pg17"

_container = None
_scratch: tuple[str, str] | None = None  # (server url, database name) to drop at exit


def _with_database(url: str, name: str) -> str:
    """`url` pointing at database `name` on the same server."""
    base, _, _ = url.rpartition("/")
    return f"{base}/{name}"


def _admin(url: str, statement: str) -> None:
    import psycopg

    with psycopg.connect(url.replace("postgresql+psycopg://", "postgresql://"), autocommit=True) as conn:
        conn.execute(statement)


def pytest_configure(config):
    global _container, _scratch
    server = os.environ.get("EDUPILOT_TEST_DATABASE_URL")
    if server:
        name = f"edupilot_test_{uuid.uuid4().hex[:10]}"
        _admin(server, f'CREATE DATABASE "{name}"')
        _scratch = (server, name)
        os.environ["DATABASE_URL"] = _with_database(server, name)
        return
    try:
        from testcontainers.community.postgres import PostgresContainer

        _container = PostgresContainer(PGVECTOR_IMAGE, driver=None).start()
    except Exception as exc:  # Docker not running, image unavailable, ...
        raise pytest.UsageError(
            "the tests need Postgres with pgvector. Start Docker, or set "
            "EDUPILOT_TEST_DATABASE_URL to a server this run may create a database on "
            f"(postgresql://user:password@host:5432/postgres). Cause: {exc}"
        ) from exc
    os.environ["DATABASE_URL"] = _container.get_connection_url()


def pytest_unconfigure(config):
    try:
        from edupilot.db import dispose_engine

        dispose_engine()
    except Exception:
        pass
    if _scratch:
        server, name = _scratch
        _admin(server, f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
    if _container is not None:
        _container.stop()


@pytest.fixture(scope="session", autouse=True)
def _schema():
    """The test database, migrated to the latest revision once per run."""
    from edupilot import db

    db.init_db()


@pytest.fixture(scope="session")
def data_dir() -> str:
    """The temporary DATA_DIR this run is pinned to."""
    return _TMP_DATA


@pytest.fixture(scope="session")
def database_url() -> str:
    """The throwaway database this run is pinned to."""
    return os.environ["DATABASE_URL"]


@pytest.fixture(scope="session")
def redis_url():
    """
    A Redis for the tests that need one: EDUPILOT_TEST_REDIS_URL (CI's
    service), or a container started on first use.
    """
    url = os.environ.get("EDUPILOT_TEST_REDIS_URL")
    if url:
        yield url
        return
    from testcontainers.community.redis import RedisContainer

    container = RedisContainer("redis:7-alpine").start()
    try:
        host = container.get_container_host_ip()
        yield f"redis://{host}:{container.get_exposed_port(6379)}/0"
    finally:
        container.stop()


@pytest.fixture
def shared_redis(redis_url, monkeypatch):
    """Point the app's shared-state client at the test Redis, emptied first."""
    import redis as redis_sync

    from edupilot.core import redis as shared

    redis_sync.Redis.from_url(redis_url).flushdb()
    monkeypatch.setattr(shared, "REDIS_URL", redis_url)
    shared._clients.clear()
    yield redis_url
    shared._clients.clear()


@pytest.fixture
def scratch_database(database_url):
    """
    An empty database next to the test one, dropped afterwards.

    For tests that need a schema of their own — migrating up and down, or
    racing several processes' startup migrations.
    """
    name = f"edupilot_scratch_{uuid.uuid4().hex[:10]}"
    _admin(database_url, f'CREATE DATABASE "{name}"')
    try:
        yield _with_database(database_url, name)
    finally:
        _admin(database_url, f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
