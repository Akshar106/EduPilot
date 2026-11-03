"""
Persistence layer.

These run against the throwaway Postgres database conftest pins the run to,
so they exercise the real schema, the real migrations and the real queries.
"""

from __future__ import annotations

import re
import threading

import pytest
from sqlalchemy import inspect

from edupilot import db
from edupilot.db import engine as db_engine
from edupilot.db.tables import metadata


def test_schema_is_idempotent():
    """init_db runs on every startup; a second call must be a no-op."""
    db.init_db()
    db.init_db()


def test_the_database_is_at_the_latest_migration():
    assert db.current_revision() == db.head_revision()


def test_the_migrations_create_exactly_the_tables_the_code_defines():
    """A table added to db/tables.py without a migration fails here, not in production."""
    present = set(inspect(db.get_engine()).get_table_names())
    assert set(metadata.tables) <= present


def _on(url: str):
    """Point the shared engine (and so init_db) at another database for a block."""
    from contextlib import contextmanager

    @contextmanager
    def scope():
        db.dispose_engine()
        original = db_engine._engine
        db_engine._engine = db_engine.create(url, pool_size=8, max_overflow=0)
        try:
            yield db_engine._engine
        finally:
            db_engine._engine.dispose()
            db_engine._engine = original

    return scope()


def test_migrations_run_down_and_back_up(scratch_database):
    """Every migration has a working downgrade, so a bad deploy can be backed out."""
    from alembic import command

    from edupilot.db.schema import alembic_config

    with _on(scratch_database) as engine:
        db.init_db()
        # The connection pins Alembic to the scratch database; without it the
        # environment would open DATABASE_URL, the run's shared test database.
        cfg = alembic_config()
        with engine.connect() as conn:
            cfg.attributes["connection"] = conn
            command.downgrade(cfg, "base")
            conn.commit()
        assert set(inspect(engine).get_table_names()) <= {"alembic_version"}
        db.init_db()
        assert db.current_revision() == db.head_revision()


def test_processes_starting_together_migrate_once(scratch_database):
    """Several API processes run init_db at once; the advisory lock serializes them."""
    errors: list[Exception] = []

    def start() -> None:
        try:
            db.init_db()
        except Exception as exc:  # pragma: no cover - the failure being tested for
            errors.append(exc)

    with _on(scratch_database):
        threads = [threading.Thread(target=start) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        assert db.current_revision() == db.head_revision()


def test_package_reexports_match_submodules():
    """`from edupilot import db` must expose everything __all__ promises."""
    for name in db.__all__:
        assert hasattr(db, name), f"db.__all__ lists {name}, which does not exist"


# ---------------------------------------------------------------------------
# Chat sessions — ownership is the whole point of this layer
# ---------------------------------------------------------------------------


def test_session_roundtrip():
    db.ensure_session("s-round", "user-1", title="hello")
    assert db.get_session_owner("s-round") == "user-1"

    mid = db.save_message("s-round", "user", "what is variance?")
    assert mid > 0

    messages = db.get_session_messages("s-round")
    assert [m["content"] for m in messages] == ["what is variance?"]


def test_ensure_session_requires_an_owner():
    """A session with no owner is unreachable and unattributable."""
    with pytest.raises(ValueError):
        db.ensure_session("s-noowner", "")


def test_list_sessions_is_scoped_to_one_user():
    db.ensure_session("s-alice", "alice")
    db.ensure_session("s-bob", "bob")

    alice_ids = {s["session_id"] for s in db.list_sessions("alice")}
    assert "s-alice" in alice_ids
    assert "s-bob" not in alice_ids


def test_delete_session_refuses_a_non_owner():
    """Ownership sits in the DELETE predicate, so a wrong user changes nothing."""
    db.ensure_session("s-owned", "owner-1")
    db.save_message("s-owned", "user", "keep me")

    assert db.delete_session("s-owned", "attacker") is False
    assert db.get_session_owner("s-owned") == "owner-1"
    assert len(db.get_session_messages("s-owned")) == 1

    assert db.delete_session("s-owned", "owner-1") is True
    assert db.get_session_owner("s-owned") is None


def test_message_metadata_survives_the_json_roundtrip():
    db.ensure_session("s-meta", "user-1")
    db.save_message(
        "s-meta",
        "assistant",
        "answer",
        intent_type="multi",
        detected_domains=["AML", "ALGO"],
        quality_score=0.91,
        pipeline_meta={"retrieval": {"chunks": 5}},
    )
    msg = db.get_session_messages("s-meta")[0]
    assert msg["detected_domains"] == ["AML", "ALGO"]
    assert msg["pipeline_meta"] == {"retrieval": {"chunks": 5}}
    assert msg["quality_score"] == pytest.approx(0.91)


def test_delete_messages_from_truncates_the_tail():
    db.ensure_session("s-trunc", "user-1")
    ids = [db.save_message("s-trunc", "user", f"m{i}") for i in range(4)]

    db.delete_messages_from("s-trunc", ids[2])
    remaining = [m["content"] for m in db.get_session_messages("s-trunc")]
    assert remaining == ["m0", "m1"]


# ---------------------------------------------------------------------------
# Self study
# ---------------------------------------------------------------------------


def test_study_session_roundtrip_and_scoping():
    db.create_ss_session("ss-1", "student-a", "Interview prep", "DSA + RAG")
    db.create_ss_session("ss-2", "student-b", "Other", None)

    assert db.get_ss_session_owner("ss-1") == "student-a"
    mine = {s["ss_session_id"] for s in db.list_ss_sessions("student-a")}
    assert mine == {"ss-1"}


def test_deleting_a_study_session_removes_its_children():
    db.create_ss_session("ss-cascade", "student-c", "Temp")
    db.save_ss_document("ss-cascade", "notes.pdf", ".pdf", 1024, 3)
    db.save_ss_message("ss-cascade", "user", "summarise")

    db.delete_ss_session("ss-cascade")

    assert db.get_ss_session("ss-cascade") is None
    assert db.list_ss_documents("ss-cascade") == []
    assert db.get_ss_messages("ss-cascade") == []


def test_uploads_to_the_course_corpus_are_logged():
    row_id = db.save_uploaded_doc("lec9.pdf", "TESTDOM", ".pdf", 12, 2048)
    logged = db.list_uploaded_docs("TESTDOM")
    assert [d["id"] for d in logged] == [row_id]
    assert logged[0]["chunk_count"] == 12


# ---------------------------------------------------------------------------
# Postgres specifics
# ---------------------------------------------------------------------------


def test_timestamps_come_back_as_utc_iso_strings():
    """The API returns them as-is, and the frontend parses ISO 8601."""
    db.ensure_session("s-time", "user-1")
    db.save_message("s-time", "user", "hi")
    session = next(s for s in db.list_sessions("user-1") if s["session_id"] == "s-time")
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\+00:00", session["updated_at"])
    assert session["message_count"] == 1


def test_nul_characters_do_not_fail_a_write():
    """Postgres refuses NUL in text and JSONB; one in model output must not lose the turn."""
    db.ensure_session("s-nul", "user-1")
    db.save_message("s-nul", "assistant", "answer", pipeline_meta={"tool": "out\x00put"})
    assert db.get_session_messages("s-nul")[0]["pipeline_meta"] == {"tool": "output"}


def test_messages_keep_their_order_within_one_second():
    """Ordered by id, not by a timestamp two turns can share."""
    db.ensure_session("s-order", "user-1")
    for i in range(5):
        db.save_message("s-order", "user", f"m{i}")
    assert [m["content"] for m in db.get_session_messages("s-order")] == [f"m{i}" for i in range(5)]


def test_conversation_memory_roundtrip():
    db.ensure_session("s-memory", "user-1")
    assert db.get_session_memory("s-memory") == ("", 0)

    mid = db.save_message("s-memory", "user", "hello")
    db.set_session_memory("s-memory", "Student greeted the tutor.", mid)
    assert db.get_session_memory("s-memory") == ("Student greeted the tutor.", mid)


def test_get_messages_after_returns_only_newer_messages():
    db.ensure_session("s-after", "user-1")
    ids = [db.save_message("s-after", "user", f"m{i}") for i in range(4)]

    later = db.get_messages_after("s-after", ids[1])
    assert [m["content"] for m in later] == ["m2", "m3"]
