"""
LangSmith tracing.

No test reaches LangSmith: the turn's root run is observed through a callback
handler, and the LangSmith client is replaced with a recording fake. What is
checked is what the app controls — that each turn's trace has a known id the
API can hand back, the labels needed to filter and compare runs, no raw user
id, and that every LangSmith call is skipped when tracing is off and cannot
fail a request when it is on.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from langchain_core.callbacks import BaseCallbackHandler

from edupilot import db
from edupilot.agents import EduPilotAgent, PipelineConfig
from edupilot.core import tracing
from edupilot.core.config import DOMAINS
from fakes import FakeModels, chunk, retriever_factory

CHUNKS = {
    "ALGO": [chunk("Variance is the average squared deviation.", "stat.pdf")],
    "ss-123": [chunk("Attention weighs tokens by relevance.", "notes.pdf")],
}


class RootRuns(BaseCallbackHandler):
    """Records the top-level chain run of each graph invocation."""

    def __init__(self) -> None:
        self.roots: list[dict] = []

    def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, **kwargs):
        if parent_run_id is None:
            self.roots.append(
                {
                    "run_id": str(run_id),
                    "name": kwargs.get("name"),
                    "tags": kwargs.get("tags") or [],
                    "metadata": kwargs.get("metadata") or {},
                }
            )


@pytest.fixture(autouse=True)
def _schema():
    db.init_db()


def make_agent() -> EduPilotAgent:
    return EduPilotAgent(DOMAINS, retriever_factory(CHUNKS), models=FakeModels())


async def test_each_turn_is_traced_under_the_run_id_it_returns():
    agent, roots = make_agent(), RootRuns()
    cfg = PipelineConfig(model="fake", enable_grounding_check=False, user_id="student-42")

    first = await agent.arun("what is variance", cfg, callbacks=[roots])
    second = await agent.arun("what is variance", cfg, callbacks=[roots])

    assert [r["run_id"] for r in roots.roots] == [first.run_id, second.run_id]
    assert first.run_id != second.run_id

    root = roots.roots[0]
    assert root["name"] == "chat_turn"
    assert "edupilot" in root["tags"]
    meta = root["metadata"]
    assert meta["model"] == "fake"
    assert meta["corrective_retrieval"] is cfg.corrective_retrieval
    # The account id never reaches the trace store; a stable pseudonym does.
    assert meta["user"] == tracing.pseudonym("student-42")
    assert "student-42" not in str(meta)


async def test_a_conversation_and_a_study_session_each_form_one_trace_thread():
    agent, roots = make_agent(), RootRuns()
    cfg = PipelineConfig(model="fake", enable_grounding_check=False)

    await agent.arun("what is variance", cfg, thread_id="chat-1", session_id="chat-1", callbacks=[roots])
    await agent.arun_study("what is attention", cfg, ss_session_id="ss-123", callbacks=[roots])

    chat, study = roots.roots
    assert chat["metadata"]["thread_id"] == "chat-1"
    assert study["name"] == "self_study_turn"
    assert study["metadata"]["thread_id"] == "ss-123"


async def test_evaluation_runs_carry_their_own_tags_and_metadata():
    agent, roots = make_agent(), RootRuns()
    cfg = PipelineConfig(
        model="fake",
        enable_grounding_check=False,
        trace_tags=("eval", "baseline"),
        trace_metadata={"case_id": "STAT-01"},
    )
    await agent.arun("what is variance", cfg, callbacks=[roots])

    root = roots.roots[0]
    assert {"eval", "baseline"} <= set(root["tags"])
    assert root["metadata"]["case_id"] == "STAT-01"


async def test_a_paused_turn_still_reports_its_run_id():
    from langgraph.checkpoint.memory import InMemorySaver

    from edupilot.agents.contracts import RouterOutput

    models = FakeModels(router=RouterOutput(domains=[], task="quiz", topics=[]))
    agent = EduPilotAgent(DOMAINS, retriever_factory(CHUNKS), models=models)
    agent.attach_checkpointer(InMemorySaver())
    roots = RootRuns()
    cfg = PipelineConfig(model="fake", enable_grounding_check=False, interactive=True)

    paused = await agent.arun("quiz me", cfg, thread_id="t-1", session_id="t-1", callbacks=[roots])
    assert paused.awaiting_clarification
    assert paused.run_id == roots.roots[0]["run_id"]


# ---------------------------------------------------------------------------
# The tracing module
# ---------------------------------------------------------------------------


class FakeLangSmith:
    def __init__(self) -> None:
        self.created: list[dict] = []
        self.updated: list[dict] = []
        self.projects_read = 0
        self.runs = SimpleNamespace(get_url=self._get_url)

    def create_feedback(self, run_id, key, **kwargs):
        self.created.append({"run_id": run_id, "key": key, **kwargs})

    def update_feedback(self, feedback_id, **kwargs):
        self.updated.append({"feedback_id": feedback_id, **kwargs})

    def read_project(self, *, project_name):
        self.projects_read += 1
        return SimpleNamespace(id="project-1")

    async def _get_url(self, run_id, *, project_id, trace_id):
        return SimpleNamespace(url=f"https://smith.example/{project_id}/{trace_id}")


@pytest.fixture
def langsmith(monkeypatch):
    fake = FakeLangSmith()
    monkeypatch.setattr(tracing, "enabled", lambda: True)
    monkeypatch.setattr(tracing, "client", lambda: fake)
    monkeypatch.setattr(tracing, "_project_id", None)
    return fake


@pytest.fixture
def fresh_env_lookup():
    """LangSmith caches environment lookups; clear the cache around a test that edits them."""
    from langsmith.utils import get_env_var

    get_env_var.cache_clear()
    yield
    get_env_var.cache_clear()


def test_tracing_needs_a_key_as_well_as_the_switch(monkeypatch, fresh_env_lookup):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
    assert tracing.enabled() is False

    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2-test")
    assert tracing.enabled() is True


def test_nothing_is_sent_when_tracing_is_off(monkeypatch):
    monkeypatch.setattr(tracing, "enabled", lambda: False)
    monkeypatch.setattr(tracing, "client", lambda: pytest.fail("client built with tracing off"))
    assert tracing.send_feedback("r", rating=1, comment=None, feedback_key="f", update=False) is False


def test_a_rating_is_sent_as_a_score_and_a_second_rating_updates_it(langsmith):
    key = tracing.feedback_id("chat", 7, "u1")
    assert key == tracing.feedback_id("chat", 7, "u1")
    assert key != tracing.feedback_id("chat", 7, "u2")

    assert tracing.send_feedback("run-1", rating=-1, comment="wrong formula", feedback_key=key, update=False)
    assert langsmith.created == [
        {
            "run_id": "run-1",
            "key": tracing.FEEDBACK_KEY,
            "score": 0,
            "comment": "wrong formula",
            "trace_id": "run-1",
            "feedback_id": key,
        }
    ]

    assert tracing.send_feedback("run-1", rating=1, comment=None, feedback_key=key, update=True)
    assert langsmith.updated == [{"feedback_id": key, "score": 1, "comment": None}]


def test_a_langsmith_failure_never_fails_the_caller(langsmith, monkeypatch):
    def boom(*args, **kwargs):
        raise ConnectionError("langsmith down")

    monkeypatch.setattr(langsmith, "create_feedback", boom)
    assert tracing.send_feedback("run-1", rating=1, comment=None, feedback_key="k", update=False) is False


async def test_trace_links_resolve_the_project_once(langsmith):
    first = await tracing.trace_url("run-1")
    second = await tracing.trace_url("run-2")
    assert first == "https://smith.example/project-1/run-1"
    assert second == "https://smith.example/project-1/run-2"
    assert langsmith.projects_read == 1
