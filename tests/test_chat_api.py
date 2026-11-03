"""
Course chat over HTTP.

Drives `/api/chat` and `/api/chat/stream` through the real app, lifespan
included (so the Postgres checkpointer is opened and attached), with the agent's
models and retrievers replaced by the fakes in tests/fakes.py.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from edupilot import db
from edupilot.agents import EduPilotAgent
from edupilot.agents import graph as agent_graph
from edupilot.core.config import DOMAINS
from edupilot.db.tables import answer_feedback
from edupilot.guardrails.grounding import GroundingReport
from fakes import FakeModels, answer_text, chunk, retriever_factory

CHUNKS = {"ALGO": [chunk("Variance is the average squared deviation.", "stat.pdf")]}


class StubCache:
    """Answer cache that never hits, so no embedding model is loaded."""

    def record_ask(self, *args, **kwargs) -> int:
        return 1

    def lookup(self, *args, **kwargs):
        return None

    def should_store(self, *args, **kwargs) -> bool:
        return False

    def store(self, *args, **kwargs) -> None:
        pass


@pytest.fixture
def client(monkeypatch):
    from edupilot.api.app import create_app
    from edupilot.core.services import services

    models = FakeModels()
    factory = retriever_factory(CHUNKS)
    agent = EduPilotAgent(DOMAINS, lambda key: factory(key), models=models)
    monkeypatch.setattr(services, "_agent", agent)
    monkeypatch.setattr(services, "_answer_cache", StubCache())

    # The claim-level grounding check loads the cross-encoder; keep the
    # citation guardrails and skip only that.
    real = agent_graph.apply_output_guardrails

    def without_claim_check(*args, **kwargs):
        kwargs["check_claims"] = False
        return real(*args, **kwargs)

    monkeypatch.setattr(agent_graph, "apply_output_guardrails", without_claim_check)
    # The answer audit uses the same check; unavailable, it falls back to the
    # (fake) model verifier.
    monkeypatch.setattr(agent_graph, "check_grounding", lambda *a, **k: GroundingReport(checked=False))

    with TestClient(create_app()) as c:
        c.models = models
        c.agent = agent
        yield c


def events(response) -> list[tuple[str, dict]]:
    out = []
    for frame in response.text.strip().split("\n\n"):
        name = data = None
        for line in frame.split("\n"):
            if line.startswith("event:"):
                name = line[6:].strip()
            elif line.startswith("data:"):
                data = json.loads(line[5:].strip())
        out.append((name, data))
    return out


def test_the_stream_ends_with_the_same_payload_as_the_json_endpoint(client):
    streamed = client.post("/api/chat/stream", json={"query": "what is variance"})
    assert streamed.status_code == 200
    assert streamed.headers["content-type"].startswith("text/event-stream")

    evs = events(streamed)
    names = [name for name, _ in evs]
    assert names[-1] == "result"
    assert "stage" in names and "token" in names and "progress" in names

    result = evs[-1][1]
    assert answer_text("ALGO") in result["final_answer"]
    assert result["detected_domains"] == ["ALGO"]
    assert result["user_message_id"] and result["assistant_message_id"]

    plain = client.post("/api/chat", json={"query": "what is variance"}).json()
    assert set(plain) == set(result)


def test_both_sides_of_a_streamed_turn_are_saved(client):
    result = events(client.post("/api/chat/stream", json={"query": "what is variance"}))[-1][1]
    messages = db.get_session_messages(result["session_id"])
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert messages[1]["content"] == result["final_answer"]


def test_a_session_remembers_its_earlier_turns(client):
    first = client.post("/api/chat", json={"query": "what is variance"}).json()
    client.post(
        "/api/chat", json={"query": "how is it computed", "session_id": first["session_id"]}
    )
    assert "Student: what is variance" in client.models.prompts[-1]


def test_editing_a_message_resets_the_conversation_memory(client):
    first = client.post("/api/chat", json={"query": "what is variance"}).json()
    sid = first["session_id"]
    second = client.post("/api/chat", json={"query": "why squared", "session_id": sid}).json()

    # Edit the second question: truncate from it, then ask something else.
    r = client.delete(f"/api/sessions/{sid}/messages/{second['user_message_id']}")
    assert r.status_code == 200
    third = client.post("/api/chat", json={"query": "what is a mean", "session_id": sid}).json()

    assert third["debug"]["memory"]["seeded_from_db"]
    prompt = client.models.prompts[-1]
    assert "Student: what is variance" in prompt
    assert "why squared" not in prompt, "a deleted turn must not be remembered"


def test_a_failure_mid_stream_becomes_an_error_event(client):
    def broken(key):
        raise RuntimeError("vector store unreachable")

    client.agent.retriever_factory = broken
    evs = events(client.post("/api/chat/stream", json={"query": "what is variance"}))

    name, data = evs[-1]
    assert name == "error"
    assert data["code"] == "internal_error"
    assert data["request_id"]
    assert "unreachable" not in data["message"], "internal detail must not reach the client"


# ---------------------------------------------------------------------------
# Agentic behaviour over HTTP
# ---------------------------------------------------------------------------


def test_a_clarifying_question_is_answered_by_the_next_message(client):
    from edupilot.agents.contracts import RouterOutput

    client.models.outputs[RouterOutput] = [
        RouterOutput(domains=[], needs_clarification=True),
        RouterOutput(domains=["ALGO"], reasoning="clarified"),
    ]
    paused = events(client.post("/api/chat/stream", json={"query": "explain this"}))[-1][1]
    assert paused["awaiting_clarification"] and paused["needs_clarification"]

    sid = paused["session_id"]
    answered = client.post("/api/chat", json={"query": "variance", "session_id": sid}).json()
    assert not answered["awaiting_clarification"]
    assert answered["detected_domains"] == ["ALGO"]
    assert [m["role"] for m in db.get_session_messages(sid)] == [
        "user", "assistant", "user", "assistant"
    ]


def test_the_profile_shows_and_forgets_what_was_learned(client):
    from edupilot.agents.contracts import RouterOutput, TopicRef

    client.models.outputs[RouterOutput] = RouterOutput(
        domains=["ALGO"], topics=[TopicRef(name="variance", domain="ALGO")], confusion=True
    )
    client.post("/api/chat", json={"query": "I don't get variance"})

    profile = client.get("/api/profile").json()
    assert [t["name"] for t in profile["topics"]] == ["variance"]
    assert [t["name"] for t in profile["weak_topics"]] == ["variance"]

    assert client.delete("/api/profile").json() == {"cleared": True}
    assert client.get("/api/profile").json()["topics"] == []


def test_a_quiz_can_be_requested_explicitly(client):
    result = client.post("/api/chat", json={"query": "variance", "task": "quiz"}).json()
    assert result["task"] == "quiz"
    assert "Practice quiz" in result["final_answer"]


def test_an_unknown_task_is_rejected():
    from pydantic import ValidationError

    from edupilot.api.schemas import ChatRequest

    with pytest.raises(ValidationError):
        ChatRequest(query="variance", task="essay")


# ---------------------------------------------------------------------------
# Feedback and traces
# ---------------------------------------------------------------------------


def _feedback_rows(message_id: int) -> list[dict]:
    with db.get_engine().connect() as conn:
        rows = conn.execute(
            select(answer_feedback).where(answer_feedback.c.message_id == message_id)
        )
        return [dict(r._mapping) for r in rows]


def test_a_student_can_rate_their_answer_and_change_their_mind(client):
    answer = client.post("/api/chat", json={"query": "what is variance"}).json()
    mid = answer["assistant_message_id"]
    assert answer["debug"]["trace"]["run_id"]
    assert answer["debug"]["trace"]["traced"] is False

    first = client.post("/api/feedback", json={"message_id": mid, "rating": -1, "comment": "too vague"})
    assert first.json() == {"recorded": True, "updated": False, "sent_to_langsmith": False}

    second = client.post("/api/feedback", json={"message_id": mid, "rating": 1}).json()
    assert second["updated"] is True

    rows = _feedback_rows(mid)
    assert len(rows) == 1, "a second rating replaces the first"
    assert rows[0]["rating"] == 1 and rows[0]["comment"] is None
    assert rows[0]["run_id"] == answer["debug"]["trace"]["run_id"]


def test_only_the_student_who_got_the_answer_can_rate_it(client):
    answer = client.post("/api/chat", json={"query": "what is variance"}).json()

    client.cookies.clear()  # a different browser, so a different anonymous identity
    r = client.post("/api/feedback", json={"message_id": answer["assistant_message_id"], "rating": 1})
    assert r.status_code == 404
    assert _feedback_rows(answer["assistant_message_id"]) == []


def test_a_question_cannot_be_rated_as_if_it_were_an_answer(client):
    answer = client.post("/api/chat", json={"query": "what is variance"}).json()
    r = client.post("/api/feedback", json={"message_id": answer["user_message_id"], "rating": 1})
    assert r.status_code == 404
    assert client.post("/api/feedback", json={"message_id": 1, "rating": 5}).status_code == 422


def test_with_tracing_on_the_rating_reaches_the_answers_trace(client, monkeypatch):
    from edupilot.core import tracing

    sent = []
    monkeypatch.setattr(tracing, "enabled", lambda: True)
    monkeypatch.setattr(tracing, "send_feedback", lambda run_id, **kw: sent.append((run_id, kw)) or True)

    answer = client.post("/api/chat", json={"query": "what is variance"}).json()
    assert answer["debug"]["trace"]["traced"] is True
    r = client.post("/api/feedback", json={"message_id": answer["assistant_message_id"], "rating": 1})

    assert r.json()["sent_to_langsmith"] is True
    run_id, kwargs = sent[0]
    assert run_id == answer["debug"]["trace"]["run_id"]
    assert kwargs["rating"] == 1 and kwargs["update"] is False


def test_the_trace_link_redirects_to_langsmith_only_when_tracing_is_on(client, monkeypatch):
    from edupilot.core import tracing

    run_id = "6f1c2c56-3d8e-4bb0-9a65-7f8d0c1b2a3e"
    assert client.get(f"/api/trace/{run_id}").status_code == 404
    assert client.get("/api/trace/not-a-uuid").status_code == 404

    async def url(rid):
        return f"https://smith.example/r/{rid}"

    monkeypatch.setattr(tracing, "enabled", lambda: True)
    monkeypatch.setattr(tracing, "trace_url", url)
    r = client.get(f"/api/trace/{run_id}", follow_redirects=False)
    assert r.status_code == 307
    assert r.headers["location"] == f"https://smith.example/r/{run_id}"


def test_a_quiz_arrives_as_data_and_survives_a_reload(client):
    result = client.post("/api/chat", json={"query": "variance", "task": "quiz"}).json()

    practice = result["practice"]
    assert practice["type"] == "quiz" and practice["items"]
    first = practice["items"][0]
    assert {"question", "options", "answer_index", "explanation", "source"} <= set(first)
    assert 0 <= first["answer_index"] < len(first["options"])

    reloaded = client.get(f"/api/sessions/{result['session_id']}").json()["messages"][-1]
    assert reloaded["pipeline_meta"]["practice"] == practice
