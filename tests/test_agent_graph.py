"""
The LangGraph agent.

Runs the real graph end to end with fake models (tests/fakes.py) and fake
retrievers, so routing, fan-out, streaming, memory and checkpointing are all
exercised without a provider, a vector store, or a model download. The
grounding check is switched off because it loads the cross-encoder; the
citation guardrails still run.
"""

from __future__ import annotations

import threading

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from sqlalchemy import text

from edupilot import db
from edupilot.agents import EduPilotAgent, PipelineConfig, prompts
from edupilot.agents.checkpoint import open_persistence
from edupilot.agents.contracts import PlannerOutput, RouterOutput, SubQuestion
from edupilot.core.config import DOMAINS
from edupilot.llm import UsageTracker
from fakes import FakeModels, answer_text, chunk, retriever_factory

CHUNKS = {
    "ALGO": [chunk("Variance is the average squared deviation.", "stat.pdf")],
    "AML": [chunk("Bias is error from wrong assumptions.", "aml.pdf")],
    "ss-123": [chunk("Attention weighs tokens by relevance.", "notes.pdf")],
}

CFG = PipelineConfig(model="fake", enable_grounding_check=False)


def multi_domain_models(**kwargs) -> FakeModels:
    return FakeModels(
        router=RouterOutput(intent_type="multi", domains=["ALGO", "AML"]),
        planner=PlannerOutput(
            sub_questions=[
                SubQuestion(question="what is variance", domain="ALGO"),
                SubQuestion(question="what is bias", domain="AML"),
            ]
        ),
        **kwargs,
    )


def make_agent(models: FakeModels | None = None, chunks=CHUNKS, log=None, **kwargs):
    return EduPilotAgent(
        DOMAINS, retriever_factory(chunks, log), models=models or FakeModels(), **kwargs
    )


@pytest.fixture(autouse=True)
def _schema():
    db.init_db()


# ---------------------------------------------------------------------------
# One turn
# ---------------------------------------------------------------------------


async def test_a_single_domain_question_is_answered_and_cited():
    log: list = []
    result = await make_agent(log=log).arun("what is variance", CFG)

    assert answer_text("ALGO") in result.final_answer
    assert result.domains == ["ALGO"]
    assert not result.refused
    assert result.sources[0]["filename"] == "stat.pdf"
    assert result.sources[0]["citation_label"]
    assert log == [("ALGO", "what is variance", None)]
    for key in ("router", "sub_questions", "retrieval", "verifier", "guardrails"):
        assert key in result.diagnostics, key


async def test_an_out_of_scope_question_is_refused_without_retrieval():
    log: list = []
    models = FakeModels(router=RouterOutput(is_course_related=False, domains=[]))
    result = await make_agent(models, log=log).arun("who won the match", CFG)

    assert result.final_answer == prompts.OUT_OF_DOMAIN_RESPONSE
    assert result.refused and not result.is_course_related
    assert log == [], "an out-of-scope question must never reach retrieval"


async def test_an_unclear_question_asks_for_clarification():
    models = FakeModels(router=RouterOutput(domains=[], needs_clarification=True))
    result = await make_agent(models).arun("explain this", CFG)

    assert result.final_answer == prompts.CLARIFICATION_RESPONSE
    assert result.needs_clarification and not result.refused


async def test_no_evidence_means_a_refusal():
    models = FakeModels()
    result = await make_agent(models, chunks={}).arun("what is variance", CFG)

    assert result.refused
    assert "could not find anything" in result.final_answer
    assert not models.prompts, "no answer may be generated without evidence"


async def test_a_router_failure_falls_back_to_keywords():
    models = FakeModels(router=RuntimeError("provider down"))
    result = await make_agent(models).arun("what is the running time of merge sort", CFG)

    assert result.diagnostics["router"]["used_fallback"]
    assert result.domains == ["ALGO"]


async def test_manual_domains_override_the_router():
    log: list = []
    await make_agent(log=log).arun("what is bias", CFG, manual_domains=["AML"])
    assert [entry[0] for entry in log] == ["AML"]


async def test_attached_filenames_are_passed_to_retrieval():
    log: list = []
    await make_agent(log=log).arun("what is variance", CFG, filenames=["stat.pdf"])
    assert log[0][2] == ["stat.pdf"]


async def test_verification_can_be_switched_off():
    models = FakeModels()
    cfg = PipelineConfig(model="fake", enable_grounding_check=False, enable_verification=False)
    result = await make_agent(models).arun("what is variance", cfg)

    assert "verifier" not in result.diagnostics
    assert "VerifierOutput" not in models.structured_calls


# ---------------------------------------------------------------------------
# Fan-out
# ---------------------------------------------------------------------------


async def test_a_multi_domain_question_fans_out_in_parallel():
    # Each branch waits at the barrier until the other arrives. Run one after
    # the other, the first would time out — so passing proves concurrency.
    barrier = threading.Barrier(2, timeout=5)

    class Log(list):
        def append(self, item):
            super().append(item)
            barrier.wait()

    log = Log()
    result = await make_agent(multi_domain_models(), log=log).arun("variance and bias", CFG)

    assert sorted(entry[0] for entry in log) == ["ALGO", "AML"]
    assert result.domains == ["ALGO", "AML"], "sub-answers keep the planner's order"
    assert "Merged answer across domains" in result.final_answer
    assert result.intent_type == "multi"


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


async def test_the_stream_reports_stages_then_tokens_then_the_result():
    events = [e async for e in make_agent().astream("what is variance", CFG)]

    kinds = [e["type"] for e in events]
    stages = [e["stage"] for e in events if e["type"] == "stage"]
    assert kinds[-1] == "result"
    assert stages[:3] == ["memory", "route", "plan"]
    assert {"research", "verify", "guardrails"} <= set(stages)
    assert {"retrieving", "answering"} <= {e["stage"] for e in events if e["type"] == "progress"}

    tokens = "".join(e["text"] for e in events if e["type"] == "token")
    assert tokens == answer_text("ALGO")
    assert answer_text("ALGO") in events[-1]["result"].final_answer


async def test_a_split_question_streams_only_the_merged_answer():
    models = multi_domain_models()
    events = [e async for e in make_agent(models).astream("variance and bias", CFG)]

    tokens = "".join(e["text"] for e in events if e["type"] == "token")
    assert tokens == models.synth, "parallel per-domain answers must not interleave"


async def test_routing_and_grading_never_reach_the_answer_stream():
    models = FakeModels()
    events = [e async for e in make_agent(models).astream("what is variance", CFG)]
    tokens = "".join(e["text"] for e in events if e["type"] == "token")
    assert "intent_type" not in tokens and "grounding_score" not in tokens


# ---------------------------------------------------------------------------
# Usage accounting
# ---------------------------------------------------------------------------


async def test_usage_counts_every_text_model_call():
    usage = UsageTracker()
    await make_agent(multi_domain_models()).arun("variance and bias", CFG, callbacks=[usage])

    stats = usage.as_dict()
    # Two domain answers and one merge are text calls; structured calls are
    # stubbed with plain runnables and report no tokens.
    assert stats["calls"] == 3
    assert stats["input_tokens"] == 30
    assert stats["by_model"] == {"fake-model": 3}


# ---------------------------------------------------------------------------
# Memory across turns
# ---------------------------------------------------------------------------


async def test_a_thread_remembers_earlier_turns():
    models = FakeModels()
    agent = make_agent(models, checkpointer=InMemorySaver())

    await agent.arun("what is variance", CFG, thread_id="t-remember")
    await agent.arun("and how is it computed", CFG, thread_id="t-remember")

    second_answer_prompt = models.prompts[-1]
    assert "Student: what is variance" in second_answer_prompt


async def test_threads_do_not_share_memory():
    models = FakeModels()
    agent = make_agent(models, checkpointer=InMemorySaver())

    await agent.arun("what is variance", CFG, thread_id="t-a")
    await agent.arun("what is a mean", CFG, thread_id="t-b")
    assert "what is variance" not in models.prompts[-1]


async def test_a_long_conversation_folds_into_a_summary():
    models = FakeModels()
    agent = make_agent(models, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "t-long"}}

    for i in range(8):
        result = await agent.arun(f"variance question {i}", CFG, thread_id="t-long")

    state = (await agent.chat_graph.aget_state(config)).values
    assert state["summary"] == models.summary
    assert result.diagnostics["memory"]["folded"] > 0
    assert len(state["messages"]) <= 12, "folded messages must leave the checkpoint"


async def test_a_turns_evidence_is_not_kept_in_the_checkpoint():
    agent = make_agent(checkpointer=InMemorySaver())
    await agent.arun("what is variance", CFG, thread_id="t-evidence")

    state = (await agent.chat_graph.aget_state({"configurable": {"thread_id": "t-evidence"}})).values
    assert state["sub_answers"] == []
    assert [m.type for m in state["messages"]] == ["human", "ai"]


async def test_a_thread_without_a_checkpoint_is_seeded_from_the_database():
    sid = "seeded-session"
    db.ensure_session(sid, "test-user")
    db.save_message(sid, "user", "what is variance")
    db.save_message(sid, "assistant", "Variance is spread.")
    current = db.save_message(sid, "user", "and standard deviation?")

    models = FakeModels()
    agent = make_agent(models, checkpointer=InMemorySaver())
    result = await agent.arun(
        "and standard deviation?", CFG,
        thread_id=sid, session_id=sid, exclude_message_ids=[current],
    )

    assert result.diagnostics["memory"]["seeded_from_db"]
    prompt = models.prompts[-1]
    assert "Student: what is variance" in prompt
    assert prompt.count("and standard deviation?") == 1, "the question is not its own history"


async def test_resetting_a_thread_forgets_it():
    models = FakeModels()
    agent = make_agent(models, checkpointer=InMemorySaver())
    await agent.arun("what is variance", CFG, thread_id="t-reset")

    await agent.reset_thread("t-reset")
    await agent.arun("what is a mean", CFG, thread_id="t-reset")
    assert "what is variance" not in models.prompts[-1]


async def test_the_postgres_checkpointer_keeps_only_the_latest_checkpoint():
    async with open_persistence() as (saver, store):
        agent = make_agent(checkpointer=saver, store=store)
        for q in ("what is variance", "why squared", "what about bias"):
            await agent.arun(q, CFG, thread_id="t-postgres")
        state = (await agent.chat_graph.aget_state({"configurable": {"thread_id": "t-postgres"}})).values

    assert len(state["messages"]) == 6, "pruning must not lose the conversation"
    with db.get_engine().connect() as conn:
        count = conn.execute(
            text("SELECT COUNT(*) FROM checkpoints WHERE thread_id = 't-postgres'")
        ).scalar_one()
    assert count == 1


async def test_a_conversation_survives_reopening_the_checkpointer():
    """What a restarted (or a second) API process sees: the same thread, from Postgres."""
    async with open_persistence() as (saver, _):
        await make_agent(checkpointer=saver).arun("what is variance", CFG, thread_id="t-reopen")

    models = FakeModels()
    async with open_persistence() as (saver, _):
        await make_agent(models, checkpointer=saver).arun("why squared", CFG, thread_id="t-reopen")
    assert "what is variance" in models.prompts[-1]


# ---------------------------------------------------------------------------
# Self Study and the sync entry point
# ---------------------------------------------------------------------------


async def test_self_study_answers_from_the_sessions_own_uploads():
    log: list = []
    models = FakeModels()
    result = await make_agent(models, log=log).arun_study(
        "what does attention do", CFG, ss_session_id="ss-123"
    )

    assert log == [("ss-123", "what does attention do", None)]
    assert result.domains == ["Self Study"]
    assert models.study_answer in result.final_answer
    assert "documents the student uploaded" in models.prompts[-1]
    assert "router" not in result.diagnostics, "Self Study is never routed"
    assert result.diagnostics["retrieval"][0]["chunks"] == 1


def test_the_sync_entry_point_runs_a_stateless_turn():
    result = make_agent().run("what is variance", CFG)
    assert answer_text("ALGO") in result.final_answer
