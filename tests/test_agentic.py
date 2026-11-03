"""
Agentic behaviour: corrective retrieval, verify -> revise, clarification by
interrupt, the calculator tool loop, tutor tasks, and the learning profile.

Like test_agent_graph.py, these run the real graph with the fakes in
tests/fakes.py — no provider, vector store, or model download.
"""

from __future__ import annotations

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from edupilot import db
from edupilot.agents import EduPilotAgent, PipelineConfig, prompts
from edupilot.agents.contracts import (
    EvidenceGrade,
    PlannerOutput,
    RouterOutput,
    SubQuestion,
    TopicRef,
    VerifierOutput,
)
from edupilot.agents.profile import empty_profile, load_profile, save_profile, update_profile
from edupilot.core.config import DOMAINS
from fakes import FakeModels, chunk, retriever_factory

VARIANCE = chunk("Variance is the average squared deviation from the mean.", "stat.pdf", 3)
SPREAD = chunk("Spread can be summarised several ways.", "stat.pdf", 9)
BIAS = chunk("Bias is error from wrong assumptions.", "aml.pdf", 2)

CFG = PipelineConfig(model="fake", enable_grounding_check=False)
CHAT = PipelineConfig(model="fake", enable_grounding_check=False, interactive=True)


def make_agent(models=None, chunks=None, log=None, **kwargs) -> EduPilotAgent:
    chunks = {"ALGO": [VARIANCE]} if chunks is None else chunks
    return EduPilotAgent(DOMAINS, retriever_factory(chunks, log), models=models or FakeModels(), **kwargs)


def unsatisfied(grounding: float) -> VerifierOutput:
    return VerifierOutput(
        grounding_score=grounding,
        coverage_score=0.9,
        unsupported_claims=["Variance was invented in 1918."],
    )


SATISFIED = VerifierOutput(grounding_score=0.95, coverage_score=0.9, is_satisfactory=True)


@pytest.fixture(autouse=True)
def _schema():
    db.init_db()


# ---------------------------------------------------------------------------
# Corrective retrieval
# ---------------------------------------------------------------------------


async def test_a_rewritten_query_finds_what_the_students_wording_missed():
    log: list = []
    chunks = {"ALGO": {"formal course terminology": [VARIANCE]}}
    result = await make_agent(chunks=chunks, log=log).arun("how spread out is my data thing", CFG)

    assert [q for _, q, _ in log] == ["how spread out is my data thing", "formal course terminology"]
    assert not result.refused
    retrieval = result.diagnostics["retrieval"][0]
    assert retrieval["rewrites"] == ["formal course terminology"]
    assert [g["relevant"] for g in retrieval["grades"]] == [0, 1]


async def test_a_follow_up_is_searched_as_the_question_it_refers_to():
    """ "It has an answer, please explain" matches nothing; what it refers to does."""
    log: list = []
    models = FakeModels(
        router=RouterOutput(domains=["ALGO"], standalone_question="What is the variance of a sample?")
    )
    result = await make_agent(models, log=log).arun("It has an answer, please explain", CFG)

    assert [q for _, q, _ in log] == ["What is the variance of a sample?"]
    assert not result.refused
    router_prompt = models.structured_prompts["RouterOutput"][0]
    assert "It has an answer, please explain" in router_prompt, "the router rewrites the student's own words"


async def test_a_question_that_stands_alone_is_searched_as_asked():
    log: list = []
    await make_agent(log=log).arun("what is variance", CFG)  # the router gives no standalone_question
    assert [q for _, q, _ in log] == ["what is variance"]


async def test_excerpts_that_only_share_vocabulary_are_dropped():
    models = FakeModels(grader=EvidenceGrade(relevant_sources=[1], sufficient=True))
    result = await make_agent(models, chunks={"ALGO": [VARIANCE, SPREAD]}).arun("what is variance", CFG)

    assert [s["page_number"] for s in result.sources] == [3]
    assert result.diagnostics["retrieval"][0]["grades"][0] == {
        "attempt": 1, "query": "what is variance", "total": 2,
        "relevant": 1, "sufficient": True, "missing": "", "graded": True,
    }  # fmt: skip


async def test_refusal_comes_only_after_the_retries_are_spent():
    log: list = []
    models = FakeModels(grader=EvidenceGrade(relevant_sources=[]))
    result = await make_agent(models, log=log).arun("what is variance", CFG)

    assert result.refused
    assert len(log) == 2, "one search, one rewritten search, then refuse"


async def test_corrective_retrieval_can_be_switched_off():
    models = FakeModels()
    cfg = PipelineConfig(model="fake", enable_grounding_check=False, corrective_retrieval=False)
    await make_agent(models).arun("what is variance", cfg)
    assert "EvidenceGrade" not in models.structured_calls


async def test_a_failed_grader_keeps_every_excerpt():
    models = FakeModels(grader=RuntimeError("provider down"))
    result = await make_agent(models, chunks={"ALGO": [VARIANCE, SPREAD]}).arun("what is variance", CFG)
    assert len(result.sources) == 2 and not result.refused


async def test_the_grader_reads_the_passage_the_search_matched():
    """A whole section reaches the answer; the grader is shown the part that matched, not its opening."""
    from langchain_core.documents import Document

    from edupilot.agents.nodes import EvidenceGrader
    from edupilot.retrieval.hybrid import HybridRetriever
    from edupilot.retrieval.rerank import RerankedChunk
    from edupilot.retrieval.vectorstore import InMemoryVectorStore, VectorRecord

    matched = "Multi-head attention runs several attention heads in parallel."
    section = "Chapter overview. " * 60 + matched  # the match lies past the summary's first 700 characters
    store = InMemoryVectorStore()
    store.upsert([VectorRecord(id="p1", values=[0.0], metadata={"text": section})], "llm")
    child = RerankedChunk(chunk_id="c1", text=matched, metadata={"parent_id": "p1"}, relevance=0.87)
    [parent] = HybridRetriever(store, embedder=None, namespace="llm")._expand_to_parents([child])
    assert parent.text.endswith(matched) and parent.metadata["matched_text"] == matched

    models = FakeModels()
    document = Document(page_content=parent.text, metadata=parent.metadata)
    await EvidenceGrader(models).grade("what is multi-head attention", [document], model="fake")
    assert matched in models.structured_prompts["EvidenceGrade"][0]


def test_a_long_section_is_cut_to_a_window_around_the_match():
    """The answer prompt and the claim check read ~2,000 characters of an excerpt; the match must be in them."""
    from edupilot.retrieval.hybrid import PARENT_WINDOW_CHARS, _window

    section = " ".join(f"word{i}" for i in range(2000))  # ~17,000 characters
    matched = " ".join(f"word{i}" for i in range(1000, 1040))
    window = _window(section, matched)
    assert matched in window and len(window) <= PARENT_WINDOW_CHARS + 2  # the two "…" marks
    assert window.startswith("…") and window.endswith("…")
    assert _window("A short slide.", "short") == "A short slide.", "a short section stays whole"
    assert _window(section, " ".join(f"word{i}" for i in range(5))).startswith("word0 "), "no gap before a match at the start"


# ---------------------------------------------------------------------------
# Verify -> revise
# ---------------------------------------------------------------------------


async def test_a_failed_verification_is_revised_and_re_verified():
    models = FakeModels(verifier=[unsatisfied(0.6), SATISFIED])
    result = await make_agent(models).arun("what is variance", CFG)

    assert models.revised in result.final_answer
    assert result.verification_revised
    audits = result.diagnostics["audits"]
    assert [a["revision"] for a in audits] == [0, 1]
    reviser_prompt = next(p for p in models.prompts if "repair an answer" in p)
    assert "Variance was invented in 1918." in reviser_prompt, "the reviser gets the findings"


async def test_a_revision_that_verifies_worse_is_thrown_away():
    models = FakeModels(verifier=[unsatisfied(0.6), unsatisfied(0.3)])
    result = await make_agent(models).arun("what is variance", CFG)

    assert models.revised not in result.final_answer
    assert not result.verification_revised


async def test_revision_is_bounded():
    models = FakeModels(verifier=[unsatisfied(0.6)])
    result = await make_agent(models).arun("what is variance", CFG)
    assert len(result.diagnostics["audits"]) == 2, "one revision, then stop"

    models = FakeModels(verifier=[unsatisfied(0.6)])
    cfg = PipelineConfig(model="fake", enable_grounding_check=False, max_revisions=0)
    result = await make_agent(models).arun("what is variance", cfg)
    assert len(result.diagnostics["audits"]) == 1


async def test_the_stream_resets_when_a_revision_replaces_the_draft():
    models = FakeModels(verifier=[unsatisfied(0.6), SATISFIED])
    events = [e async for e in make_agent(models).astream("what is variance", CFG)]

    kinds = [e["type"] for e in events]
    assert "reset" in kinds
    after_reset = "".join(e["text"] for e in events[kinds.index("reset"):] if e["type"] == "token")
    assert after_reset == models.revised


@pytest.fixture
def claim_scores(monkeypatch):
    """A stand-in cross-encoder: every (claim, excerpt) pair scores `scores["value"]`."""
    from edupilot.guardrails import grounding

    scores = {"value": 0.9, "calls": 0}

    def score(pairs):
        scores["calls"] += 1
        return [scores["value"]] * len(pairs)

    monkeypatch.setattr(grounding, "_score_pairs", score)
    monkeypatch.setattr(grounding, "_RECENT", grounding.OrderedDict())
    return scores


SUPPORTED = {"ALGO": "Variance is the average squared deviation of the data from its mean [Source 1]."}
CHECKED = PipelineConfig(model="fake")


async def test_a_draft_whose_claims_check_out_needs_no_model_verifier(claim_scores):
    models = FakeModels(verifier=[unsatisfied(0.6), SATISFIED], answers=SUPPORTED)
    events = [e async for e in make_agent(models).astream("what is variance", CHECKED)]
    result = next(e for e in events if e["type"] == "result")["result"]

    assert "VerifierOutput" not in models.structured_calls
    assert "reset" not in [e["type"] for e in events], "the answer streams once"
    assert not result.verification_revised and result.guardrail_action == "pass"
    assert [a["auditor"] for a in result.diagnostics["audits"]] == ["claim check"]
    assert claim_scores["calls"] == 1, "the guardrails reuse the audit's score"


async def test_unsupported_claims_are_revised_from_the_claim_check_findings(claim_scores):
    claim_scores["value"] = 0.0
    models = FakeModels(verifier=[unsatisfied(0.6), SATISFIED], answers=SUPPORTED)
    result = await make_agent(models).arun("what is variance", CHECKED)

    assert "VerifierOutput" not in models.structured_calls
    assert result.verification_revised and models.revised in result.final_answer
    reviser_prompt = next(p for p in models.prompts if "repair an answer" in p)
    assert SUPPORTED["ALGO"] in reviser_prompt, "the reviser is told which claim failed"


async def test_a_question_in_several_parts_is_still_audited_by_the_model(claim_scores):
    models = FakeModels(
        router=RouterOutput(intent_type="multi", domains=["ALGO", "AML"]),
        planner=PlannerOutput(
            sub_questions=[
                SubQuestion(question="what is variance", domain="ALGO"),
                SubQuestion(question="what is bias", domain="AML"),
            ]
        ),
    )
    agent = make_agent(models, chunks={"ALGO": [VARIANCE], "AML": [BIAS]})
    await agent.arun("variance and bias", CHECKED)
    assert "VerifierOutput" in models.structured_calls, "only the model judges coverage of each part"


# ---------------------------------------------------------------------------
# Clarification by interrupt
# ---------------------------------------------------------------------------


async def test_an_unclear_question_pauses_and_the_reply_resumes_it():
    models = FakeModels(
        router=[
            RouterOutput(domains=[], needs_clarification=True),
            RouterOutput(domains=["ALGO"], reasoning="clarified"),
        ]
    )
    agent = make_agent(models, checkpointer=InMemorySaver())

    paused = await agent.arun("explain this", CHAT, thread_id="t-clarify")
    assert paused.awaiting_clarification and paused.needs_clarification
    assert paused.final_answer == prompts.CLARIFICATION_RESPONSE
    assert await agent.pending_clarification("t-clarify") == prompts.CLARIFICATION_RESPONSE

    resumed = await agent.arun("variance of a sample", CHAT, thread_id="t-clarify")
    assert not resumed.awaiting_clarification
    assert resumed.domains == ["ALGO"]
    assert await agent.pending_clarification("t-clarify") is None
    assert "explain this (variance of a sample)" in models.structured_prompts["RouterOutput"][-1]

    state = (await agent.chat_graph.aget_state({"configurable": {"thread_id": "t-clarify"}})).values
    assert [m.content for m in state["messages"]][:3] == [
        "explain this", prompts.CLARIFICATION_RESPONSE, "variance of a sample"
    ]


async def test_without_a_thread_an_unclear_question_gets_the_fixed_reply():
    models = FakeModels(router=RouterOutput(domains=[], needs_clarification=True))
    result = await make_agent(models).arun("explain this", CFG)
    assert result.needs_clarification and not result.awaiting_clarification


async def test_only_one_clarifying_question_per_turn():
    models = FakeModels(router=RouterOutput(domains=[], needs_clarification=True))
    agent = make_agent(models, checkpointer=InMemorySaver())
    await agent.arun("explain this", CHAT, thread_id="t-twice")
    second = await agent.arun("that thing", CHAT, thread_id="t-twice")

    assert second.needs_clarification and not second.awaiting_clarification
    assert await agent.pending_clarification("t-twice") is None


async def test_resetting_a_thread_drops_a_paused_turn():
    models = FakeModels(router=RouterOutput(domains=[], needs_clarification=True))
    agent = make_agent(models, checkpointer=InMemorySaver())
    await agent.arun("explain this", CHAT, thread_id="t-drop")
    await agent.reset_thread("t-drop")
    assert await agent.pending_clarification("t-drop") is None


# ---------------------------------------------------------------------------
# Calculator tool loop
# ---------------------------------------------------------------------------


async def test_the_answerer_can_calculate_and_cite_the_result():
    models = FakeModels(calculation="variance([2, 4, 4, 4, 5, 5, 7, 9])")
    result = await make_agent(models).arun("what is the variance of 2,4,4,4,5,5,7,9", CFG)

    assert "4.57143" in result.final_answer
    assert "[Source 2]" in result.final_answer, "one excerpt, then the calculation"
    calc = [s for s in result.sources if s["kind"] == "calculation"]
    assert calc and calc[0]["source_num"] == 2 and calc[0]["filename"] is None
    assert result.diagnostics["retrieval"][0]["calculations"] == [
        "variance([2, 4, 4, 4, 5, 5, 7, 9]) = 4.57143"
    ]


async def test_tools_can_be_switched_off():
    models = FakeModels(calculation="mean([1, 2])")
    cfg = PipelineConfig(model="fake", enable_grounding_check=False, enable_tools=False)
    await make_agent(models).arun("what is variance", cfg)
    assert not any(models.tool_bindings)


# ---------------------------------------------------------------------------
# Tutor tasks
# ---------------------------------------------------------------------------


async def test_a_quiz_is_written_from_retrieved_evidence_with_citations():
    log: list = []
    models = FakeModels(
        router=RouterOutput(task="quiz", domains=["ALGO"], topics=[TopicRef(name="variance", domain="ALGO")])
    )
    result = await make_agent(models, log=log).arun("quiz me on variance", CFG)

    assert result.task == "quiz"
    assert log[0][1] == "variance", "retrieval searches the topic, not 'quiz me on'"
    assert "Practice quiz" in result.final_answer
    assert "<details>" in result.final_answer and "[Source 1]" in result.final_answer
    assert not any("teaching assistant" in p for p in models.prompts), "no answer is written"
    assert "verifier" not in result.diagnostics


async def test_flashcards_are_written_from_retrieved_evidence():
    models = FakeModels(
        router=RouterOutput(task="flashcards", domains=["ALGO"], topics=[TopicRef(name="variance", domain="ALGO")])
    )
    result = await make_agent(models).arun("flashcards on variance", CFG)
    assert result.task == "flashcards"
    assert "Average squared deviation from the mean [Source 1]" in result.final_answer


async def test_the_caller_can_force_a_task():
    models = FakeModels()
    result = await make_agent(models).arun("variance", CFG, task="flashcards")
    assert result.task == "flashcards"


async def test_quiz_me_with_no_topic_and_no_history_asks_which_topic():
    models = FakeModels(router=RouterOutput(task="quiz", domains=[]))
    agent = make_agent(models, checkpointer=InMemorySaver())
    result = await agent.arun("quiz me", CHAT, thread_id="t-quiz")
    assert result.awaiting_clarification
    assert result.final_answer == prompts.QUIZ_TOPIC_QUESTION


async def test_quiz_me_with_no_topic_uses_the_students_weak_topics():
    store = InMemoryStore()
    profile = empty_profile()
    for _ in range(2):
        profile = update_profile(profile, topics=[{"name": "variance", "domain": "ALGO"}], confusion=True)
    await save_profile(store, "student-1", profile)

    log: list = []
    models = FakeModels(router=RouterOutput(task="quiz", domains=[]))
    agent = make_agent(models, log=log, checkpointer=InMemorySaver(), store=store)
    cfg = PipelineConfig(model="fake", enable_grounding_check=False, interactive=True, user_id="student-1")
    result = await agent.arun("quiz me", cfg, thread_id="t-weak")

    assert not result.awaiting_clarification
    assert log[0][:2] == ("ALGO", "variance")
    assert "difficult: variance" in models.structured_prompts["QuizOutput"][0]


async def test_self_study_detects_a_quiz_request():
    models = FakeModels()
    result = await make_agent(models, chunks={"ss-1": [VARIANCE]}).arun_study(
        "quiz me on my notes", CFG, ss_session_id="ss-1"
    )
    assert result.task == "quiz"
    assert "Practice quiz" in result.final_answer


# ---------------------------------------------------------------------------
# Learning profile
# ---------------------------------------------------------------------------


async def test_each_turn_updates_the_students_profile():
    store = InMemoryStore()
    models = FakeModels(
        router=RouterOutput(
            domains=["ALGO"],
            topics=[TopicRef(name="variance", domain="ALGO")],
            confusion=True,
            requested_depth="concise",
        )
    )
    agent = make_agent(models, store=store)
    cfg = PipelineConfig(model="fake", enable_grounding_check=False, user_id="student-2")

    await agent.arun("I still don't get variance, briefly?", cfg)
    profile = await load_profile(store, "student-2")
    assert profile["topics"]["ALGO|variance"]["confusion"] == 1
    assert profile["depth"] == "concise"

    await agent.arun("what is variance", cfg)
    answer_prompt = models.prompts[-1]
    assert "<student_profile>" in answer_prompt and "variance (ALGO)" in answer_prompt


async def test_anonymous_runs_keep_no_profile():
    store = InMemoryStore()
    await make_agent(store=store).arun("what is variance", CFG)
    assert await store.asearch(("students",)) == []


# ---------------------------------------------------------------------------
# Citations across domains
# ---------------------------------------------------------------------------


async def test_each_domains_citations_are_numbered_after_the_previous_domain():
    models = FakeModels(
        router=RouterOutput(intent_type="multi", domains=["ALGO", "AML"]),
        planner=PlannerOutput(
            sub_questions=[
                SubQuestion(question="what is variance", domain="ALGO"),
                SubQuestion(question="what is bias", domain="AML"),
            ]
        ),
        answers={"ALGO": "Variance is spread [Source 1].", "AML": "Bias is error [Source 1]."},
    )
    result = await make_agent(models, chunks={"ALGO": [VARIANCE], "AML": [BIAS]}).arun(
        "variance and bias", CFG
    )

    synth_prompt = next(p for p in models.prompts if "merge several already-grounded" in p)
    assert "Variance is spread [Source 1]." in synth_prompt
    assert "Bias is error [Source 2]." in synth_prompt, "AML's first source follows ALGO's one"
    assert [s["source_num"] for s in result.sources] == [1, 2]


@pytest.mark.parametrize(
    ("request_text", "topic"),
    [
        ("quiz me on hypothesis testing", "hypothesis testing"),
        ("Make 10 flashcards about transformers", "transformers"),
        ("quiz me (variance)", "variance"),
        ("can you give me some practice questions on SQL joins?", "SQL joins"),
        ("quiz me on my notes", "main ideas, key definitions and important results"),
    ],
)
def test_a_practice_request_is_searched_by_its_topic(request_text, topic):
    from edupilot.agents.nodes import topic_of

    assert topic_of(request_text) == topic


async def test_without_router_topics_a_quiz_still_searches_the_topic():
    log: list = []
    models = FakeModels(router=RuntimeError("provider down"))
    result = await make_agent(models, log=log).arun("quiz me on heapsort", CFG)

    assert result.task == "quiz"
    assert ("ALGO", "heapsort", None) in log
