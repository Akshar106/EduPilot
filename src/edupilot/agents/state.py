"""
Graph state
===========
What flows between the nodes of the agent graph.

Two lifetimes share one state object:

  * **Conversation state** — `messages` and `summary` — is persisted by the
    checkpointer between turns, keyed by `thread_id = session_id`. This is the
    model's working memory of the conversation.
  * **Turn state** — everything else — belongs to one question. It is reset
    when a turn starts and its heavy parts (retrieved documents) are cleared
    when the turn ends, so a stored checkpoint holds the conversation rather
    than the last turn's evidence. A turn paused for clarification keeps its
    state in the checkpoint until the student's reply resumes it.

Reducers decide how concurrent writes combine. `sub_answers` is written by one
`research` branch per sub-question running in parallel, so it concatenates;
`diagnostics` is written piecewise by every node, so it merges. Both treat
`None` as "reset", which is how a turn clears what the previous one left.

Each `research` branch is itself a small graph (`agents/research.py`) with its
own state, `ResearchState`; only its `sub_answers` flow back to the parent.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


def append_or_reset(left: list | None, right: list | None) -> list:
    """Concatenate writes, e.g. from parallel branches. `None` clears the list."""
    if right is None:
        return []
    return (left or []) + list(right)


def merge_dicts(left: dict | None, right: dict | None) -> dict:
    """Shallow-merge per-node diagnostics. `None` clears the dict."""
    if right is None:
        return {}
    return {**(left or {}), **right}


class SubAnswer(TypedDict):
    """One domain's grounded answer, as produced by a `research` branch."""

    domain: str
    question: str
    answer: str
    documents: list[Document]
    """Evidence for `answer`: retrieved excerpts, then any calculator results."""
    labels: list[str]
    """`labels[i]` is the citation label for [Source i+1] in `answer`."""
    refused: bool
    latency_ms: int
    retrieval: dict[str, Any]
    """Retrieval diagnostics for the debug panel, including grading and rewrites."""


class ResearchTask(TypedDict, total=False):
    """The payload `Send` hands to one `research` branch."""

    question: str
    domain: str
    retriever_key: str
    """What the retriever factory is called with: a domain code for course
    chat, a study-session id for Self Study."""
    history: list[dict]
    summary: str
    profile_block: str
    filenames: list[str] | None
    self_study: bool
    task: str
    """"answer" retrieves and answers; "quiz"/"flashcards" only retrieve, and
    the tutor node writes from the combined evidence."""


class ResearchState(ResearchTask, total=False):
    """Working state of one `research` branch."""

    query: str
    """The search query in use: the question, or its latest rewrite."""
    attempts: int
    documents: list[Document]
    retrieval: dict[str, Any]
    grade: dict[str, Any]
    grades: Annotated[list[dict], append_or_reset]
    rewrites: Annotated[list[str], append_or_reset]
    next_step: str
    """Where the last decision point sent the branch; read by the edges."""
    answer_messages: Annotated[list[AnyMessage], add_messages]
    """The answering conversation: prompt, model turns, calculator results."""
    tool_rounds: int
    calculations: Annotated[list[Document], append_or_reset]
    answer_started: float
    sub_answers: Annotated[list[SubAnswer], append_or_reset]


class ResearchOutput(TypedDict):
    """What a `research` branch hands back to the parent graph."""

    sub_answers: Annotated[list[SubAnswer], append_or_reset]


class AgentState(TypedDict, total=False):
    # --- conversation: persisted by the checkpointer across turns -----------
    messages: Annotated[list[AnyMessage], add_messages]
    summary: str

    # --- turn input ----------------------------------------------------------
    query: str
    started_at: float
    """`time.perf_counter()` when the turn began, for the total-latency figure."""
    session_id: str | None
    """Used only to seed a thread that has no checkpoint yet from the database."""
    exclude_message_ids: list[int]
    """Database rows already written for this turn, kept out of its own history."""
    manual_domains: list[str] | None
    filenames: list[str] | None
    retriever_key: str | None
    """Self Study only: the study session whose uploads to search."""
    history: list[dict]
    """Recent turns as `{role, content}`. Produced by the memory node for
    course chat; supplied by the caller for Self Study."""
    task: str
    """"answer", "quiz" or "flashcards". Set by the caller to force a task,
    otherwise by the router."""

    # --- turn working state --------------------------------------------------
    profile: dict[str, Any]
    decision: dict[str, Any]
    original_query: str
    clarification: dict[str, str]
    """`{"asked", "reply"}` once a paused turn has been resumed."""
    clarifications: int
    clarify_message: str
    sub_questions: list[dict[str, str]]
    sub_answers: Annotated[list[SubAnswer], append_or_reset]
    answers: list[SubAnswer]
    """`sub_answers` in sub-question order, citations renumbered globally."""
    merged: str
    practice: dict | None
    """Structured quiz or flashcards, when the tutor wrote the answer."""
    audits: Annotated[list[dict], append_or_reset]
    """One summary per verifier pass: the first draft, then each revision."""
    needs_revision: bool
    revisions: int
    best: dict[str, Any]
    """The best-verified draft so far: `{"answer", "grounding", "coverage", "revision"}`."""
    verdict: dict[str, Any]
    diagnostics: Annotated[dict[str, Any], merge_dicts]

    # --- turn output -----------------------------------------------------------
    result: dict[str, Any]


#: Written into the input of every turn so nothing carries over from the last.
TURN_RESET: dict[str, Any] = {
    "decision": {},
    "original_query": "",
    "clarification": {},
    "clarifications": 0,
    "clarify_message": "",
    "sub_questions": [],
    "sub_answers": None,
    "answers": [],
    "merged": "",
    "practice": None,
    "audits": None,
    "needs_revision": False,
    "revisions": 0,
    "best": {},
    "verdict": {},
    "diagnostics": None,
    "result": {},
}
