"""
EduPilot agents
===============
The agents, the LangGraph graph that runs them, and their prompts.

    memory -> route -> (clarify) -> plan -> [research] x N -> collect
           -> synthesize -> verify <-> revise | tutor -> guardrails -> learn

    from edupilot.agents import EduPilotAgent, PipelineConfig

    agent = EduPilotAgent(DOMAINS, get_retriever)
    result = await agent.arun("what is a p-value", PipelineConfig(model="..."))

`result.grounding_score` is None when grounding was not measured — it is never
defaulted, so a displayed score always reflects a real check.
"""

from .context import PipelineConfig
from .contracts import (
    EvidenceBlock,
    EvidenceGrade,
    FlashcardsOutput,
    PlannerOutput,
    QueryRewrite,
    QuizOutput,
    RouterOutput,
    SourceFence,
    VerifierOutput,
    build_evidence,
    build_evidence_summary,
    chunk_text,
    citation_label,
    format_history,
    is_refusal,
    make_fence,
    parse_json_response,
    strip_refusal_marker,
)
from .graph import (
    SELF_STUDY_DOMAIN,
    EduPilotAgent,
    PipelineResult,
    build_sources,
)
from .memory import fold_memory, format_memory, seed_from_db
from .nodes import (
    ANSWER_STREAM_TAG,
    Answerer,
    EvidenceGrader,
    Planner,
    QueryRewriter,
    Reviser,
    Router,
    RouterDecision,
    Synthesizer,
    Tutor,
    Verifier,
    detect_task,
)
from .profile import format_profile, update_profile, weak_topics
from .prompts import REFUSAL_MARKER
from .research import ResearchGraph
from .state import AgentState, ResearchState, ResearchTask, SubAnswer
from .tools import calculate, evaluate

__all__ = [
    "ANSWER_STREAM_TAG",
    "REFUSAL_MARKER",
    "SELF_STUDY_DOMAIN",
    "AgentState",
    "Answerer",
    "EduPilotAgent",
    "EvidenceBlock",
    "EvidenceGrade",
    "EvidenceGrader",
    "FlashcardsOutput",
    "PipelineConfig",
    "PipelineResult",
    "Planner",
    "PlannerOutput",
    "QueryRewrite",
    "QueryRewriter",
    "QuizOutput",
    "ResearchGraph",
    "ResearchState",
    "ResearchTask",
    "Reviser",
    "Router",
    "RouterDecision",
    "RouterOutput",
    "SourceFence",
    "SubAnswer",
    "Synthesizer",
    "Tutor",
    "Verifier",
    "VerifierOutput",
    "build_evidence",
    "build_evidence_summary",
    "build_sources",
    "calculate",
    "chunk_text",
    "citation_label",
    "detect_task",
    "evaluate",
    "fold_memory",
    "format_history",
    "format_memory",
    "format_profile",
    "is_refusal",
    "make_fence",
    "parse_json_response",
    "seed_from_db",
    "strip_refusal_marker",
    "update_profile",
    "weak_topics",
]
