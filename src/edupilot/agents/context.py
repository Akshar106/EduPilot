"""
Run context
===========
`PipelineConfig` is the per-request configuration every node reads: LangGraph
passes it to the graph and its subgraphs as the run context
(`context_schema`), so it never sits in persisted state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from edupilot.core.config import (
    CORRECTIVE_RETRIEVAL,
    ENABLE_TUTOR_TOOLS,
    LLM_MAX_TOKENS_GENERATE,
    LLM_MAX_TOKENS_SYNTH,
    MAX_QUERY_REWRITES,
    MAX_REVISIONS,
)
from edupilot.retrieval import RetrievalConfig


@dataclass
class PipelineConfig:
    """Per-request settings, passed to every node as the graph's run context."""

    model: str = ""
    verify_model: str = ""

    # Sourced from config so there is one place to tune generation length.
    max_tokens_answer: int = LLM_MAX_TOKENS_GENERATE
    max_tokens_synth: int = LLM_MAX_TOKENS_SYNTH
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    enable_verification: bool = True
    enable_grounding_check: bool = True
    max_parallel_domains: int = 4

    corrective_retrieval: bool = CORRECTIVE_RETRIEVAL
    max_query_rewrites: int = MAX_QUERY_REWRITES
    max_revisions: int = MAX_REVISIONS
    enable_tools: bool = ENABLE_TUTOR_TOOLS

    interactive: bool = False
    """The caller can carry on a paused turn — chat over a checkpointed
    thread. When False an unclear question gets a fixed reply instead of a
    clarifying question the caller could never answer."""
    user_id: str | None = None
    """Whose learning profile to read and update. None for anonymous runs
    such as evaluation."""

    trace_tags: tuple[str, ...] = ()
    """Extra LangSmith tags for this run, e.g. the evaluation experiment."""
    trace_metadata: dict[str, Any] = field(default_factory=dict)
    """Extra LangSmith metadata for this run, e.g. the evaluation case id."""
