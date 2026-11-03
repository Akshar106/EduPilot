"""
LangSmith experiments
=====================
Runs the test suite as LangSmith experiments, so configurations can be
compared side by side — per case, per metric, with latency and token cost —
instead of as two terminal printouts.

    edupilot-evaluate --langsmith
    edupilot-evaluate --langsmith --variant baseline --variant no-corrective

The test cases live in LangSmith as a dataset, kept in step with `TEST_CASES`
by `sync_dataset` and keyed by case id, so re-running never duplicates them.
Each variant is one experiment on that dataset: the pipeline the API runs, with
some `PipelineConfig` fields changed, scored by the same rules as the local
runner (`score_result`). Every pipeline run is traced under its experiment, so
a failing case opens straight into the steps that produced it.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from edupilot.core.config import DEFAULT_MODEL, EMBEDDING_MODEL, RERANKER_MODEL

from .cases import TEST_CASES, TestCase, TestResult
from .runner import score_result

DATASET_NAME = "edupilot-golden-cases"

#: Configurations worth comparing: the default, and each agentic behaviour
#: switched off on its own, to see what it costs and what it buys.
VARIANTS: dict[str, dict[str, Any]] = {
    "baseline": {},
    "no-corrective": {"corrective_retrieval": False},
    "no-revision": {"max_revisions": 0},
    "no-tools": {"enable_tools": False},
    "lean": {"corrective_retrieval": False, "max_revisions": 0, "enable_tools": False},
}


@dataclass
class ExperimentReport:
    variant: str
    experiment: str
    results: list[TestResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def example_id(case_id: str) -> str:
    """Stable example id per case, so updates replace rather than duplicate."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"edupilot-case:{case_id}"))


def to_example(tc: TestCase) -> dict[str, Any]:
    return {
        "id": example_id(tc.id),
        "inputs": {"query": tc.query},
        "outputs": {
            "expected_intent": tc.expected_intent,
            "expected_domains": tc.expected_domains,
            "expected_behavior": tc.expected_behavior,
            "relevant_keywords": tc.relevant_keywords,
            "gold_answer": tc.gold_answer,
        },
        "metadata": {"case_id": tc.id, "name": tc.name, "category": tc.category},
    }


def sync_dataset(client, cases: Sequence[TestCase] = TEST_CASES, name: str = DATASET_NAME) -> str:
    """
    Make the LangSmith dataset match `cases`, and return its id.

    Missing cases are added and edited ones updated. Examples for cases no
    longer in the suite are left alone — deleting data in someone's LangSmith
    workspace is not this command's call — and are simply not evaluated.
    """
    if client.has_dataset(dataset_name=name):
        dataset = client.read_dataset(dataset_name=name)
    else:
        dataset = client.create_dataset(
            name, description="EduPilot test cases: routing, retrieval, grounding and refusals."
        )

    existing = {str(ex.id): ex for ex in client.list_examples(dataset_id=dataset.id)}
    added = []
    for tc in cases:
        wanted = to_example(tc)
        current = existing.get(wanted["id"])
        if current is None:
            added.append(wanted)
            continue
        metadata = current.metadata or {}
        if (
            current.inputs != wanted["inputs"]
            or current.outputs != wanted["outputs"]
            or any(metadata.get(k) != v for k, v in wanted["metadata"].items())
        ):
            client.update_example(
                wanted["id"],
                inputs=wanted["inputs"],
                outputs=wanted["outputs"],
                metadata={**metadata, **wanted["metadata"]},
                dataset_id=dataset.id,
            )
    if added:
        client.create_examples(dataset_id=dataset.id, examples=added)
    return str(dataset.id)


# ---------------------------------------------------------------------------
# Target and evaluator
# ---------------------------------------------------------------------------


def make_target(
    pipeline_fn: Callable, *, model: str, top_k: int, rerank_top_k: int, enable_verification: bool
) -> Callable[[dict], dict]:
    """The function LangSmith calls once per example: one stateless pipeline run."""

    def target(inputs: dict) -> dict:
        started = time.perf_counter()
        result = pipeline_fn(
            query=inputs["query"],
            model=model,
            top_k=top_k,
            rerank_top_k=rerank_top_k,
            enable_verification=enable_verification,
        )
        return {**result, "latency_ms": round((time.perf_counter() - started) * 1000, 1)}

    return target


def make_evaluator(
    cases_by_id: dict[str, TestCase], *, model: str, collected: list[TestResult]
) -> Callable:
    """
    One evaluator reporting every metric, scored by `score_result`.

    `collected` receives each `TestResult`, so the CLI can print the same
    scorecard as a local run once the experiment finishes.
    """

    def edupilot_metrics(run, example) -> dict:
        tc = cases_by_id[example.metadata["case_id"]]
        outputs = run.outputs or {}
        scored = score_result(tc, outputs, latency_ms=outputs.get("latency_ms", 0.0), model=model)
        if run.error:
            scored.passed = False
            scored.error = str(run.error)
        collected.append(scored)

        results = [
            {"key": "passed", "score": int(scored.passed), "comment": scored.behavior_notes},
            {"key": "intent_match", "score": int(scored.intent_match)},
            {"key": "domain_match", "score": int(scored.domain_match)},
            {"key": "retrieval_hit_rate", "score": scored.retrieval_hit_rate},
            {"key": "citation_accuracy", "score": scored.citation_accuracy},
            {"key": "faithfulness", "score": scored.faithfulness_score},
            {"key": "grounding", "score": scored.grounding_score},
        ]
        # 0.0 also means "the relevance model could not load"; recording it
        # would drag the average down for a missing dependency.
        if scored.answer_relevance > 0:
            results.append({"key": "answer_relevance", "score": scored.answer_relevance})
        return {"results": results}

    return edupilot_metrics


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


def run_experiments(
    variants: Sequence[str],
    *,
    cases: Sequence[TestCase] = TEST_CASES,
    model: str = DEFAULT_MODEL,
    top_k: int,
    rerank_top_k: int,
    enable_verification: bool = True,
    max_concurrency: int = 2,
    prefix: str = "edupilot",
    client=None,
    pipeline_factory: Callable[..., Callable] | None = None,
) -> list[ExperimentReport]:
    """
    Run each variant as an experiment over `cases`. Needs LANGSMITH_API_KEY.

    `max_concurrency` is low by default: each case makes several LLM calls,
    and free-tier provider rate limits are the usual bottleneck.
    """
    from langsmith.run_helpers import tracing_context

    from .live import build_pipeline_fn

    unknown = [v for v in variants if v not in VARIANTS]
    if unknown:
        raise ValueError(f"unknown variant(s) {unknown}; choose from {sorted(VARIANTS)}")

    if client is None:
        from langsmith import Client

        client = Client()
    pipeline_factory = pipeline_factory or build_pipeline_fn

    dataset_id = sync_dataset(client)
    cases_by_id = {tc.id: tc for tc in cases}
    example_ids = [example_id(tc.id) for tc in cases]

    reports = []
    for variant in variants:
        overrides = VARIANTS[variant]
        collected: list[TestResult] = []
        pipeline_fn = pipeline_factory(overrides=overrides, trace_tags=("eval", variant))
        # Pipeline runs are traced under the experiment whether or not
        # LANGSMITH_TRACING is set for the app.
        with tracing_context(enabled=True):
            results = client.evaluate(
                make_target(
                    pipeline_fn,
                    model=model,
                    top_k=top_k,
                    rerank_top_k=rerank_top_k,
                    enable_verification=enable_verification,
                ),
                data=client.list_examples(dataset_id=dataset_id, example_ids=example_ids),
                evaluators=[make_evaluator(cases_by_id, model=model, collected=collected)],
                experiment_prefix=f"{prefix}-{variant}",
                description=f"EduPilot suite, variant {variant!r}",
                metadata={
                    "variant": variant,
                    "model": model,
                    "embedding_model": EMBEDDING_MODEL,
                    "reranker_model": RERANKER_MODEL,
                    "top_k": top_k,
                    "rerank_top_k": rerank_top_k,
                    "verification": enable_verification,
                    **overrides,
                },
                max_concurrency=max_concurrency,
            )
        reports.append(
            ExperimentReport(
                variant=variant,
                experiment=getattr(results, "experiment_name", f"{prefix}-{variant}"),
                results=collected,
            )
        )
    return reports
