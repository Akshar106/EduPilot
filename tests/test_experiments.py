"""
LangSmith experiments.

The LangSmith client is a fake that stores examples in memory and runs
`evaluate` the way LangSmith does — target per example, then the evaluators on
the resulting run — so dataset sync, variant wiring and scoring are checked
without a network call.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from edupilot.evaluation import TEST_CASES
from edupilot.evaluation.experiments import (
    DATASET_NAME,
    VARIANTS,
    example_id,
    run_experiments,
    sync_dataset,
    to_example,
)

CASES = [tc for tc in TEST_CASES if tc.category != "edge-case"][:2]


class FakeLangSmith:
    def __init__(self) -> None:
        self.datasets: dict[str, str] = {}
        self.examples: dict[str, SimpleNamespace] = {}
        self.created = self.updated = 0
        self.experiments: list[dict] = []

    # -- datasets ------------------------------------------------------
    def has_dataset(self, *, dataset_name):
        return dataset_name in self.datasets

    def read_dataset(self, *, dataset_name):
        return SimpleNamespace(id=self.datasets[dataset_name])

    def create_dataset(self, name, *, description=None):
        self.datasets[name] = f"ds-{name}"
        return SimpleNamespace(id=self.datasets[name])

    def list_examples(self, *, dataset_id, example_ids=None):
        wanted = set(example_ids) if example_ids is not None else None
        return [ex for eid, ex in self.examples.items() if wanted is None or eid in wanted]

    def create_examples(self, *, dataset_id, examples):
        for ex in examples:
            self.examples[ex["id"]] = SimpleNamespace(**ex)
            self.created += 1

    def update_example(self, example_id, *, inputs, outputs, metadata, dataset_id):
        self.examples[example_id] = SimpleNamespace(
            id=example_id, inputs=inputs, outputs=outputs, metadata=metadata
        )
        self.updated += 1

    # -- experiments ---------------------------------------------------
    def evaluate(self, target, *, data, evaluators, experiment_prefix, metadata, **kwargs):
        rows = []
        for example in data:
            outputs = target(example.inputs)
            run = SimpleNamespace(outputs=outputs, error=None)
            rows.append([ev(run, example) for ev in evaluators])
        self.experiments.append({"prefix": experiment_prefix, "metadata": metadata, "rows": rows})
        return SimpleNamespace(experiment_name=f"{experiment_prefix}-0001")


def test_the_dataset_is_created_once_and_updated_in_place():
    client = FakeLangSmith()
    sync_dataset(client, CASES)
    assert client.created == len(CASES)
    assert set(client.examples) == {example_id(tc.id) for tc in CASES}

    sync_dataset(client, CASES)
    assert client.created == len(CASES) and client.updated == 0, "re-sync must not duplicate"

    edited = to_example(CASES[0])
    client.examples[edited["id"]].outputs = {**edited["outputs"], "expected_intent": "stale"}
    sync_dataset(client, CASES)
    assert client.updated == 1
    assert client.examples[edited["id"]].outputs == edited["outputs"]


def answering_pipeline(**factory_kwargs):
    """A pipeline_fn that answers every case correctly, recording its overrides."""
    answering_pipeline.calls.append(factory_kwargs)
    by_query = {tc.query: tc for tc in TEST_CASES}

    def pipeline_fn(query, model, top_k, rerank_top_k, enable_verification):
        tc = by_query[query]
        return {
            "final_answer": f"{' '.join(tc.relevant_keywords)} explained here [Source 1].",
            "intent_type": tc.expected_intent if tc.expected_intent != "any" else "single",
            "detected_domains": list(tc.expected_domains),
            "is_course_related": True,
            "needs_clarification": False,
            "quality_score": 0.9,
            "sources": [{"source_num": 1, "text": " ".join(tc.relevant_keywords)}],
            "debug": {},
        }

    return pipeline_fn


answering_pipeline.calls = []


def test_each_variant_is_one_experiment_with_its_config_recorded(monkeypatch):
    from edupilot.evaluation import runner

    # The faithfulness judge and relevance model are LLM / model calls.
    monkeypatch.setattr(runner, "faithfulness", lambda **kw: 0.9)
    monkeypatch.setattr(runner, "answer_relevance", lambda q, a: 0.0)
    answering_pipeline.calls.clear()
    client = FakeLangSmith()

    reports = run_experiments(
        ["baseline", "no-corrective"],
        cases=CASES,
        model="fake",
        top_k=12,
        rerank_top_k=8,
        client=client,
        pipeline_factory=answering_pipeline,
    )

    assert [e["prefix"] for e in client.experiments] == ["edupilot-baseline", "edupilot-no-corrective"]
    assert client.experiments[1]["metadata"]["corrective_retrieval"] is False
    assert answering_pipeline.calls[1] == {
        "overrides": VARIANTS["no-corrective"],
        "trace_tags": ("eval", "no-corrective"),
    }

    baseline = reports[0]
    assert baseline.experiment == "edupilot-baseline-0001"
    assert len(baseline.results) == len(CASES) and all(r.passed for r in baseline.results)

    scores = {r["key"]: r["score"] for r in client.experiments[0]["rows"][0][0]["results"]}
    assert scores["passed"] == 1 and scores["citation_accuracy"] == 1.0
    assert scores["retrieval_hit_rate"] == 1.0
    assert "answer_relevance" not in scores, "an unavailable metric is omitted, not zeroed"


def test_an_unknown_variant_is_refused_before_anything_runs():
    client = FakeLangSmith()
    with pytest.raises(ValueError, match="unknown variant"):
        run_experiments(["fastest"], cases=CASES, top_k=12, rerank_top_k=8, client=client)
    assert DATASET_NAME not in client.datasets
