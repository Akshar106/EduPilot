"""
Embedding and reranking backends.

Hosted (Pinecone Inference) and local (sentence-transformers) backends share
one interface; these tests run both against fakes, so no model is downloaded
and no request leaves the machine. Token counting is replaced with a word
count where a test needs exact budgets.
"""

from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from edupilot.retrieval import embeddings, vectorstore
from edupilot.retrieval.embeddings import (
    EmbeddingCache,
    PineconeEmbedder,
    SentenceTransformerEmbedder,
    TokenBudgetError,
    build_embedder,
)

# The package re-exports a function named `rerank`, which shadows the module.
rerank = importlib.import_module("edupilot.retrieval.rerank")


@pytest.fixture
def word_counts(monkeypatch):
    """Count tokens as words, so budget tests do not need a tokenizer download."""
    from edupilot.ingestion import chunking

    monkeypatch.setattr(chunking, "get_token_counter", lambda name=None: lambda t: len(t.split()))


class FakeInference:
    def __init__(self, dimension: int = 1024) -> None:
        self.dimension = dimension
        self.embed_calls: list[dict] = []
        self.rerank_calls: list[dict] = []
        self.scores: list[float] = []
        self.fail = False

    def embed(self, *, model, inputs, parameters):
        self.embed_calls.append({"model": model, "inputs": list(inputs), "parameters": parameters})
        # Deliberately not unit length: the embedder must normalize.
        return SimpleNamespace(
            data=[SimpleNamespace(values=[3.0] + [0.0] * (self.dimension - 1)) for _ in inputs]
        )

    def rerank(self, *, model, query, documents, **kwargs):
        if self.fail:
            raise ConnectionError("inference unavailable")
        self.rerank_calls.append({"model": model, "query": query, "documents": documents, **kwargs})
        return SimpleNamespace(
            data=[SimpleNamespace(index=i, score=s) for i, s in enumerate(self.scores)]
        )


def hosted(inference: FakeInference, **kwargs) -> PineconeEmbedder:
    return PineconeEmbedder("llama-text-embed-v2", client=SimpleNamespace(inference=inference), **kwargs)


# ---------------------------------------------------------------------------
# Hosted embeddings
# ---------------------------------------------------------------------------


def test_the_model_picks_its_backend():
    assert isinstance(build_embedder("llama-text-embed-v2", api_key="k"), PineconeEmbedder)
    assert isinstance(build_embedder("BAAI/bge-small-en-v1.5"), SentenceTransformerEmbedder)


def test_passages_and_queries_are_embedded_as_such_without_truncation(word_counts):
    inference = FakeInference()
    embedder = hosted(inference)

    docs = embedder.embed_documents([f"chunk {i}" for i in range(100)])
    query = embedder.embed_query("what is variance")

    assert docs.shape == (100, 1024) and query.shape == (1024,)
    assert np.allclose(np.linalg.norm(docs, axis=1), 1.0), "vectors must be unit length"
    passage_calls = [c for c in inference.embed_calls if c["parameters"]["input_type"] == "passage"]
    assert [len(c["inputs"]) for c in passage_calls] == [64, 36], "batched under the request limit"
    assert all(c["parameters"]["truncate"] == "NONE" for c in inference.embed_calls)
    assert inference.embed_calls[-1]["parameters"]["input_type"] == "query"


def test_a_repeated_query_is_not_sent_twice(word_counts):
    inference = FakeInference()
    embedder = hosted(inference)
    embedder.embed_query("what is variance")
    embedder.embed_query("what is variance")
    assert len(inference.embed_calls) == 1


def test_cached_passages_are_not_re_embedded(word_counts):
    inference = FakeInference()
    embedder = hosted(inference, cache=EmbeddingCache())
    embedder.embed_documents(["a chunk", "another chunk"])
    embedder.embed_documents(["a chunk", "another chunk", "a new chunk"])
    assert [c["inputs"] for c in inference.embed_calls] == [
        ["a chunk", "another chunk"],
        ["a new chunk"],
    ]


def test_over_budget_text_is_refused_before_any_request(word_counts):
    inference = FakeInference()
    with pytest.raises(TokenBudgetError):
        hosted(inference).embed_documents(["word " * 2100])
    assert inference.embed_calls == []


def test_a_wrong_sized_response_is_an_error_not_a_bad_index(word_counts):
    with pytest.raises(ValueError, match="shape"):
        hosted(FakeInference(dimension=384)).embed_documents(["a chunk"])


def test_chunks_are_sized_with_the_stand_in_tokenizer():
    embedder = build_embedder("llama-text-embed-v2", api_key="k")
    assert embedder.tokenizer_name == "BAAI/bge-small-en-v1.5"
    assert embedder.max_tokens == 2048


# ---------------------------------------------------------------------------
# Local embeddings keep their behaviour after the refactor
# ---------------------------------------------------------------------------


class FakeSentenceTransformer:
    def __init__(self) -> None:
        self.encoded: list[list[str]] = []
        self.tokenizer = SimpleNamespace(encode=lambda t, **kw: t.split())

    def encode(self, texts, **kwargs):
        self.encoded.append(list(texts))
        return np.ones((len(texts), 384), dtype=np.float32) / np.sqrt(384)


def test_the_local_backend_still_prefixes_queries_only():
    embedder = SentenceTransformerEmbedder("BAAI/bge-small-en-v1.5")
    embedder._model = FakeSentenceTransformer()

    embedder.embed_documents(["a passage"])
    embedder.embed_query("what is variance")

    passage, query = embedder._model.encoded
    assert passage == ["a passage"]
    assert query == [embedder.spec.query_prefix + "what is variance"]


# ---------------------------------------------------------------------------
# Hosted reranking
# ---------------------------------------------------------------------------

CANDIDATES = [
    ("c1", "Variance measures spread.", {"filename": "stat.pdf"}, 0.4),
    ("c2", "Bias is systematic error.", {"filename": "aml.pdf"}, 0.3),
    ("c3", "Unrelated text about cooking.", {"filename": "misc.pdf"}, 0.2),
]


@pytest.fixture
def inference(monkeypatch):
    fake = FakeInference()
    monkeypatch.setattr(rerank, "_pinecone", lambda: SimpleNamespace(inference=fake))
    return fake


def test_a_hosted_reranker_scores_and_filters_in_one_request(inference):
    inference.scores = [0.91, 0.30, 0.001]
    kept = rerank.rerank(
        "what is variance", CANDIDATES, top_k=5, model_name="bge-reranker-v2-m3"
    )

    assert [c.chunk_id for c in kept] == ["c1", "c2"], "below the absolute floor is dropped"
    assert kept[0].relevance == pytest.approx(0.91)
    (call,) = inference.rerank_calls
    assert call["model"] == "bge-reranker-v2-m3"
    assert [d["id"] for d in call["documents"]] == ["c1", "c2", "c3"]


def test_a_hosted_request_never_exceeds_its_document_limit(inference):
    many = [(f"c{i}", f"variance text {i}", {}, 1.0 - i / 200) for i in range(130)]
    inference.scores = [0.9] * rerank.HOSTED_MAX_DOCUMENTS
    kept = rerank.rerank("variance", many, top_k=200, model_name="bge-reranker-v2-m3")

    (call,) = inference.rerank_calls
    assert len(call["documents"]) == rerank.HOSTED_MAX_DOCUMENTS
    assert {c.chunk_id for c in kept} <= {f"c{i}" for i in range(100)}, "the fused-rank tail is dropped"


def test_the_configured_reranker_is_used_by_default(inference, monkeypatch):
    from edupilot.core import config

    monkeypatch.setattr(config, "RERANKER_MODEL", "bge-reranker-v2-m3")
    inference.scores = [0.9, 0.2, 0.1]
    rerank.rerank("what is variance", CANDIDATES, top_k=5)
    assert inference.rerank_calls, "RERANKER_MODEL must reach the reranker"


def test_a_failed_hosted_call_falls_back_to_keyword_scoring(inference):
    inference.fail = True
    kept = rerank.rerank("variance spread", CANDIDATES, top_k=5, model_name="bge-reranker-v2-m3")
    assert kept and kept[0].chunk_id == "c1"


def test_missing_local_models_are_detected_once(monkeypatch):
    monkeypatch.setattr(rerank, "_model", None)
    monkeypatch.setattr(rerank, "_not_installed", False)
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)  # import now fails

    assert rerank.get_cross_encoder() == (None, None)
    assert rerank._not_installed is True
    monkeypatch.delitem(sys.modules, "sentence_transformers")
    assert rerank.get_cross_encoder() == (None, None), "not retried on every request"


# ---------------------------------------------------------------------------
# The index must match the embedding model
# ---------------------------------------------------------------------------


def test_an_index_built_with_another_embedding_model_is_refused():
    store = vectorstore.PineconeVectorStore(
        api_key="k", index_name="edupilot-v3", dimension=1024, create_if_missing=False
    )
    store._client = SimpleNamespace(
        describe_index=lambda name: SimpleNamespace(dimension=384),
        Index=lambda name: pytest.fail("a mismatched index must not be opened"),
    )
    with pytest.raises(vectorstore.VectorStoreError, match="384-dimensional"):
        store.index  # noqa: B018 - property access opens the index


def test_a_matching_index_opens():
    store = vectorstore.PineconeVectorStore(
        api_key="k", index_name="edupilot-v3", dimension=1024, create_if_missing=False
    )
    store._client = SimpleNamespace(
        describe_index=lambda name: SimpleNamespace(dimension=1024),
        Index=lambda name: f"handle:{name}",
    )
    assert store.index == "handle:edupilot-v3"


def test_the_registry_knows_the_hosted_model():
    spec = embeddings.get_model_spec("llama-text-embed-v2")
    assert (spec.provider, spec.dimension) == ("pinecone", 1024)
