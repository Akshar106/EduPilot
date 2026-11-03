"""
The pgvector store and the index version pointer.

Runs against the run's throwaway Postgres. Each test gets its own index
version (its own table), dropped afterwards, so tests cannot see each other's
vectors.
"""

from __future__ import annotations

import uuid

import numpy as np
import pytest
from sqlalchemy import delete

from edupilot import db
from edupilot.db.tables import index_versions
from edupilot.retrieval import (
    BM25Encoder,
    IndexPointer,
    InMemoryVectorStore,
    PgVectorStore,
    VectorRecord,
    VectorStoreError,
)
from edupilot.retrieval.pgvector import table_name

DIM = 8


def unit(*hot: int, dim: int = DIM) -> list[float]:
    v = np.zeros(dim, dtype=np.float32)
    v[list(hot)] = 1.0
    return (v / np.linalg.norm(v)).tolist()


@pytest.fixture
def store():
    name = f"test-{uuid.uuid4().hex[:8]}-v1"
    s = PgVectorStore(index_name=name, dimension=DIM)
    yield s
    s.drop_index(name)


def records() -> list[VectorRecord]:
    return [
        VectorRecord("a", unit(0), {"text": "variance spread", "filename": "stat.pdf", "page": 1, "is_parent": False, "confidence": 0.9}, sparse={11: 1.2, 12: 0.4}),
        VectorRecord("b", unit(0, 1), {"text": "bias error", "filename": "aml.pdf", "page": 2, "is_parent": False, "confidence": 0.5}, sparse={13: 1.0}),
        VectorRecord("c", unit(2), {"text": "parent window", "filename": "stat.pdf", "page": 1, "is_parent": True}),
        VectorRecord("d", unit(3), {"text": "attention", "filename": "llm.pdf", "page": 7, "is_parent": False, "confidence": 0.7}, sparse={11: 0.2, 14: 2.0}),
    ]


# ---------------------------------------------------------------------------
# Reads and writes
# ---------------------------------------------------------------------------


def test_dense_search_ranks_by_cosine_similarity(store):
    store.upsert(records(), "ns")
    hits = store.query(unit(0), top_k=2, namespace="ns")
    assert [h.id for h in hits] == ["a", "b"]
    assert hits[0].score == pytest.approx(1.0, abs=1e-5)
    assert hits[0].metadata["filename"] == "stat.pdf"


def test_namespaces_are_isolated(store):
    store.upsert(records(), "aml")
    store.upsert([VectorRecord("z", unit(0), {"text": "other"})], "ss_123")
    assert {h.id for h in store.query(unit(0), top_k=10, namespace="ss_123")} == {"z"}
    assert "z" not in {h.id for h in store.query(unit(0), top_k=10, namespace="aml")}


@pytest.mark.parametrize(
    "flt",
    [
        {"filename": {"$eq": "stat.pdf"}},
        {"filename": "stat.pdf"},
        {"is_parent": {"$ne": True}},
        {"filename": {"$in": ["aml.pdf", "llm.pdf"]}},
        {"filename": {"$nin": ["aml.pdf"]}},
        {"confidence": {"$gte": 0.7}},
        {"confidence": {"$lt": 0.7}},
        {"page": {"$gt": 1}},
        {"$and": [{"is_parent": {"$ne": True}}, {"filename": {"$in": ["stat.pdf", "llm.pdf"]}}]},
        {"$or": [{"filename": "aml.pdf"}, {"page": {"$lte": 1}}]},
        {"filename": {"$in": []}},
    ],
)
def test_filters_match_the_in_memory_store(store, flt):
    """Same Pinecone-style semantics as the store the other tests run on."""
    memory = InMemoryVectorStore()
    memory.upsert(records(), "ns")
    store.upsert(records(), "ns")
    expected = {h.id for h in memory.query(unit(0), top_k=10, namespace="ns", metadata_filter=flt)}
    got = {h.id for h in store.query(unit(0), top_k=10, namespace="ns", metadata_filter=flt)}
    assert got == expected


def test_lexical_matches_join_the_results_through_rank_fusion(store):
    store.upsert(records(), "ns")
    # Dense alone would rank d last; its strong BM25 match on term 14 lifts it.
    dense = [h.id for h in store.query(unit(0), top_k=4, namespace="ns")]
    hybrid = [h.id for h in store.query(unit(0), top_k=4, namespace="ns", sparse={14: 3.0})]
    assert dense.index("d") > hybrid.index("d")
    assert hybrid[0] in {"a", "d"}


def test_a_lexical_only_hit_is_returned(store):
    """A chunk sharing the query's words but not its meaning still reaches the reranker."""
    store.upsert(records(), "ns")
    hits = store.query(unit(5), top_k=4, namespace="ns", sparse={13: 1.0})
    # Every vector is orthogonal to the query, so only the lexical list ranks b.
    assert hits[0].id == "b"


def test_a_filter_on_a_small_namespace_still_returns_k_results(store):
    """
    HNSW filters while walking the graph (iterative scan), so a few matches
    among many vectors are found rather than lost to a post-filter.
    """
    rng = np.random.default_rng(0)
    bulk = [
        VectorRecord(f"n{i}", (v / np.linalg.norm(v)).tolist(), {"text": "noise", "filename": "big.pdf"})
        for i, v in enumerate(rng.normal(size=(600, DIM)).astype(np.float32))
    ]
    store.upsert(bulk, "ns")
    store.upsert(
        [VectorRecord(f"m{i}", unit(i % DIM), {"text": "mine", "filename": "mine.pdf"}) for i in range(5)],
        "ns",
    )
    hits = store.query(unit(0), top_k=5, namespace="ns", metadata_filter={"filename": "mine.pdf"})
    assert len(hits) == 5


def test_upsert_replaces_by_id(store):
    store.upsert(records(), "ns")
    store.upsert([VectorRecord("a", unit(4), {"text": "rewritten"})], "ns")
    assert store.fetch(["a"], "ns")["a"].text == "rewritten"
    assert store.stats()["namespaces"] == {"ns": 4}


def test_fetch_returns_only_what_exists(store):
    store.upsert(records(), "ns")
    got = store.fetch(["a", "c", "missing", "a"], "ns")
    assert set(got) == {"a", "c"}
    assert got["c"].metadata["is_parent"] is True


def test_deletes_by_id_filter_and_namespace(store):
    store.upsert(records(), "ns")
    store.upsert(records(), "other")
    store.delete("ns", ids=["a"])
    store.delete("ns", metadata_filter={"filename": {"$eq": "stat.pdf"}})
    assert set(store.fetch(["a", "b", "c", "d"], "ns")) == {"b", "d"}
    store.delete("ns", delete_all=True)
    assert store.stats()["namespaces"] == {"other": 4}


def test_reading_an_index_that_was_never_written_is_empty():
    s = PgVectorStore(index_name=f"never-{uuid.uuid4().hex[:8]}-v1", dimension=DIM)
    assert s.query(unit(0), top_k=3, namespace="ns") == []
    assert s.fetch(["a"], "ns") == {}
    assert not s.exists()
    assert s.stats()["total_vectors"] == 0 and s.stats()["exists"] is False


def test_an_index_built_with_another_embedding_model_is_refused(store):
    store.upsert(records(), "ns")
    wider = PgVectorStore(index_name=store.index_name, dimension=DIM * 2)
    with pytest.raises(VectorStoreError, match="EMBEDDING_MODEL"):
        wider.query(unit(0, dim=DIM * 2), top_k=1, namespace="ns")


def test_nul_characters_in_chunk_text_are_dropped(store):
    store.upsert([VectorRecord("x", unit(0), {"text": "a\x00b"})], "ns")
    assert store.fetch(["x"], "ns")["x"].text == "ab"


def test_study_uploads_are_copied_into_a_new_version(store):
    store.upsert(records(), "aml")
    store.upsert([VectorRecord("s1", unit(1), {"text": "my notes"})], "ss_abc")
    new = PgVectorStore(index_name=store.index_name.replace("-v1", "-v2"), dimension=DIM)
    try:
        assert new.copy_namespaces(store.index_name, prefix="ss_") == 1
        assert new.stats()["namespaces"] == {"ss_abc": 1}
        assert new.fetch(["s1"], "ss_abc")["s1"].text == "my notes"
    finally:
        new.drop_index(new.index_name)


def test_drop_removes_the_table(store):
    store.upsert(records(), "ns")
    store.drop_index(store.index_name)
    assert not store.exists()


def test_table_names_are_safe_identifiers():
    assert table_name("edupilot-v3") == "vectors_edupilot_v3"
    assert table_name("Ed U; drop table--v1") == "vectors_ed_u_drop_table_v1"


# ---------------------------------------------------------------------------
# Index versions
# ---------------------------------------------------------------------------


@pytest.fixture
def pointer():
    """A pointer over an empty version table, restored afterwards."""
    with db.transaction() as conn:
        saved = [dict(r._mapping) for r in conn.execute(index_versions.select())]
        conn.execute(delete(index_versions))
    yield IndexPointer("edupilot", backend="pgvector")
    with db.transaction() as conn:
        conn.execute(delete(index_versions))
        if saved:
            conn.execute(index_versions.insert(), saved)


def test_versions_start_at_one_and_count_unpromoted_builds(pointer):
    assert pointer.active is None and pointer.version == 0
    assert pointer.next_version_name() == (1, "edupilot-v1")
    pointer.begin("edupilot-v1", 1, embed_model="bge", dimension=384)
    # v1 was built but never promoted; the next rebuild must not reuse it.
    assert pointer.next_version_name() == (2, "edupilot-v2")


def test_promotion_retires_the_previous_version(pointer):
    pointer.promote("edupilot-v1", 1, note="first")
    pointer.promote("edupilot-v2", 2, note="second")
    assert pointer.active == "edupilot-v2" and pointer.version == 2
    assert [v["name"] for v in pointer.history()] == ["edupilot-v1"]
    statuses = {v["name"]: v["status"] for v in pointer.records()}
    assert statuses == {"edupilot-v1": "retired", "edupilot-v2": "active"}


def test_rolling_back_promotes_the_retired_version_again(pointer):
    pointer.promote("edupilot-v1", 1)
    pointer.promote("edupilot-v2", 2)
    pointer.promote("edupilot-v1", 1)
    assert pointer.active == "edupilot-v1"
    assert [v["name"] for v in pointer.history()] == ["edupilot-v2"]


def test_the_bm25_table_belongs_to_its_version(pointer):
    encoder = BM25Encoder().fit(["variance spread mean", "bias error model"])
    pointer.begin("edupilot-v1", 1, embed_model="bge", dimension=384)
    pointer.save_sparse_encoder("edupilot-v1", encoder.to_dict())
    assert pointer.sparse_encoder() is None  # not live yet
    pointer.promote("edupilot-v1", 1)
    loaded = BM25Encoder.from_dict(pointer.sparse_encoder())
    assert loaded.encode_query("variance") == pytest.approx(encoder.encode_query("variance"))


def test_a_cached_pointer_sees_another_processs_promotion_after_refresh(pointer):
    cached = IndexPointer("edupilot", backend="pgvector", cache_seconds=60)
    assert cached.active is None
    pointer.promote("edupilot-v1", 1)  # "another process"
    assert cached.active is None, "served from the cache within the window"
    cached.refresh()
    assert cached.active == "edupilot-v1"


def test_services_follow_a_promotion_without_a_restart(pointer, monkeypatch):
    """The sparse encoder, retrievers and answer cache all move to the new version."""
    from edupilot.core.services import Services

    services = Services()
    monkeypatch.setattr(services, "_pointer", IndexPointer("edupilot", backend="pgvector"))
    assert services.sparse_encoder is None

    encoder = BM25Encoder().fit(["variance spread mean", "bias error model"])
    pointer.begin("edupilot-v1", 1, embed_model="bge", dimension=384)
    pointer.save_sparse_encoder("edupilot-v1", encoder.to_dict())
    pointer.promote("edupilot-v1", 1)

    assert services.sparse_encoder is not None
    assert services.sparse_encoder.encode_query("bias") == pytest.approx(encoder.encode_query("bias"))
