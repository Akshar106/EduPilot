"""
Vector store
============
A provider-agnostic interface over the vector database, a Pinecone
implementation, and an in-memory one for tests. The pgvector implementation,
which the app uses by default, is in retrieval/pgvector.py.

What this fixes in the previous `retriever.py`:

  * **Filtering happened in Python.** `source_filter` over-fetched 9x the
    candidates and discarded the rest client-side, then silently fell back to
    *unfiltered* results when nothing matched — so "search only this
    attachment" quietly searched everything. Filters are now pushed down to
    the engine.
  * **No retry.** A single transient 5xx failed the whole request. Upserts and
    queries now retry with exponential backoff.
  * **No index versioning.** Re-indexing mutated the live index in place, so
    the system served a half-built index during any re-ingest. Indexes are now
    versioned (`edupilot-v3`) behind a pointer (retrieval/versions.py) that is
    flipped only once a rebuild completes — blue/green, with instant rollback.
  * **Unbounded upserts.** Vectors were sent in fixed batches of 100 with no
    regard for Pinecone's 2 MB request cap; a batch of large-metadata chunks
    could exceed it. Batches are now sized by payload bytes.

A Pinecone index uses `metric="dotproduct"`, which is required for native
sparse-dense hybrid search. Because all embeddings are L2-normalized, the dot
product equals cosine similarity, so dense ranking is unchanged.
"""

from __future__ import annotations

import json
import logging
import random
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .versions import IndexPointer

logger = logging.getLogger(__name__)

#: Pinecone caps a single upsert request at 2 MB. Stay well under it.
MAX_UPSERT_BYTES = 1_500_000
#: ...and at 1000 vectors per request.
MAX_UPSERT_VECTORS = 250
#: Pinecone caps metadata at 40 KB per vector.
MAX_METADATA_BYTES = 38_000

_RETRYABLE = ("429", "500", "502", "503", "504", "timeout", "temporarily", "unavailable")


class VectorStoreError(RuntimeError):
    """Vector store operation failed after exhausting retries."""


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class VectorRecord:
    """One vector to upsert."""

    id: str
    values: Sequence[float]
    metadata: dict[str, Any] = field(default_factory=dict)
    sparse: dict[int, float] | None = None
    """Sparse term weights (term index -> weight) for hybrid search."""


@dataclass
class SearchHit:
    """One search result."""

    id: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    values: list[float] | None = None

    @property
    def text(self) -> str:
        return str(self.metadata.get("text", ""))


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------


class VectorStore(ABC):
    """Minimal interface the retrieval layer depends on."""

    @abstractmethod
    def upsert(self, records: Sequence[VectorRecord], namespace: str) -> int:
        """Insert or replace vectors. Returns the count written."""

    @abstractmethod
    def query(
        self,
        vector: Sequence[float],
        top_k: int,
        namespace: str,
        *,
        metadata_filter: dict | None = None,
        sparse: dict[int, float] | None = None,
        include_values: bool = False,
    ) -> list[SearchHit]:
        """Search. `metadata_filter` is pushed down to the engine, not applied locally."""

    @abstractmethod
    def fetch(self, ids: Sequence[str], namespace: str) -> dict[str, SearchHit]:
        """Fetch vectors by ID. Used for small-to-big parent expansion."""

    @abstractmethod
    def delete(
        self,
        namespace: str,
        *,
        ids: Sequence[str] | None = None,
        delete_all: bool = False,
        metadata_filter: dict | None = None,
    ) -> None:
        """Delete by ID, by filter, or wipe a namespace."""

    @abstractmethod
    def stats(self) -> dict:
        """Index statistics — vector counts per namespace."""

    def document_titles(self, namespace: str) -> dict[str, str]:
        """
        Each document's title, by filename, as ingestion stored it with the
        chunks. Empty where the engine cannot list it cheaply; callers fall
        back to the file name.
        """
        return {}


# ---------------------------------------------------------------------------
# Metadata sanitation
# ---------------------------------------------------------------------------


def sanitize_metadata(
    meta: dict[str, Any], *, max_bytes: int | None = MAX_METADATA_BYTES
) -> dict[str, Any]:
    """
    Coerce a metadata dict into what Pinecone accepts.

    Pinecone permits only str, int, float, bool, and list[str]. Nulls and
    nested structures are rejected at write time, so they are converted or
    dropped here rather than failing an entire batch. pgvector applies the
    same rules, so a chunk's metadata reads back identically from either
    backend, but has no size cap (`max_bytes=None`).
    """
    out: dict[str, Any] = {}
    for key, value in meta.items():
        if value is None:
            continue
        if isinstance(value, str):
            # Postgres refuses NUL in text and JSONB.
            out[key] = value.replace("\x00", "")
        elif isinstance(value, bool) or isinstance(value, (int, float)):
            out[key] = value
        elif isinstance(value, (list, tuple)):
            out[key] = [str(v) for v in value if v is not None]
        else:
            out[key] = json.dumps(value, default=str)

    if max_bytes is None:
        return out

    # Enforce the per-vector metadata cap, trimming the text field first since
    # it is both the largest and the only one that degrades gracefully.
    encoded = len(json.dumps(out, default=str).encode("utf-8"))
    if encoded > max_bytes and "text" in out:
        overflow = encoded - max_bytes
        text = str(out["text"])
        out["text"] = text[: max(0, len(text) - overflow - 64)]
        out["text_truncated"] = True
        logger.warning("metadata over %d bytes; truncated text field", max_bytes)
    return out


def _retry(operation: str, fn, *, attempts: int = 4, base_delay: float = 0.5):
    """
    Run `fn`, retrying transient failures with exponential backoff and jitter.

    Non-transient errors (bad request, auth) raise immediately — retrying them
    only delays a failure the caller needs to see.
    """
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:
            last = exc
            message = str(exc).lower()
            if not any(token in message for token in _RETRYABLE):
                raise VectorStoreError(f"{operation} failed: {exc}") from exc
            if attempt == attempts - 1:
                break
            delay = base_delay * (2**attempt) + random.uniform(0, 0.25)
            logger.warning(
                "%s failed (attempt %d/%d): %s — retrying in %.1fs",
                operation, attempt + 1, attempts, exc, delay,
            )
            time.sleep(delay)
    raise VectorStoreError(f"{operation} failed after {attempts} attempts: {last}") from last


# ---------------------------------------------------------------------------
# Pinecone implementation
# ---------------------------------------------------------------------------


class PineconeVectorStore(VectorStore):
    """
    Pinecone-backed vector store.

    Args:
        index_name: Concrete index to use. Omit to resolve via `pointer`.
        pointer: Version pointer for blue/green index selection.
        dimension: Embedding dimension; used only when creating an index.
        metric: `dotproduct` is required for native sparse-dense hybrid. With
            L2-normalized vectors it is equivalent to cosine.
    """

    def __init__(
        self,
        *,
        api_key: str,
        index_name: str | None = None,
        pointer: IndexPointer | None = None,
        dimension: int = 384,
        metric: str = "dotproduct",
        cloud: str = "aws",
        region: str = "us-east-1",
        create_if_missing: bool = True,
    ) -> None:
        if not api_key:
            raise ValueError("PINECONE_API_KEY is not set")
        if not index_name and not pointer:
            raise ValueError("provide index_name or pointer")

        self._api_key = api_key
        self._pointer = pointer
        self._explicit_name = index_name
        self.dimension = dimension
        self.metric = metric
        self.cloud = cloud
        self.region = region
        self.create_if_missing = create_if_missing

        self._client = None
        self._index = None
        self._index_name: str | None = None
        # RLock, not Lock. `index` holds this lock and then calls
        # `ensure_index`, which reads `self.client` — and `client` acquires the
        # same lock to build the SDK handle. With a plain Lock that is a
        # self-deadlock: the thread blocks forever waiting for a lock it is
        # already holding, at 0% CPU, with no other thread able to release it.
        #
        # It only triggers on a store whose client has not been built yet,
        # which is why serving traffic never hit it and every `--rebuild` hung:
        # `store_for_index()` returns a fresh instance, so the first upsert
        # takes exactly this path. Same reasoning as Services._lock.
        self._lock = threading.RLock()

    # -- connection --------------------------------------------------------

    @property
    def client(self):
        if self._client is None:
            with self._lock:
                if self._client is None:
                    from pinecone import Pinecone

                    self._client = Pinecone(api_key=self._api_key)
        return self._client

    @property
    def index_name(self) -> str:
        if self._explicit_name:
            return self._explicit_name
        assert self._pointer is not None
        active = self._pointer.active
        if not active:
            # No rebuild has been promoted yet — start at v1.
            _, name = self._pointer.next_version_name()
            return name
        return active

    @property
    def index(self):
        """The live index handle, created on first use under a lock."""
        name = self.index_name
        if self._index is not None and self._index_name == name:
            return self._index
        with self._lock:
            if self._index is None or self._index_name != name:
                if self.create_if_missing:
                    self.ensure_index(name)
                self._check_dimension(name)
                self._index = self.client.Index(name)
                self._index_name = name
        return self._index

    def _check_dimension(self, name: str) -> None:
        """
        Refuse an index built with a different embedding model.

        Changing EMBEDDING_MODEL changes the vector size, and querying the old
        index with the new model fails on every request with a Pinecone error
        that does not say why. This names the cause and the fix once, at open.
        """
        try:
            actual = self.client.describe_index(name).dimension
        except Exception:
            return  # unknown; a real mismatch still surfaces on the first query
        if actual and int(actual) != self.dimension:
            raise VectorStoreError(
                f"index {name} holds {actual}-dimensional vectors, but the configured "
                f"embedding model produces {self.dimension}. EMBEDDING_MODEL must match the "
                "model the index was built with: switch it back, or build a new index "
                "version with `edupilot-reindex --rebuild` and promote it."
            )

    def ensure_index(self, name: str, *, wait_seconds: int = 120) -> None:
        """Create `name` if absent and block until it reports ready."""
        existing = {i.name for i in self.client.list_indexes()}
        if name in existing:
            return

        from pinecone import ServerlessSpec

        logger.info(
            "creating Pinecone index %s (dim=%d, metric=%s)", name, self.dimension, self.metric
        )
        self.client.create_index(
            name=name,
            dimension=self.dimension,
            metric=self.metric,
            spec=ServerlessSpec(cloud=self.cloud, region=self.region),
        )

        deadline = time.time() + wait_seconds
        while time.time() < deadline:
            try:
                if self.client.describe_index(name).status.get("ready", False):
                    logger.info("index %s ready", name)
                    return
            except Exception:  # index not yet visible
                pass
            time.sleep(2)
        raise VectorStoreError(f"index {name} did not become ready in {wait_seconds}s")

    # -- writes ------------------------------------------------------------

    @staticmethod
    def _to_payload(record: VectorRecord) -> dict:
        payload: dict[str, Any] = {
            "id": record.id,
            "values": list(record.values),
            "metadata": sanitize_metadata(record.metadata),
        }
        if record.sparse:
            payload["sparse_values"] = {
                "indices": [int(i) for i in record.sparse],
                "values": [float(v) for v in record.sparse.values()],
            }
        return payload

    @staticmethod
    def _batches(payloads: list[dict]) -> Iterable[list[dict]]:
        """
        Split payloads into requests bounded by both count and byte size.

        The previous code used a flat batch of 100 regardless of payload size,
        which can breach Pinecone's 2 MB request cap once chunk text is
        carried in metadata.
        """
        batch: list[dict] = []
        size = 0
        for payload in payloads:
            encoded = len(json.dumps(payload, default=str).encode("utf-8"))
            if batch and (size + encoded > MAX_UPSERT_BYTES or len(batch) >= MAX_UPSERT_VECTORS):
                yield batch
                batch, size = [], 0
            batch.append(payload)
            size += encoded
        if batch:
            yield batch

    def upsert(self, records: Sequence[VectorRecord], namespace: str) -> int:
        if not records:
            return 0
        payloads = [self._to_payload(r) for r in records]
        written = 0
        for batch in self._batches(payloads):
            _retry(
                f"upsert({namespace}, n={len(batch)})",
                lambda b=batch: self.index.upsert(vectors=b, namespace=namespace),
            )
            written += len(batch)
        logger.debug("upserted %d vectors into %s/%s", written, self.index_name, namespace)
        return written

    def delete(
        self,
        namespace: str,
        *,
        ids: Sequence[str] | None = None,
        delete_all: bool = False,
        metadata_filter: dict | None = None,
    ) -> None:
        if delete_all:
            _retry(
                f"delete_all({namespace})",
                lambda: self.index.delete(delete_all=True, namespace=namespace),
            )
            return
        if metadata_filter:
            _retry(
                f"delete_by_filter({namespace})",
                lambda: self.index.delete(filter=metadata_filter, namespace=namespace),
            )
            return
        if ids:
            for i in range(0, len(ids), 500):
                batch = list(ids[i : i + 500])
                _retry(
                    f"delete_ids({namespace}, n={len(batch)})",
                    lambda b=batch: self.index.delete(ids=b, namespace=namespace),
                )

    # -- reads -------------------------------------------------------------

    def query(
        self,
        vector: Sequence[float],
        top_k: int,
        namespace: str,
        *,
        metadata_filter: dict | None = None,
        sparse: dict[int, float] | None = None,
        include_values: bool = False,
    ) -> list[SearchHit]:
        kwargs: dict[str, Any] = {
            "vector": list(vector),
            "top_k": max(1, top_k),
            "namespace": namespace,
            "include_metadata": True,
            "include_values": include_values,
        }
        if metadata_filter:
            kwargs["filter"] = metadata_filter
        if sparse:
            kwargs["sparse_vector"] = {
                "indices": [int(i) for i in sparse],
                "values": [float(v) for v in sparse.values()],
            }

        response = _retry(f"query({namespace}, k={top_k})", lambda: self.index.query(**kwargs))
        return [
            SearchHit(
                id=m.id,
                score=float(m.score),
                metadata=dict(m.metadata or {}),
                values=list(m.values) if include_values and getattr(m, "values", None) else None,
            )
            for m in response.matches
        ]

    def fetch(self, ids: Sequence[str], namespace: str) -> dict[str, SearchHit]:
        if not ids:
            return {}
        out: dict[str, SearchHit] = {}
        unique = list(dict.fromkeys(ids))
        for i in range(0, len(unique), 100):
            batch = unique[i : i + 100]
            response = _retry(
                f"fetch({namespace}, n={len(batch)})",
                lambda b=batch: self.index.fetch(ids=b, namespace=namespace),
            )
            for vid, vec in (response.vectors or {}).items():
                out[vid] = SearchHit(id=vid, score=1.0, metadata=dict(vec.metadata or {}))
        return out

    def stats(self) -> dict:
        raw = _retry("describe_index_stats", lambda: self.index.describe_index_stats())
        namespaces = {k: v.get("vector_count", 0) for k, v in (raw.get("namespaces") or {}).items()}
        return {
            "index": self.index_name,
            "dimension": raw.get("dimension"),
            "total_vectors": raw.get("total_vector_count", 0),
            "namespaces": namespaces,
        }


# ---------------------------------------------------------------------------
# Local in-memory implementation (tests, CI, offline development)
# ---------------------------------------------------------------------------


class InMemoryVectorStore(VectorStore):
    """
    Dependency-free VectorStore for tests.

    Implements the same filter semantics as Pinecone for the operators the
    project uses ($eq, $ne, $in, $nin, $gte, $lte, $gt, $lt), so tests exercise
    real filter behaviour without a network call.
    """

    def __init__(self) -> None:
        self._data: dict[str, dict[str, VectorRecord]] = {}

    def upsert(self, records: Sequence[VectorRecord], namespace: str) -> int:
        bucket = self._data.setdefault(namespace, {})
        for r in records:
            bucket[r.id] = r
        return len(records)

    @staticmethod
    def _matches(meta: dict, flt: dict | None) -> bool:
        if not flt:
            return True
        for key, condition in flt.items():
            if key == "$and":
                if not all(InMemoryVectorStore._matches(meta, c) for c in condition):
                    return False
                continue
            if key == "$or":
                if not any(InMemoryVectorStore._matches(meta, c) for c in condition):
                    return False
                continue

            value = meta.get(key)
            if not isinstance(condition, dict):
                if value != condition:
                    return False
                continue
            for op, operand in condition.items():
                if op == "$eq" and value != operand:
                    return False
                if op == "$ne" and value == operand:
                    return False
                if op == "$in" and value not in operand:
                    return False
                if op == "$nin" and value in operand:
                    return False
                if op == "$gte" and not (value is not None and value >= operand):
                    return False
                if op == "$lte" and not (value is not None and value <= operand):
                    return False
                if op == "$gt" and not (value is not None and value > operand):
                    return False
                if op == "$lt" and not (value is not None and value < operand):
                    return False
        return True

    def query(
        self,
        vector: Sequence[float],
        top_k: int,
        namespace: str,
        *,
        metadata_filter: dict | None = None,
        sparse: dict[int, float] | None = None,
        include_values: bool = False,
    ) -> list[SearchHit]:
        import numpy as np

        bucket = self._data.get(namespace, {})
        q = np.asarray(vector, dtype=np.float32)
        scored: list[SearchHit] = []
        for record in bucket.values():
            if not self._matches(record.metadata, metadata_filter):
                continue
            score = float(np.dot(q, np.asarray(record.values, dtype=np.float32)))
            if sparse and record.sparse:
                score += sum(w * record.sparse.get(i, 0.0) for i, w in sparse.items())
            scored.append(
                SearchHit(
                    id=record.id,
                    score=score,
                    metadata=dict(record.metadata),
                    values=list(record.values) if include_values else None,
                )
            )
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:top_k]

    def fetch(self, ids: Sequence[str], namespace: str) -> dict[str, SearchHit]:
        bucket = self._data.get(namespace, {})
        return {
            i: SearchHit(id=i, score=1.0, metadata=dict(bucket[i].metadata))
            for i in ids
            if i in bucket
        }

    def delete(
        self,
        namespace: str,
        *,
        ids: Sequence[str] | None = None,
        delete_all: bool = False,
        metadata_filter: dict | None = None,
    ) -> None:
        if delete_all:
            self._data.pop(namespace, None)
            return
        bucket = self._data.get(namespace, {})
        if metadata_filter:
            for key in [k for k, r in bucket.items() if self._matches(r.metadata, metadata_filter)]:
                bucket.pop(key, None)
        for i in ids or []:
            bucket.pop(i, None)

    def document_titles(self, namespace: str) -> dict[str, str]:
        return {
            r.metadata["filename"]: r.metadata["doc_title"]
            for r in self._data.get(namespace, {}).values()
            if r.metadata.get("filename") and r.metadata.get("doc_title")
        }

    def stats(self) -> dict:
        return {
            "index": "in-memory",
            "total_vectors": sum(len(b) for b in self._data.values()),
            "namespaces": {k: len(v) for k, v in self._data.items()},
        }
