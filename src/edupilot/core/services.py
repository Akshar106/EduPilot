"""
Composition root
================
Builds and holds the singletons: embedder, vector store, sparse encoder,
retrievers, indexer, agent graph, and user store.

Kept out of `edupilot.api` so the HTTP layer contains routing and nothing
else, and so tests can assemble the same object graph against
`InMemoryVectorStore` without starting a server.

Everything stateful lives in Postgres, including which index version is live
and its BM25 table. Each process checks the live version (cached for a few
seconds) and reloads what depends on it when it changes, so a promotion made
by the reindex CLI or another instance reaches every process without a
restart.

Every singleton is created under a lock. The previous code initialized module
globals (`_pipeline`, `_retrievers`, `_embedding_model`, `_pinecone_index`)
with no synchronization while a four-worker thread pool called into them, so
two concurrent first-requests could each start loading the same model.
"""

from __future__ import annotations

import logging
import threading

from edupilot.core.config import (
    DOMAINS,
    EMBEDDING_MODEL,
    INDEX_BASE,
    LLM_SIMULATED,
    PINECONE_API_KEY,
    PINECONE_CLOUD,
    PINECONE_METRIC,
    PINECONE_REGION,
    VECTOR_BACKEND,
)
from edupilot.ingestion.chunking import ChunkingConfig
from edupilot.retrieval import (
    BM25Encoder,
    Embedder,
    EmbeddingCache,
    HybridRetriever,
    Indexer,
    IndexPointer,
    IndexRegistry,
    InMemoryVectorStore,
    PgVectorStore,
    PineconeVectorStore,
    VectorStore,
    build_embedder,
)
from edupilot.security import UserStore

#: How long a process may serve the live index version from memory before
#: asking the database again — the delay before a promotion reaches it.
POINTER_CACHE_SECONDS = 5.0

logger = logging.getLogger(__name__)


class Services:
    """
    Lazily-constructed application object graph.

    Nothing is built at import time — model loads and network calls happen on
    first use, so `import services` stays cheap and tests can substitute
    components before anything is instantiated.
    """

    def __init__(self, *, use_in_memory_store: bool = False) -> None:
        self._use_in_memory = use_in_memory_store
        # RLock, not Lock. The properties compose: `store` holds the lock and
        # then reads `embedder.dimension`, `indexer` reads `store`, `embedder`,
        # `registry` and `sparse_encoder`, and `_retriever_for` reads three of
        # them. With a plain Lock the first such access deadlocks the process
        # on itself, because a thread cannot re-acquire a non-reentrant lock it
        # already holds.
        self._lock = threading.RLock()

        self._embedder: Embedder | None = None
        self._store: VectorStore | None = None
        self._pointer: IndexPointer | None = None
        self._sparse: BM25Encoder | None = None
        self._registry: IndexRegistry | None = None
        self._indexer: Indexer | None = None
        self._users: UserStore | None = None
        self._agent = None
        self._answer_cache = None
        self._retrievers: dict[str, HybridRetriever] = {}
        # The index version the sparse encoder, retrievers and answer cache
        # were built for. False means "not loaded yet" (None is "no index").
        self._loaded_for: str | None | bool = False

    # ------------------------------------------------------------------
    # Core components
    # ------------------------------------------------------------------

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            with self._lock:
                if self._embedder is None:
                    # Local or hosted, per the model's registry entry.
                    self._embedder = build_embedder(
                        EMBEDDING_MODEL,
                        cache=EmbeddingCache(),
                        strict=True,
                        api_key=PINECONE_API_KEY,
                    )
        return self._embedder

    @property
    def index_pointer(self) -> IndexPointer:
        if self._pointer is None:
            with self._lock:
                if self._pointer is None:
                    self._pointer = IndexPointer(
                        INDEX_BASE, backend=VECTOR_BACKEND, cache_seconds=POINTER_CACHE_SECONDS
                    )
        return self._pointer

    @property
    def store(self) -> VectorStore:
        if self._store is None:
            with self._lock:
                if self._store is None:
                    self._store = self._build_store()
        return self._store

    def _build_store(self, index_name: str | None = None) -> VectorStore:
        if self._use_in_memory:
            logger.warning("using in-memory vector store — data is not persisted")
            return InMemoryVectorStore()
        pointer = None if index_name else self.index_pointer
        if VECTOR_BACKEND == "pinecone":
            return PineconeVectorStore(
                api_key=PINECONE_API_KEY,
                index_name=index_name,
                pointer=pointer,
                dimension=self.embedder.dimension,
                metric=PINECONE_METRIC,
                cloud=PINECONE_CLOUD,
                region=PINECONE_REGION,
            )
        if VECTOR_BACKEND != "pgvector":
            raise ValueError(f"VECTOR_BACKEND must be pgvector or pinecone, not {VECTOR_BACKEND!r}")
        return PgVectorStore(
            index_name=index_name, pointer=pointer, dimension=self.embedder.dimension
        )

    def store_for_index(self, index_name: str) -> VectorStore:
        """A store bound to a specific index version. Used during a rebuild."""
        return self._build_store(index_name)

    @property
    def sparse_encoder(self) -> BM25Encoder | None:
        """
        The live index version's fitted BM25 encoder, or None when there is none.

        Absence is not an error: without it, hybrid search degrades to
        dense-only, which still works. It is logged once per version rather
        than on every query.
        """
        self._sync_index_version()
        return self._sparse

    def _sync_index_version(self) -> None:
        """
        Rebuild what depends on the live index version if it has changed.

        The sparse encoder belongs to the version that produced the stored
        sparse vectors; retrievers hold the encoder; answer-cache entries are
        bound to the version that answered them. All three are dropped
        together when the live version moves.
        """
        if self._use_in_memory:
            if self._loaded_for is False:
                self._loaded_for = None
            return
        active = self.index_pointer.active
        if active == self._loaded_for:
            return
        with self._lock:
            if active == self._loaded_for:
                return
            self._sparse = BM25Encoder.from_dict(self.index_pointer.sparse_encoder(active)) if active else None
            if active and self._sparse is None:
                logger.warning(
                    "index %s has no fitted BM25 encoder — lexical matching is disabled "
                    "until you run `edupilot-reindex --rebuild`",
                    active,
                )
            if self._loaded_for is not False:
                # Only a change drops anything: on the first load there is
                # nothing stale yet.
                logger.info("live index changed %s -> %s", self._loaded_for, active)
                self._retrievers.clear()
                self._indexer = None
                # Rebound on next access, which makes every entry from the
                # previous corpus unservable.
                self._answer_cache = None
            self._loaded_for = active

    def reload_sparse_encoder(self) -> None:
        """Re-read the live version now, without waiting out the pointer cache."""
        if self._pointer is not None:
            self._pointer.refresh()
        with self._lock:
            self._retrievers.clear()
            self._indexer = None
            self._answer_cache = None
            self._loaded_for = False
        self._sync_index_version()

    @property
    def registry(self) -> IndexRegistry:
        if self._registry is None:
            with self._lock:
                if self._registry is None:
                    self._registry = IndexRegistry()
        return self._registry

    @property
    def chunking(self) -> ChunkingConfig:
        return ChunkingConfig(
            model_name=self.embedder.tokenizer_name,
            model_window=self.embedder.max_tokens,
            max_tokens=min(448, self.embedder.max_tokens - 64),
        )

    @property
    def indexer(self) -> Indexer:
        self._sync_index_version()
        if self._indexer is None:
            with self._lock:
                if self._indexer is None:
                    self._indexer = Indexer(
                        self.store,
                        self.embedder,
                        self.registry,
                        sparse_encoder=self.sparse_encoder,
                        chunking=self.chunking,
                    )
        return self._indexer

    @property
    def answer_cache(self):
        """
        Frequency-gated answer cache, bound to the active index version.

        Built lazily and rebound on `reload_sparse_encoder`, so a promoted
        rebuild cannot leave entries pointing at a retired corpus.
        """
        self._sync_index_version()
        if self._answer_cache is None:
            with self._lock:
                if self._answer_cache is None:
                    from edupilot.retrieval.answer_cache import AnswerCache

                    self._answer_cache = AnswerCache(
                        self.embedder, index_version=self.index_pointer.active or ""
                    )
                    removed = self._answer_cache.sweep()
                    if removed:
                        logger.info("answer cache: swept %d stale entries", removed)
        return self._answer_cache

    @property
    def users(self) -> UserStore:
        if self._users is None:
            with self._lock:
                if self._users is None:
                    self._users = UserStore()
        return self._users

    # ------------------------------------------------------------------
    # Retrievers
    # ------------------------------------------------------------------

    def retriever(self, domain: str) -> HybridRetriever:
        """Retriever for a course domain, cached per namespace."""
        namespace = DOMAINS[domain]["namespace"]
        return self._retriever_for(namespace)

    def study_retriever(self, ss_session_id: str) -> HybridRetriever:
        """
        Retriever for one study session's uploads.

        The namespace derives from the session id, so one student's uploads
        are physically separated from another's rather than filtered apart at
        query time.
        """
        return self._retriever_for(f"ss_{ss_session_id.replace('-', '')}")

    def _retriever_for(self, namespace: str) -> HybridRetriever:
        self._sync_index_version()
        cached = self._retrievers.get(namespace)
        if cached is not None:
            return cached
        with self._lock:
            if namespace not in self._retrievers:
                self._retrievers[namespace] = HybridRetriever(
                    self.store,
                    self.embedder,
                    namespace,
                    sparse_encoder=self.sparse_encoder,
                )
            return self._retrievers[namespace]

    # ------------------------------------------------------------------
    # Agent
    # ------------------------------------------------------------------

    @property
    def agent(self):
        """
        The LangGraph agent. Its checkpointer is attached by the app's
        lifespan, which owns the async connection it needs.
        """
        if self._agent is None:
            with self._lock:
                if self._agent is None:
                    from edupilot.agents import EduPilotAgent

                    if LLM_SIMULATED:
                        from edupilot.llm.simulated import simulated_llm

                        logger.warning("LLM_SIMULATED is on — every model call is simulated")
                        self._agent = EduPilotAgent(
                            DOMAINS, self._agent_retriever, models=simulated_llm
                        )
                    else:
                        self._agent = EduPilotAgent(DOMAINS, self._agent_retriever)
        return self._agent

    def _agent_retriever(self, key: str) -> HybridRetriever:
        """Domain codes resolve to course namespaces; anything else is a study session."""
        return self.retriever(key) if key in DOMAINS else self.study_retriever(key)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def health(self) -> dict:
        """
        Report readiness of every dependency this app actually uses.

        The previous check validated GEMINI_API_KEY and PINECONE_API_KEY but
        not GROQ_API_KEY — and the default model is Groq, so it reported "ok"
        on a deployment where every chat request would fail.
        """
        import os

        from edupilot import db
        from edupilot.core.config import (
            AVAILABLE_MODELS,
            DEFAULT_MODEL,
            GROQ_MODELS,
            RERANKER_MODEL,
        )
        from edupilot.retrieval.embeddings import get_model_spec
        from edupilot.retrieval.rerank import HOSTED_RERANKERS

        checks: dict[str, dict] = {}

        needs_groq = DEFAULT_MODEL in GROQ_MODELS or any(m in GROQ_MODELS for m in AVAILABLE_MODELS)
        needs_gemini = any(m.startswith("gemini") for m in AVAILABLE_MODELS)

        checks["groq_api_key"] = {
            "required": needs_groq and not LLM_SIMULATED,
            "present": bool(os.getenv("GROQ_API_KEY")),
        }
        checks["gemini_api_key"] = {
            "required": needs_gemini and not LLM_SIMULATED,
            "present": bool(os.getenv("GEMINI_API_KEY")),
        }
        # Pinecone serves the vectors only with that backend, but also hosts
        # the optional hosted embedding and reranking models.
        uses_pinecone = (
            VECTOR_BACKEND == "pinecone"
            or get_model_spec(EMBEDDING_MODEL).provider == "pinecone"
            or RERANKER_MODEL in HOSTED_RERANKERS
        )
        checks["pinecone_api_key"] = {
            "required": uses_pinecone and not self._use_in_memory,
            "present": bool(PINECONE_API_KEY),
        }
        checks["jwt_secret"] = {
            "required": True,
            "present": bool(os.getenv("JWT_SECRET_KEY")),
        }
        database = db.ping()
        checks["database"] = {"required": True, "present": database}
        active = self.index_pointer.active if database else None
        checks["vector_index"] = {
            "required": True,
            "present": bool(active) or self._use_in_memory,
        }
        checks["sparse_encoder"] = {
            "required": False,
            "present": self.sparse_encoder is not None if active else False,
        }

        # A missing required dependency is a hard failure, not a warning.
        failing = [k for k, v in checks.items() if v["required"] and not v["present"]]
        return {
            "status": "ok" if not failing else "degraded",
            "failing": failing,
            "checks": checks,
            "domains": list(DOMAINS),
            "embedding_model": EMBEDDING_MODEL,
            "vector_backend": VECTOR_BACKEND,
            "active_index": active,
            "llm": "simulated" if LLM_SIMULATED else "live",
        }


#: Application-wide instance.
services = Services()
