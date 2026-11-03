"""
EduPilot retrieval
==================
Embeddings, vector storage, sparse encoding, query transformation, hybrid
search, and reranking.

    from edupilot.retrieval import HybridRetriever, RetrievalConfig, get_embedder

    result = retriever.retrieve("what is a p-value", config=RetrievalConfig())
    if result.is_empty:
        ...  # no supporting evidence — refuse rather than answer

`RetrievalResult.chunks` being empty is a meaningful outcome, not an error: it
means nothing in the knowledge base cleared the relevance floor.
"""

from .answer_cache import AnswerCache, CacheHit, normalize_question
from .embeddings import (
    Embedder,
    EmbeddingCache,
    PineconeEmbedder,
    SentenceTransformerEmbedder,
    TokenBudgetError,
    build_embedder,
    get_embedder,
    get_model_spec,
    reset_embedder,
)
from .hybrid import HybridRetriever, RetrievalConfig, RetrievalResult
from .indexer import Indexer, IndexRegistry, IndexResult, course_files, source_type_of
from .lc_retriever import EduPilotRetriever, RetrievalOutcome, to_document
from .pgvector import PgVectorStore
from .query_transform import TransformedQuery, expand_acronyms, transform_query
from .rerank import RerankedChunk, rerank, score_summary
from .sparse import BM25Encoder
from .vectorstore import (
    InMemoryVectorStore,
    PineconeVectorStore,
    SearchHit,
    VectorRecord,
    VectorStore,
    VectorStoreError,
)
from .versions import IndexPointer

__all__ = [
    "AnswerCache",
    "CacheHit",
    "normalize_question",
    "BM25Encoder",
    "Embedder",
    "EmbeddingCache",
    "EduPilotRetriever",
    "HybridRetriever",
    "IndexPointer",
    "IndexRegistry",
    "IndexResult",
    "Indexer",
    "InMemoryVectorStore",
    "PineconeEmbedder",
    "PgVectorStore",
    "PineconeVectorStore",
    "RerankedChunk",
    "RetrievalConfig",
    "RetrievalOutcome",
    "RetrievalResult",
    "SearchHit",
    "SentenceTransformerEmbedder",
    "TokenBudgetError",
    "TransformedQuery",
    "VectorRecord",
    "VectorStore",
    "VectorStoreError",
    "build_embedder",
    "course_files",
    "expand_acronyms",
    "get_embedder",
    "get_model_spec",
    "rerank",
    "reset_embedder",
    "score_summary",
    "source_type_of",
    "to_document",
    "transform_query",
]
