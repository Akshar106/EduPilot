"""
Embedding layer
===============
Pluggable embedders with an explicit token budget, an asymmetric query
instruction, and a persistent cache.

Why this replaces the previous inline `SentenceTransformer("all-MiniLM-L6-v2")`:

  * **Truncation was silent.** MiniLM's window is 256 tokens; chunks were
    ~1,000. `sentence-transformers` truncates without warning, so 47.9% of the
    indexed corpus never reached the model. `TokenBudgetError` now makes that
    a loud failure instead of invisible data loss.
  * **BGE is asymmetric.** bge-*-en-v1.5 expects an instruction prefix on the
    *query* side only. Embedding queries and passages identically — as the old
    code did — leaves measurable retrieval quality on the table.
  * **No caching.** Every re-index re-encoded every chunk. Embeddings are
    keyed by (model, content hash), so re-ingesting an unchanged document is
    nearly free.

bge-small-en-v1.5 is 384-dimensional, matching MiniLM.

Two backends share one interface: `SentenceTransformerEmbedder` runs a model
in-process (the local-models extra), and `PineconeEmbedder` calls a hosted
Pinecone Inference model, which needs no torch and no weights. `build_embedder`
picks one from the model's registry entry.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Iterable, Sequence
from typing import Literal

import numpy as np
from sqlalchemy import Engine, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from edupilot.db.tables import embedding_cache as cache_table

logger = logging.getLogger(__name__)


class TokenBudgetError(ValueError):
    """Raised when text exceeds the embedding model's window.

    Silent truncation is the failure this whole module exists to prevent, so
    over-long input is an error rather than something quietly clipped.
    """


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------


class EmbeddingModelSpec:
    """
    Static facts about an embedding model that callers need before loading it.

    `provider` is "local" (sentence-transformers, in-process) or "pinecone"
    (Pinecone Inference). `tokenizer` is the HuggingFace tokenizer used to
    *count* tokens for budgets and chunk sizing; it defaults to the model's
    own, and a hosted model names a local stand-in.
    """

    __slots__ = (
        "name",
        "dimension",
        "max_tokens",
        "query_prefix",
        "document_prefix",
        "provider",
        "tokenizer",
    )

    def __init__(
        self,
        name: str,
        dimension: int,
        max_tokens: int,
        query_prefix: str = "",
        document_prefix: str = "",
        provider: str = "local",
        tokenizer: str | None = None,
    ) -> None:
        self.name = name
        self.dimension = dimension
        self.max_tokens = max_tokens
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.provider = provider
        self.tokenizer = tokenizer or name


#: Models this project has validated. `query_prefix` follows each model card.
MODEL_REGISTRY: dict[str, EmbeddingModelSpec] = {
    "BAAI/bge-small-en-v1.5": EmbeddingModelSpec(
        name="BAAI/bge-small-en-v1.5",
        dimension=384,
        max_tokens=512,
        query_prefix="Represent this sentence for searching relevant passages: ",
    ),
    "BAAI/bge-base-en-v1.5": EmbeddingModelSpec(
        name="BAAI/bge-base-en-v1.5",
        dimension=768,
        max_tokens=512,
        query_prefix="Represent this sentence for searching relevant passages: ",
    ),
    # Retained so an existing index can still be read during a migration.
    "all-MiniLM-L6-v2": EmbeddingModelSpec(
        name="all-MiniLM-L6-v2",
        dimension=384,
        max_tokens=256,
    ),
    # Hosted on Pinecone Inference. Asymmetric through `input_type` rather than
    # a text prefix. Its window is 2,048 tokens; budgets are counted with the
    # bge tokenizer, whose WordPiece vocabulary splits English into more
    # tokens than Llama's, so a count under the limit is under it for the
    # model too. Chunks stay sized for bge (CHUNK_MAX_TOKENS), which keeps an
    # A/B comparison against the local model about the embedder alone.
    "llama-text-embed-v2": EmbeddingModelSpec(
        name="llama-text-embed-v2",
        dimension=1024,
        max_tokens=2048,
        provider="pinecone",
        tokenizer="BAAI/bge-small-en-v1.5",
    ),
}


def get_model_spec(name: str) -> EmbeddingModelSpec:
    """Look up a model's spec, falling back to conservative defaults."""
    if name in MODEL_REGISTRY:
        return MODEL_REGISTRY[name]
    logger.warning("Unregistered embedding model %r — assuming 384d/512tok", name)
    return EmbeddingModelSpec(name=name, dimension=384, max_tokens=512)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


class EmbeddingCache:
    """
    Embedding cache in Postgres, keyed by (model, text).

    Vectors are stored as raw float32 bytes. Shared by every process using the
    database, so a document embedded once by the CLI is a cache hit for the
    API, and the reverse.

    Args:
        engine: The SQLAlchemy engine. Defaults to the app's shared one.
    """

    def __init__(self, engine: Engine | None = None) -> None:
        self._engine = engine

    @property
    def engine(self) -> Engine:
        if self._engine is None:
            from edupilot.db import get_engine

            self._engine = get_engine()
        return self._engine

    @staticmethod
    def make_key(model: str, text: str) -> str:
        return hashlib.sha256(f"{model}\x00{text}".encode()).hexdigest()

    def get_many(self, keys: Sequence[str]) -> dict[str, np.ndarray]:
        if not keys:
            return {}
        out: dict[str, np.ndarray] = {}
        with self.engine.connect() as conn:
            for i in range(0, len(keys), 1000):
                rows = conn.execute(
                    select(cache_table.c.key, cache_table.c.dim, cache_table.c.vec).where(
                        cache_table.c.key.in_(list(keys[i : i + 1000]))
                    )
                )
                for key, dim, blob in rows:
                    out[key] = np.frombuffer(blob, dtype=np.float32, count=dim)
        return out

    def put_many(self, items: Iterable[tuple[str, str, np.ndarray]]) -> None:
        rows = [
            {"key": key, "model": model, "dim": int(vec.shape[0]),
             "vec": vec.astype(np.float32).tobytes()}
            for key, model, vec in items
        ]
        if not rows:
            return
        statement = pg_insert(cache_table)
        statement = statement.on_conflict_do_update(
            index_elements=["key"],
            set_={"model": statement.excluded.model, "dim": statement.excluded.dim,
                  "vec": statement.excluded.vec},
        )
        with self.engine.begin() as conn:
            for i in range(0, len(rows), 500):
                conn.execute(statement, rows[i : i + 500])

    def size(self) -> int:
        with self.engine.connect() as conn:
            return conn.execute(select(func.count()).select_from(cache_table)).scalar_one()


# ---------------------------------------------------------------------------
# Embedder interface
# ---------------------------------------------------------------------------


class Embedder(ABC):
    """
    Interface every embedding backend implements.

    The base class owns what must behave identically across backends: the
    token-budget guard, the document cache, and the query/passage asymmetry.
    A backend supplies tokenization and the encoding call itself.
    """

    spec: EmbeddingModelSpec
    cache: EmbeddingCache | None = None
    strict: bool = True

    @property
    def dimension(self) -> int:
        return self.spec.dimension

    @property
    def model_name(self) -> str:
        return self.spec.name

    @property
    def max_tokens(self) -> int:
        return self.spec.max_tokens

    @property
    def tokenizer_name(self) -> str:
        """HuggingFace tokenizer whose counts this embedder's budgets use."""
        return self.spec.tokenizer

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Token count under the tokenizer budgets are measured with."""

    @abstractmethod
    def _encode(
        self, texts: Sequence[str], batch_size: int, *, kind: Literal["passage", "query"]
    ) -> np.ndarray:
        """Encode already-prefixed texts. Returns (n, dim) L2-normalized float32."""

    # -- budget enforcement ------------------------------------------------

    def _check_budget(
        self,
        texts: Sequence[str],
        label: str,
        counts: Sequence[int] | None = None,
    ) -> None:
        """
        Reject input that would be silently truncated.

        Checked before encoding so the failure names the offending text
        instead of surfacing later as unexplained retrieval misses.

        Args:
            counts: Token counts the caller already computed. The chunker
                measures every chunk to size it, so re-tokenizing here doubles
                the tokenization work for no new information.
        """
        offenders: list[tuple[int, int]] = []
        for i, t in enumerate(texts):
            n = counts[i] if counts is not None else self.count_tokens(t)
            if n > self.spec.max_tokens:
                offenders.append((i, n))

        if not offenders:
            return

        preview = ", ".join(f"#{i}={n}tok" for i, n in offenders[:5])
        message = (
            f"{len(offenders)}/{len(texts)} {label} exceed the {self.spec.max_tokens}-token "
            f"window of {self.spec.name} ({preview}). They would be silently truncated."
        )
        if self.strict:
            raise TokenBudgetError(message)
        logger.warning("%s — truncating (strict=False)", message)

    # -- public API --------------------------------------------------------

    def embed_documents(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 64,
        token_counts: Sequence[int] | None = None,
    ) -> np.ndarray:
        """
        Embed passages for indexing. Returns (n, dim) L2-normalized float32.

        Args:
            token_counts: Precomputed token counts, one per text. Only trusted
                when this model has no document prefix — otherwise the prefix
                would make the caller's counts wrong, and a wrong count here
                reintroduces exactly the silent truncation this guard exists
                to prevent.
        """
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)

        prefixed = [self.spec.document_prefix + t for t in texts]
        usable = (
            token_counts
            if token_counts is not None
            and not self.spec.document_prefix
            and len(token_counts) == len(texts)
            else None
        )
        self._check_budget(prefixed, "documents", counts=usable)

        if self.cache is None:
            return self._encode(prefixed, batch_size, kind="passage")

        out = np.zeros((len(prefixed), self.dimension), dtype=np.float32)
        keys = [EmbeddingCache.make_key(self.spec.name, t) for t in prefixed]
        hits = self.cache.get_many(keys)
        misses = [i for i, k in enumerate(keys) if k not in hits]

        for i, k in enumerate(keys):
            if k in hits:
                out[i] = hits[k]

        if misses:
            fresh = self._encode([prefixed[i] for i in misses], batch_size, kind="passage")
            for slot, i in enumerate(misses):
                out[i] = fresh[slot]
            self.cache.put_many(
                (keys[i], self.spec.name, out[i]) for i in misses
            )

        logger.debug(
            "embedded %d documents (%d cached, %d computed)",
            len(texts), len(texts) - len(misses), len(misses),
        )
        return out

    def embed_query(self, text: str) -> np.ndarray:
        """
        Embed a query, applying the model's asymmetric query handling.

        bge-*-en-v1.5 is trained with an instruction on the query side only;
        omitting it measurably degrades retrieval. Queries are short, so the
        budget check is advisory rather than strict.
        """
        prefixed = self.spec.query_prefix + text
        if self.count_tokens(prefixed) > self.spec.max_tokens:
            logger.warning("query exceeds %d tokens; truncating", self.spec.max_tokens)
        return self._encode([prefixed], 1, kind="query")[0]


class SentenceTransformerEmbedder(Embedder):
    """
    `sentence-transformers` backend, run in-process.

    Args:
        model_name: HuggingFace model id.
        cache: Optional embedding cache. Documents are cached; queries are not
            (they are one-off and cheap).
        strict: When True, over-budget text raises `TokenBudgetError`. When
            False it is truncated with a warning — used only for reading an
            index built by an older, smaller-window model.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        *,
        cache: EmbeddingCache | None = None,
        strict: bool = True,
        device: str | None = None,
    ) -> None:
        self.spec = get_model_spec(model_name)
        self.cache = cache
        self.strict = strict
        self._device = device
        self._model = None
        self._lock = threading.Lock()

    def _get_model(self):
        """Load the model once, under a lock so concurrent requests can't race."""
        if self._model is None:
            with self._lock:
                if self._model is None:
                    try:
                        from sentence_transformers import SentenceTransformer
                    except ImportError as exc:
                        raise RuntimeError(
                            f"{self.spec.name} runs locally and needs the local-models extra "
                            "(`uv sync --extra local-models`), or set EMBEDDING_MODEL to a "
                            "hosted model such as llama-text-embed-v2"
                        ) from exc

                    logger.info("loading embedding model %s", self.spec.name)
                    from .inference import configure_threads

                    configure_threads()
                    model = SentenceTransformer(self.spec.name, device=self._device)
                    # Make the model's own window match what we validate against.
                    model.max_seq_length = self.spec.max_tokens
                    self._model = model
        return self._model

    def count_tokens(self, text: str) -> int:
        tok = self._get_model().tokenizer
        return len(tok.encode(text, add_special_tokens=True, truncation=False))

    def _encode(
        self, texts: Sequence[str], batch_size: int, *, kind: Literal["passage", "query"]
    ) -> np.ndarray:
        # `kind` is carried by the text prefix for these models.
        vecs = self._get_model().encode(
            list(texts),
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(vecs, dtype=np.float32)


class PineconeEmbedder(Embedder):
    """
    Pinecone Inference backend: a hosted model, so no torch and no weights.

    Over-length input is rejected server-side as well as by the budget check
    (`truncate="NONE"`), keeping the no-silent-truncation rule. Recent query
    vectors are kept briefly, because one turn embeds the same question for
    the answer cache and for each retrieval probe — each a network round trip
    to a hosted model.

    Args:
        api_key: Pinecone API key, the same one the vector store uses.
        client: An existing `Pinecone` client to reuse (tests pass a fake).
    """

    #: Inputs per request; the hosted embedding models accept up to 96.
    MAX_BATCH = 96
    #: Distinct recent queries kept in memory.
    QUERY_CACHE_SIZE = 256

    def __init__(
        self,
        model_name: str = "llama-text-embed-v2",
        *,
        api_key: str = "",
        cache: EmbeddingCache | None = None,
        strict: bool = True,
        client=None,
    ) -> None:
        self.spec = get_model_spec(model_name)
        self.cache = cache
        self.strict = strict
        self._api_key = api_key
        self._client = client
        self._lock = threading.Lock()
        self._recent: OrderedDict[str, np.ndarray] = OrderedDict()

    def _inference(self):
        if self._client is None:
            with self._lock:
                if self._client is None:
                    if not self._api_key:
                        raise RuntimeError(f"{self.spec.name} is hosted and needs PINECONE_API_KEY")
                    from pinecone import Pinecone

                    self._client = Pinecone(api_key=self._api_key)
        return self._client.inference

    def count_tokens(self, text: str) -> int:
        from edupilot.ingestion.chunking import get_token_counter

        return get_token_counter(self.spec.tokenizer)(text)

    def _encode(
        self, texts: Sequence[str], batch_size: int, *, kind: Literal["passage", "query"]
    ) -> np.ndarray:
        out = np.zeros((len(texts), self.dimension), dtype=np.float32)
        step = min(max(batch_size, 1), self.MAX_BATCH)
        for start in range(0, len(texts), step):
            batch = list(texts[start : start + step])
            response = self._inference().embed(
                model=self.spec.name,
                inputs=batch,
                parameters={
                    "input_type": kind,
                    "truncate": "NONE",
                    "dimension": self.dimension,
                },
            )
            vecs = np.asarray([e.values for e in response.data], dtype=np.float32)
            if vecs.shape != (len(batch), self.dimension):
                raise ValueError(
                    f"{self.spec.name} returned vectors of shape {vecs.shape}, "
                    f"expected {(len(batch), self.dimension)}"
                )
            # Normalized here rather than trusted: the dotproduct index metric
            # equals cosine only for unit vectors.
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            out[start : start + len(batch)] = vecs / np.maximum(norms, 1e-12)
        return out

    def embed_query(self, text: str) -> np.ndarray:
        with self._lock:
            cached = self._recent.get(text)
            if cached is not None:
                self._recent.move_to_end(text)
                return cached
        vec = super().embed_query(text)
        with self._lock:
            self._recent[text] = vec
            if len(self._recent) > self.QUERY_CACHE_SIZE:
                self._recent.popitem(last=False)
        return vec


def build_embedder(
    model_name: str,
    *,
    cache: EmbeddingCache | None = None,
    strict: bool = True,
    api_key: str | None = None,
) -> Embedder:
    """The backend that serves `model_name`, per its registry entry."""
    spec = get_model_spec(model_name)
    if spec.provider == "pinecone":
        if api_key is None:
            from edupilot.core.config import PINECONE_API_KEY

            api_key = PINECONE_API_KEY
        return PineconeEmbedder(model_name, api_key=api_key, cache=cache, strict=strict)
    return SentenceTransformerEmbedder(model_name, cache=cache, strict=strict)


# ---------------------------------------------------------------------------
# Process-wide singleton
# ---------------------------------------------------------------------------

_embedder: Embedder | None = None
_embedder_lock = threading.Lock()


def get_embedder(model_name: str | None = None, *, strict: bool = True) -> Embedder:
    """
    Return the shared embedder, constructing it on first use.

    Double-checked locking: the previous code initialized module globals with
    no synchronization while a 4-worker thread pool called into them, which
    could load the model several times concurrently.
    """
    global _embedder
    if _embedder is not None:
        return _embedder

    with _embedder_lock:
        if _embedder is None:
            from edupilot.core.config import EMBEDDING_MODEL

            _embedder = build_embedder(
                model_name or EMBEDDING_MODEL, cache=EmbeddingCache(), strict=strict
            )
    return _embedder


def reset_embedder() -> None:
    """Drop the singleton. Test-support only."""
    global _embedder
    with _embedder_lock:
        _embedder = None
