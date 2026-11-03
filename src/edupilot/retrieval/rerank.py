"""
Reranking
=========
Cross-encoder reranking of retrieval candidates.

Three problems with the previous reranker:

  * **It was off on the main path.** `rerank()` defaulted to `mode="keyword"`
    and the Chat pipeline never passed a mode, so every course question was
    reranked by Jaccard token overlap. The cross-encoder ran only in Self
    Study. The "advanced reranking" in the architecture diagram was not
    actually in the request path.
  * **Scores were not comparable.** Keyword mode produced 0-1 Jaccard values
    while cross-encoder mode produced raw logits in roughly [-11, +11]. One
    `confidence_threshold` setting was applied to both, which is why the code
    had to pass `-5.0` in one place and `0.20` in another for the same
    parameter. Cross-encoder scores are now squashed to 0-1 with a sigmoid,
    so a single threshold means the same thing everywhere.
  * **The empty-result path was inconsistent.** Keyword mode kept exactly one
    chunk when nothing cleared the threshold; cross-encoder mode kept `top_k`.
    Both now return nothing, and the caller decides — silently substituting
    low-relevance context is what produces confident answers off bad evidence.

The reranker is RERANKER_MODEL: a local cross-encoder (the local-models
extra), or a model hosted on Pinecone Inference (`HOSTED_RERANKERS`), which
needs no torch. Either way scores land on the same 0-1 scale and go through
the same floors.
"""

from __future__ import annotations

import logging
import math
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)

RerankMode = Literal["cross_encoder", "keyword", "none"]

#: Preferred cross-encoder, then fallbacks. bge-reranker-base is markedly
#: stronger on technical text than the ms-marco MiniLM it replaces.
CROSS_ENCODER_CANDIDATES = (
    "BAAI/bge-reranker-base",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
)

#: Rerankers served by Pinecone Inference. All return relevance in [0, 1].
HOSTED_RERANKERS = frozenset({"bge-reranker-v2-m3", "pinecone-rerank-v0", "cohere-rerank-3.5"})
#: Documents one hosted rerank request accepts.
HOSTED_MAX_DOCUMENTS = 100

#: Sigmoid temperature applied to models that emit raw logits.
_LOGIT_TEMPERATURE = 2.0

#: Absolute relevance floors, per model.
#:
#: Model-specific because the scales genuinely differ: bge-reranker-base
#: applies a Sigmoid internally and emits calibrated probabilities, while
#: ms-marco emits raw logits that are squashed first.
#:
#: The bge value is calibrated, not guessed. Measured over 12 in-domain and 6
#: out-of-domain queries against the course corpus:
#:
#:     in-domain      0.881 .. 0.9998   (worst: "explain the margin in SVM")
#:     out-of-domain  0.000037 .. 0.0028 (worst: "what is the capital of France")
#:
#: a 317x gap. 0.05 is the geometric mean of the two boundaries, leaving ~18x
#: headroom against false accepts and ~17x against false rejects.
#:
#: bge-reranker-v2-m3 is the hosted successor in the same family, and Pinecone
#: returns its scores as sigmoid probabilities like bge-reranker-base, so it
#: starts from the same floor. That value is provisional — not yet measured
#: against this corpus — and worth re-checking with the evaluation suite
#: before switching production to it.
DEFAULT_ABSOLUTE_FLOORS = {
    "BAAI/bge-reranker-base": 0.05,
    "BAAI/bge-reranker-large": 0.05,
    "cross-encoder/ms-marco-MiniLM-L-6-v2": 0.30,
    "bge-reranker-v2-m3": 0.05,
}
_FALLBACK_FLOOR = 0.05

#: Keep only candidates scoring at least this fraction of the best candidate.
#:
#: Deliberately small. Cross-encoder scores are extremely peaked — a strong
#: match lands near 0.99 while a useful supporting chunk may sit at 0.1 — so a
#: 10% relative floor would discard almost all supporting context and leave
#: the generator with a single chunk. The absolute floor does the real
#: rejecting; this only trims the far tail.
DEFAULT_RELATIVE_FLOOR = 0.01


@dataclass
class RerankedChunk:
    """A candidate with its post-rerank relevance score in [0, 1]."""

    chunk_id: str
    text: str
    metadata: dict
    relevance: float
    retrieval_score: float = 0.0

    @property
    def source_file(self) -> str:
        return str(self.metadata.get("source_file", ""))


def _sigmoid(x: float) -> float:
    """Numerically stable logistic, used to normalize raw cross-encoder logits."""
    z = x / _LOGIT_TEMPERATURE
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _emits_probabilities(model) -> bool:
    """
    True when the model already applies a sigmoid, so `predict` returns 0-1.

    Getting this wrong is not cosmetic: applying a second sigmoid to values
    already in [0, 1] compresses every score into [0.50, 0.62], which makes
    any relevance threshold unreachable and lets irrelevant chunks through as
    "evidence". Checked explicitly rather than inferred from the value range,
    which a single unlucky batch can fool.
    """
    for attr in ("activation_fn", "activation_fct", "default_activation_function"):
        fn = getattr(model, attr, None)
        if fn is not None:
            return "sigmoid" in type(fn).__name__.lower()
    return False


def _absolute_floor(model_name: str | None, override: float | None) -> float:
    if override is not None:
        return override
    return DEFAULT_ABSOLUTE_FLOORS.get(model_name or "", _FALLBACK_FLOOR)


# ---------------------------------------------------------------------------
# Cross-encoder
# ---------------------------------------------------------------------------

_model = None
_model_name: str | None = None
_model_lock = threading.Lock()
#: sentence-transformers is not installed. Permanent for the process, so it is
#: remembered rather than re-discovered (and re-logged) on every request.
_not_installed = False


def get_cross_encoder(preferred: str | None = None):
    """
    Load a cross-encoder, trying the preferred model then the fallbacks.

    Returns (model, name) or (None, None) if none can be loaded. Loading
    happens once, under a lock, so concurrent requests cannot each start
    their own download.
    """
    global _model, _model_name, _not_installed
    if _model is not None:
        return _model, _model_name
    if _not_installed:
        return None, None

    with _model_lock:
        if _model is not None:
            return _model, _model_name

        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            _not_installed = True
            logger.warning(
                "no local cross-encoder: sentence-transformers is not installed "
                "(the local-models extra)"
            )
            return None, None

        candidates = ([preferred] if preferred else []) + list(CROSS_ENCODER_CANDIDATES)
        for name in candidates:
            if not name or name in HOSTED_RERANKERS:
                continue
            try:
                from .inference import configure_threads

                configure_threads()
                logger.info("loading cross-encoder %s", name)
                _model = CrossEncoder(name, max_length=512)
                _model_name = name
                return _model, _model_name
            except Exception as exc:
                logger.warning("could not load cross-encoder %s: %s", name, exc)

    logger.error("no cross-encoder available — falling back to keyword reranking")
    return None, None


def rerank_cross_encoder(
    query: str,
    candidates: Sequence[tuple[str, str, dict, float]],
    *,
    model_name: str | None = None,
    batch_size: int = 32,
) -> list[RerankedChunk] | None:
    """
    Score (query, passage) pairs with a cross-encoder.

    Args:
        candidates: (chunk_id, text, metadata, retrieval_score) tuples.

    Returns:
        Chunks sorted by relevance, or None if no cross-encoder is available
        so the caller can fall back.
    """
    model, name = get_cross_encoder(model_name)
    if model is None:
        return None

    from .inference import model_slot

    pairs = [(query, text) for _, text, _, _ in candidates]
    try:
        with model_slot("rerank"):
            scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
    except Exception as exc:
        logger.warning("cross-encoder scoring failed: %s", exc)
        return None

    # Only squash models that emit raw logits; probability-emitting models are
    # already on the right scale.
    normalize = (lambda s: s) if _emits_probabilities(model) else _sigmoid

    out = [
        RerankedChunk(
            chunk_id=cid,
            text=text,
            metadata=meta,
            relevance=float(normalize(float(score))),
            retrieval_score=retrieval,
        )
        for (cid, text, meta, retrieval), score in zip(candidates, scores)
    ]
    out.sort(key=lambda c: c.relevance, reverse=True)
    return out


# ---------------------------------------------------------------------------
# Hosted (Pinecone Inference)
# ---------------------------------------------------------------------------

_hosted_client = None
_hosted_lock = threading.Lock()


def _pinecone():
    """The shared Pinecone client for inference calls, or None without a key."""
    global _hosted_client
    if _hosted_client is None:
        with _hosted_lock:
            if _hosted_client is None:
                from edupilot.core.config import PINECONE_API_KEY

                if not PINECONE_API_KEY:
                    return None
                from pinecone import Pinecone

                _hosted_client = Pinecone(api_key=PINECONE_API_KEY)
    return _hosted_client


def rerank_hosted(
    query: str,
    candidates: Sequence[tuple[str, str, dict, float]],
    *,
    model_name: str,
) -> list[RerankedChunk] | None:
    """
    Score candidates with a Pinecone-hosted reranker, in one request.

    Returns None when the call fails so the caller can fall back. Over-long
    passages are truncated at the end: unlike embedding, where truncation
    silently drops indexed text, here it only shortens what one score sees.

    Retrieval can fuse more candidates than one request accepts. They arrive
    best-first by fused rank, so the tail beyond `HOSTED_MAX_DOCUMENTS` is
    dropped rather than paid for with a second request per question.
    """
    client = _pinecone()
    if client is None:
        logger.warning("hosted reranker %s needs PINECONE_API_KEY", model_name)
        return None
    candidates = list(candidates)[:HOSTED_MAX_DOCUMENTS]
    try:
        response = client.inference.rerank(
            model=model_name,
            query=query,
            documents=[{"id": cid, "text": text} for cid, text, _, _ in candidates],
            rank_fields=["text"],
            return_documents=False,
            top_n=len(candidates),
            parameters={"truncate": "END"},
        )
    except Exception as exc:
        logger.warning("hosted reranker %s failed: %s", model_name, exc)
        return None

    scores = {item.index: float(item.score) for item in response.data}
    out = [
        RerankedChunk(
            chunk_id=cid,
            text=text,
            metadata=meta,
            relevance=scores.get(i, 0.0),
            retrieval_score=retrieval,
        )
        for i, (cid, text, meta, retrieval) in enumerate(candidates)
    ]
    out.sort(key=lambda c: c.relevance, reverse=True)
    return out


# ---------------------------------------------------------------------------
# Keyword fallback
# ---------------------------------------------------------------------------


def _keyword_relevance(query_tokens: set[str], text: str) -> float:
    """
    Coverage of the query's terms by the passage, in [0, 1].

    Coverage (what fraction of the query is present) rather than the previous
    Jaccard similarity, which penalized long passages for being long — the
    union term grows with passage length, so a thorough chunk scored lower
    than a short one containing the same answer.
    """
    from .sparse import tokenize

    if not query_tokens:
        return 0.0
    text_tokens = set(tokenize(text))
    if not text_tokens:
        return 0.0
    return len(query_tokens & text_tokens) / len(query_tokens)


def rerank_keyword(
    query: str,
    candidates: Sequence[tuple[str, str, dict, float]],
) -> list[RerankedChunk]:
    """Lexical fallback used when no cross-encoder can be loaded."""
    from .sparse import tokenize

    query_tokens = set(tokenize(query))
    out = [
        RerankedChunk(
            chunk_id=cid,
            text=text,
            metadata=meta,
            # Blend lexical coverage with the retrieval score so the hybrid
            # signal is not thrown away entirely on the fallback path.
            relevance=0.7 * _keyword_relevance(query_tokens, text) + 0.3 * min(retrieval, 1.0),
            retrieval_score=retrieval,
        )
        for cid, text, meta, retrieval in candidates
    ]
    out.sort(key=lambda c: c.relevance, reverse=True)
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def rerank(
    query: str,
    candidates: Sequence[tuple[str, str, dict, float]],
    *,
    top_k: int,
    mode: RerankMode = "cross_encoder",
    min_relevance: float | None = None,
    relative_floor: float = DEFAULT_RELATIVE_FLOOR,
    model_name: str | None = None,
) -> list[RerankedChunk]:
    """
    Rerank retrieval candidates and apply two relevance floors.

    Args:
        candidates: (chunk_id, text, metadata, retrieval_score) tuples.
        top_k: Maximum chunks to return.
        mode: "cross_encoder" (default), "keyword", or "none" to keep the
            retrieval order.
        min_relevance: Absolute floor. Defaults to a per-model value, because
            reranker score scales are not comparable across models.
        relative_floor: Drop candidates below this fraction of the best score.
        model_name: Reranker to use. Defaults to RERANKER_MODEL.

    Returns:
        Up to `top_k` chunks, best first. **May be empty** — that means the
        retrieved evidence is not relevant to the question, and the caller
        must treat it as "no evidence" rather than answering from the best of
        a bad set.
    """
    if not candidates:
        return []
    if model_name is None:
        from edupilot.core.config import RERANKER_MODEL

        model_name = RERANKER_MODEL

    used_model: str | None = None
    if mode == "none":
        ranked = [
            RerankedChunk(chunk_id=c, text=t, metadata=m, relevance=min(s, 1.0), retrieval_score=s)
            for c, t, m, s in candidates
        ]
        ranked.sort(key=lambda c: c.relevance, reverse=True)
    elif mode == "keyword":
        ranked = rerank_keyword(query, candidates)
        used_model = "keyword"
    else:
        hosted = model_name in HOSTED_RERANKERS
        ranked = (
            rerank_hosted(query, candidates, model_name=model_name)
            if hosted
            else rerank_cross_encoder(query, candidates, model_name=model_name)
        )
        if ranked is None:
            ranked = rerank_keyword(query, candidates)
            used_model = "keyword"
        else:
            used_model = model_name if hosted else _model_name

    if not ranked:
        return []

    floor = _absolute_floor(used_model, min_relevance)
    best = ranked[0].relevance

    if best < floor:
        logger.info(
            "no candidate cleared the absolute floor (best %.5f < %.5f) for %r — no evidence",
            best, floor, query[:60],
        )
        return []

    cutoff = max(floor, best * relative_floor)
    kept = [c for c in ranked if c.relevance >= cutoff]
    logger.debug(
        "rerank kept %d/%d (best %.5f, cutoff %.5f)", len(kept), len(ranked), best, cutoff
    )
    return kept[:top_k]


def score_summary(chunks: Sequence[RerankedChunk]) -> dict:
    """Debug-friendly summary of a reranked set."""
    if not chunks:
        return {"count": 0}
    scores = [c.relevance for c in chunks]
    return {
        "count": len(chunks),
        "top": round(max(scores), 4),
        "min": round(min(scores), 4),
        "mean": round(sum(scores) / len(scores), 4),
        "sources": [c.metadata.get("filename", "?") for c in chunks],
    }
