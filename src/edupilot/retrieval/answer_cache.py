"""
Frequency-gated semantic answer cache
=====================================
Caches answers to questions that are actually asked repeatedly, and nothing
else.

Two decisions distinguish it from a plain response cache:

**Frequency gate.** Every question increments a counter; only on the
`PROMOTE_AFTER`-th ask is the answer stored. A class of students asks a long
tail of one-off questions, and storing those buys nothing while making stale
answers more likely. The counter is cheap — a row per normalized question —
and the payload is only written for questions that earn it.

**Semantic matching.** A hit requires cosine similarity above
`SIMILARITY_FLOOR` against a cached question's embedding, so "what is a
p-value" reaches "explain p-values". pgvector finds the nearest cached
question in the database rather than every entry being loaded and compared
in Python. Exact matching would almost never fire on
real phrasing. The floor is deliberately high: serving a confidently wrong
answer to a *near-miss* question is worse than the cache missing.

**Invalidation.** Every entry records the index version that produced it. A
rebuild promotes a new version, so entries from an older corpus stop matching
and are swept — a re-index can never serve answers grounded in documents that
have since changed.

Only answers that were not refused and that carry a measured grounding score
above `MIN_GROUNDING_TO_CACHE` are stored: a refusal or a weakly-grounded
answer is exactly what a corpus fix is meant to change.
"""

from __future__ import annotations

import logging
import re
import time
import unicodedata
from dataclasses import dataclass
from typing import Any

import numpy as np
from sqlalchemy import Engine, and_, delete, exists, func, or_, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from edupilot.db.tables import answer_cache as cache

logger = logging.getLogger(__name__)

#: Asks required before an answer is stored. The first two are counted only.
PROMOTE_AFTER = 3

#: Cosine similarity required to reuse a cached answer.
#:
#: Calibrated for normalized bge-small embeddings, where unrelated questions
#: land near 0.6-0.75 and genuine paraphrases sit above 0.93. Lowering this
#: trades correctness for hit rate — the wrong direction for a tutor.
SIMILARITY_FLOOR = 0.95

#: Entries older than this are ignored and swept, even if still matching.
TTL_SECONDS = 7 * 24 * 3600

#: Never cache an answer whose grounding was not measured or was weak.
MIN_GROUNDING_TO_CACHE = 0.7

_WHITESPACE = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]")


def normalize_question(text: str) -> str:
    """
    Canonical form used for the frequency counter.

    Deliberately aggressive — this only decides whether two asks count as the
    same *for counting*. Serving a cached answer additionally requires the
    semantic check, so an over-eager normalization here cannot by itself cause
    a wrong answer.
    """
    text = unicodedata.normalize("NFKC", text).lower().strip()
    text = _PUNCT.sub(" ", text)
    return _WHITESPACE.sub(" ", text).strip()


@dataclass
class CacheHit:
    payload: dict[str, Any]
    similarity: float
    ask_count: int
    age_seconds: float


class AnswerCache:
    """
    Answer cache in Postgres, shared by every API process. Thread-safe.

    Args:
        embedder: Anything with `embed_query(str) -> np.ndarray`, normalized.
        index_version: Active vector index name; entries from another version
            are never served.
        engine: The SQLAlchemy engine. Defaults to the app's shared one.
    """

    def __init__(self, embedder, index_version: str = "", *, engine: Engine | None = None) -> None:
        from edupilot.db import get_engine

        self._embedder = embedder
        self._index_version = index_version or "unversioned"
        self.engine = engine or get_engine()

    def _scope(self, domains: list[str] | None, model: str) -> str:
        """
        Cache key namespace.

        Domains and model are part of it because the same question answered
        for a different domain set, or by a different model, is a different
        answer. The index version makes a rebuild invalidate everything.
        """
        doms = ",".join(sorted(domains or []))
        return f"{doms}|{model}|{self._index_version}"

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def lookup(self, question: str, *, domains: list[str] | None, model: str) -> CacheHit | None:
        """Return a cached answer for a semantically equivalent question, or None."""
        scope = self._scope(domains, model)
        now = time.time()
        live = and_(
            cache.c.scope == scope,
            cache.c.payload.is_not(None),
            cache.c.created_at > now - TTL_SECONDS,
        )

        with self.engine.connect() as conn:
            if not conn.execute(select(exists().where(live))).scalar():
                return None  # nothing cached in this scope; skip the embedding

        try:
            probe = np.asarray(self._embedder.embed_query(question), dtype=np.float32)
        except Exception:
            logger.warning("cache lookup could not embed the query", exc_info=True)
            return None

        # The nearest cached question, by inner product: vectors are
        # normalized, so that is cosine similarity. Entries embedded by a
        # model with another dimension cannot be compared and are skipped.
        distance = cache.c.embedding.max_inner_product(probe.tolist())
        with self.engine.connect() as conn:
            row = conn.execute(
                select(
                    cache.c.id, cache.c.payload, cache.c.ask_count, cache.c.created_at,
                    (-distance).label("similarity"),
                )
                .where(live, func.vector_dims(cache.c.embedding) == int(probe.shape[0]))
                .order_by(distance)
                .limit(1)
            ).first()

        if row is None or row.similarity < SIMILARITY_FLOOR:
            return None

        with self.engine.begin() as conn:
            conn.execute(
                update(cache)
                .where(cache.c.id == row.id)
                .values(hit_count=cache.c.hit_count + 1, last_hit_at=now)
            )

        logger.info("answer cache hit (similarity %.3f) for %r", row.similarity, question[:60])
        return CacheHit(
            payload=row.payload,
            similarity=float(row.similarity),
            ask_count=int(row.ask_count),
            age_seconds=now - float(row.created_at),
        )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def record_ask(self, question: str, *, domains: list[str] | None, model: str) -> int:
        """
        Count this ask. Returns the running total for the question.

        One upsert, so concurrent asks on different API processes all count.
        """
        norm = normalize_question(question)
        if not norm:
            return 0
        statement = pg_insert(cache).values(
            norm_query=norm, scope=self._scope(domains, model), ask_count=1, created_at=time.time()
        )
        statement = statement.on_conflict_do_update(
            index_elements=["norm_query", "scope"],
            set_={"ask_count": cache.c.ask_count + 1},
        ).returning(cache.c.ask_count)
        with self.engine.begin() as conn:
            return int(conn.execute(statement).scalar_one())

    def should_store(self, ask_count: int, payload: dict[str, Any]) -> bool:
        """
        Whether this answer has earned a cache entry.

        Refusals and weakly-grounded answers are excluded: those are precisely
        the answers a corpus fix is supposed to improve, and freezing them for
        a week would hide the fix.
        """
        if ask_count < PROMOTE_AFTER:
            return False
        if payload.get("refused"):
            return False
        grounding = payload.get("grounding_score")
        if grounding is None or float(grounding) < MIN_GROUNDING_TO_CACHE:
            return False
        return bool(str(payload.get("final_answer", "")).strip())

    def store(
        self, question: str, payload: dict[str, Any], *, domains: list[str] | None, model: str
    ) -> bool:
        """Promote a counted question to a full cache entry. Returns True if stored."""
        norm = normalize_question(question)
        if not norm:
            return False
        scope = self._scope(domains, model)

        try:
            vec = np.asarray(self._embedder.embed_query(question), dtype=np.float32)
        except Exception:
            logger.warning("could not embed question for caching", exc_info=True)
            return False

        # Store the response only — never the diagnostics blob, which carries
        # chunk text and timings specific to the run that produced it.
        slim = {
            k: payload.get(k)
            for k in (
                "final_answer", "intent_type", "detected_domains", "is_course_related",
                "needs_clarification", "refused", "grounding_score", "guardrail_action",
                "sources",
            )
        }

        with self.engine.begin() as conn:
            conn.execute(
                update(cache)
                .where(cache.c.norm_query == norm, cache.c.scope == scope)
                .values(
                    embedding=vec.tolist(),
                    payload=slim,
                    index_version=self._index_version,
                    created_at=time.time(),
                )
            )
        logger.info("cached answer for %r (scope=%s)", question[:60], scope)
        return True

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def sweep(self) -> int:
        """Delete expired entries and anything from a superseded index. Returns rows removed."""
        cutoff = time.time() - TTL_SECONDS
        with self.engine.begin() as conn:
            result = conn.execute(
                delete(cache).where(
                    or_(
                        cache.c.created_at < cutoff,
                        and_(
                            cache.c.payload.is_not(None),
                            cache.c.index_version.is_distinct_from(self._index_version),
                        ),
                    )
                )
            )
            return result.rowcount

    def stats(self) -> dict[str, Any]:
        with self.engine.connect() as conn:
            row = conn.execute(
                select(
                    func.count().label("tracked"),
                    func.count(cache.c.payload).label("cached"),
                    func.coalesce(func.sum(cache.c.hit_count), 0).label("hits"),
                )
            ).one()
        return {
            "tracked_questions": int(row.tracked or 0),
            "cached_answers": int(row.cached or 0),
            "total_hits": int(row.hits or 0),
            "promote_after": PROMOTE_AFTER,
            "similarity_floor": SIMILARITY_FLOOR,
            "ttl_days": round(TTL_SECONDS / 86400, 1),
            "index_version": self._index_version,
        }


__all__ = [
    "MIN_GROUNDING_TO_CACHE",
    "PROMOTE_AFTER",
    "SIMILARITY_FLOOR",
    "TTL_SECONDS",
    "AnswerCache",
    "CacheHit",
    "normalize_question",
]
