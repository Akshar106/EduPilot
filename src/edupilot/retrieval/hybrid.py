"""
Hybrid retriever
================
Search across one namespace: query transformation, engine-side sparse+dense
hybrid search, rank fusion across probes, cross-encoder reranking, and
small-to-big parent expansion.

Differences from the previous `DomainRetriever`:

  * **Filters are pushed down.** `source_filter` becomes a Pinecone metadata
    filter instead of a Python list comprehension over over-fetched results.
    The old code also fell back to *unfiltered* results when a filter matched
    nothing, so restricting a search to one attachment silently searched the
    whole corpus. A filter that matches nothing now returns nothing.
  * **Lexical search is engine-side.** BM25 term weights are stored as sparse
    vectors, so one query does both halves of the hybrid. No in-process index
    to rebuild, and it works with multiple workers.
  * **Reranking is on.** The Chat path reranked by token overlap; the
    cross-encoder now runs on every path.
  * **Empty means empty.** When nothing clears the relevance floor the
    retriever returns nothing instead of substituting the best of a bad set.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from edupilot.core import metrics
from edupilot.core.config import MAX_BOOK_EXCERPTS, MAX_EXCERPTS_PER_DOCUMENT

from .embeddings import Embedder
from .query_transform import LLMCallable, TransformedQuery, transform_query
from .rerank import RerankedChunk, RerankMode, rerank, score_summary
from .sparse import BM25Encoder
from .vectorstore import SearchHit, VectorStore

logger = logging.getLogger(__name__)

#: RRF damping constant. 60 is the value from the original Cormack et al. paper.
RRF_K = 60

#: Relative trust in each probe type during fusion.
PROBE_WEIGHTS = {"primary": 1.0, "variant": 0.7, "hyde": 0.7}

#: Of each probe's candidates, the share reserved for textbook passages when
#: the book limit is on; the rest go to the course's own material.
BOOK_CANDIDATE_SHARE = 0.25

#: Most of a parent section an excerpt carries: the matched passage and the
#: text around it. Book sections run to 6,500 characters, and everything
#: downstream reads only a part of each. The answer prompt shares its evidence
#: budget out (about 2,000 characters each for eight excerpts) and the claim
#: check's cross-encoder reads 512 tokens, both from the start, so a match
#: deep in a section reached neither, and answers filled the gap from the
#: model's own knowledge. 2,000 fits both.
PARENT_WINDOW_CHARS = 2000


def _window(section: str, matched: str, limit: int = PARENT_WINDOW_CHARS) -> str:
    """`matched` with as much of `section` around it as fits in `limit` characters."""
    if len(section) <= limit:
        return section
    at = section.find(matched)
    if at < 0:
        return section[:limit] + "…"
    if len(matched) >= limit:
        return matched
    start = max(0, at - (limit - len(matched)) // 2)
    end = min(len(section), start + limit)
    start = max(0, end - limit)
    # Cut at whitespace, so no word is split.
    if start > 0:
        start = section.find(" ", start, at) + 1 or start
    if end < len(section):
        cut = section.rfind(" ", at + len(matched), end)
        end = cut if cut > 0 else end
    return ("…" if start > 0 else "") + section[start:end].strip() + ("…" if end < len(section) else "")


@dataclass
class RetrievalConfig:
    """Knobs for one retrieval call."""

    top_k: int = 6
    """Chunks handed to the generator after reranking."""
    candidate_multiplier: int = 5
    """Candidates fetched per probe, as a multiple of top_k."""
    rerank_mode: RerankMode = "cross_encoder"
    min_relevance: float | None = None
    """
    Absolute relevance floor. None uses the reranker's calibrated default —
    score scales differ by model, so a hardcoded number is wrong for all but
    one of them. See retrieval/rerank.py.
    """
    relative_floor: float = 0.01
    """Drop candidates scoring below this fraction of the best candidate."""
    use_hybrid: bool = True
    """Attach sparse BM25 weights to the query for lexical matching."""
    use_multi_query: bool = False
    use_hyde: bool = False
    expand_to_parents: bool = True
    """Swap each matched child for its wider parent window before generating."""
    max_candidates: int = 120
    """Hard cap on candidates sent to the cross-encoder, for latency."""
    max_book_excerpts: int | None = MAX_BOOK_EXCERPTS
    """
    Textbook excerpts allowed among the `top_k` while course material can fill
    the rest; more are used only when it cannot. None lifts the limit.
    """
    max_excerpts_per_document: int | None = MAX_EXCERPTS_PER_DOCUMENT
    """The same rule for any single file. None: no per-file limit."""


@dataclass
class RetrievalResult:
    """Retrieved evidence plus the diagnostics the debug panel renders."""

    chunks: list[RerankedChunk] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return not self.chunks

    @property
    def sources(self) -> list[str]:
        seen: list[str] = []
        for c in self.chunks:
            name = str(c.metadata.get("filename", ""))
            if name and name not in seen:
                seen.append(name)
        return seen


def compose_evidence(
    chunks: Sequence[RerankedChunk],
    top_k: int,
    *,
    max_book: int | None = None,
    max_per_document: int | None = None,
) -> tuple[list[RerankedChunk], dict[str, int]]:
    """
    The `top_k` excerpts to answer from: course material first, books to
    support it.

    A textbook covers every topic, at length, so by relevance alone it wins
    most questions outright — in the LLM course, one 370-page book was the top
    source for half of all questions, including topics with their own
    lecture. Students are taught from the lectures, and an answer should be
    too.

    The first pass takes excerpts in relevance order but holds back books past
    `max_book` (and any file past `max_per_document`). The second fills the
    remaining slots from what was held back, still by relevance. So a question
    the lectures cover is answered from them, with a book adding depth, and a
    question only the book covers still gets a full set of excerpts rather
    than too few. Every excerpt has already cleared the relevance floor.

    Returns the excerpts — course material first, then books, each most
    relevant first — and counts for diagnostics.
    """
    selected: list[RerankedChunk] = []
    held: list[RerankedChunk] = []
    books = 0
    per_document: dict[str, int] = {}
    for chunk in chunks:
        if len(selected) >= top_k:
            break
        document = str(chunk.metadata.get("filename") or "")
        is_book = chunk.metadata.get("source_type") == "book"
        if (max_book is not None and is_book and books >= max_book) or (
            max_per_document is not None and per_document.get(document, 0) >= max_per_document
        ):
            held.append(chunk)
            continue
        selected.append(chunk)
        books += is_book
        per_document[document] = per_document.get(document, 0) + 1

    filled = 0
    for chunk in held:
        if len(selected) >= top_k:
            break
        selected.append(chunk)
        filled += 1

    # Course material first, then books, each by relevance: [Source 1] is the
    # lecture's account, and the book reads as the supporting reference.
    selected.sort(key=lambda c: (c.metadata.get("source_type") == "book", -c.relevance))
    return selected, {
        "book_excerpts": sum(c.metadata.get("source_type") == "book" for c in selected),
        "held_back": len(held) - filled,
        "filled_from_held_back": filled,
    }


class HybridRetriever:
    """
    Retrieval over a single vector-store namespace.

    Args:
        store: Vector store backend.
        embedder: Dense embedder.
        namespace: Namespace to search (one per domain, or per study session).
        sparse_encoder: Fitted BM25 encoder. None disables the lexical half.
    """

    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        namespace: str,
        *,
        sparse_encoder: BM25Encoder | None = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.namespace = namespace
        self.sparse_encoder = sparse_encoder

    # ------------------------------------------------------------------
    # Filters
    # ------------------------------------------------------------------

    @staticmethod
    def build_filter(
        *,
        filenames: Sequence[str] | None = None,
        exclude_parents: bool = True,
        min_confidence: float | None = None,
        extra: dict | None = None,
    ) -> dict | None:
        """
        Build a Pinecone metadata filter.

        `exclude_parents` keeps parent windows out of search results: they are
        stored for context expansion but must never be matched directly, or a
        single parent would crowd out the children it contains.
        """
        clauses: list[dict] = []
        if filenames:
            clauses.append({"filename": {"$in": list(filenames)}})
        if exclude_parents:
            clauses.append({"is_parent": {"$ne": True}})
        if min_confidence is not None:
            clauses.append({"confidence": {"$gte": min_confidence}})
        if extra:
            clauses.append(extra)

        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def _search_lists(
        self,
        probe: str,
        top_k: int,
        metadata_filter: dict | None,
        use_hybrid: bool,
        *,
        book_share: float | None,
    ) -> list[list[SearchHit]]:
        """
        Search for one probe; with `book_share`, course material and books separately.

        A textbook can outnumber a course's lecture chunks twenty to one, so one
        search's top candidates are nearly all book, and the lectures never
        reach the reranker to be preferred. Searching each on its own, with
        books given `book_share` of the `top_k` candidate slots, keeps the
        reranker's workload the same while guaranteeing the lectures a
        hearing. Where there are no books the course search keeps every slot.
        """
        dense = self.embedder.embed_query(probe).tolist()
        sparse = None
        if use_hybrid and self.sparse_encoder and self.sparse_encoder.fitted:
            sparse = self.sparse_encoder.encode_query(probe) or None

        def search(k: int, flt: dict | None) -> list[SearchHit]:
            return self.store.query(
                vector=dense, top_k=k, namespace=self.namespace, metadata_filter=flt, sparse=sparse
            )

        if book_share is None:
            return [search(top_k, metadata_filter)]

        def with_filter(clause: dict) -> dict:
            return {"$and": [metadata_filter, clause]} if metadata_filter else clause

        book_k = max(1, int(top_k * book_share))
        books = search(book_k, with_filter({"source_type": {"$eq": "book"}}))
        course = search(top_k, with_filter({"source_type": {"$ne": "book"}}))
        if books:
            course = course[: top_k - len(books)]
        return [course, books] if books else [course]

    @staticmethod
    def _fuse(ranked_lists: list[tuple[str, list[SearchHit]]]) -> dict[str, tuple[float, SearchHit]]:
        """
        Reciprocal Rank Fusion across probes.

        RRF combines rankings without needing their scores to be on a common
        scale, which matters here because a HyDE probe and the literal query
        produce systematically different similarity magnitudes.
        """
        fused: dict[str, float] = {}
        best: dict[str, SearchHit] = {}

        for kind, hits in ranked_lists:
            weight = PROBE_WEIGHTS.get(kind, 0.7)
            for rank, hit in enumerate(hits, start=1):
                fused[hit.id] = fused.get(hit.id, 0.0) + weight / (RRF_K + rank)
                if hit.id not in best or hit.score > best[hit.id].score:
                    best[hit.id] = hit

        return {cid: (score, best[cid]) for cid, score in fused.items()}

    def retrieve(
        self,
        query: str,
        *,
        config: RetrievalConfig | None = None,
        filenames: Sequence[str] | None = None,
        metadata_filter: dict | None = None,
        llm: LLMCallable | None = None,
    ) -> RetrievalResult:
        """
        Retrieve evidence for a query.

        Returns a `RetrievalResult` whose `chunks` may legitimately be empty —
        that is the signal that the knowledge base does not cover the question,
        and the caller must refuse rather than answer unsupported.
        """
        cfg = config or RetrievalConfig()
        diagnostics: dict[str, Any] = {"namespace": self.namespace, "query": query}

        transformed: TransformedQuery = transform_query(
            query,
            llm=llm,
            use_multi_query=cfg.use_multi_query,
            use_hyde=cfg.use_hyde,
        )
        diagnostics["probes"] = transformed.probes
        diagnostics["acronyms_expanded"] = transformed.expanded != transformed.original

        combined_filter = self.build_filter(filenames=filenames, extra=metadata_filter)
        diagnostics["filter"] = combined_filter

        per_probe = min(cfg.top_k * cfg.candidate_multiplier, cfg.max_candidates)
        ranked_lists: list[tuple[str, list[SearchHit]]] = []
        started = time.perf_counter()

        for i, probe in enumerate(transformed.probes):
            if i == 0:
                kind = "primary"
            elif transformed.hyde and probe == transformed.hyde:
                kind = "hyde"
            else:
                kind = "variant"
            try:
                lists = self._search_lists(
                    probe,
                    per_probe,
                    combined_filter,
                    cfg.use_hybrid,
                    book_share=BOOK_CANDIDATE_SHARE if cfg.max_book_excerpts is not None else None,
                )
            except Exception as exc:
                logger.warning("probe %r failed: %s", probe[:60], exc)
                continue
            ranked_lists.extend((kind, hits) for hits in lists)

        metrics.RETRIEVAL_STAGE.labels("search").observe(time.perf_counter() - started)

        if not ranked_lists:
            diagnostics["error"] = "all probes failed"
            return RetrievalResult(chunks=[], diagnostics=diagnostics)

        fused = self._fuse(ranked_lists)
        diagnostics["candidates"] = len(fused)

        if not fused:
            diagnostics["reason"] = "no vectors matched the filter"
            return RetrievalResult(chunks=[], diagnostics=diagnostics)

        # Rerank the best candidates by fused rank.
        ordered = sorted(fused.items(), key=lambda kv: kv[1][0], reverse=True)[: cfg.max_candidates]
        candidates = [
            (cid, hit.text, hit.metadata, score) for cid, (score, hit) in ordered if hit.text
        ]

        # When expanding to parents, several top children often share one
        # parent and collapse into a single context unit. Rerank a wider pool
        # so the caller still receives `top_k` *distinct* units afterwards.
        rerank_k = cfg.top_k * 3 if cfg.expand_to_parents else cfg.top_k

        started = time.perf_counter()
        reranked = rerank(
            # Rerank against the student's actual words, not the acronym-expanded
            # probe: expansion helps recall at the search stage but adds terms
            # the student never wrote, which skews cross-encoder relevance.
            query=query,
            candidates=candidates,
            top_k=rerank_k,
            mode=cfg.rerank_mode,
            min_relevance=cfg.min_relevance,
            relative_floor=cfg.relative_floor,
        )
        metrics.RETRIEVAL_STAGE.labels("rerank").observe(time.perf_counter() - started)

        if not reranked:
            diagnostics["reason"] = "no candidate cleared the relevance floor"
            diagnostics["reranked"] = {"count": 0}
            return RetrievalResult(chunks=[], diagnostics=diagnostics)

        if cfg.expand_to_parents:
            expanded = self._expand_to_parents(reranked)
            diagnostics["children_matched"] = len(reranked)
            diagnostics["parents_after_dedup"] = len(expanded)
            reranked = expanded

        reranked, diagnostics["composition"] = compose_evidence(
            reranked,
            cfg.top_k,
            max_book=cfg.max_book_excerpts,
            max_per_document=cfg.max_excerpts_per_document,
        )
        diagnostics["reranked"] = score_summary(reranked)
        return RetrievalResult(chunks=reranked, diagnostics=diagnostics)

    # ------------------------------------------------------------------
    # Small-to-big
    # ------------------------------------------------------------------

    def _expand_to_parents(self, chunks: list[RerankedChunk]) -> list[RerankedChunk]:
        """
        Replace matched children with their parent windows.

        Retrieval matches a precise child; generation reads better with the
        surrounding context. Where several children share a parent, the parent
        is emitted once at the best child's relevance, which also removes the
        near-duplicate context the old pipeline fed the model. A long parent
        is cut to a window around the best child (`PARENT_WINDOW_CHARS`).
        """
        parent_ids = [
            str(c.metadata["parent_id"]) for c in chunks if c.metadata.get("parent_id")
        ]
        if not parent_ids:
            return chunks

        try:
            parents = self.store.fetch(parent_ids, self.namespace)
        except Exception as exc:
            logger.warning("parent expansion failed, using children: %s", exc)
            return chunks

        out: list[RerankedChunk] = []
        used_parents: set[str] = set()

        for chunk in chunks:
            pid = str(chunk.metadata.get("parent_id") or "")
            parent = parents.get(pid)
            if not parent or not parent.metadata.get("text"):
                out.append(chunk)
                continue
            if pid in used_parents:
                continue  # a sibling already contributed this parent
            used_parents.add(pid)

            meta = dict(parent.metadata)
            meta["expanded_from_child"] = chunk.chunk_id
            # The part of the section the search matched: a section can run to
            # thousands of characters, and what a quick read needs is here.
            meta["matched_text"] = chunk.text
            out.append(
                RerankedChunk(
                    chunk_id=pid,
                    text=_window(str(parent.metadata["text"]), chunk.text),
                    metadata=meta,
                    relevance=chunk.relevance,
                    retrieval_score=chunk.retrieval_score,
                )
            )
        return out
