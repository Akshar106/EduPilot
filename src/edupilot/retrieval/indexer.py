"""
Indexing
========
Writes documents into the vector store: extract, chunk, embed, encode sparse
weights, upsert.

Properties the previous ingest path lacked:

  * **Idempotency.** Documents are identified by a SHA-256 of their bytes.
    Re-ingesting an unchanged file is a no-op; re-ingesting a *changed* file
    deletes the old version's vectors before writing the new ones. Previously
    dedup was by chunk ID only, so editing a document left its stale chunks in
    the index forever.
  * **Isolation.** Chunk IDs are scoped by namespace and content hash, so two
    study sessions uploading the same filename cannot collide. The old scheme
    (`SELF_STUDY_<stem>_<n>`) collided on every shared filename, and because
    `self_study_chunks.chunk_id` is globally UNIQUE with `INSERT OR IGNORE`,
    the second session's chunks were silently dropped from SQLite.
  * **Atomic rebuilds.** `edupilot-reindex --rebuild` writes into a fresh
    index version and promotes it only on success, so a failed re-index
    cannot leave the live index half-populated.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sqlalchemy import Engine, delete, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from edupilot.db.rows import row_dict
from edupilot.db.tables import indexed_documents as documents
from edupilot.ingestion import DocumentChunk, chunk_document, extract_document
from edupilot.ingestion.chunking import ChunkingConfig

from .embeddings import Embedder
from .sparse import BM25Encoder
from .vectorstore import VectorRecord, VectorStore

logger = logging.getLogger(__name__)


@dataclass
class IndexResult:
    """Outcome of indexing one document."""

    filename: str
    content_hash: str
    chunks_indexed: int = 0
    parents_indexed: int = 0
    skipped: bool = False
    """True when the document was already indexed at this exact content hash."""
    replaced_version: bool = False
    """True when an older version of the same file was removed first."""
    error: str | None = None
    duration_ms: int = 0
    extraction: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class IndexRegistry:
    """
    Tracks which documents are indexed, at which content hash, in which
    namespace.

    This is what makes re-ingest idempotent and lets a changed file replace
    its own prior vectors without wiping the namespace.

    Args:
        engine: The SQLAlchemy engine. Defaults to the app's shared one.
    """

    def __init__(self, engine: Engine | None = None) -> None:
        from edupilot.db import get_engine

        self.engine = engine or get_engine()

    def get(self, namespace: str, filename: str) -> dict | None:
        with self.engine.connect() as conn:
            row = conn.execute(
                select(documents).where(
                    documents.c.namespace == namespace, documents.c.filename == filename
                )
            ).first()
        return row_dict(row) if row else None

    def record(
        self,
        *,
        namespace: str,
        filename: str,
        source_path: str,
        content_hash: str,
        chunk_count: int,
        parent_count: int,
        embed_model: str,
        chunker: str,
        source_type: str = "lecture",
    ) -> None:
        values = {
            "source_path": source_path,
            "content_hash": content_hash,
            "chunk_count": chunk_count,
            "parent_count": parent_count,
            "embed_model": embed_model,
            "chunker": chunker,
            "source_type": source_type,
        }
        statement = pg_insert(documents).values(namespace=namespace, filename=filename, **values)
        statement = statement.on_conflict_do_update(
            index_elements=["namespace", "filename"],
            set_={**values, "indexed_at": func.now()},
        )
        with self.engine.begin() as conn:
            conn.execute(statement)

    def forget(self, namespace: str, filename: str) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                delete(documents).where(
                    documents.c.namespace == namespace, documents.c.filename == filename
                )
            )

    def prune(self, keep: set[tuple[str, str]], *, skip_prefix: str = "ss_") -> int:
        """
        Forget every document not in `keep`, except namespaces starting with `skip_prefix`.

        After a full rebuild, a file removed from the corpus (or a whole course
        dropped from DOMAINS) is no longer in the live index, so its registry
        row only misleads. Study-session rows are left alone: a rebuild carries
        those vectors over rather than re-indexing them. Returns rows removed.
        """
        removed = 0
        with self.engine.begin() as conn:
            rows = conn.execute(select(documents.c.namespace, documents.c.filename)).all()
            for namespace, filename in rows:
                if namespace.startswith(skip_prefix) or (namespace, filename) in keep:
                    continue
                conn.execute(
                    delete(documents).where(
                        documents.c.namespace == namespace, documents.c.filename == filename
                    )
                )
                removed += 1
        return removed

    def list_documents(self, namespace: str | None = None) -> list[dict]:
        query = select(documents).order_by(documents.c.namespace, documents.c.filename)
        if namespace:
            query = query.where(documents.c.namespace == namespace)
        with self.engine.connect() as conn:
            return [row_dict(r) for r in conn.execute(query)]


# ---------------------------------------------------------------------------
# Course folders
# ---------------------------------------------------------------------------

#: Subfolders of a course whose files are textbooks rather than course
#: material. Retrieval uses the difference: a book covers every topic in
#: depth and would otherwise crowd the lectures out of the evidence.
BOOK_FOLDERS = frozenset({"book", "books", "textbook", "textbooks"})


def source_type_of(path: Path, root: Path) -> str:
    """"book" for a file under one of BOOK_FOLDERS in `root`, else "lecture"."""
    try:
        parts = path.relative_to(root).parts[:-1]
    except ValueError:
        return "lecture"
    return "book" if any(part.lower() in BOOK_FOLDERS for part in parts) else "lecture"


def course_files(root: str | Path) -> list[tuple[Path, str]]:
    """
    Every indexable file in a course folder, subfolders included, with its
    source type.

    Documents are identified by file name within a course, so a second file
    with a name already seen is skipped (with a warning) rather than
    silently replacing the first one's vectors.
    """
    from edupilot.ingestion import SUPPORTED_EXTENSIONS

    base = Path(root)
    if not base.is_dir():
        return []
    found: list[tuple[Path, str]] = []
    seen: dict[str, Path] = {}
    for path in sorted(base.rglob("*")):
        relative = path.relative_to(base).parts
        if not path.is_file() or any(part.startswith(".") for part in relative):
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if path.name in seen:
            logger.warning(
                "skipping %s: a file named %s is already in this course (%s)",
                path, path.name, seen[path.name],
            )
            continue
        seen[path.name] = path
        found.append((path, source_type_of(path, base)))
    return found


# ---------------------------------------------------------------------------
# Indexer
# ---------------------------------------------------------------------------


class Indexer:
    """
    Writes documents into a namespace of the vector store.

    Args:
        store: Vector store backend.
        embedder: Dense embedder.
        registry: Document registry for idempotency.
        sparse_encoder: Fitted BM25 encoder. When absent, only dense vectors
            are written and hybrid search degrades to semantic-only.
        chunking: Chunking parameters. Defaults are matched to the embedder.
    """

    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        registry: IndexRegistry,
        *,
        sparse_encoder: BM25Encoder | None = None,
        chunking: ChunkingConfig | None = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.registry = registry
        self.sparse_encoder = sparse_encoder
        self.chunking = chunking or ChunkingConfig(
            model_name=embedder.tokenizer_name,
            model_window=embedder.max_tokens,
            max_tokens=min(448, embedder.max_tokens - 64),
        )

    # ------------------------------------------------------------------
    # Vector construction
    # ------------------------------------------------------------------

    def _build_records(
        self, chunks: Sequence[DocumentChunk], source_type: str = "lecture"
    ) -> list[VectorRecord]:
        """
        Embed children, derive parent vectors, and attach sparse weights.

        Parents are stored so they can be fetched for context expansion, but
        they are never embedded: a parent runs to ~1,400 tokens, well past the
        model window, and embedding it would either fail the token budget or
        reintroduce silent truncation. Their vector is the L2-normalized mean
        of their children — a valid centroid, and they are excluded from
        search by an `is_parent` metadata filter regardless.
        """
        children = [c for c in chunks if not c.is_parent]
        parents = [c for c in chunks if c.is_parent]

        records: list[VectorRecord] = []
        child_vectors: dict[str, np.ndarray] = {}

        if children:
            # The chunker already measured every chunk; reuse those counts so
            # the budget check does not tokenize the whole corpus a second time.
            vectors = self.embedder.embed_documents(
                [c.text for c in children],
                token_counts=[c.token_count for c in children],
            )
            for chunk, vector in zip(children, vectors):
                child_vectors[chunk.chunk_id] = vector
                sparse = (
                    self.sparse_encoder.encode_document(chunk.text)
                    if self.sparse_encoder and self.sparse_encoder.fitted
                    else None
                )
                records.append(
                    VectorRecord(
                        id=chunk.chunk_id,
                        values=vector.tolist(),
                        sparse=sparse or None,
                        metadata={
                            **chunk.metadata,
                            "text": chunk.text,
                            "is_parent": False,
                            "source_type": source_type,
                        },
                    )
                )

        for parent in parents:
            ids = parent.metadata.get("child_ids") or []
            members = [child_vectors[i] for i in ids if i in child_vectors]
            if members:
                centroid = np.mean(members, axis=0)
                norm = float(np.linalg.norm(centroid))
                vector = (centroid / norm) if norm > 0 else centroid
            else:
                vector = np.zeros(self.embedder.dimension, dtype=np.float32)
            records.append(
                VectorRecord(
                    id=parent.chunk_id,
                    values=vector.astype(np.float32).tolist(),
                    sparse=None,
                    metadata={
                        **parent.metadata,
                        "text": parent.text,
                        "is_parent": True,
                        "source_type": source_type,
                    },
                )
            )

        return records

    # ------------------------------------------------------------------
    # Single document
    # ------------------------------------------------------------------

    def index_document(
        self,
        path: str | Path,
        *,
        namespace: str,
        domain: str,
        force: bool = False,
        prepared_chunks: list[DocumentChunk] | None = None,
        source_type: str = "lecture",
    ) -> IndexResult:
        """
        Index one document into `namespace`.

        Skips work when the file's content hash already matches what is
        indexed. When the hash differs, the previous version's vectors are
        deleted before the new ones are written, so a corrected slide deck
        cannot leave stale text retrievable.

        Args:
            prepared_chunks: Already-chunked output for this file. Supplying
                it skips extraction — used by a full rebuild, which has to
                chunk the whole corpus up front to fit BM25 and would
                otherwise extract every document a second time.
            source_type: "lecture" or "book", stored on every chunk.
        """
        started = time.perf_counter()
        p = Path(path)
        filename = p.name

        try:
            from edupilot.ingestion.models import file_content_hash

            content_hash = file_content_hash(p)
        except OSError as exc:
            return IndexResult(filename=filename, content_hash="", error=str(exc))

        existing = self.registry.get(namespace, filename)
        unchanged = (
            existing
            and existing["content_hash"] == content_hash
            # Moving a file into or out of book/ changes how it is retrieved.
            and existing.get("source_type", "lecture") == source_type
        )
        if unchanged and not force:
            logger.info("%s unchanged (hash %s) — skipping", filename, content_hash[:12])
            return IndexResult(
                filename=filename,
                content_hash=content_hash,
                chunks_indexed=existing["chunk_count"],
                skipped=True,
                duration_ms=int((time.perf_counter() - started) * 1000),
            )

        replaced = False
        if existing:
            logger.info(
                "%s changed (%s -> %s) — removing previous version",
                filename, existing["content_hash"][:12], content_hash[:12],
            )
            self._delete_document_vectors(namespace, filename, existing["content_hash"])
            replaced = True

        # A caller that has already chunked this file (a full rebuild, which
        # must chunk everything up front to fit BM25) passes the chunks in
        # rather than paying for a second extraction. On this corpus that is
        # ~4 minutes saved per rebuild.
        doc = None
        if prepared_chunks is not None:
            chunks = prepared_chunks
        else:
            try:
                doc = extract_document(p)
                chunks = chunk_document(doc, domain, config=self.chunking, scope=namespace)
            except Exception as exc:
                logger.exception("extraction failed for %s", filename)
                return IndexResult(
                    filename=filename,
                    content_hash=content_hash,
                    error=f"{type(exc).__name__}: {exc}",
                    duration_ms=int((time.perf_counter() - started) * 1000),
                )

        children = [c for c in chunks if not c.is_parent]
        parents = [c for c in chunks if c.is_parent]
        if not children:
            return IndexResult(
                filename=filename,
                content_hash=content_hash,
                error="document produced no chunks (empty or unreadable)",
                duration_ms=int((time.perf_counter() - started) * 1000),
            )

        try:
            records = self._build_records(chunks, source_type)
            self.store.upsert(records, namespace=namespace)
        except Exception as exc:
            logger.exception("upsert failed for %s", filename)
            return IndexResult(
                filename=filename,
                content_hash=content_hash,
                error=f"{type(exc).__name__}: {exc}",
                duration_ms=int((time.perf_counter() - started) * 1000),
            )

        self.registry.record(
            namespace=namespace,
            filename=filename,
            source_path=str(p),
            content_hash=content_hash,
            chunk_count=len(children),
            parent_count=len(parents),
            embed_model=self.embedder.model_name,
            chunker=self.chunking.model_name,
            source_type=source_type,
        )

        from edupilot.ingestion import extraction_report

        # Only available when this call did the extraction; a caller supplying
        # prepared chunks already has its own report.
        report = extraction_report(doc) if doc is not None else {}

        return IndexResult(
            filename=filename,
            content_hash=content_hash,
            chunks_indexed=len(children),
            parents_indexed=len(parents),
            replaced_version=replaced,
            duration_ms=int((time.perf_counter() - started) * 1000),
            extraction=report,
        )

    def _delete_document_vectors(self, namespace: str, filename: str, content_hash: str) -> None:
        """
        Remove a document's vectors by metadata filter.

        Filtering on both filename and the *old* content hash means a
        concurrent write of the new version cannot be deleted by this call.
        """
        try:
            self.store.delete(
                namespace,
                metadata_filter={
                    "$and": [
                        {"filename": {"$eq": filename}},
                        {"doc_hash": {"$eq": content_hash}},
                    ]
                },
            )
        except Exception as exc:
            # Serverless Pinecone does not support delete-by-filter on all
            # tiers; fall back to filename-only deletion. pgvector always does.
            logger.warning("filtered delete failed (%s) — retrying by filename", exc)
            try:
                self.store.delete(namespace, metadata_filter={"filename": {"$eq": filename}})
            except Exception as exc2:
                logger.error("could not delete old vectors for %s: %s", filename, exc2)

    def remove_document(self, namespace: str, filename: str) -> bool:
        """Delete a document's vectors and forget it. Returns True if it existed."""
        existing = self.registry.get(namespace, filename)
        if not existing:
            return False
        self._delete_document_vectors(namespace, filename, existing["content_hash"])
        self.registry.forget(namespace, filename)
        return True

    # ------------------------------------------------------------------
    # Bulk
    # ------------------------------------------------------------------

    def index_directory(
        self,
        directory: str | Path,
        *,
        namespace: str,
        domain: str,
        force: bool = False,
        prepared: dict[str, list[DocumentChunk]] | None = None,
    ) -> list[IndexResult]:
        """
        Index every supported document in a course folder, subfolders
        included; files under a book/ subfolder are indexed as books.

        Args:
            prepared: {absolute_path: chunks} for files already chunked by the
                caller, so extraction is not repeated.
        """
        root = Path(directory)
        if not root.exists():
            logger.warning("knowledge base directory %s does not exist", root)
            return []

        results = []
        for p, source_type in course_files(root):
            results.append(
                self.index_document(
                    p,
                    namespace=namespace,
                    domain=domain,
                    force=force,
                    prepared_chunks=(prepared or {}).get(str(p)),
                    source_type=source_type,
                )
            )
        return results
