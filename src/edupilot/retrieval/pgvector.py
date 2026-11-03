"""
pgvector store
==============
The vector store in Postgres, next to everything else the app keeps.

Each index version is its own table (`vectors_edupilot_v3`), mirroring a
Pinecone index per version: a rebuild fills a fresh table while the live one
keeps serving, promotion flips the pointer, and the old table is dropped
whenever the operator is done with it. A table per version is also what lets
a rebuild change the embedding model — an HNSW index needs one fixed
dimension per column.

Columns:

  namespace  one per course domain, or `ss_<id>` per study session
  id         chunk id, unique within its namespace
  embedding  vector(D): the dense embedding, L2-normalized
  sparse     sparsevec: BM25 term weights (retrieval/sparse.py)
  terms      int[]: the sparse vector's term ids, for the inverted index
  metadata   jsonb: chunk text, file, page, parent id, ...

Search is two queries and a fusion:

  * **Dense**: HNSW over inner product. Vectors are normalized, so inner
    product is cosine similarity — the same ranking as the Pinecone
    `dotproduct` index. HNSW applies the namespace and metadata filters while
    it walks the graph (pgvector's iterative scan), so a filter that matches
    a small study session still returns k results rather than whatever
    survived of the k nearest overall.
  * **Lexical**: exact BM25 scoring of the rows sharing at least one term
    with the query, found through a GIN index on `terms`. That is an inverted
    index, so the cost tracks how many chunks contain the query's words, not
    the size of the corpus.
  * **Reciprocal rank fusion** of the two lists. Pinecone summed a cosine in
    [0, 1] and an unbounded BM25 score, so whichever scale ran larger decided
    the order; RRF needs no common scale. The cross-encoder reranks the
    result anyway.

Metadata filters use the same Pinecone-style operators as the other stores
($eq, $ne, $in, $nin, $gt, $gte, $lt, $lte, $and, $or), translated to JSONB
containment where possible so the GIN index on `metadata` serves them.
"""

from __future__ import annotations

import logging
import re
import threading
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from pgvector import SparseVector
from pgvector.sqlalchemy import SPARSEVEC, VECTOR
from sqlalchemy import (
    Column,
    ColumnElement,
    Engine,
    Index,
    Integer,
    MetaData,
    Table,
    Text,
    and_,
    delete,
    false,
    func,
    not_,
    or_,
    select,
    text,
    true,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.dialects.postgresql import insert as pg_insert

from .vectorstore import SearchHit, VectorRecord, VectorStore, VectorStoreError, sanitize_metadata

if TYPE_CHECKING:
    from .versions import IndexPointer

logger = logging.getLogger(__name__)

#: Size of the hashed BM25 term space (retrieval/sparse.py).
SPARSE_DIMENSIONS = 2**20

#: RRF damping constant, as in retrieval/hybrid.py (Cormack et al.).
RRF_K = 60

#: Rows per INSERT. Each row carries its chunk text, so this bounds a
#: statement at a few MB.
UPSERT_BATCH = 500

#: HNSW graph parameters: pgvector's defaults, written down so a change is
#: deliberate. ef_search is set per query from the requested k.
HNSW_M = 16
HNSW_EF_CONSTRUCTION = 64
MIN_EF_SEARCH = 40

_RANGE_OPS = {"$gt": "__gt__", "$gte": "__ge__", "$lt": "__lt__", "$lte": "__le__"}


def table_name(index_name: str) -> str:
    """The table holding an index version: `edupilot-v3` -> `vectors_edupilot_v3`."""
    slug = re.sub(r"[^a-z0-9_]+", "_", index_name.lower()).strip("_")
    name = f"vectors_{slug}"
    if len(name) > 63:  # Postgres truncates identifiers past 63 bytes
        raise ValueError(f"index name {index_name!r} is too long for a table name")
    return name


def define_table(index_name: str, dimension: int) -> Table:
    """The SQLAlchemy definition of one index version's table, with its indexes."""
    name = table_name(index_name)
    table = Table(
        name,
        MetaData(),
        Column("namespace", Text, primary_key=True),
        Column("id", Text, primary_key=True),
        Column("embedding", VECTOR(dimension), nullable=False),
        Column("sparse", SPARSEVEC(SPARSE_DIMENSIONS)),
        Column("terms", ARRAY(Integer)),
        Column("metadata", JSONB, nullable=False, server_default=text("'{}'::jsonb")),
    )
    Index(
        f"{name}_embedding_hnsw",
        table.c.embedding,
        postgresql_using="hnsw",
        postgresql_with={"m": HNSW_M, "ef_construction": HNSW_EF_CONSTRUCTION},
        postgresql_ops={"embedding": "vector_ip_ops"},
    )
    Index(f"{name}_terms_gin", table.c.terms, postgresql_using="gin")
    Index(
        f"{name}_metadata_gin",
        table.c.metadata,
        postgresql_using="gin",
        postgresql_ops={"metadata": "jsonb_path_ops"},
    )
    return table


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------


def compile_filter(table: Table, flt: dict | None) -> ColumnElement[bool]:
    """
    A Pinecone-style metadata filter as a SQL condition on `table.metadata`.

    Semantics match InMemoryVectorStore: `$ne` and `$nin` match a row that
    lacks the field, range operators do not.
    """
    if not flt:
        return true()
    meta = table.c.metadata
    clauses: list[ColumnElement[bool]] = []
    for key, condition in flt.items():
        if key == "$and":
            clauses.append(and_(*(compile_filter(table, c) for c in condition)))
            continue
        if key == "$or":
            clauses.append(or_(*(compile_filter(table, c) for c in condition)))
            continue
        if not isinstance(condition, dict):
            condition = {"$eq": condition}
        for op, operand in condition.items():
            if op == "$eq":
                clauses.append(meta.contains({key: operand}))
            elif op == "$ne":
                clauses.append(not_(meta.contains({key: operand})))
            elif op == "$in":
                clauses.append(or_(*(meta.contains({key: v}) for v in operand)) if operand else false())
            elif op == "$nin":
                clauses.append(and_(*(not_(meta.contains({key: v})) for v in operand)))
            elif op in _RANGE_OPS:
                clauses.append(getattr(meta[key].as_float(), _RANGE_OPS[op])(operand))
            else:
                raise ValueError(f"unsupported metadata filter operator {op!r}")
    return and_(*clauses)


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class PgVectorStore(VectorStore):
    """
    pgvector-backed vector store.

    Args:
        engine: The SQLAlchemy engine. Defaults to the app's shared one.
        index_name: Index version to use. Omit to follow `pointer`.
        pointer: Version pointer for blue/green index selection.
        dimension: Embedding dimension of the configured model. A table built
            with a different one is refused rather than queried.
    """

    def __init__(
        self,
        *,
        engine: Engine | None = None,
        index_name: str | None = None,
        pointer: IndexPointer | None = None,
        dimension: int = 384,
    ) -> None:
        if not index_name and not pointer:
            raise ValueError("provide index_name or pointer")
        from edupilot.db import get_engine

        self.engine = engine or get_engine()
        self._explicit_name = index_name
        self._pointer = pointer
        self.dimension = dimension
        self._tables: dict[str, Table] = {}
        self._ready: set[str] = set()
        self._lock = threading.Lock()
        self._iterative_scan: bool | None = None

    # -- tables ------------------------------------------------------------

    @property
    def index_name(self) -> str:
        if self._explicit_name:
            return self._explicit_name
        assert self._pointer is not None
        active = self._pointer.active
        if not active:
            # Nothing promoted yet; the first rebuild will create v1.
            return self._pointer.next_version_name()[1]
        return active

    def _table(self, name: str) -> Table:
        table = self._tables.get(name)
        if table is None:
            table = self._tables[name] = define_table(name, self.dimension)
        return table

    def _existing_dimension(self, conn, name: str) -> int | None:
        """Dimension of the `embedding` column, or None when the table does not exist."""
        declared = conn.execute(
            text(
                "SELECT format_type(a.atttypid, a.atttypmod) FROM pg_attribute a "
                "WHERE a.attrelid = to_regclass(:t) AND a.attname = 'embedding'"
            ),
            {"t": table_name(name)},
        ).scalar()
        if declared is None:
            return None
        match = re.search(r"\((\d+)\)", declared)
        return int(match.group(1)) if match else None

    def _check(self, conn, name: str, *, create: bool) -> bool:
        """
        Make sure `name`'s table is usable: right dimension, and created if asked.

        Returns False when the table does not exist and `create` is off, which
        reads treat as an empty index. Checked once per table per process.
        """
        if name in self._ready:
            return True
        existing = self._existing_dimension(conn, name)
        if existing is None and not create:
            return False
        if existing is not None and existing != self.dimension:
            raise VectorStoreError(
                f"index {name} holds {existing}-dimensional vectors, but the configured "
                f"embedding model produces {self.dimension}. EMBEDDING_MODEL must match the "
                "model the index was built with: switch it back, or build a new index "
                "version with `edupilot-reindex --rebuild` and promote it."
            )
        if existing is None:
            # Two processes may write to a new index at once (two students'
            # first uploads); the lock makes one create it and the other wait.
            # Not marked ready: if this transaction rolls back, so does the
            # CREATE, and the next call has to find that out.
            conn.execute(text("SELECT pg_advisory_xact_lock(hashtext(:t))"), {"t": table_name(name)})
            self._table(name).create(conn, checkfirst=True)
            logger.info("created vector table %s (dim=%d)", table_name(name), self.dimension)
            return True
        with self._lock:
            self._ready.add(name)
        return True

    def ensure_index(self, name: str | None = None) -> None:
        """Create the table for `name` (default: the current index) if absent."""
        with self.engine.begin() as conn:
            self._check(conn, name or self.index_name, create=True)

    def exists(self, name: str | None = None) -> bool:
        with self.engine.connect() as conn:
            return self._existing_dimension(conn, name or self.index_name) is not None

    def drop_index(self, name: str) -> None:
        """Drop an index version's table. The caller must not drop the live one."""
        with self.engine.begin() as conn:
            conn.execute(text(f'DROP TABLE IF EXISTS "{table_name(name)}"'))
        with self._lock:
            self._ready.discard(name)
        logger.info("dropped vector table %s", table_name(name))

    def copy_namespaces(self, source: str, *, prefix: str) -> int:
        """
        Copy every namespace starting with `prefix` from index `source` into this one.

        A rebuild re-indexes the course corpus from its files, but students'
        study uploads exist only in the index they were written to. Copying
        them keeps every study session searchable across a promotion. Needs
        both versions to share the embedding dimension. Returns rows copied.
        """
        target = self.index_name
        with self.engine.begin() as conn:
            source_dim = self._existing_dimension(conn, source)
            if source_dim is None:
                return 0
            if source_dim != self.dimension:
                logger.warning(
                    "not copying %s* from %s: %d-dimensional vectors, the new index is %d",
                    prefix, source, source_dim, self.dimension,
                )
                return 0
            self._check(conn, target, create=True)
            src, dst = self._table(source), self._table(target)
            statement = pg_insert(dst).from_select(
                [c.name for c in dst.c],
                select(*src.c).where(src.c.namespace.startswith(prefix, autoescape=True)),
            ).on_conflict_do_nothing()
            # SQLAlchemy reports an INSERT's row count only when asked to.
            copied = conn.execute(statement.execution_options(preserve_rowcount=True)).rowcount
        logger.info("copied %d vectors (%s*) from %s into %s", copied, prefix, source, target)
        return copied

    # -- writes ------------------------------------------------------------

    @staticmethod
    def _row(record: VectorRecord, namespace: str) -> dict[str, Any]:
        sparse = {int(i): float(v) for i, v in (record.sparse or {}).items() if v}
        return {
            "namespace": namespace,
            "id": record.id,
            "embedding": [float(v) for v in record.values],
            "sparse": SparseVector(sparse, SPARSE_DIMENSIONS) if sparse else None,
            "terms": sorted(sparse) if sparse else None,
            "metadata": sanitize_metadata(record.metadata, max_bytes=None),
        }

    def upsert(self, records: Sequence[VectorRecord], namespace: str) -> int:
        if not records:
            return 0
        name = self.index_name
        table = self._table(name)
        # Last write wins for a repeated id, as with Pinecone; Postgres would
        # otherwise reject an upsert that touches one row twice.
        rows = list({r.id: self._row(r, namespace) for r in records}.values())
        statement = pg_insert(table)
        statement = statement.on_conflict_do_update(
            index_elements=["namespace", "id"],
            set_={c: statement.excluded[c] for c in ("embedding", "sparse", "terms", "metadata")},
        )
        with self.engine.begin() as conn:
            self._check(conn, name, create=True)
            for i in range(0, len(rows), UPSERT_BATCH):
                conn.execute(statement, rows[i : i + UPSERT_BATCH])
        logger.debug("upserted %d vectors into %s/%s", len(rows), name, namespace)
        return len(rows)

    def delete(
        self,
        namespace: str,
        *,
        ids: Sequence[str] | None = None,
        delete_all: bool = False,
        metadata_filter: dict | None = None,
    ) -> None:
        name = self.index_name
        table = self._table(name)
        if delete_all:
            condition = true()
        elif metadata_filter:
            condition = compile_filter(table, metadata_filter)
        elif ids:
            condition = table.c.id.in_(list(ids))
        else:
            return
        with self.engine.begin() as conn:
            if not self._check(conn, name, create=False):
                return
            conn.execute(delete(table).where(table.c.namespace == namespace, condition))

    # -- reads -------------------------------------------------------------

    def _supports_iterative_scan(self, conn) -> bool:
        """pgvector 0.8 added filtered HNSW scans that keep going until k rows match."""
        if self._iterative_scan is None:
            version = conn.execute(
                text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
            ).scalar() or "0"
            major, minor = (int(p) for p in (version.split(".") + ["0"])[:2])
            self._iterative_scan = (major, minor) >= (0, 8)
            if not self._iterative_scan:
                logger.warning(
                    "pgvector %s has no iterative index scans; filtered searches may "
                    "return fewer than k results. Upgrade to 0.8 or newer.",
                    version,
                )
        return self._iterative_scan

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
        name = self.index_name
        table = self._table(name)
        k = max(1, top_k)
        where = and_(table.c.namespace == namespace, compile_filter(table, metadata_filter))
        extra = [table.c.embedding] if include_values else []

        query_vector = [float(v) for v in vector]
        distance = table.c.embedding.max_inner_product(query_vector)
        dense_query = (
            select(table.c.id, table.c.metadata, (-distance).label("score"), *extra)
            .where(where)
            .order_by(distance)
            .limit(k)
        )

        weights = {int(i): float(v) for i, v in (sparse or {}).items() if v}
        lexical_query = None
        if weights:
            query_sparse = SparseVector(weights, SPARSE_DIMENSIONS)
            lexical_distance = table.c.sparse.max_inner_product(query_sparse)
            lexical_query = (
                select(table.c.id, table.c.metadata, (-lexical_distance).label("score"), *extra)
                .where(where, table.c.terms.overlap(sorted(weights)))
                .order_by(lexical_distance)
                .limit(k)
            )

        with self.engine.begin() as conn:
            if not self._check(conn, name, create=False):
                return []
            # SET LOCAL lasts until this transaction ends, so a pooled
            # connection goes back to the pool with the defaults.
            conn.execute(text(f"SET LOCAL hnsw.ef_search = {max(MIN_EF_SEARCH, 2 * k)}"))
            if self._supports_iterative_scan(conn):
                conn.execute(text("SET LOCAL hnsw.iterative_scan = relaxed_order"))
            dense = conn.execute(dense_query).all()
            lexical = conn.execute(lexical_query).all() if lexical_query is not None else []

        def hit(row, score: float) -> SearchHit:
            values = [float(v) for v in row.embedding] if include_values else None
            return SearchHit(id=row.id, score=score, metadata=dict(row.metadata), values=values)

        # relaxed_order can return near-ties slightly out of order; the
        # scores are exact, so restore the order before ranking.
        dense = sorted(dense, key=lambda r: r.score, reverse=True)
        if not lexical:
            return [hit(r, float(r.score)) for r in dense]

        lexical = sorted(lexical, key=lambda r: r.score, reverse=True)
        fused: dict[str, float] = {}
        rows: dict[str, Any] = {}
        for ranked in (dense, lexical):
            for rank, row in enumerate(ranked, start=1):
                fused[row.id] = fused.get(row.id, 0.0) + 1.0 / (RRF_K + rank)
                rows.setdefault(row.id, row)
        best = sorted(fused, key=fused.__getitem__, reverse=True)[:k]
        return [hit(rows[i], fused[i]) for i in best]

    def fetch(self, ids: Sequence[str], namespace: str) -> dict[str, SearchHit]:
        if not ids:
            return {}
        name = self.index_name
        table = self._table(name)
        with self.engine.connect() as conn:
            if not self._check(conn, name, create=False):
                return {}
            rows = conn.execute(
                select(table.c.id, table.c.metadata).where(
                    table.c.namespace == namespace, table.c.id.in_(list(dict.fromkeys(ids)))
                )
            ).all()
        return {r.id: SearchHit(id=r.id, score=1.0, metadata=dict(r.metadata)) for r in rows}

    def document_titles(self, namespace: str) -> dict[str, str]:
        name = self.index_name
        table = self._table(name)
        filename = table.c.metadata["filename"].astext
        title = table.c.metadata["doc_title"].astext
        with self.engine.connect() as conn:
            if not self._check(conn, name, create=False):
                return {}
            rows = conn.execute(
                select(filename, func.max(title))
                .where(table.c.namespace == namespace, filename.isnot(None))
                .group_by(filename)
            ).all()
        return {f: t for f, t in rows if t}

    def stats(self) -> dict:
        name = self.index_name
        table = self._table(name)
        with self.engine.connect() as conn:
            # An index nothing has been written to yet reads as empty, so the
            # Knowledge Base page works on a fresh install.
            exists = self._check(conn, name, create=False)
            rows = (
                conn.execute(select(table.c.namespace, func.count()).group_by(table.c.namespace)).all()
                if exists
                else []
            )
        namespaces = {ns: int(n) for ns, n in rows}
        return {
            "index": name,
            "table": table_name(name),
            "exists": exists,
            "dimension": self.dimension,
            "total_vectors": sum(namespaces.values()),
            "namespaces": namespaces,
        }
