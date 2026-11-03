"""
Tables
======
Every table the app owns, as SQLAlchemy Core definitions. Alembic migrations
(db/migrations) create and change them; this module is what the queries are
written against and what `--autogenerate` diffs.

  • users, refresh_tokens       — accounts (security/auth.py)
  • chat_sessions, chat_messages — course chat, owned by a user
  • self_study_*                 — each student's private sessions and uploads
  • uploaded_documents           — admin uploads to the course corpus
  • answer_feedback              — students' 👍/👎 on answers
  • ingest_jobs                  — uploads waiting for, or done with, indexing
  • indexed_documents            — which file is indexed, at which hash, where
  • index_versions               — vector index versions and which one is live
  • embedding_cache              — passage embeddings by (model, text)
  • answer_cache                 — answers to frequently asked questions

Not here, because they are created at runtime rather than by a migration: one
vector table per index version (retrieval/pgvector.py), and LangGraph's
checkpoint and store tables (agents/checkpoint.py).

`user_id` columns on sessions carry no foreign key on purpose: in open mode an
anonymous browser identity owns its sessions without a `users` row.
"""

from __future__ import annotations

from pgvector.sqlalchemy import VECTOR
from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Column,
    Float,
    ForeignKey,
    Identity,
    Index,
    Integer,
    LargeBinary,
    MetaData,
    Table,
    Text,
    UniqueConstraint,
    false,
    func,
    text,
    true,
)
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP

#: Deterministic constraint names, so migrations can refer to them.
metadata = MetaData(
    naming_convention={
        "ix": "ix_%(table_name)s_%(column_0_N_name)s",
        "uq": "uq_%(table_name)s_%(column_0_N_name)s",
        "ck": "ck_%(table_name)s_%(constraint_name)s",
        "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
        "pk": "pk_%(table_name)s",
    }
)


def _timestamp(name: str, *, default: bool = True, nullable: bool = False) -> Column:
    return Column(
        name,
        TIMESTAMP(timezone=True),
        nullable=nullable,
        server_default=func.now() if default else None,
    )


def _id() -> Column:
    return Column("id", BigInteger, Identity(), primary_key=True)


# ---------------------------------------------------------------------------
# Accounts
# ---------------------------------------------------------------------------

users = Table(
    "users",
    metadata,
    Column("user_id", Text, primary_key=True),
    Column("email", Text, nullable=False, unique=True),
    Column("password_hash", Text, nullable=False),
    Column("role", Text, nullable=False, server_default="student"),
    Column("display_name", Text, nullable=False, server_default=""),
    _timestamp("created_at"),
    _timestamp("last_login_at", default=False, nullable=True),
    Column("is_active", Boolean, nullable=False, server_default=true()),
)

refresh_tokens = Table(
    "refresh_tokens",
    metadata,
    Column("token_hash", Text, primary_key=True),
    Column(
        "user_id",
        Text,
        ForeignKey("users.user_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    ),
    _timestamp("expires_at", default=False),
    _timestamp("created_at"),
    Column("revoked", Boolean, nullable=False, server_default=false()),
)

# ---------------------------------------------------------------------------
# Course chat
# ---------------------------------------------------------------------------

chat_sessions = Table(
    "chat_sessions",
    metadata,
    _id(),
    Column("session_id", Text, nullable=False, unique=True),
    Column("user_id", Text, nullable=False),
    Column("title", Text),
    # Rolling conversation memory: older turns compacted into a digest, and
    # the last message id already folded in, so summarizing is incremental.
    Column("summary", Text),
    Column("summary_through_id", BigInteger),
    _timestamp("created_at"),
    _timestamp("updated_at"),
    Index("ix_chat_sessions_user_updated", "user_id", text("updated_at DESC")),
)

chat_messages = Table(
    "chat_messages",
    metadata,
    _id(),
    Column(
        "session_id",
        Text,
        ForeignKey("chat_sessions.session_id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("role", Text, nullable=False),
    Column("content", Text, nullable=False),
    _timestamp("timestamp"),
    Column("intent_type", Text),
    Column("detected_domains", JSONB),
    Column("quality_score", Float),
    Column("pipeline_meta", JSONB),
    CheckConstraint("role IN ('user', 'assistant')", name="role"),
    Index("ix_chat_messages_session", "session_id", "id"),
)

# ---------------------------------------------------------------------------
# Self Study
# ---------------------------------------------------------------------------

self_study_sessions = Table(
    "self_study_sessions",
    metadata,
    _id(),
    Column("ss_session_id", Text, nullable=False, unique=True),
    Column("user_id", Text, nullable=False),
    Column("name", Text, nullable=False),
    Column("description", Text),
    _timestamp("created_at"),
    _timestamp("updated_at"),
    Index("ix_self_study_sessions_user_updated", "user_id", text("updated_at DESC")),
)

self_study_documents = Table(
    "self_study_documents",
    metadata,
    _id(),
    Column(
        "ss_session_id",
        Text,
        ForeignKey("self_study_sessions.ss_session_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    ),
    Column("filename", Text, nullable=False),
    Column("file_type", Text),
    Column("file_size_bytes", BigInteger),
    Column("chunk_count", Integer, nullable=False, server_default="0"),
    _timestamp("upload_timestamp"),
    # queued -> indexing -> ready | failed, as the ingestion worker gets to it.
    Column("status", Text, nullable=False, server_default="ready"),
    Column("error", Text),
    CheckConstraint("status IN ('queued', 'indexing', 'ready', 'failed')", name="status"),
)

self_study_messages = Table(
    "self_study_messages",
    metadata,
    _id(),
    Column(
        "ss_session_id",
        Text,
        ForeignKey("self_study_sessions.ss_session_id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("role", Text, nullable=False),
    Column("content", Text, nullable=False),
    _timestamp("timestamp"),
    Column("quality_score", Float),
    Column("pipeline_meta", JSONB),
    CheckConstraint("role IN ('user', 'assistant')", name="role"),
    Index("ix_self_study_messages_session", "ss_session_id", "id"),
)

# ---------------------------------------------------------------------------
# Course corpus and feedback
# ---------------------------------------------------------------------------

uploaded_documents = Table(
    "uploaded_documents",
    metadata,
    _id(),
    Column("filename", Text, nullable=False),
    Column("domain", Text, nullable=False, index=True),
    _timestamp("upload_timestamp"),
    Column("file_type", Text),
    Column("chunk_count", Integer, nullable=False, server_default="0"),
    Column("file_size_bytes", BigInteger),
)

# A student's rating of one answer. `kind` says which messages table
# `message_id` refers to; `run_id` is the answer's LangSmith trace, if any.
answer_feedback = Table(
    "answer_feedback",
    metadata,
    _id(),
    Column("kind", Text, nullable=False),
    Column("message_id", BigInteger, nullable=False),
    Column("user_id", Text, nullable=False),
    Column("rating", Integer, nullable=False),
    Column("comment", Text),
    Column("run_id", Text),
    _timestamp("created_at"),
    CheckConstraint("kind IN ('chat', 'self_study')", name="kind"),
    CheckConstraint("rating IN (-1, 1)", name="rating"),
    UniqueConstraint("kind", "message_id", "user_id"),
)

# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

# Uploads waiting to be indexed, claimed by workers with FOR UPDATE SKIP
# LOCKED (ingestion/jobs.py). The file travels in `content` so a worker on
# another machine needs no shared disk; it is cleared once the job finishes.
ingest_jobs = Table(
    "ingest_jobs",
    metadata,
    _id(),
    Column("kind", Text, nullable=False),
    Column("status", Text, nullable=False, server_default="queued"),
    Column("user_id", Text, nullable=False),
    Column("namespace", Text, nullable=False),
    Column("domain", Text, nullable=False),
    Column("filename", Text, nullable=False),
    Column("file_type", Text),
    Column("size_bytes", BigInteger),
    Column(
        "ss_session_id",
        Text,
        ForeignKey("self_study_sessions.ss_session_id", ondelete="CASCADE"),
    ),
    Column("document_id", BigInteger, ForeignKey("self_study_documents.id", ondelete="CASCADE")),
    Column("content", LargeBinary),
    Column("result", JSONB),
    Column("error", Text),
    Column("attempts", Integer, nullable=False, server_default="0"),
    _timestamp("created_at"),
    _timestamp("started_at", default=False, nullable=True),
    _timestamp("finished_at", default=False, nullable=True),
    CheckConstraint("kind IN ('study', 'course')", name="kind"),
    CheckConstraint("status IN ('queued', 'running', 'done', 'failed')", name="status"),
    # What a worker scans for: only the queued rows, oldest first.
    Index("ix_ingest_jobs_queued", "id", postgresql_where=text("status = 'queued'")),
)

indexed_documents = Table(
    "indexed_documents",
    metadata,
    _id(),
    Column("namespace", Text, nullable=False),
    Column("filename", Text, nullable=False),
    Column("source_path", Text, nullable=False),
    Column("content_hash", Text, nullable=False),
    Column("chunk_count", Integer, nullable=False, server_default="0"),
    Column("parent_count", Integer, nullable=False, server_default="0"),
    Column("embed_model", Text, nullable=False, server_default=""),
    Column("chunker", Text, nullable=False, server_default=""),
    # "lecture" for a course's own material, "book" for a textbook (a file in
    # a book/ subfolder); retrieval prefers the former.
    Column("source_type", Text, nullable=False, server_default="lecture"),
    _timestamp("indexed_at"),
    UniqueConstraint("namespace", "filename"),
)

# One row per vector index version. `status` moves building -> active ->
# retired; at most one row is active, which is what serves traffic. The fitted
# BM25 table belongs to the version whose sparse vectors it produced.
index_versions = Table(
    "index_versions",
    metadata,
    Column("version", Integer, primary_key=True, autoincrement=False),
    Column("name", Text, nullable=False, unique=True),
    Column("backend", Text, nullable=False),
    Column("embed_model", Text, nullable=False, server_default=""),
    Column("dimension", Integer),
    Column("status", Text, nullable=False, server_default="building"),
    Column("note", Text, nullable=False, server_default=""),
    Column("sparse_encoder", JSONB),
    _timestamp("created_at"),
    _timestamp("promoted_at", default=False, nullable=True),
    _timestamp("retired_at", default=False, nullable=True),
    CheckConstraint("status IN ('building', 'active', 'retired')", name="status"),
    Index(
        "uq_index_versions_one_active",
        "status",
        unique=True,
        postgresql_where=text("status = 'active'"),
    ),
)

# ---------------------------------------------------------------------------
# Caches
# ---------------------------------------------------------------------------

# Passage embeddings keyed by sha256(model, text), so re-indexing an unchanged
# document re-embeds nothing. Raw float32 bytes: exact, and half the size of
# the JSON a vector would otherwise round-trip through.
embedding_cache = Table(
    "embedding_cache",
    metadata,
    Column("key", Text, primary_key=True),
    Column("model", Text, nullable=False),
    Column("dim", Integer, nullable=False),
    Column("vec", LargeBinary, nullable=False),
    _timestamp("created_at"),
)

# Frequency-gated answer cache (retrieval/answer_cache.py). `embedding` has no
# fixed dimension because it follows the configured embedding model; lookups
# are exact scans of one scope, which stays small by design.
answer_cache = Table(
    "answer_cache",
    metadata,
    _id(),
    Column("norm_query", Text, nullable=False),
    Column("scope", Text, nullable=False),
    Column("ask_count", Integer, nullable=False, server_default="1"),
    Column("embedding", VECTOR()),
    Column("payload", JSONB),
    Column("index_version", Text),
    Column("created_at", Float, nullable=False),
    Column("last_hit_at", Float),
    Column("hit_count", Integer, nullable=False, server_default="0"),
    UniqueConstraint("norm_query", "scope"),
    Index("ix_answer_cache_scope_created", "scope", "created_at"),
)
