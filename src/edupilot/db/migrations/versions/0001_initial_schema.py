"""initial schema

Every table in db/tables.py, as of the move from SQLite to Postgres.

Revision ID: 0001
Revises:
Create Date: 2026-11-02
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import VECTOR
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # pgvector: answer_cache.embedding here, and the per-version vector tables
    # the vector store creates at runtime.
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table('answer_cache',
    sa.Column('id', sa.BigInteger(), sa.Identity(always=False), nullable=False),
    sa.Column('norm_query', sa.Text(), nullable=False),
    sa.Column('scope', sa.Text(), nullable=False),
    sa.Column('ask_count', sa.Integer(), server_default='1', nullable=False),
    sa.Column('embedding', VECTOR(), nullable=True),
    sa.Column('payload', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('index_version', sa.Text(), nullable=True),
    sa.Column('created_at', sa.Float(), nullable=False),
    sa.Column('last_hit_at', sa.Float(), nullable=True),
    sa.Column('hit_count', sa.Integer(), server_default='0', nullable=False),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_answer_cache')),
    sa.UniqueConstraint('norm_query', 'scope', name=op.f('uq_answer_cache_norm_query_scope'))
    )
    op.create_index('ix_answer_cache_scope_created', 'answer_cache', ['scope', 'created_at'], unique=False)
    op.create_table('answer_feedback',
    sa.Column('id', sa.BigInteger(), sa.Identity(always=False), nullable=False),
    sa.Column('kind', sa.Text(), nullable=False),
    sa.Column('message_id', sa.BigInteger(), nullable=False),
    sa.Column('user_id', sa.Text(), nullable=False),
    sa.Column('rating', sa.Integer(), nullable=False),
    sa.Column('comment', sa.Text(), nullable=True),
    sa.Column('run_id', sa.Text(), nullable=True),
    sa.Column('created_at', postgresql.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint("kind IN ('chat', 'self_study')", name=op.f('ck_answer_feedback_kind')),
    sa.CheckConstraint('rating IN (-1, 1)', name=op.f('ck_answer_feedback_rating')),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_answer_feedback')),
    sa.UniqueConstraint('kind', 'message_id', 'user_id', name=op.f('uq_answer_feedback_kind_message_id_user_id'))
    )
    op.create_table('chat_sessions',
    sa.Column('id', sa.BigInteger(), sa.Identity(always=False), nullable=False),
    sa.Column('session_id', sa.Text(), nullable=False),
    sa.Column('user_id', sa.Text(), nullable=False),
    sa.Column('title', sa.Text(), nullable=True),
    sa.Column('summary', sa.Text(), nullable=True),
    sa.Column('summary_through_id', sa.BigInteger(), nullable=True),
    sa.Column('created_at', postgresql.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', postgresql.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_chat_sessions')),
    sa.UniqueConstraint('session_id', name=op.f('uq_chat_sessions_session_id'))
    )
    op.create_index('ix_chat_sessions_user_updated', 'chat_sessions', ['user_id', sa.literal_column('updated_at DESC')], unique=False)
    op.create_table('embedding_cache',
    sa.Column('key', sa.Text(), nullable=False),
    sa.Column('model', sa.Text(), nullable=False),
    sa.Column('dim', sa.Integer(), nullable=False),
    sa.Column('vec', sa.LargeBinary(), nullable=False),
    sa.Column('created_at', postgresql.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.PrimaryKeyConstraint('key', name=op.f('pk_embedding_cache'))
    )
    op.create_table('index_versions',
    sa.Column('version', sa.Integer(), autoincrement=False, nullable=False),
    sa.Column('name', sa.Text(), nullable=False),
    sa.Column('backend', sa.Text(), nullable=False),
    sa.Column('embed_model', sa.Text(), server_default='', nullable=False),
    sa.Column('dimension', sa.Integer(), nullable=True),
    sa.Column('status', sa.Text(), server_default='building', nullable=False),
    sa.Column('note', sa.Text(), server_default='', nullable=False),
    sa.Column('sparse_encoder', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('created_at', postgresql.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('promoted_at', postgresql.TIMESTAMP(timezone=True), nullable=True),
    sa.Column('retired_at', postgresql.TIMESTAMP(timezone=True), nullable=True),
    sa.CheckConstraint("status IN ('building', 'active', 'retired')", name=op.f('ck_index_versions_status')),
    sa.PrimaryKeyConstraint('version', name=op.f('pk_index_versions')),
    sa.UniqueConstraint('name', name=op.f('uq_index_versions_name'))
    )
    op.create_index('uq_index_versions_one_active', 'index_versions', ['status'], unique=True, postgresql_where=sa.text("status = 'active'"))
    op.create_table('indexed_documents',
    sa.Column('id', sa.BigInteger(), sa.Identity(always=False), nullable=False),
    sa.Column('namespace', sa.Text(), nullable=False),
    sa.Column('filename', sa.Text(), nullable=False),
    sa.Column('source_path', sa.Text(), nullable=False),
    sa.Column('content_hash', sa.Text(), nullable=False),
    sa.Column('chunk_count', sa.Integer(), server_default='0', nullable=False),
    sa.Column('parent_count', sa.Integer(), server_default='0', nullable=False),
    sa.Column('embed_model', sa.Text(), server_default='', nullable=False),
    sa.Column('chunker', sa.Text(), server_default='', nullable=False),
    sa.Column('indexed_at', postgresql.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_indexed_documents')),
    sa.UniqueConstraint('namespace', 'filename', name=op.f('uq_indexed_documents_namespace_filename'))
    )
    op.create_table('self_study_sessions',
    sa.Column('id', sa.BigInteger(), sa.Identity(always=False), nullable=False),
    sa.Column('ss_session_id', sa.Text(), nullable=False),
    sa.Column('user_id', sa.Text(), nullable=False),
    sa.Column('name', sa.Text(), nullable=False),
    sa.Column('description', sa.Text(), nullable=True),
    sa.Column('created_at', postgresql.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', postgresql.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_self_study_sessions')),
    sa.UniqueConstraint('ss_session_id', name=op.f('uq_self_study_sessions_ss_session_id'))
    )
    op.create_index('ix_self_study_sessions_user_updated', 'self_study_sessions', ['user_id', sa.literal_column('updated_at DESC')], unique=False)
    op.create_table('uploaded_documents',
    sa.Column('id', sa.BigInteger(), sa.Identity(always=False), nullable=False),
    sa.Column('filename', sa.Text(), nullable=False),
    sa.Column('domain', sa.Text(), nullable=False),
    sa.Column('upload_timestamp', postgresql.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('file_type', sa.Text(), nullable=True),
    sa.Column('chunk_count', sa.Integer(), server_default='0', nullable=False),
    sa.Column('file_size_bytes', sa.BigInteger(), nullable=True),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_uploaded_documents'))
    )
    op.create_index(op.f('ix_uploaded_documents_domain'), 'uploaded_documents', ['domain'], unique=False)
    op.create_table('users',
    sa.Column('user_id', sa.Text(), nullable=False),
    sa.Column('email', sa.Text(), nullable=False),
    sa.Column('password_hash', sa.Text(), nullable=False),
    sa.Column('role', sa.Text(), server_default='student', nullable=False),
    sa.Column('display_name', sa.Text(), server_default='', nullable=False),
    sa.Column('created_at', postgresql.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('last_login_at', postgresql.TIMESTAMP(timezone=True), nullable=True),
    sa.Column('is_active', sa.Boolean(), server_default=sa.text('true'), nullable=False),
    sa.PrimaryKeyConstraint('user_id', name=op.f('pk_users')),
    sa.UniqueConstraint('email', name=op.f('uq_users_email'))
    )
    op.create_table('chat_messages',
    sa.Column('id', sa.BigInteger(), sa.Identity(always=False), nullable=False),
    sa.Column('session_id', sa.Text(), nullable=False),
    sa.Column('role', sa.Text(), nullable=False),
    sa.Column('content', sa.Text(), nullable=False),
    sa.Column('timestamp', postgresql.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('intent_type', sa.Text(), nullable=True),
    sa.Column('detected_domains', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('quality_score', sa.Float(), nullable=True),
    sa.Column('pipeline_meta', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.CheckConstraint("role IN ('user', 'assistant')", name=op.f('ck_chat_messages_role')),
    sa.ForeignKeyConstraint(['session_id'], ['chat_sessions.session_id'], name=op.f('fk_chat_messages_session_id_chat_sessions'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_chat_messages'))
    )
    op.create_index('ix_chat_messages_session', 'chat_messages', ['session_id', 'id'], unique=False)
    op.create_table('refresh_tokens',
    sa.Column('token_hash', sa.Text(), nullable=False),
    sa.Column('user_id', sa.Text(), nullable=False),
    sa.Column('expires_at', postgresql.TIMESTAMP(timezone=True), nullable=False),
    sa.Column('created_at', postgresql.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('revoked', sa.Boolean(), server_default=sa.text('false'), nullable=False),
    sa.ForeignKeyConstraint(['user_id'], ['users.user_id'], name=op.f('fk_refresh_tokens_user_id_users'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('token_hash', name=op.f('pk_refresh_tokens'))
    )
    op.create_index(op.f('ix_refresh_tokens_user_id'), 'refresh_tokens', ['user_id'], unique=False)
    op.create_table('self_study_documents',
    sa.Column('id', sa.BigInteger(), sa.Identity(always=False), nullable=False),
    sa.Column('ss_session_id', sa.Text(), nullable=False),
    sa.Column('filename', sa.Text(), nullable=False),
    sa.Column('file_type', sa.Text(), nullable=True),
    sa.Column('file_size_bytes', sa.BigInteger(), nullable=True),
    sa.Column('chunk_count', sa.Integer(), server_default='0', nullable=False),
    sa.Column('upload_timestamp', postgresql.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['ss_session_id'], ['self_study_sessions.ss_session_id'], name=op.f('fk_self_study_documents_ss_session_id_self_study_sessions'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_self_study_documents'))
    )
    op.create_index(op.f('ix_self_study_documents_ss_session_id'), 'self_study_documents', ['ss_session_id'], unique=False)
    op.create_table('self_study_messages',
    sa.Column('id', sa.BigInteger(), sa.Identity(always=False), nullable=False),
    sa.Column('ss_session_id', sa.Text(), nullable=False),
    sa.Column('role', sa.Text(), nullable=False),
    sa.Column('content', sa.Text(), nullable=False),
    sa.Column('timestamp', postgresql.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('quality_score', sa.Float(), nullable=True),
    sa.Column('pipeline_meta', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.CheckConstraint("role IN ('user', 'assistant')", name=op.f('ck_self_study_messages_role')),
    sa.ForeignKeyConstraint(['ss_session_id'], ['self_study_sessions.ss_session_id'], name=op.f('fk_self_study_messages_ss_session_id_self_study_sessions'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_self_study_messages'))
    )
    op.create_index('ix_self_study_messages_session', 'self_study_messages', ['ss_session_id', 'id'], unique=False)


def downgrade() -> None:
    op.drop_index('ix_self_study_messages_session', table_name='self_study_messages')
    op.drop_table('self_study_messages')
    op.drop_index(op.f('ix_self_study_documents_ss_session_id'), table_name='self_study_documents')
    op.drop_table('self_study_documents')
    op.drop_index(op.f('ix_refresh_tokens_user_id'), table_name='refresh_tokens')
    op.drop_table('refresh_tokens')
    op.drop_index('ix_chat_messages_session', table_name='chat_messages')
    op.drop_table('chat_messages')
    op.drop_table('users')
    op.drop_index(op.f('ix_uploaded_documents_domain'), table_name='uploaded_documents')
    op.drop_table('uploaded_documents')
    op.drop_index('ix_self_study_sessions_user_updated', table_name='self_study_sessions')
    op.drop_table('self_study_sessions')
    op.drop_table('indexed_documents')
    op.drop_index('uq_index_versions_one_active', table_name='index_versions', postgresql_where=sa.text("status = 'active'"))
    op.drop_table('index_versions')
    op.drop_table('embedding_cache')
    op.drop_index('ix_chat_sessions_user_updated', table_name='chat_sessions')
    op.drop_table('chat_sessions')
    op.drop_table('answer_feedback')
    op.drop_index('ix_answer_cache_scope_created', table_name='answer_cache')
    op.drop_table('answer_cache')
