"""ingestion jobs

Uploads are indexed by a worker instead of inside the request: a queue table
for the jobs, and a status on study documents while they wait.

Revision ID: 0002
Revises: 0001
Create Date: 2026-11-02
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0002"
down_revision: str | Sequence[str] | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table('ingest_jobs',
    sa.Column('id', sa.BigInteger(), sa.Identity(always=False), nullable=False),
    sa.Column('kind', sa.Text(), nullable=False),
    sa.Column('status', sa.Text(), server_default='queued', nullable=False),
    sa.Column('user_id', sa.Text(), nullable=False),
    sa.Column('namespace', sa.Text(), nullable=False),
    sa.Column('domain', sa.Text(), nullable=False),
    sa.Column('filename', sa.Text(), nullable=False),
    sa.Column('file_type', sa.Text(), nullable=True),
    sa.Column('size_bytes', sa.BigInteger(), nullable=True),
    sa.Column('ss_session_id', sa.Text(), nullable=True),
    sa.Column('document_id', sa.BigInteger(), nullable=True),
    sa.Column('content', sa.LargeBinary(), nullable=True),
    sa.Column('result', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('error', sa.Text(), nullable=True),
    sa.Column('attempts', sa.Integer(), server_default='0', nullable=False),
    sa.Column('created_at', postgresql.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('started_at', postgresql.TIMESTAMP(timezone=True), nullable=True),
    sa.Column('finished_at', postgresql.TIMESTAMP(timezone=True), nullable=True),
    sa.CheckConstraint("kind IN ('study', 'course')", name=op.f('ck_ingest_jobs_kind')),
    sa.CheckConstraint("status IN ('queued', 'running', 'done', 'failed')", name=op.f('ck_ingest_jobs_status')),
    sa.ForeignKeyConstraint(['document_id'], ['self_study_documents.id'], name=op.f('fk_ingest_jobs_document_id_self_study_documents'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['ss_session_id'], ['self_study_sessions.ss_session_id'], name=op.f('fk_ingest_jobs_ss_session_id_self_study_sessions'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_ingest_jobs'))
    )
    op.create_index('ix_ingest_jobs_queued', 'ingest_jobs', ['id'], unique=False, postgresql_where=sa.text("status = 'queued'"))
    op.add_column('self_study_documents', sa.Column('status', sa.Text(), server_default='ready', nullable=False))
    op.add_column('self_study_documents', sa.Column('error', sa.Text(), nullable=True))
    op.create_check_constraint(
        op.f('ck_self_study_documents_status'),
        'self_study_documents',
        "status IN ('queued', 'indexing', 'ready', 'failed')",
    )


def downgrade() -> None:
    op.drop_constraint(op.f('ck_self_study_documents_status'), 'self_study_documents', type_='check')
    op.drop_column('self_study_documents', 'error')
    op.drop_column('self_study_documents', 'status')
    op.drop_index('ix_ingest_jobs_queued', table_name='ingest_jobs', postgresql_where=sa.text("status = 'queued'"))
    op.drop_table('ingest_jobs')
