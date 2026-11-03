"""document source type

Whether an indexed document is the course's own material or a textbook, so
retrieval can prefer lectures and cite books as books.

Revision ID: 0003
Revises: 0002
Create Date: 2026-11-02
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: str | Sequence[str] | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "indexed_documents",
        sa.Column("source_type", sa.Text(), server_default="lecture", nullable=False),
    )


def downgrade() -> None:
    op.drop_column("indexed_documents", "source_type")
