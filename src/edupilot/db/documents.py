"""
Course knowledge-base uploads
=============================
A record of every file an admin uploaded to the shared corpus. What is
actually indexed, and at which content hash, is the document registry's job
(retrieval/indexer.py); this is the upload log.
"""

from __future__ import annotations

from sqlalchemy import insert, select

from .engine import get_engine, transaction
from .rows import row_dict
from .tables import uploaded_documents


def save_uploaded_doc(
    filename: str,
    domain: str,
    file_type: str,
    chunk_count: int,
    file_size_bytes: int,
) -> int:
    """Insert a row for an uploaded document. Returns the new row id."""
    with transaction() as conn:
        return conn.execute(
            insert(uploaded_documents)
            .values(
                filename=filename,
                domain=domain,
                file_type=file_type,
                chunk_count=chunk_count,
                file_size_bytes=file_size_bytes,
            )
            .returning(uploaded_documents.c.id)
        ).scalar_one()


def list_uploaded_docs(domain: str | None = None) -> list[dict]:
    """Return uploaded documents, newest first, optionally for one domain."""
    query = select(uploaded_documents).order_by(uploaded_documents.c.id.desc())
    if domain:
        query = query.where(uploaded_documents.c.domain == domain)
    with get_engine().connect() as conn:
        return [row_dict(r) for r in conn.execute(query)]
