"""
Operational database
====================
Postgres-backed persistence for conversations, study sessions, uploads and
feedback. Import the package, not its modules::

    from edupilot import db

    db.init_db()
    db.save_message(session_id, "user", text)

The split mirrors the concerns in the schema: `chat` (conversations),
`documents` (uploads to the shared course corpus), `self_study` (private
per-student sessions) and `feedback`. `engine` owns the connection pool,
`tables` the table definitions, and `schema` the Alembic migrations.
"""

from .chat import (
    delete_messages_from,
    delete_session,
    ensure_session,
    get_messages_after,
    get_session_memory,
    get_session_messages,
    get_session_owner,
    list_sessions,
    save_message,
    set_session_memory,
    update_session_title,
)
from .documents import list_uploaded_docs, save_uploaded_doc
from .engine import dispose_engine, get_engine, ping, transaction
from .feedback import get_rated_message, save_feedback
from .schema import current_revision, head_revision, init_db
from .self_study import (
    create_ss_session,
    delete_ss_document_record,
    delete_ss_session,
    get_ss_document,
    get_ss_messages,
    get_ss_session,
    get_ss_session_owner,
    list_ss_documents,
    list_ss_sessions,
    save_ss_document,
    save_ss_message,
    touch_ss_session,
)

__all__ = [
    # engine / schema
    "dispose_engine",
    "get_engine",
    "ping",
    "transaction",
    "current_revision",
    "head_revision",
    "init_db",
    # chat
    "delete_messages_from",
    "delete_session",
    "ensure_session",
    "get_messages_after",
    "get_session_memory",
    "get_session_messages",
    "get_session_owner",
    "set_session_memory",
    "list_sessions",
    "save_message",
    "update_session_title",
    # feedback
    "get_rated_message",
    "save_feedback",
    # documents
    "list_uploaded_docs",
    "save_uploaded_doc",
    # self study
    "create_ss_session",
    "delete_ss_document_record",
    "delete_ss_session",
    "get_ss_document",
    "get_ss_messages",
    "get_ss_session",
    "get_ss_session_owner",
    "list_ss_documents",
    "list_ss_sessions",
    "save_ss_document",
    "save_ss_message",
    "touch_ss_session",
]
