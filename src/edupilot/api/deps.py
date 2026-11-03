"""
Shared route dependencies
=========================
The pieces every router needs: authenticated-user annotations, the worker pool
that keeps blocking pipeline work off the event loop, and the two small
validators that would otherwise be copy-pasted across modules.
"""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated

from fastapi import Depends

from edupilot.core.config import (
    AVAILABLE_MODELS,
    ENABLE_HYDE,
    ENABLE_MULTI_QUERY,
    ENABLE_PARENT_EXPANSION,
    MAX_BOOK_EXCERPTS,
    MAX_EXCERPTS_PER_DOCUMENT,
)
from edupilot.core.observability import request_id_var
from edupilot.security import AppError, ErrorCode, User, admin_user, current_user

#: Blocking work the async routes still do — the answer cache's embeddings,
#: indexing uploads, password hashing — is dispatched
#: here so it never stalls the event loop. (The agent graph moves its own
#: blocking steps onto threads and awaits model calls directly.)
#:
#: Created on first use rather than at import: the lifespan shuts the pool
#: down, and an app started again in the same process (the test suite does
#: this) needs a fresh one rather than a dead pool that rejects every task.
_executor: ThreadPoolExecutor | None = None
_executor_lock = threading.Lock()


def _pool() -> ThreadPoolExecutor:
    global _executor
    with _executor_lock:
        if _executor is None:
            _executor = ThreadPoolExecutor(max_workers=6, thread_name_prefix="edupilot")
        return _executor


async def run_blocking(fn, *args, **kwargs):
    """
    Run a synchronous call in the worker pool, preserving the request id.

    The context variable does not cross a thread boundary on its own, so it is
    re-set inside the worker; without this every log line emitted from the
    pipeline would be attributed to no request at all.
    """
    rid = request_id_var.get()

    def wrapped():
        token = request_id_var.set(rid)
        try:
            return fn(*args, **kwargs)
        finally:
            request_id_var.reset(token)

    return await asyncio.get_running_loop().run_in_executor(_pool(), wrapped)


def shutdown_executor() -> None:
    """Release the worker pool. Called from the app's lifespan teardown."""
    global _executor
    with _executor_lock:
        if _executor is not None:
            _executor.shutdown(wait=False)
            _executor = None


CurrentUser = Annotated[User, Depends(current_user)]
AdminUser = Annotated[User, Depends(admin_user)]


def validate_model(name: str) -> str:
    """Reject unknown model names rather than passing them to a provider."""
    if name not in AVAILABLE_MODELS:
        raise AppError(
            code=ErrorCode.VALIDATION_FAILED,
            message=f"Unknown model '{name}'.",
            details={"available": AVAILABLE_MODELS},
        )
    return name


def retrieval_config(top_k: int, rerank_top_k: int):
    """Build a RetrievalConfig from the per-request knobs plus global feature flags."""
    from edupilot.retrieval import RetrievalConfig

    return RetrievalConfig(
        top_k=rerank_top_k,
        candidate_multiplier=max(3, top_k // 2),
        use_multi_query=ENABLE_MULTI_QUERY,
        use_hyde=ENABLE_HYDE,
        expand_to_parents=ENABLE_PARENT_EXPANSION,
        max_book_excerpts=MAX_BOOK_EXCERPTS,
        max_excerpts_per_document=MAX_EXCERPTS_PER_DOCUMENT,
    )


__all__ = [
    "AdminUser",
    "CurrentUser",
    "retrieval_config",
    "run_blocking",
    "shutdown_executor",
    "validate_model",
]
