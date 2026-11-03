"""
LangChain retriever
===================
`HybridRetriever` behind LangChain's `BaseRetriever` interface.

The retrieval stack itself is unchanged: engine-side sparse+dense search, RRF
across probes, cross-encoder reranking with a calibrated floor, and
small-to-big parent expansion. LangChain's stock retrievers do less, so this
adapter wraps it rather than replacing it. What the wrapper adds is the
standard interface: results come back as `Document`s carrying their relevance
and citation metadata, `ainvoke` works from the async agent graph, and the
same object can later be handed to a tool-calling agent as-is.

`search` / `asearch` also return the retrieval diagnostics the debug panel
renders, which the `BaseRetriever` contract has no slot for.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field

from .hybrid import HybridRetriever, RetrievalConfig
from .rerank import RerankedChunk


@dataclass
class RetrievalOutcome:
    """Retrieved documents plus the diagnostics the debug panel renders."""

    documents: list[Document] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return not self.documents


def to_document(chunk: RerankedChunk) -> Document:
    """A reranked chunk as a `Document`, with its scores folded into metadata."""
    return Document(
        id=chunk.chunk_id,
        page_content=chunk.text,
        metadata={
            **chunk.metadata,
            "chunk_id": chunk.chunk_id,
            "relevance": chunk.relevance,
            "retrieval_score": chunk.retrieval_score,
        },
    )


class EduPilotRetriever(BaseRetriever):
    """
    One namespace's hybrid retriever as a LangChain retriever.

    An empty result is meaningful, not an error: nothing in the namespace
    cleared the relevance floor, and the caller must refuse rather than answer.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    retriever: HybridRetriever
    config: RetrievalConfig = Field(default_factory=RetrievalConfig)
    filenames: list[str] | None = None
    """Restrict the search to these source files (pushed down as a filter)."""
    query_llm: Callable[[str, str], str] | None = None
    """`(system, user) -> text`, used only when HyDE or multi-query is enabled."""

    def search(self, query: str) -> RetrievalOutcome:
        result = self.retriever.retrieve(
            query, config=self.config, filenames=self.filenames, llm=self.query_llm
        )
        return RetrievalOutcome(
            documents=[to_document(c) for c in result.chunks],
            diagnostics=result.diagnostics,
        )

    async def asearch(self, query: str) -> RetrievalOutcome:
        # The vector-store client and the cross-encoder are synchronous; a
        # worker thread keeps them off the event loop so other requests keep
        # moving while this one searches.
        return await asyncio.to_thread(self.search, query)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        return self.search(query).documents

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> list[Document]:
        return (await self.asearch(query)).documents
