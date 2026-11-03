"""
Test doubles for the agent graph.

`FakeModels` stands in for `edupilot.llm.build_llm`: structured-output calls
return canned schema instances (or the result of a function of the prompt),
and text calls go to a chat model that picks its reply from the system prompt
and streams it word by word — including tool calls, when tools are bound — so
streaming and the tool loop are exercised without a provider.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import RunnableLambda

from edupilot.agents.contracts import (
    EvidenceGrade,
    Flashcard,
    FlashcardsOutput,
    PlannerOutput,
    QueryRewrite,
    QuizOutput,
    QuizQuestion,
    RouterOutput,
    VerifierOutput,
)
from edupilot.core.config import DOMAINS
from edupilot.retrieval import HybridRetriever, RerankedChunk, RetrievalResult

_USAGE = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}


class ScriptedChatModel(BaseChatModel):
    """A chat model whose reply — text, or an AIMessage with tool calls — is a function of the prompt."""

    reply: Callable[[list[BaseMessage]], str | AIMessage]

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        out = self.reply(messages)
        message = out if isinstance(out, AIMessage) else AIMessage(content=out)
        message.usage_metadata = _USAGE
        message.response_metadata = {"model_name": "fake-model"}
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        out = self.reply(messages)
        if isinstance(out, AIMessage) and out.tool_calls:
            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content="",
                    tool_call_chunks=[
                        {"name": c["name"], "args": json.dumps(c["args"]), "id": c["id"], "index": i}
                        for i, c in enumerate(out.tool_calls)
                    ],
                )
            )
            return
        text = out.content if isinstance(out, AIMessage) else out
        words = str(text).split(" ")
        for i, word in enumerate(words):
            last = i == len(words) - 1
            # Like the real providers, usage arrives on the final chunk.
            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content=word + ("" if last else " "),
                    usage_metadata=_USAGE if last else None,
                    response_metadata={"model_name": "fake-model"} if last else {},
                )
            )


def answer_text(domain: str) -> str:
    return f"{domain} answer: the concept is explained here [Source 1]."


def _source_numbers(prompt: str) -> list[int]:
    return sorted({int(n) for n in re.findall(r"\[Source (\d+)\]", prompt)})


def _grade_everything(prompt: str) -> EvidenceGrade:
    return EvidenceGrade(relevant_sources=_source_numbers(prompt), sufficient=True)


DEFAULT_QUIZ = QuizOutput(
    title="Practice quiz",
    questions=[
        QuizQuestion(
            question="What does variance measure?",
            options=["Spread around the mean", "The middle value"],
            answer_index=0,
            explanation="It is the average squared deviation",
            source=1,
        ),
        QuizQuestion(
            question="Why square the deviations?",
            options=["To keep them positive", "To shrink them"],
            answer_index=0,
            explanation="Squaring stops positive and negative deviations cancelling",
            source=1,
        ),
    ],
)

DEFAULT_CARDS = FlashcardsOutput(
    title="Flashcards",
    cards=[Flashcard(front="Variance", back="Average squared deviation from the mean", source=1)],
)


class FakeModels:
    """
    Replacement for `build_llm`, recording which prompts it was given.

    Structured outputs may be given as an instance, an exception to raise, a
    function of the prompt text, or a list consumed one call at a time (the
    last item repeats).
    """

    def __init__(
        self,
        *,
        router: Any = None,
        planner: Any = None,
        verifier: Any = None,
        grader: Any = None,
        rewriter: Any = None,
        quiz: Any = None,
        flashcards: Any = None,
        synth: str = "Merged answer across domains [Source 1].",
        summary: str = "Student asked about variance.",
        study_answer: str = "Your notes say attention weighs tokens [Source 1].",
        revised: str = "Revised answer, now fully supported [Source 1].",
        calculation: str | None = None,
        answers: dict[str, str] | None = None,
    ) -> None:
        self.outputs: dict[type, Any] = {
            RouterOutput: router or RouterOutput(domains=["ALGO"], reasoning="stats question"),
            PlannerOutput: planner or PlannerOutput(),
            VerifierOutput: verifier
            or VerifierOutput(
                grounding_score=0.9, coverage_score=0.8, quality_score=0.85, is_satisfactory=True
            ),
            EvidenceGrade: grader or _grade_everything,
            QueryRewrite: rewriter or QueryRewrite(query="formal course terminology"),
            QuizOutput: quiz or DEFAULT_QUIZ,
            FlashcardsOutput: flashcards or DEFAULT_CARDS,
        }
        self.synth = synth
        self.summary = summary
        self.study_answer = study_answer
        self.revised = revised
        self.calculation = calculation
        self.answers = answers or {}
        self.prompts: list[str] = []
        self.structured_calls: list[str] = []
        self.structured_prompts: dict[str, list[str]] = {}
        self.tool_bindings: list[bool] = []

    def __call__(self, model: str, *, max_tokens: int, schema: type | None = None, tools=None):
        if schema is not None:
            return RunnableLambda(lambda prompt: self._structured(schema, prompt))
        self.tool_bindings.append(bool(tools))
        return ScriptedChatModel(reply=lambda messages: self._reply(messages, bool(tools)))

    def _structured(self, schema: type, prompt: Any):
        name = schema.__name__
        text = prompt.to_string() if hasattr(prompt, "to_string") else str(prompt)
        self.structured_calls.append(name)
        self.structured_prompts.setdefault(name, []).append(text)
        value = self.outputs[schema]
        if isinstance(value, list):
            value = value.pop(0) if len(value) > 1 else value[0]
        if isinstance(value, Exception):
            raise value
        if callable(value) and not isinstance(value, type):
            return value(text)
        return value

    def _reply(self, messages: list[BaseMessage], tools_bound: bool) -> str | AIMessage:
        system = str(messages[0].content)
        user = str(messages[1].content) if len(messages) > 1 else ""
        self.prompts.append(system + "\n" + user)
        if "merge several already-grounded" in system:
            return self.synth
        if "running summary of a tutoring conversation" in system:
            return self.summary
        if "repair an answer that an audit found problems" in system:
            return self.revised

        tool_results = [m for m in messages if isinstance(m, ToolMessage)]
        if self.calculation and tools_bound and not tool_results:
            return AIMessage(
                content="",
                tool_calls=[{"name": "calculate", "args": {"expression": self.calculation}, "id": "call-1"}],
            )
        if tool_results:
            cite = re.search(r"\[Source \d+\]", str(tool_results[-1].content))
            return f"The calculated result is in {tool_results[-1].content.splitlines()[0]} {cite.group(0) if cite else ''}."

        if "documents the student uploaded" in system:
            return self.study_answer
        for code in DOMAINS:
            if f"({code})" in system:
                return self.answers.get(code, answer_text(code))
        return "unexpected prompt"


def chunk(text: str, filename: str = "lecture.pdf", page: int = 1) -> RerankedChunk:
    return RerankedChunk(
        chunk_id=f"{filename}-{page}",
        text=text,
        metadata={"filename": filename, "page_number": page},
        relevance=0.8,
        retrieval_score=0.5,
    )


class FakeRetriever(HybridRetriever):
    """
    Returns fixed chunks per retriever key and records the queries it saw.

    `chunks_by_key` values may also map a query to its chunks, for tests where
    a rewritten query finds what the original did not.
    """

    def __init__(self, chunks_by_key: dict, key: str, log: list) -> None:
        self.chunks = chunks_by_key.get(key, [])
        self.key = key
        self.log = log

    def retrieve(self, query, *, config=None, filenames=None, metadata_filter=None, llm=None):
        self.log.append((self.key, query, filenames))
        chunks = self.chunks.get(query, []) if isinstance(self.chunks, dict) else self.chunks
        return RetrievalResult(chunks=list(chunks), diagnostics={"namespace": self.key})


def retriever_factory(chunks_by_key: dict, log: list | None = None):
    log = [] if log is None else log
    return lambda key: FakeRetriever(chunks_by_key, key, log)
