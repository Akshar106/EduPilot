"""
Simulated models
================
Stand-ins for every model call, for load tests (LLM_SIMULATED=true).

A load test against the real providers measures their rate limits — Groq's
free tier allows a few dozen requests a minute — not this system. These models
take a realistic time to answer, stream their tokens, and return valid
structured outputs, so everything else in a turn runs for real: routing logic,
the graph, embedding, pgvector search, reranking, guardrails, checkpoints and
the database writes.

Timing per call:
  time to first token   SIMULATED_LLM_LATENCY_MS (jittered ±25%)
  streaming             SIMULATED_LLM_TOKENS_PER_SECOND

Waits are `asyncio.sleep`, so a simulated call occupies the event loop no more
than a real network call does. Refused in production (api/app.py).
"""

from __future__ import annotations

import asyncio
import random
import re
from collections.abc import AsyncIterator, Iterator
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda

from edupilot.core.config import (
    DOMAINS,
    SIMULATED_LLM_LATENCY_MS,
    SIMULATED_LLM_TOKENS_PER_SECOND,
)

MODEL_NAME = "simulated"

#: Roughly what a grounded answer runs to.
ANSWER_WORDS = 140


def _latency() -> float:
    return SIMULATED_LLM_LATENCY_MS / 1000 * random.uniform(0.75, 1.25)


def _text(prompt: Any) -> str:
    if hasattr(prompt, "to_string"):
        return prompt.to_string()
    if isinstance(prompt, list):
        return "\n".join(str(getattr(m, "content", m)) for m in prompt)
    return str(prompt)


def _sources(text: str) -> list[int]:
    return sorted({int(n) for n in re.findall(r"\[Source (\d+)\]", text)}) or [1]


def _question(text: str) -> str:
    match = re.search(r"<question>\s*(.*?)\s*</question>", text, re.S)
    return (match.group(1) if match else text)[:500]


def _domains_for(question: str) -> list[str]:
    lowered = question.lower()
    return [
        code for code, cfg in DOMAINS.items()
        if any(kw.lower() in lowered for kw in cfg.get("keywords", []))
    ]


def _structured(schema: type, prompt: Any):
    """A plausible, valid instance of `schema` for this prompt."""
    from edupilot.agents import contracts as c

    text = _text(prompt)
    if schema is c.RouterOutput:
        question = _question(text)
        domains = _domains_for(question) or [next(iter(DOMAINS))]
        return c.RouterOutput(
            intent_type="multi" if len(domains) > 1 else "single",
            domains=domains,
            is_course_related=True,
            reasoning="simulated",
            topics=[c.TopicRef(name=question[:40], domain=domains[0])],
        )
    if schema is c.PlannerOutput:
        question = _question(text)
        return c.PlannerOutput(
            sub_questions=[
                c.SubQuestion(question=question, domain=d) for d in (_domains_for(question) or list(DOMAINS)[:1])
            ]
        )
    if schema is c.EvidenceGrade:
        return c.EvidenceGrade(relevant_sources=_sources(text), sufficient=True)
    if schema is c.QueryRewrite:
        return c.QueryRewrite(query=_question(text))
    if schema is c.VerifierOutput:
        return c.VerifierOutput(
            grounding_score=0.9, coverage_score=0.85, quality_score=0.88, is_satisfactory=True
        )
    if schema is c.QuizOutput:
        return c.QuizOutput(
            title="Practice quiz",
            questions=[
                c.QuizQuestion(
                    question=f"Simulated question {i + 1}",
                    options=["The option the excerpt supports", "A distractor"],
                    answer_index=0,
                    explanation="From the excerpt.",
                    source=1,
                )
                for i in range(3)
            ],
        )
    if schema is c.FlashcardsOutput:
        return c.FlashcardsOutput(
            title="Flashcards",
            cards=[c.Flashcard(front=f"Term {i + 1}", back="Its definition, from the excerpt.", source=1) for i in range(3)],
        )
    return schema()


_EXCERPT = re.compile(r"\[Source (\d+)\][^\n]*\n(.*?)(?=\n\n\[Source \d+\]|<<<END-SOURCE|\Z)", re.S)
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def _answer(text: str) -> str:
    """
    Sentences lifted from the prompt's own excerpts, each cited.

    A grounded answer, so the guardrails pass it as they would a real one and
    the turn takes the full path — verification, caching — rather than ending
    in a refusal. Without excerpts (summaries, synthesis), a fixed reply.
    """
    sentences: list[str] = []
    for number, body in _EXCERPT.findall(text):
        for sentence in _SENTENCE.split(" ".join(body.split()))[:2]:
            if len(sentence.split()) >= 4:
                sentences.append(f"{sentence.rstrip('.')} [Source {number}].")
        if sum(len(x.split()) for x in sentences) >= ANSWER_WORDS:
            break
    if sentences:
        return " ".join(sentences)
    cite = " ".join(f"[Source {n}]" for n in _sources(text)[:2])
    return f"A summary of the points above {cite}."


class SimulatedChatModel(BaseChatModel):
    """Answers after a delay, citing the sources in its prompt, token by token."""

    @property
    def _llm_type(self) -> str:
        return "simulated"

    @property
    def _identifying_params(self) -> dict:
        return {"model_name": MODEL_NAME}

    def _message(self, messages: list[BaseMessage]) -> AIMessage:
        text = _answer(_text(messages))
        return AIMessage(
            content=text,
            usage_metadata={"input_tokens": 1200, "output_tokens": len(text.split()), "total_tokens": 1200 + len(text.split())},
            response_metadata={"model_name": MODEL_NAME},
        )

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        import time

        time.sleep(_latency())
        return ChatResult(generations=[ChatGeneration(message=self._message(messages))])

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        await asyncio.sleep(_latency())
        return ChatResult(generations=[ChatGeneration(message=self._message(messages))])

    def _stream(self, messages, stop=None, run_manager=None, **kwargs) -> Iterator[ChatGenerationChunk]:
        raise NotImplementedError("the simulated model only streams asynchronously")

    async def _astream(
        self, messages, stop=None, run_manager=None, **kwargs
    ) -> AsyncIterator[ChatGenerationChunk]:
        message = self._message(messages)
        await asyncio.sleep(_latency())
        words = str(message.content).split(" ")
        gap = 1 / max(1, SIMULATED_LLM_TOKENS_PER_SECOND)
        for i, word in enumerate(words):
            last = i == len(words) - 1
            chunk = ChatGenerationChunk(
                message=AIMessageChunk(
                    content=word + ("" if last else " "),
                    usage_metadata=message.usage_metadata if last else None,
                    response_metadata={"model_name": MODEL_NAME} if last else {},
                )
            )
            if run_manager:
                await run_manager.on_llm_new_token(chunk.text, chunk=chunk)
            yield chunk
            await asyncio.sleep(gap)


async def _structured_after_delay(schema: type, prompt: Any):
    await asyncio.sleep(_latency())
    return _structured(schema, prompt)


def simulated_llm(model: str, *, max_tokens: int, schema: type | None = None, tools=None) -> Runnable:
    """Drop-in for `build_llm`: the same signature, simulated calls."""
    if schema is not None:
        return RunnableLambda(
            lambda prompt: _structured(schema, prompt),
            afunc=lambda prompt: _structured_after_delay(schema, prompt),
        )
    return SimulatedChatModel()
