"""
Chat models
===========
Every model call goes through a LangChain chat model built here, wrapped in
the same fallback chain the system has always used.

Replaces the hand-rolled `client.py`. What moved where:

  * **Provider SDKs -> LangChain chat models.** `ChatGroq` and
    `ChatGoogleGenerativeAI` replace direct `groq` / `google-genai` calls, so
    every node in the agent graph speaks one message type and can stream
    tokens.
  * **Retry -> the provider SDK.** The Gemini SDK retries 429 and 5xx
    responses with backoff, and `MAX_RETRIES_PER_MODEL` keeps the wait bounded
    before the chain moves on. Groq does not retry at all: its waits run to
    tens of seconds, and the next model has quota now (`GROQ_MAX_RETRIES`).
  * **Fallback -> `with_fallbacks`.** The order is unchanged: same-provider
    alternatives first (a quota hit is usually per-model), then the other
    vendor (an outage is usually per-provider).
  * **JSON parsing -> structured output.** Agents that return JSON bind a
    Pydantic schema with `with_structured_output`. A malformed reply now fails
    and falls through to the next model instead of being regex-repaired.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import threading
import weakref
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable

from edupilot.core.config import (
    AVAILABLE_MODELS,
    DEFAULT_MODEL,
    GROQ_MODELS,
    LLM_MAX_TOKENS_CLASSIFY,
)

logger = logging.getLogger(__name__)

#: SDK-level retries per model before the fallback chain moves on. Two keeps a
#: rate-limited model from pinning a request for long when another model in
#: the chain could answer now.
MAX_RETRIES_PER_MODEL = 2

#: Groq answers a 429 with the wait until the model's per-minute token budget
#: refills, often 20-40 s, and its SDK sleeps that out: measured, one turn spent
#: two minutes retrying. Each Groq model has a budget of its own, so the next
#: model in the chain can answer at once; no retry is the faster path.
GROQ_MAX_RETRIES = 0

#: Seconds before one model call is abandoned.
REQUEST_TIMEOUT_S = 60.0

#: Low and fixed: routing, grading and grounded answering want consistency.
TEMPERATURE = 0.1

_GROQ_PREFIXES = ("openai/", "qwen", "llama", "meta-llama/", "mixtral", "gemma", "deepseek", "kimi", "moonshot", "allam")


class LLMError(RuntimeError):
    """Every model in the fallback chain failed, or none could be built."""


def is_groq_model(model: str) -> bool:
    return model in GROQ_MODELS or model.lower().startswith(_GROQ_PREFIXES)


def is_gemini_model(model: str) -> bool:
    return model.lower().startswith("gemini")


def _api_key_for(model: str) -> str:
    if is_groq_model(model):
        return os.getenv("GROQ_API_KEY", "")
    if is_gemini_model(model):
        return os.getenv("GEMINI_API_KEY", "")
    return ""


def build_fallback_chain(model: str) -> list[str]:
    """
    Models to try, in order, starting with the requested one.

    Same-provider alternatives come first (a quota hit is usually per-model),
    then the other provider, since a provider-wide outage needs a different
    vendor rather than a different model.
    """
    chain = [model]
    same = [m for m in AVAILABLE_MODELS if m != model and is_groq_model(m) == is_groq_model(model)]
    other = [m for m in AVAILABLE_MODELS if m != model and is_groq_model(m) != is_groq_model(model)]
    for candidate in same + other:
        if candidate not in chain:
            chain.append(candidate)
    return chain


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

#: Chat models cached per event loop. A model holds an async HTTP client whose
#: pooled connections belong to the loop that opened them; sharing one across
#: loops (the API's loop and an evaluation run's `asyncio.run`) fails with
#: "Event loop is closed". Keying on the loop gives each its own clients, and
#: the weak reference drops them when the loop is collected.
_by_loop: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_no_loop: dict[tuple[str, int], BaseChatModel] = {}
_cache_lock = threading.Lock()


def _cache_for_current_loop() -> dict[tuple[str, int], BaseChatModel]:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return _no_loop
    with _cache_lock:
        return _by_loop.setdefault(loop, {})


# Reasoning models think before they answer, and the thinking counts against
# max_tokens. Routing, grading and grounded answering need little of it, so
# effort is kept low, and the thinking is kept out of the answer text.


def _groq_reasoning(model: str) -> dict[str, Any]:
    name = model.lower()
    if name.startswith("openai/gpt-oss"):
        return {"reasoning_effort": "low"}  # the reasoning arrives apart from the answer
    if name.startswith("qwen/qwen3"):
        return {"reasoning_format": "hidden"}  # else <think> blocks land in the answer
    return {}


def _structured_options(model: str) -> dict[str, Any]:
    """
    How a model is asked for structured output. gpt-oss on Groq often writes
    the JSON as its reply instead of calling the output tool, and Groq rejects
    that reply (`tool_use_failed`): the 20b model failed every routing call
    this way. Its JSON-schema mode takes the same JSON as the reply itself.
    """
    if model.lower().startswith("openai/gpt-oss"):
        return {"method": "json_schema"}
    return {}


def _gemini_thinking(model: str) -> dict[str, Any]:
    name = model.lower()
    if name.startswith("gemini-2.5"):
        return {"thinking_budget": 0}
    if name.startswith("gemini-3"):
        # Gemini 3 replaced the budget with a level. "low" still spent ~350 of
        # a 512-token routing budget thinking; "minimal" spends none (Flash only).
        return {"thinking_level": "minimal" if "flash" in name else "low"}
    return {}


def chat_model(model: str, *, max_tokens: int) -> BaseChatModel:
    """One provider's chat model, with no fallback. Cached per event loop."""
    cache = _cache_for_current_loop()
    key = (model, max_tokens)
    cached = cache.get(key)
    if cached is not None:
        return cached

    if is_groq_model(model):
        from langchain_groq import ChatGroq

        built: BaseChatModel = ChatGroq(
            model=model,
            temperature=TEMPERATURE,
            max_tokens=max_tokens,
            max_retries=GROQ_MAX_RETRIES,
            timeout=REQUEST_TIMEOUT_S,
            api_key=_api_key_for(model),
            **_groq_reasoning(model),
        )
    elif is_gemini_model(model):
        from langchain_google_genai import ChatGoogleGenerativeAI

        # Gemini 3 fixes its own sampling and warns on every call that sets it.
        sampling = {} if model.lower().startswith("gemini-3") else {"temperature": TEMPERATURE}
        built = ChatGoogleGenerativeAI(
            model=model,
            **sampling,
            max_output_tokens=max_tokens,
            max_retries=MAX_RETRIES_PER_MODEL,
            timeout=REQUEST_TIMEOUT_S,
            google_api_key=_api_key_for(model),
            **_gemini_thinking(model),
        )
    else:
        raise LLMError(f"Unknown model provider for {model!r}")

    cache[key] = built
    return built


def build_llm(
    model: str,
    *,
    max_tokens: int,
    schema: type | None = None,
    tools: list | None = None,
) -> Runnable:
    """
    The requested model plus its fallback chain, as one runnable.

    Models whose provider has no API key configured are left out of the chain
    rather than failing at call time — the old client skipped them the same
    way, one LLMError at a time.

    Args:
        model: Preferred model. Provider is inferred from the name.
        max_tokens: Output cap for every model in the chain.
        schema: Optional Pydantic model. When given, each model is bound with
            `with_structured_output(schema)` and the runnable returns an
            instance of it.
        tools: Optional LangChain tools to bind with `bind_tools`, so replies
            may be tool calls. Ignored when `schema` is given.

    Raises:
        LLMError: no model in the chain has a configured provider key.
    """
    chain = [m for m in build_fallback_chain(model) if _api_key_for(m)]
    if not chain:
        raise LLMError(f"No API key is configured for {model!r} or any fallback model")

    runnables: list[Runnable] = []
    for name in chain:
        base = chat_model(name, max_tokens=max_tokens)
        if schema:
            runnables.append(base.with_structured_output(schema, **_structured_options(name)))
        elif tools:
            runnables.append(base.bind_tools(tools))
        else:
            runnables.append(base)

    primary, *fallbacks = runnables
    return primary.with_fallbacks(fallbacks) if fallbacks else primary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def to_messages(messages: list[dict], system: str | None = None) -> list[BaseMessage]:
    """Convert `{"role", "content"}` dicts into LangChain messages."""
    out: list[BaseMessage] = [SystemMessage(content=system)] if system else []
    for m in messages:
        cls = AIMessage if m.get("role") == "assistant" else HumanMessage
        out.append(cls(content=str(m.get("content", ""))))
    return out


#: Typography some models use where the pipeline parses plain ASCII. gpt-oss
#: cites as 【Source 1】 and hyphenates with U+2011; the citation checks, the
#: grounding check and the UI's citation links all expect [Source 1].
#: Mapped per character, so streamed tokens come out right as well.
_PLAIN = str.maketrans({"【": "[", "】": "]", "\u2010": "-", "\u2011": "-", "\u00a0": " ", "\u202f": " "})

#: A line reference inside a citation: [Source 1†L4-L9].
_CITATION_SUFFIX = re.compile(r"(\[Source\s+\d+)\s*†[^\]\n]*\]")


def message_text(message: Any) -> str:
    """Plain text of a model reply, whether it carries a string or content blocks."""
    text = message.text if isinstance(message, BaseMessage) else str(message or "")
    return _CITATION_SUFFIX.sub(r"\1]", text.translate(_PLAIN))


def call_llm(
    messages: list[dict],
    system: str | None = None,
    model: str = DEFAULT_MODEL,
    max_tokens: int = LLM_MAX_TOKENS_CLASSIFY,
) -> str:
    """
    Synchronous one-shot completion, for callers outside the agent graph.

    The agent graph calls models directly; this remains for the evaluation
    judge and the query-transform probes, which run in plain threads.

    Raises:
        LLMError: every model in the chain failed.
    """
    try:
        reply = build_llm(model, max_tokens=max_tokens).invoke(to_messages(messages, system))
    except LLMError:
        raise
    except Exception as exc:
        raise LLMError(f"All models failed for {model!r}. Last error: {exc}") from exc
    return message_text(reply)


def parse_json_response(raw: str) -> dict:
    """Re-exported from agents.contracts so callers need only one import."""
    from edupilot.agents.contracts import parse_json_response as _parse

    return _parse(raw)
