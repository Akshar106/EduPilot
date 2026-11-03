"""
LLM layer
=========
LangChain chat models across Groq and Gemini, with provider fallback and
per-request token accounting::

    from edupilot.llm import UsageTracker, build_llm

    usage = UsageTracker()
    llm = build_llm("openai/gpt-oss-120b", max_tokens=800)
    reply = await llm.ainvoke(messages, config={"callbacks": [usage]})
    tokens = usage.as_dict()

`call_llm` remains as a synchronous one-shot helper for code outside the agent
graph.
"""

from .models import (
    LLMError,
    build_fallback_chain,
    build_llm,
    call_llm,
    chat_model,
    is_gemini_model,
    is_groq_model,
    message_text,
    parse_json_response,
    to_messages,
)
from .usage import UsageTracker

__all__ = [
    "LLMError",
    "UsageTracker",
    "build_fallback_chain",
    "build_llm",
    "call_llm",
    "chat_model",
    "is_gemini_model",
    "is_groq_model",
    "message_text",
    "parse_json_response",
    "to_messages",
]
