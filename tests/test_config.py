"""Configuration and path resolution."""

from __future__ import annotations

from pathlib import Path

from edupilot.core import config


def test_data_paths_live_under_data_dir():
    """The files the app writes resolve under DATA_DIR, so one mount covers them."""
    data_dir = config.DATA_DIR
    for path in (config.KNOWLEDGE_BASE_DIR,):
        assert data_dir in path.parents or path == data_dir, f"{path} escapes DATA_DIR"


def test_data_dir_is_redirected_by_conftest():
    """The env override must win, or a test run would touch real data."""
    assert "edupilot-test-data-" in str(config.DATA_DIR)


def test_database_is_redirected_by_conftest(database_url):
    """Likewise the database: a test run must never reach the developer's own."""
    assert config.DATABASE_URL == database_url
    assert not config.DATABASE_URL.endswith("/edupilot")


def test_static_dir_lives_inside_the_package():
    """Web assets ship with the package, not with the working directory."""
    assert config.PACKAGE_DIR in config.STATIC_DIR.parents
    assert config.STATIC_DIR.name == "static"


def test_every_domain_is_fully_specified():
    required = {
        "name", "abbr", "color", "knowledge_base_path",
        "namespace", "description", "keywords",
    }
    assert config.DOMAINS, "at least one domain must be configured"
    for key, domain in config.DOMAINS.items():
        missing = required - set(domain)
        assert not missing, f"domain {key} is missing {missing}"
        assert domain["keywords"], f"domain {key} has no keywords"


def test_domain_namespaces_are_unique():
    """Two domains sharing a namespace would silently merge their corpora."""
    namespaces = [d["namespace"] for d in config.DOMAINS.values()]
    assert len(namespaces) == len(set(namespaces))


def test_knowledge_base_paths_are_unique_and_under_the_kb_dir():
    paths = [Path(d["knowledge_base_path"]) for d in config.DOMAINS.values()]
    assert len(paths) == len(set(paths))
    for path in paths:
        assert config.KNOWLEDGE_BASE_DIR in path.parents


def test_default_model_is_available():
    assert config.DEFAULT_MODEL in config.AVAILABLE_MODELS
    assert config.VERIFY_MODEL in config.AVAILABLE_MODELS


def test_groq_models_are_a_subset_of_available():
    assert set(config.GROQ_MODELS) <= set(config.AVAILABLE_MODELS)


def test_every_available_model_has_a_provider():
    from edupilot.llm import is_gemini_model, is_groq_model

    for model in config.AVAILABLE_MODELS:
        assert is_groq_model(model) != is_gemini_model(model), model
        assert is_groq_model(model) == (model in config.GROQ_MODELS), model


def test_reasoning_models_keep_their_thinking_low_and_out_of_the_answer():
    from edupilot.llm.models import _gemini_thinking, _groq_reasoning

    assert _groq_reasoning("openai/gpt-oss-120b") == {"reasoning_effort": "low"}
    assert _groq_reasoning("qwen/qwen3.8-27b") == {"reasoning_format": "hidden"}
    assert _gemini_thinking("gemini-3.5-flash") == {"thinking_level": "minimal"}
    assert _gemini_thinking("gemini-2.5-flash") == {"thinking_budget": 0}


def test_gpt_oss_returns_structured_output_as_json_schema():
    """With tool calling, gpt-oss-20b answered every routing call as plain JSON and Groq rejected it."""
    from edupilot.llm.models import _structured_options

    assert _structured_options("openai/gpt-oss-20b") == {"method": "json_schema"}
    assert _structured_options("qwen/qwen3.8-27b") == {}


def test_a_retired_model_says_so_instead_of_something_went_wrong():
    from edupilot.security.errors import ErrorCode, classify_upstream

    groq = RuntimeError("Error code: 404 - {'error': {'message': 'The model `llama-3.3-70b-versatile` "
                        "does not exist or you do not have access to it.', 'code': 'model_not_found'}}")
    gemini = RuntimeError("404 NOT_FOUND. models/gemini-2.0-flash is not found for API version v1beta")
    for exc in (groq, gemini):
        error = classify_upstream(exc)
        assert error.code == ErrorCode.UPSTREAM_UNAVAILABLE
        assert "no longer serves it" in error.message


def test_cors_is_not_a_wildcard():
    """allow_origins=['*'] with credentials lets any site call the API as the user."""
    assert "*" not in config.CORS_ALLOWED_ORIGINS


def test_model_typography_becomes_what_the_citation_checks_parse():
    from langchain_core.messages import AIMessage

    from edupilot.guardrails.citations import CITATION
    from edupilot.llm import message_text

    text = message_text(AIMessage(content="Bias\u2011variance\u00a0trade\u2011off【Source 1】 and【Source 2†L4-L9】."))
    assert text == "Bias-variance trade-off[Source 1] and[Source 2]."
    assert [m.group(1) for m in CITATION.finditer(text)] == ["1", "2"]
    # Streamed a token at a time, the brackets still come out plain.
    assert "".join(message_text(t) for t in ("【", "Source", " 3", "】")) == "[Source 3]"
