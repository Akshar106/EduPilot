"""
HTTP surface.

Asserts the shape of the API — which routes exist, which are protected, and
what the request schemas accept — without starting a server or calling an LLM.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from edupilot.api.app import create_app
from edupilot.api.routes import ROUTERS
from edupilot.api.schemas import ChatRequest, RegisterRequest, StudyChatRequest

#: Every route the frontend depends on. A rename here is a breaking change,
#: so it is pinned rather than derived.
EXPECTED_ROUTES = {
    ("GET", "/"),
    ("POST", "/api/auth/register"),
    ("POST", "/api/auth/login"),
    ("POST", "/api/auth/refresh"),
    ("POST", "/api/auth/logout"),
    ("GET", "/api/auth/me"),
    ("GET", "/api/health"),
    ("GET", "/api/config"),
    ("POST", "/api/chat"),
    ("POST", "/api/chat/stream"),
    ("GET", "/api/sessions"),
    ("POST", "/api/sessions"),
    ("GET", "/api/sessions/{session_id}"),
    ("DELETE", "/api/sessions/{session_id}"),
    ("DELETE", "/api/sessions/{session_id}/messages/{message_id}"),
    ("GET", "/api/kb/status"),
    ("GET", "/api/kb/materials"),
    ("GET", "/api/kb/documents"),
    ("POST", "/api/kb/upload"),
    ("DELETE", "/api/kb/{domain}/{filename}"),
    ("GET", "/api/documents/{domain}/{filename}"),
    ("POST", "/api/self-study/sessions"),
    ("GET", "/api/self-study/sessions"),
    ("GET", "/api/self-study/sessions/{ss_session_id}"),
    ("DELETE", "/api/self-study/sessions/{ss_session_id}"),
    ("POST", "/api/self-study/sessions/{ss_session_id}/upload"),
    ("DELETE", "/api/self-study/sessions/{ss_session_id}/documents/{doc_id}"),
    ("GET", "/api/jobs/{job_id}"),
    ("GET", "/metrics"),
    ("POST", "/api/self-study/chat"),
    ("GET", "/api/profile"),
    ("DELETE", "/api/profile"),
    ("POST", "/api/feedback"),
    ("GET", "/api/trace/{run_id}"),
}


@pytest.fixture(scope="module")
def app():
    return create_app()


def _routes(app) -> set[tuple[str, str]]:
    """
    Every (method, path) the app serves, including routes nested in routers.

    FastAPI 0.141 / Starlette 1.6 stopped flattening `include_router` into
    `app.routes`. It now inserts an `_IncludedRouter` wrapper whose own `path`
    is None and which exposes the router as `original_router` rather than
    `.routes`. A flat walk therefore sees only `/docs` and `/openapi.json` and
    reports every route as missing — while the app serves all of them.

    Handling both shapes keeps this working across versions rather than
    pinning one.
    """
    found: set[tuple[str, str]] = set()
    seen: set[int] = set()

    def walk(routes) -> None:
        for route in routes or ():
            if id(route) in seen:
                continue
            seen.add(id(route))

            path = getattr(route, "path", None)
            for method in getattr(route, "methods", None) or ():
                if method not in ("HEAD", "OPTIONS") and path:
                    found.add((method, path))

            # Mounts expose `.routes`; included routers expose the router they
            # wrapped. Router paths already carry their prefix.
            walk(getattr(route, "routes", None))
            wrapped = getattr(route, "original_router", None)
            if wrapped is not None:
                walk(getattr(wrapped, "routes", None))

    walk(app.routes)
    return found


def test_every_expected_route_is_registered(app):
    missing = EXPECTED_ROUTES - _routes(app)
    assert not missing, f"routes disappeared: {sorted(missing)}"


def test_no_unexpected_api_routes(app):
    """Catches a route added without a corresponding test or doc update."""
    actual_api = {(m, p) for m, p in _routes(app) if p.startswith("/api/")}
    expected_api = {(m, p) for m, p in EXPECTED_ROUTES if p.startswith("/api/")}
    assert actual_api - expected_api == set()


def test_all_routers_are_registered():
    """A router left out of ROUTERS silently serves nothing."""
    from edupilot.api.routes import (
        auth,
        chat,
        knowledge_base,
        profile,
        self_study,
        sessions,
        system,
    )

    for module in (auth, chat, sessions, knowledge_base, self_study, profile, system):
        assert module.router in ROUTERS, f"{module.__name__} is not registered"


def test_openapi_schema_builds(app):
    """A duplicated operation id or bad annotation blows up here first."""
    schema = app.openapi()
    assert schema["info"]["title"] == "EduPilot API"
    assert schema["paths"]


# ---------------------------------------------------------------------------
# Schemas — the input bounds that keep prompt cost and abuse in check
# ---------------------------------------------------------------------------


def test_chat_query_cannot_be_empty():
    with pytest.raises(ValidationError):
        ChatRequest(query="")


def test_chat_query_is_length_bounded():
    from edupilot.core.config import MAX_QUERY_CHARS

    with pytest.raises(ValidationError):
        ChatRequest(query="x" * (MAX_QUERY_CHARS + 1))


def test_chat_top_k_is_bounded():
    with pytest.raises(ValidationError):
        ChatRequest(query="hi", top_k=999)
    with pytest.raises(ValidationError):
        ChatRequest(query="hi", rerank_top_k=0)


def test_chat_defaults_come_from_config():
    from edupilot.core.config import DEFAULT_MODEL, DEFAULT_RERANK_TOP_K, DEFAULT_TOP_K

    req = ChatRequest(query="what is variance?")
    assert req.model == DEFAULT_MODEL
    assert req.top_k == DEFAULT_TOP_K
    assert req.rerank_top_k == DEFAULT_RERANK_TOP_K
    assert req.session_id is None


def test_no_schema_accepts_a_caller_supplied_owner():
    """Ownership must come from the token; a body field would reintroduce the IDOR."""
    for model in (ChatRequest, StudyChatRequest, RegisterRequest):
        assert "user_id" not in model.model_fields
        assert "owner" not in model.model_fields


def test_register_password_has_a_floor_and_a_bcrypt_ceiling():
    with pytest.raises(ValidationError):
        RegisterRequest(email="a@b.co", password="short")
    with pytest.raises(ValidationError):
        RegisterRequest(email="a@b.co", password="x" * 73)
    RegisterRequest(email="a@b.co", password="a-long-enough-password")


# ---------------------------------------------------------------------------
# Static assets must revalidate
# ---------------------------------------------------------------------------


@pytest.fixture
def frontend(tmp_path, monkeypatch):
    """An app serving a miniature Next.js export: two pages and one hashed bundle."""
    from edupilot.api import app as app_module
    from edupilot.api.routes import system

    (tmp_path / "_next/static/chunks").mkdir(parents=True)
    (tmp_path / "study").mkdir()
    (tmp_path / "index.html").write_text("<html>chat</html>")
    (tmp_path / "study/index.html").write_text("<html>study</html>")
    (tmp_path / "404.html").write_text("<html>not found</html>")
    (tmp_path / "_next/static/chunks/app-3f2a.js").write_text("console.log(1)")
    monkeypatch.setattr(app_module, "STATIC_DIR", tmp_path)
    monkeypatch.setattr(system, "STATIC_DIR", tmp_path)
    return app_module.create_app()


def test_pages_revalidate_and_hashed_bundles_are_cached_for_good(frontend):
    """
    A page keeps its URL across deploys, so it must revalidate or a browser
    shows the previous deploy; a hashed bundle never changes at its URL.
    """
    from fastapi.testclient import TestClient

    with TestClient(frontend) as client:
        for path in ("/", "/study/"):
            r = client.get(path)
            assert r.status_code == 200, path
            assert r.headers["cache-control"] == "no-cache", path
        bundle = client.get("/_next/static/chunks/app-3f2a.js")
        assert bundle.status_code == 200
        assert bundle.headers["cache-control"] == "public, max-age=31536000, immutable"


def test_unchanged_pages_revalidate_cheaply(frontend):
    """`no-cache` must still allow a 304 — otherwise it costs a full re-download."""
    from fastapi.testclient import TestClient

    with TestClient(frontend) as client:
        first = client.get("/study/")
        etag = first.headers.get("etag")
        assert etag, "no ETag means every revalidation re-sends the body"
        second = client.get("/study/", headers={"If-None-Match": etag})
        assert second.status_code == 304
        assert not second.content


def test_routes_resolve_like_the_static_export(frontend):
    from fastapi.testclient import TestClient

    with TestClient(frontend) as client:
        redirect = client.get("/study", follow_redirects=False)
        assert redirect.status_code in (301, 307, 308)
        assert redirect.headers["location"].endswith("/study/")
        assert client.get("/study").text == "<html>study</html>"
        missing = client.get("/no-such-page/")
        assert missing.status_code == 404 and "not found" in missing.text


def test_an_unknown_api_path_gets_json_not_the_frontend(frontend):
    from fastapi.testclient import TestClient

    with TestClient(frontend) as client:
        r = client.get("/api/no-such-route")
    assert r.status_code == 404
    assert r.headers["content-type"].startswith("application/json")


def test_without_a_built_frontend_the_root_says_how_to_build_it(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from edupilot.api import app as app_module
    from edupilot.api.routes import system

    monkeypatch.setattr(app_module, "STATIC_DIR", tmp_path)
    monkeypatch.setattr(system, "STATIC_DIR", tmp_path)
    with TestClient(app_module.create_app()) as client:
        body = client.get("/").json()
    assert "npm run build" in body["frontend"]
