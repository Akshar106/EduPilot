"""
Application factory
===================
Assembles the FastAPI app: lifespan, middleware, error handling, static
assets, and router registration. Route logic lives in `routes/`; this module
contains none of it.

Run with::

    uvicorn edupilot.api.app:app --reload --port 8000

Required environment (see .env.example):
    GROQ_API_KEY, DATABASE_URL, JWT_SECRET_KEY
"""

from __future__ import annotations

import asyncio
import logging
import secrets
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from edupilot import db
from edupilot.api.deps import shutdown_executor
from edupilot.api.routes import ROUTERS
from edupilot.core import metrics
from edupilot.core import redis as shared_redis
from edupilot.core.config import (
    AUTH_REQUIRED,
    BOOTSTRAP_ADMIN_EMAIL,
    BOOTSTRAP_ADMIN_PASSWORD,
    CORS_ALLOWED_ORIGINS,
    INGEST_WORKER,
    IS_PRODUCTION,
    LLM_SIMULATED,
    STATIC_DIR,
)
from edupilot.core.observability import configure_logging, request_id_var
from edupilot.core.services import services
from edupilot.security import AppError, AuthError, ErrorCode, Role
from edupilot.security.deps import ANON_COOKIE, ANON_COOKIE_MAX_AGE

configure_logging()
logger = logging.getLogger("edupilot.api")


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------


def _bootstrap_admin() -> None:
    """
    Create the first admin account when the user table is empty.

    Without it there is no way to reach the admin-only knowledge-base routes
    on a fresh deployment.
    """
    if services.users.count_users() > 0:
        return
    if not (BOOTSTRAP_ADMIN_EMAIL and BOOTSTRAP_ADMIN_PASSWORD):
        if not AUTH_REQUIRED:
            # Single-user mode already grants admin, so there is nothing to
            # bootstrap and nothing to warn about.
            return
        logger.warning(
            "No users exist and BOOTSTRAP_ADMIN_EMAIL/PASSWORD are unset — "
            "knowledge-base administration is unreachable. Set both and restart."
        )
        return
    try:
        user = services.users.create_user(
            BOOTSTRAP_ADMIN_EMAIL, BOOTSTRAP_ADMIN_PASSWORD, role=Role.ADMIN
        )
        logger.info("bootstrapped admin account %s", user.email)
    except (AuthError, ValueError) as exc:
        logger.error("could not bootstrap admin: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if LLM_SIMULATED and IS_PRODUCTION:
        raise RuntimeError(
            "LLM_SIMULATED=true is for load tests and is refused in production: "
            "every answer would be placeholder text."
        )

    # Migrations run before anything touches a table. Several processes may
    # start together; the first migrates and the rest wait for it.
    await asyncio.to_thread(db.init_db)

    if not AUTH_REQUIRED:
        logger.warning(
            "open mode: sign-in is OFF. Each browser gets its own anonymous "
            "identity, so conversations stay separate. Anonymous callers are "
            "%s. Set EDUPILOT_AUTH_REQUIRED=true to require sign-in.",
            "ADMIN (development)" if not IS_PRODUCTION
            else "students — knowledge-base writes are refused",
        )

    await asyncio.to_thread(_bootstrap_admin)

    health = await asyncio.to_thread(services.health)
    if health["status"] != "ok":
        logger.error("startup health degraded — missing: %s", health["failing"])
    else:
        logger.info("startup OK — index=%s", health["active_index"])

    from edupilot.core import tracing

    if tracing.enabled():
        logger.info("LangSmith tracing ON — project %r", tracing.project_name())

    # The checkpointer and profile store hold an async connection pool that
    # belongs to this event loop, so they are opened here and live exactly as
    # long as the app. Imported here, like the chat route's agent imports, so
    # importing the app stays cheap.
    from edupilot.agents.checkpoint import open_persistence

    worker = None
    if INGEST_WORKER == "embedded":
        from edupilot.ingestion.jobs import Worker

        worker = Worker(lambda: services.indexer)
        worker.start()
    else:
        logger.info("uploads are indexed by separate edupilot-worker processes")

    async with open_persistence() as (checkpointer, store):
        services.agent.attach_checkpointer(checkpointer, store=store)
        yield
    if worker is not None:
        await asyncio.to_thread(worker.stop)
    # Traces upload from a background thread; without this the last few turns
    # before a restart never reach LangSmith.
    await asyncio.to_thread(tracing.flush)
    await shared_redis.close()
    shutdown_executor()
    db.dispose_engine()


# ---------------------------------------------------------------------------
# Static assets
# ---------------------------------------------------------------------------


class _FrontendFiles(StaticFiles):
    """
    The Next.js static export (frontend/), served from STATIC_DIR.

    Bundles under `_next/static/` are named by content hash, so a URL never
    changes meaning and can be cached for a year. Everything else — the HTML
    pages and the payloads client-side navigation fetches — keeps one URL
    across deploys and must revalidate. `no-cache` does not mean "do not
    store": an unchanged page costs a 304 and no body. Without any
    Cache-Control a browser applies heuristic freshness and can show a
    previous deploy for minutes.
    """

    async def get_response(self, path: str, scope):  # type: ignore[override]
        # An unknown API path must get the JSON error envelope, not the
        # frontend's HTML 404 page.
        if path == "api" or path.startswith("api/"):
            raise StarletteHTTPException(status_code=404)
        return await super().get_response(path, scope)

    def file_response(self, full_path, stat_result, scope, status_code=200):  # type: ignore[override]
        response = super().file_response(full_path, stat_result, scope, status_code)
        immutable = "/_next/static/" in scope.get("path", "")
        response.headers["Cache-Control"] = (
            "public, max-age=31536000, immutable" if immutable else "no-cache"
        )
        return response


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    """Build the application. Separate from the module-level `app` so tests
    can construct an isolated instance."""
    app = FastAPI(
        title="EduPilot API",
        description="Multi-agent, source-grounded educational RAG",
        version="2.0.0",
        lifespan=lifespan,
        docs_url=None if IS_PRODUCTION else "/docs",
        redoc_url=None,
    )

    app.add_middleware(
        CORSMiddleware,
        # Previously allow_origins=["*"], which lets any site call this API
        # with the user's credentials attached.
        allow_origins=CORS_ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type"],
        max_age=600,
    )

    _register_middleware(app)
    _register_error_handlers(app)

    for router in ROUTERS:
        app.include_router(router)

    if (STATIC_DIR / "index.html").exists():
        # Last, so every API route matches first. Absolute path: a relative
        # directory resolves against the process CWD and breaks whenever the
        # server is started from anywhere but the project root.
        app.mount("/", _FrontendFiles(directory=str(STATIC_DIR), html=True), name="frontend")

    return app


def _register_middleware(app: FastAPI) -> None:
    @app.middleware("http")
    async def anonymous_identity(request: Request, call_next):
        """
        Give every browser its own identity when sign-in is disabled.

        Without this, all unauthenticated callers would share one user id, and
        the ownership checks — which are working correctly — would happily show
        each visitor everyone else's conversations, because they really would
        all be the same owner.

        The cookie carries a random id and nothing else. It authorizes nothing
        beyond "these are my own chats", so it is not signed; forging one gets
        you a different empty session list.
        """
        if AUTH_REQUIRED:
            return await call_next(request)

        anon_id = request.cookies.get(ANON_COOKIE, "")
        issued = False
        if not anon_id or len(anon_id) < 16 or not anon_id.isalnum():
            anon_id = secrets.token_hex(16)
            issued = True
        request.state.anon_id = anon_id

        response = await call_next(request)
        if issued:
            response.set_cookie(
                ANON_COOKIE,
                anon_id,
                httponly=True,
                secure=IS_PRODUCTION,
                samesite="lax",
                max_age=ANON_COOKIE_MAX_AGE,
            )
        return response

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        """Attach a request id to every log line and response, and time the request."""
        rid = uuid.uuid4().hex[:12]
        token = request_id_var.set(rid)
        started = time.perf_counter()
        status = 500
        metrics.HTTP_IN_FLIGHT.inc()
        try:
            response = await call_next(request)
            status = response.status_code
        finally:
            request_id_var.reset(token)
            metrics.HTTP_IN_FLIGHT.dec()
            elapsed = time.perf_counter() - started
            route = _route_label(request)
            if route != "/metrics":
                metrics.HTTP_REQUESTS.labels(request.method, route, str(status)).inc()
                metrics.HTTP_LATENCY.labels(request.method, route).observe(elapsed)
        response.headers["X-Request-ID"] = rid
        logger.info(
            "%s %s -> %d (%.0fms)",
            request.method, request.url.path, response.status_code, elapsed * 1000,
        )
        return response


def _route_label(request: Request) -> str:
    """
    The matched route's path template, for metric labels.

    Never the raw path: `/api/sessions/<uuid>` per conversation would create a
    new time series for every session.
    """
    route = request.scope.get("route")
    path = getattr(route, "path", None)
    if path:
        return path
    if route is not None:  # the static frontend mount
        return "static"
    return "unmatched"


def _register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
        exc.request_id = request_id_var.get() or exc.request_id
        exc.log()
        headers = {}
        if exc.code in (ErrorCode.RATE_LIMITED, ErrorCode.OVERLOADED) and "retry_after" in exc.details:
            headers["Retry-After"] = str(exc.details["retry_after"])
        return JSONResponse(exc.to_response(), status_code=exc.status_code, headers=headers)

    @app.exception_handler(Exception)
    async def unhandled_handler(request: Request, exc: Exception) -> JSONResponse:
        """
        Catch-all that never leaks internals.

        The previous code returned `f"Pipeline error: {exc}"`, which could
        include index names, file paths, and provider error bodies.
        """
        error = AppError(code=ErrorCode.INTERNAL, internal=f"{type(exc).__name__}: {exc}")
        error.request_id = request_id_var.get() or error.request_id
        error.__cause__ = exc
        error.log()
        return JSONResponse(error.to_response(), status_code=500)


#: The ASGI application uvicorn serves.
app = create_app()
