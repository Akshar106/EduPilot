"""Unauthenticated surface: the SPA entry point, health, and client config."""

from __future__ import annotations

import secrets

from fastapi import APIRouter, Header
from fastapi.responses import FileResponse, JSONResponse, Response

from edupilot.api.deps import run_blocking
from edupilot.core import metrics
from edupilot.core.config import (
    AVAILABLE_MODELS,
    DEFAULT_MODEL,
    DEFAULT_RERANK_TOP_K,
    DEFAULT_TOP_K,
    DOMAINS,
    GROQ_MODELS,
    MAX_QUERY_CHARS,
    METRICS_TOKEN,
    STATIC_DIR,
)
from edupilot.core.services import services
from edupilot.security import AppError, ErrorCode

router = APIRouter(tags=["system"])


@router.get("/", include_in_schema=False)
async def root():
    index = STATIC_DIR / "index.html"
    if not index.exists():
        return JSONResponse(
            {
                "service": "EduPilot API",
                "docs": "/docs",
                "frontend": "not built — run `npm ci && npm run build` in frontend/",
            }
        )
    # The page names its bundles by content hash; the page itself must revalidate.
    return FileResponse(index, headers={"Cache-Control": "no-cache"})


@router.get("/api/health")
async def health():
    return await run_blocking(services.health)


@router.get("/metrics", include_in_schema=False)
async def prometheus_metrics(authorization: str | None = Header(default=None)):
    """
    Prometheus metrics for this process.

    Open unless METRICS_TOKEN is set, in which case the scraper must send it as
    a bearer token. Route timings are not secret, but a public deployment has
    no reason to publish them.
    """
    if METRICS_TOKEN and not secrets.compare_digest(authorization or "", f"Bearer {METRICS_TOKEN}"):
        raise AppError(code=ErrorCode.UNAUTHENTICATED, internal="metrics scrape without the token")
    body, content_type = await run_blocking(metrics.render)
    return Response(body, media_type=content_type)


@router.get("/api/config")
async def get_config():
    """Everything the frontend needs to render before its first real request."""
    return {
        "available_models": AVAILABLE_MODELS,
        "groq_models": GROQ_MODELS,
        "default_model": DEFAULT_MODEL,
        "domains": {
            k: {
                "name": v["name"],
                "abbr": v["abbr"],
                "color": v["color"],
                "description": v["description"],
            }
            for k, v in DOMAINS.items()
        },
        "defaults": {
            "top_k": DEFAULT_TOP_K,
            "rerank_top_k": DEFAULT_RERANK_TOP_K,
            "max_query_chars": MAX_QUERY_CHARS,
        },
    }
