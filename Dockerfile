# EduPilot — container image
# =============================================================================
# Targets Hugging Face Spaces (Docker SDK) but is plain Docker, so it runs
# unchanged on Fly, Render, or any VPS.
#
# Dependencies are installed with uv from uv.lock, so the image runs exactly
# the versions CI tested.
#
# LOCAL_MODELS picks where embedding and reranking run:
#
#   true (default)  bge-small-en-v1.5 and bge-reranker-base run in-process.
#                   Their weights (~1.2 GB) are baked in, because downloading
#                   them on first request means the first visitor waits
#                   minutes and a restart re-downloads. Torch is the CPU-only
#                   build — the default wheel pulls the entire CUDA stack.
#                   Peak memory serving a query is ~1.1 GB; give it 2 GB.
#
#   false           Hosted Pinecone Inference models. No torch and no weights,
#                   so the image is a fraction of the size and boots in
#                   seconds. Requires EMBEDDING_MODEL / RERANKER_MODEL set to
#                   hosted models and an index built with them — see
#                   "Hosted embeddings" in the README.
#
#   docker build --build-arg LOCAL_MODELS=false -t edupilot:hosted .
# =============================================================================

# ---------------------------------------------------------------------------
# Frontend: the Next.js app, built to a static export
#
# Only the export reaches the final image — no Node runtime ships.
# ---------------------------------------------------------------------------
FROM node:24-slim AS web

ENV NEXT_TELEMETRY_DISABLED=1
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
# `next build`, not `npm run build`: the npm script also copies the export
# into src/, which does not exist in this stage.
RUN npx next build

# ---------------------------------------------------------------------------
# Application image
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS base

COPY --from=ghcr.io/astral-sh/uv:0.12.22 /uv /uvx /bin/

ARG LOCAL_MODELS=true

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Compile at install so the first request does not pay for it; copy rather
    # than hard-link, since there is no uv cache in the image to link from.
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    # Use the image's interpreter; never download another one.
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:$PATH" \
    # Set before the tokenizers Rust extension initializes. The chunker and the
    # embedder both touch a tokenizer in one process, and re-entering rayon's
    # pool from a second call site deadlocks — see core/config.py.
    TOKENIZERS_PARALLELISM=false \
    # Where the baked weights live. HF_HOME must match at build and run time or
    # the runtime re-downloads everything.
    HF_HOME=/opt/models \
    SENTENCE_TRANSFORMERS_HOME=/opt/models

# tesseract: OCR fallback for scanned PDF pages (ingestion only, but the
#            import path touches it).
# libgl1/libglib: PyMuPDF and Pillow runtime deps.
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        libgl1 \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces runs the container as uid 1000 and mounts persistent storage at
# /data owned by that user. Only the directories it writes to are its own, and
# they are empty when chowned here: `chown -R` over an installed tree copies
# every file into a new layer, which doubled the image — once for the
# virtualenv, again for the baked model weights.
RUN useradd -m -u 1000 edupilot \
    && mkdir -p /data /opt/models \
    && chown edupilot:edupilot /data /opt/models

WORKDIR /app

# ---------------------------------------------------------------------------
# Dependencies
#
# Third-party packages only, from the lockfile. Copying just the packaging
# metadata keeps this layer cached across source edits. pyproject.toml pins
# torch to the CPU index on Linux, so the CUDA build is never considered.
# ---------------------------------------------------------------------------
COPY pyproject.toml uv.lock README.md ./
RUN if [ "$LOCAL_MODELS" = "true" ]; then EXTRAS="--extra local-models"; else EXTRAS=""; fi \
    && uv sync --locked --no-dev --no-install-project $EXTRAS

# ---------------------------------------------------------------------------
# Bake the models (local mode only)
#
# Runs before the source copy so editing code does not re-download 1.2 GB.
# Same ids as core/config.py: EMBEDDING_MODEL and RERANKER_MODEL. Downloaded
# as the runtime user, so the weights are owned by it without a chown.
# ---------------------------------------------------------------------------
USER edupilot
RUN if [ "$LOCAL_MODELS" = "true" ]; then \
        python -c "from sentence_transformers import CrossEncoder, SentenceTransformer; \
SentenceTransformer('BAAI/bge-small-en-v1.5'); CrossEncoder('BAAI/bge-reranker-base'); \
print('models cached under /opt/models')"; \
    else \
        echo "hosted models: nothing to bake"; \
    fi
USER root

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
COPY src/ src/
COPY deploy/ deploy/
COPY --from=web /web/out/ src/edupilot/web/static/
# Now the project itself, editable as before, against the installed packages.
RUN if [ "$LOCAL_MODELS" = "true" ]; then EXTRAS="--extra local-models"; else EXTRAS=""; fi \
    && uv sync --locked --no-dev $EXTRAS

# The application tree and virtualenv stay root-owned and read-only to the
# server; everything it writes goes under EDUPILOT_DATA_DIR.
USER edupilot

# Open mode: the bundled frontend has no login screen, so sign-in is off and
# each browser gets its own anonymous identity instead. Anonymous callers are
# students in production, so knowledge-base writes are refused — manage the
# corpus with `edupilot-reindex`, or set EDUPILOT_AUTH_REQUIRED=true once a
# login UI exists.
ENV EDUPILOT_DATA_DIR=/data \
    EDUPILOT_ENV=production \
    EDUPILOT_AUTH_REQUIRED=false \
    PORT=7860

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT}/api/health" || exit 1

ENTRYPOINT ["/app/deploy/entrypoint.sh"]
