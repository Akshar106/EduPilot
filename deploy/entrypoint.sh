#!/usr/bin/env sh
# EduPilot container entrypoint.
#
# Checks the one setting the app cannot start without, then execs the server.
#
# Everything the app keeps — accounts, conversations, the vector index and
# which version of it is live, checkpoints, caches — is in the Postgres named
# by DATABASE_URL, and the app migrates that database itself at startup. The
# data volume holds only the course corpus; student uploads travel to the
# ingestion worker through the database too.

set -eu

DATA_DIR="${EDUPILOT_DATA_DIR:-/data}"
PORT="${PORT:-7860}"

mkdir -p "$DATA_DIR/knowledge_base"

# Fail loudly rather than retrying a database that was never configured: the
# built-in default points at localhost, which inside a container is nothing.
if [ -z "${DATABASE_URL:-}" ]; then
    echo "ERROR: DATABASE_URL is not set." >&2
    echo "       Point it at Postgres with the pgvector extension, e.g." >&2
    echo "       postgresql://user:password@host:5432/edupilot" >&2
    exit 1
fi

echo "EduPilot starting: data=$DATA_DIR port=$PORT env=${EDUPILOT_ENV:-development}"

# exec so uvicorn is PID 1 and receives SIGTERM directly — otherwise the
# platform's stop signal goes to the shell and the server is killed instead of
# shut down, dropping in-flight requests.
#
# PYTHON is overridable so this same script can be run outside the container
# (against a virtualenv) to rehearse the deployment boot path.
exec "${PYTHON:-python}" -m uvicorn edupilot.api.app:app \
    --host 0.0.0.0 \
    --port "$PORT" \
    --workers 1 \
    --timeout-keep-alive 65
