#!/bin/sh
set -e

# If the first argument is "serve" or empty, start the FastAPI Web Studio server
if [ "$#" -eq 0 ] || [ "$1" = "serve" ]; then
    HOST="${FORGE_HOST:-0.0.0.0}"
    PORT="${FORGE_PORT:-8000}"
    WORKERS="${FORGE_WORKERS:-4}"
    echo "[FORGE] Starting Production Inverse-Design Platform on ${HOST}:${PORT} with ${WORKERS} workers..."
    exec python -m uvicorn forge.api.server:app --host "$HOST" --port "$PORT" --workers "$WORKERS"
fi

# Otherwise, pass all arguments directly to the FORGE CLI dispatcher
exec python -m forge "$@"

