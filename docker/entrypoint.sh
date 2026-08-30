#!/usr/bin/env bash
# PrismaCoder container entrypoint.
#   api      - FastAPI inference server   (default)
#   ui       - Gradio UI
#   verifier - standalone verifier worker (POST /verify)
set -euo pipefail

ROLE="${1:-api}"

case "$ROLE" in
  api)
    exec python3 -m uvicorn inference.server.app:app --host 0.0.0.0 --port 8000
    ;;
  ui)
    exec python3 inference/gradio_app.py
    ;;
  verifier)
    exec python3 docker/verifier/worker.py --port 9100
    ;;
  *)
    echo "unknown role: $ROLE (api|ui|verifier)" >&2
    exit 2
    ;;
esac
