#!/usr/bin/env bash
# Local product launcher. Does not train, publish, or pull a model implicitly.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
PYTHON_BIN="${PYTHON_BIN:-$SCRIPT_DIR/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Create runtime: python3 -m venv .venv && .venv/bin/python -m pip install -r requirements-dev.txt"
  exit 1
fi
export MPLCONFIGDIR="${MPLCONFIGDIR:-$SCRIPT_DIR/.cache/matplotlib}"
case "${1:-app}" in
  app) exec "$PYTHON_BIN" app.py ;;
  review) shift; exec "$PYTHON_BIN" review_cli.py "$@" ;;
  benchmark) shift; exec "$PYTHON_BIN" benchmark.py "$@" ;;
  test) shift; exec "$PYTHON_BIN" -m pytest tests -q "$@" ;;
  server) exec "$PYTHON_BIN" -m uvicorn server.app:app --host 127.0.0.1 --port "${PORT:-8000}" ;;
  train) exec "$PYTHON_BIN" train_grpo.py ;;
  *) echo "Usage: ./run.sh [app|review|benchmark|test|server|train] [arguments]"; exit 2 ;;
esac
