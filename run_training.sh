#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"
if [ -n "${ZERO_PYTHON:-}" ]; then
  PYTHON="$ZERO_PYTHON"
elif [ -x "$PROJECT_DIR/.venv/bin/python" ]; then
  PYTHON="$PROJECT_DIR/.venv/bin/python"
else
  PYTHON=python3
fi
exec "$PYTHON" -u train.py --resume_from auto "$@"
