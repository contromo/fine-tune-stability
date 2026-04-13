#!/usr/bin/env bash
set -euo pipefail

VENV_PATH="${VENV_PATH:-.venv}"
PYTHON_BIN="$VENV_PATH/bin/python"

uv venv "$VENV_PATH"

if [[ "${1:-}" == "--train" ]]; then
  uv pip install --python "$PYTHON_BIN" -e ".[train]"
else
  uv pip install --python "$PYTHON_BIN" -e .
fi
