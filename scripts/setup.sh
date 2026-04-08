#!/usr/bin/env bash
set -euo pipefail

uv venv .venv

if [[ "${1:-}" == "--train" ]]; then
  uv pip install --python .venv/bin/python -e ".[train]"
else
  uv pip install --python .venv/bin/python -e .
fi
