#!/usr/bin/env bash
set -euo pipefail

uv venv .venv
uv pip install --python .venv/bin/python -e .
