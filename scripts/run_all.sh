#!/usr/bin/env bash
set -euo pipefail

mkdir -p results
./scripts/setup.sh
./.venv/bin/python scripts/run_pretrain.py --output results/pretrain_manifest.json
./.venv/bin/python scripts/run_sweep.py --output results/sweep_manifest.json
./.venv/bin/python scripts/run_diagnostic.py --write-sample results/example_eval_log.jsonl
./.venv/bin/python scripts/run_diagnostic.py --eval-log results/example_eval_log.jsonl --output results/diagnostic_summary.json
