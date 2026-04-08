#!/usr/bin/env bash
set -euo pipefail

mkdir -p results
./scripts/setup.sh --train
./.venv/bin/python scripts/run_pretrain.py --output-dir results/runs/pretrain_smoke --train-steps 16 --eval-interval 16 --num-envs 2 --eval-episodes 2 --batch-size 8 --min-replay-size 8 --episode-length 16
./.venv/bin/python scripts/run_finetune.py --checkpoint results/runs/pretrain_smoke/checkpoint --output-dir results/runs/finetune_smoke --n-step 1 --critic-width 256 --seed 0 --train-steps 96 --eval-interval 32 --num-envs 2 --eval-episodes 2 --batch-size 8 --min-replay-size 8 --diagnostic-min-transitions 16 --diagnostic-minibatches 2 --diagnostic-batch-size 8 --episode-length 16
./.venv/bin/python scripts/run_sweep.py --output results/sweep_manifest.json
./.venv/bin/python scripts/run_diagnostic.py --eval-log results/runs/finetune_smoke/eval_log.jsonl --output results/diagnostic_summary.json
