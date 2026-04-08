# Fine-Tuning Stability Atlas

Public repository for the Stability Atlas codebase. It now provides both the dependency-light core atlas modules and a first vertical-slice Brax/MuJoCo Playground training runtime:

- n-step transition aggregation with timeout-aware bootstrapping
- recent-transition cyclic buffer for instability diagnostics
- collapse thresholds, TD-error summary statistics, and early-warning trigger logic
- a sibling `atlas_training` package for Go1 SAC pretraining and shifted-domain fine-tuning
- experiment sweep generation for the preregistered atlas grid
- lightweight tests plus a dependency-gated training smoke path

## Status

The repo keeps the reusable scientific logic in `atlas/` and isolates Brax/JAX/MuJoCo Playground code in `atlas_training/` so pure-Python installs and tests do not depend on the training stack.

## Layout

- `atlas/`: pure core library modules
- `atlas_training/`: Brax/MuJoCo Playground runtime for the one-cell vertical slice
- `docs/`: methodology and integration notes
- `experiments/`: experiment-level notes and placeholders
- `scripts/`: local runner utilities
- `tests/`: unit tests for the core logic
- `results/`: generated manifests and summaries

## Quickstart

Create a local virtual environment and install the package:

```bash
./scripts/setup.sh
```

Install with the training stack:

```bash
./scripts/setup.sh --train
```

Run the unit tests:

```bash
python3 -m unittest discover -s tests -v
```

Generate the preregistered sweep manifest:

```bash
python3 scripts/run_sweep.py --output results/sweep_manifest.json
```

Run a small pretrain checkpoint:

```bash
python3 scripts/run_pretrain.py --output-dir results/runs/pretrain_go1
```

Fine-tune under the shifted domain:

```bash
python3 scripts/run_finetune.py --checkpoint results/runs/pretrain_go1/checkpoint --output-dir results/runs/finetune_go1
```

Compute diagnostic metrics from an eval log:

```bash
python3 scripts/run_diagnostic.py --eval-log results/runs/finetune_go1/eval_log.jsonl --output results/diagnostic_summary.json
```

Write a sample eval log schema:

```bash
python3 scripts/run_diagnostic.py --write-sample results/example_eval_log.jsonl
```

Run the scaffold end-to-end:

```bash
./scripts/run_all.sh
```

## Integration Notes

The core code assumes one-step transitions use the standard SAC-style bootstrap multiplier convention:

- continuing transition: `discount = gamma`
- environment terminal: `discount = 0.0`
- time-limit truncation: `discount = gamma`

If your existing stack stores raw done masks instead, convert them before inserting into the atlas helpers.

The runtime contract and hook points are documented in [`docs/integration.md`](docs/integration.md).
