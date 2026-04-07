# Fine-Tuning Stability Atlas

Public repository for the Stability Atlas scaffold. It currently provides the reusable core logic and project structure needed to integrate with a Brax SAC training stack:

- n-step transition aggregation with timeout-aware bootstrapping
- recent-transition cyclic buffer for instability diagnostics
- collapse thresholds, TD-error summary statistics, and early-warning trigger logic
- experiment sweep generation for the preregistered atlas grid
- lightweight scripts and tests so the repo is runnable before MuJoCo Playground integration lands

## Status

The current implementation focuses on stable, testable core modules rather than bundling a full Brax SAC fork. The next integration step is to call the `atlas.nstep` and `atlas.diagnostics` helpers from the actual SAC actor and replay loop.

## Layout

- `atlas/`: core library modules
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

Run the unit tests:

```bash
python3 -m unittest discover -s tests -v
```

Generate the preregistered sweep manifest:

```bash
python3 scripts/run_sweep.py --output results/sweep_manifest.json
```

Generate a placeholder pretrain manifest:

```bash
python3 scripts/run_pretrain.py --output results/pretrain_manifest.json
```

Compute diagnostic metrics from an eval log:

```bash
python3 scripts/run_diagnostic.py --eval-log results/example_eval_log.jsonl --output results/diagnostic_summary.json
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

The intended Brax hook points are documented in [`docs/integration.md`](docs/integration.md).
