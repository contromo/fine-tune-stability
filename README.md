# Fine-Tuning Stability Atlas

`fine-tune-stability` is a small research codebase for studying collapse during sample-efficient fine-tuning of robotic RL policies. The repository is split into:

- `atlas/`: pure, dependency-light research utilities
- `atlas_training/`: the Brax + MuJoCo Playground vertical slice for Go1 SAC pretrain/fine-tune runs

The current implementation is designed to answer one concrete question well: can we pretrain a Go1 locomotion policy, shift the domain at fine-tune time, log recent-buffer TD diagnostics, and recover the artifacts needed for a stability-atlas analysis.

## What Is Here

- naive off-policy n-step aggregation with timeout-aware flushing
- recent-transition diagnostics based on TD-error variance and q95
- collapse-threshold utilities and warning-score logic
- a one-cell Brax/MuJoCo Playground training slice for `Go1JoystickFlatTerrain`
- reproducible runner scripts for pretrain, fine-tune, pilot calibration, sweep manifest generation, and diagnostic summarization

## Repo Layout

- `atlas/`
  Pure helpers for transitions, n-step aggregation, time-limit handling, recent buffers, and diagnostics.
- `atlas_training/`
  Runtime code for environment construction, SAC wiring, checkpoint save/load, and fine-tune diagnostics.
- `scripts/`
  CLI entrypoints and the one-command smoke flow.
- `tests/`
  Pure-Python tests plus dependency-gated runtime coverage.
- `docs/`
  Methodology notes and Brax integration details.

## Install

Base install:

```bash
./scripts/setup.sh
```

Training install:

```bash
./scripts/setup.sh --train
```

Notes:

- `.[train]` intentionally resolves the default `jax` package so CPU smoke runs work out of the box.
- GPU users should replace or upgrade that JAX install with the platform-specific wheel set for their accelerator.
- `playground` is the distribution name that provides the `mujoco_playground` import path used by this repo.
- Brax is intentionally constrained to `>=0.14.2,<0.15` because `atlas_training` depends on some 0.14.x evaluator and replay internals.

## Quickstart

Run the pure-Python test suite:

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

Fine-tune from that checkpoint under the shifted domain:

```bash
python3 scripts/run_finetune.py \
  --checkpoint results/runs/pretrain_go1/checkpoint \
  --output-dir results/runs/finetune_go1
```

Run the pilot calibration gate with a shared pretrain, three fine-tune seeds, and one extremal throughput probe:

```bash
python3 scripts/run_pilot.py \
  --output-dir results/runs/pilot_gate \
  --pretrain-steps 1000000
```

Summarize diagnostic logs:

```bash
python3 scripts/run_diagnostic.py \
  --eval-log results/runs/finetune_go1/eval_log.jsonl \
  --output results/diagnostic_summary.json
```

Run the end-to-end smoke flow:

```bash
./scripts/run_all.sh
```

The smoke flow uses deliberately tiny training settings for runtime reasons; it is a wiring check, not a meaningful experiment.

## Artifacts

Pretrain writes:

- `results/runs/<run_id>/config.json`
- `results/runs/<run_id>/summary.json`
- `results/runs/<run_id>/checkpoint/checkpoint.msgpack`

Fine-tune writes:

- `results/runs/<run_id>/config.json`
- `results/runs/<run_id>/pretrain_baseline.json`
- `results/runs/<run_id>/eval_log.jsonl`
- `results/runs/<run_id>/summary.json`

Pilot writes:

- `results/runs/<pilot_id>/pilot_report.json`
- `results/runs/<pilot_id>/shared_pretrain/...`
- `results/runs/<pilot_id>/seed_<seed>/...`
- `results/runs/<pilot_id>/extreme_probe/summary.json`

`summary.json` includes `warning_triggered` as a convenience summary field. The canonical per-eval warning signal remains `score` in `eval_log.jsonl`.
`pilot_report.json` includes an explicit `proceed`, `adjust`, or `fail` gate decision plus the shared-pretrain caveat and conservative sweep budget bound.

## Important Runtime Conventions

One-step transitions use bootstrap-multiplier semantics:

- continuing step: `discount = gamma`
- true terminal: `discount = 0.0`
- time-limit truncation: `discount = gamma`

The vertical slice also relies on these diagnostic conventions:

- TD diagnostics are computed from the recent-transition buffer only
- the first two eligible eval checkpoints are warmup-only and do not emit rows
- `eval_index` starts at `0` on the first emitted post-warmup row
- eval scheduling fires on the first step at-or-past each target interval, so non-divisible `eval_interval` values still work

## Testing

Always run:

```bash
python3 -m unittest discover -s tests -v
```

GitHub Actions runs this pure-Python suite on pull requests and pushes to `main` / `vertical_slice`.

Dependency-gated runtime tests can be run with the training environment:

```bash
./.venv/bin/python -m unittest tests.test_training_runtime -v
```

The heavy smoke test in `tests/test_training_smoke.py` is disabled by default and only runs when:

```bash
FINE_TUNE_STABILITY_RUN_TRAINING_SMOKE=1 python3 -m unittest tests.test_training_smoke -v
```

That smoke now covers both the original vertical slice and a tiny pilot run, including the shared pretrain, one fine-tune seed, and the extremal throughput probe.

## Further Notes

- Methodology details live in [`docs/methodology.md`](docs/methodology.md).
- Brax integration assumptions and hook points live in [`docs/integration.md`](docs/integration.md).
- Project-specific contributor guidance for agents and maintainers lives in [`AGENTS.md`](AGENTS.md).
