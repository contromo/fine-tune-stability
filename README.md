# fine-tune-stability

A reproducible benchmark for **fine-tuning-under-shift** of robotic RL policies. One
locomotion environment (`Go1JoystickFlatTerrain` via MuJoCo Playground), one severe
dynamics shift (friction × payload), and **three warning signals instrumented per eval
row** so alternative collapse-detection heuristics can be compared on the same runs.

The accompanying paper (post-mortem) reports a negative result on the canonical
TD-variance warning signal under a severe shift. The repository is designed to let
others plug in their own signals against the same logged diagnostics.

> **Status (2026-04):** 1 environment × 1 shift × 3 signals, small seed counts, `n=1`-only
> warning pilots. The primary warning result is negative; see [benchmark/README.md](benchmark/README.md)
> for confounds and limits. The value on offer is the infrastructure, not a new SOTA.

## Signal catalog

Every post-warmup eval row in `eval_log.jsonl` carries:

| Field | Formula | Where |
| --- | --- | --- |
| `score` | `log(var_recent_TD) − log(var_warmup_TD)` | canonical TD-variance warning ([atlas/diagnostics.py](atlas/diagnostics.py)) |
| `actor_kl_drift` | diagonal-Gaussian KL from pretrained to current policy on a probe batch | [atlas_training/signals.py](atlas_training/signals.py) |
| `q_magnitude_drift` | `log(mean(min(\|Q1\|,\|Q2\|))_current) − log(…)_pretrained` on a probe batch | [atlas_training/signals.py](atlas_training/signals.py) |

Only `score` has a calibrated trigger threshold (`log(3)`). Alternative signals produce
threshold-free ROC-AUC against collapse-horizon labels by default; pass
`--trigger-threshold` to `scripts/run_diagnostic.py` to additionally compute
`first_warning_eval` / `lead_time_evals` for them.

**Adding a new signal:** (1) add the function to [atlas_training/signals.py](atlas_training/signals.py),
(2) add the optional field to `EvalLogRow` in [atlas_training/diagnostics.py](atlas_training/diagnostics.py),
(3) wire it into `_run_training_loop` in [atlas_training/runtime.py](atlas_training/runtime.py)
so the per-eval value is written to `eval_log.jsonl`, (4) analyze via
`python scripts/run_diagnostic.py --score-field your_signal_name`.

## Quickstart (CPU smoke)

```bash
./scripts/setup.sh --train
./scripts/run_all.sh
```

The smoke flow uses tiny training settings (wiring check, not an experiment). End-to-end
CPU runtime on macOS is under 15 minutes.

Run the test suite:

```bash
python3 -m unittest discover tests
```

## Benchmark data

The frozen paper data lives in [benchmark/data/](benchmark/data/) with a SHA-256
manifest. Every number in the paper is traceable to one of these files.

- [`benchmark/README.md`](benchmark/README.md) — data card, schemas, confounds.
- `benchmark/data/manifest.sha256` — verify via `cd benchmark/data && shasum -a 256 -c manifest.sha256`.

Schema stability is enforced by `tests/test_benchmark_data.py`.

## Full-scale reproducibility (GPU recipe)

The headline numbers in the paper were produced on a single A100 via Runpod. This is a
documented recipe, not a one-command reproduction.

```bash
# 1. Install the training stack on a GPU host (replace jax install with CUDA wheel).
./scripts/setup.sh --train
pip install --upgrade "jax[cuda12]==0.9.2"

# 2. Pretrain a checkpoint (~2 GPU-hours at 2M steps).
python3 scripts/run_pretrain.py --output-dir results/runs/pretrain_go1

# 3. Generate the fine-tune sweep manifest.
python3 scripts/run_sweep.py --output results/sweep_manifest.json

# 4. Execute fine-tune seeds from the manifest (one invocation per seed, or batched via
#    your scheduler of choice). Each seed takes ~3 GPU-hours at 2M fine-tune steps.
python3 scripts/run_finetune.py \
  --checkpoint results/runs/pretrain_go1/checkpoint \
  --output-dir results/runs/finetune_go1_seed0

# 5. Summarize the canonical TD-variance signal.
python3 scripts/run_diagnostic.py \
  --eval-log results/runs/finetune_go1_seed0/eval_log.jsonl \
  --output results/diagnostic_summary.json

# 6. Analyze an alternative signal (AUC for any field; lead time requires a threshold).
python3 scripts/run_diagnostic.py \
  --eval-log results/runs/finetune_go1_seed0/eval_log.jsonl \
  --score-field actor_kl_drift \
  --trigger-threshold 0.5 \
  --output results/diag_actor_kl.json
```

Budgeted cost at $1.50/GPU-hr for the v4 pilot (3 seeds × pretrain + fine-tune): ~$20.

## Install details

```bash
./scripts/setup.sh           # base install
./scripts/setup.sh --train   # training stack
VENV_PATH=/abs/path/.venv ./scripts/setup.sh --train  # alternative venv
```

Notes:

- `.[train]` pins `brax 0.14.2`, `jax 0.9.2`, `mujoco 3.6.0`, `mujoco-mjx 3.6.0`,
  `playground 0.2.0`.
- Default install resolves the CPU JAX wheel so smoke runs work out of the box.
- GPU users replace the JAX install with the CUDA wheel for their accelerator.
- Brax is pinned to `0.14.2` because `atlas_training` depends on 0.14.x evaluator and
  replay internals.

### Runpod / network filesystems

On Runpod, `/workspace` is often a slow network mount. Keep the venv and UV cache on
local container disk:

```bash
mkdir -p /root/.venvs /root/.cache/uv
export UV_CACHE_DIR=/root/.cache/uv UV_LINK_MODE=copy
VENV_PATH=/root/.venvs/fine-tune-stability ./scripts/setup.sh --train
ln -sfn /root/.venvs/fine-tune-stability .venv
```

## Repo layout

- `atlas/` — pure, dependency-light research utilities (n-step aggregation, recent
  buffers, TD diagnostics, KL helper).
- `atlas_training/` — Brax + MuJoCo Playground vertical slice (SAC pretrain/fine-tune,
  probe-context signals, eval-log emission).
- `benchmark/data/` — frozen paper snapshot with SHA manifest and data card.
- `scripts/` — CLI entrypoints (`run_pretrain`, `run_finetune`, `run_pilot`,
  `run_diagnostic`, `run_sweep`) and the smoke flow `run_all.sh`.
- `tests/` — pure-Python tests plus dependency-gated runtime coverage.
- `docs/` — methodology, Brax integration notes, pilot runbook.

## Artifacts produced by training

- Pretrain: `config.json`, `summary.json`, `checkpoint/checkpoint.msgpack`.
- Fine-tune: `config.json`, `pretrain_baseline.json`, `eval_log.jsonl`, `summary.json`
  (includes `probe_size` when the alt-signal probe is active).
- Pilot: `preflight.json`, `pilot.log`, `pilot_report.json`, per-seed outputs,
  `extreme_probe/summary.json`, decision note under `docs/decisions/`.

## Runtime conventions

One-step transitions use bootstrap-multiplier semantics:
- continuing step → `discount = gamma`
- true terminal → `discount = 0.0`
- time-limit truncation → `discount = gamma`

Diagnostic conventions:
- TD diagnostics are computed from the recent-transition buffer only.
- The first two eligible eval checkpoints are warmup-only (no rows emitted).
- `eval_index` starts at `0` on the first post-warmup row.
- The alt-signal probe batch is sampled once at fine-tune start via
  `replay_buffer.sample(...)`, advancing the replay RNG by one sample before the main
  loop begins. This is intentional and logged as `probe_size` in the finetune summary.

## Testing

```bash
python3 -m unittest discover tests                       # pure-Python, always runs
./.venv/bin/python -m unittest tests.test_training_runtime  # requires training stack
FINE_TUNE_STABILITY_RUN_TRAINING_SMOKE=1 \                  # gated heavy smoke
  python3 -m unittest tests.test_training_smoke
```

## Further notes

- Methodology: [`docs/methodology.md`](docs/methodology.md).
- Brax integration assumptions: [`docs/integration.md`](docs/integration.md).
- Pilot operating instructions: [`docs/pilot_runbook.md`](docs/pilot_runbook.md).
- Scheduler handoff after a successful pilot:
  ```bash
  python3 scripts/run_sweep.py \
    --from-pilot-report results/runs/pilot_gate/pilot_report.json \
    --output results/sweep_manifest.json
  ```
- Do not generate any sweep or sensitivity manifest from a pilot report that ended in
  `adjust` or `fail`.

## Citation

```
@software{fine_tune_stability_2026,
  author = {Manav Mehra},
  title = {fine-tune-stability: A Benchmark for Warning-Signal Evaluation under
           Dynamics Shift},
  year = {2026},
  url = {https://github.com/manavm0/fine-tune-stability}
}
```

See also [`CITATION.cff`](CITATION.cff).
