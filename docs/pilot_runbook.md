# Pilot Runbook

This is the canonical operator flow for running the pilot calibration gate on a single host.

## 1. Install the training stack

```bash
./scripts/setup.sh --train
```

The default `jax` dependency installed by `.[train]` is the CPU build so local smoke runs work by default. For a real GPU pilot, replace that install with the accelerator-specific JAX/JAXlib wheel set that matches the target host before proceeding.

On Runpod or any host where the repo lives on a network-mounted workspace, put the virtualenv on local container disk and keep only the checkout/results on the network volume:

```bash
mkdir -p /root/.venvs /root/.cache/uv
export UV_CACHE_DIR=/root/.cache/uv
export UV_LINK_MODE=copy
VENV_PATH=/root/.venvs/fine-tune-stability ./scripts/setup.sh --train
ln -sfn /root/.venvs/fine-tune-stability .venv
```

If the target host is a GPU box, replace the default CPU JAX wheel afterwards with the accelerator-specific build that matches the image:

```bash
uv pip install --python .venv/bin/python --upgrade "jax[cuda12]==0.9.2"
```

## 2. Validate the host

Standalone preflight:

```bash
python3 scripts/preflight_pilot.py --profile production
```

If the host is CPU-only and you intentionally want to validate the flow anyway:

```bash
python3 scripts/preflight_pilot.py --profile production --allow-cpu
```

`preflight.json` is written to `results/runs/pilot_gate/preflight.json` by default. Running the real pilot later reruns preflight and overwrites that file intentionally so the embedded provenance matches the actual pilot run.

## 3. Run the GPU smoke pilot first

Validate the production path cheaply on the real host before committing to the full pilot:

```bash
python3 scripts/run_pilot.py \
  --profile production \
  --run-id pilot_gpu_smoke \
  --output-dir results/runs/pilot_gpu_smoke \
  --seeds 0,1 \
  --pretrain-steps 50000 \
  --eval-interval 10000 \
  --fine-tune-steps 50016 \
  --baseline-eval-episodes 10 \
  --throughput-probe-updates 50
```

Smoke success criteria:

- `preflight.json` reports the expected GPU backend and devices
- shared pretrain completes
- both fine-tune seeds emit at least 3 post-warmup eval rows
- `extreme_probe/summary.json` exists and reports finite throughput
- `pilot_report.json` is written
- no OOM, backend, or MuJoCo/Brax wiring failures occur

The smoke run's `decision` field is advisory only. Use the structural criteria above to decide whether to continue to the full pilot.

## 4. Run the production-profile pilot

```bash
python3 scripts/run_pilot.py --profile production
```

Useful overrides:

```bash
python3 scripts/run_pilot.py \
  --profile production \
  --output-dir results/runs/pilot_gate_alt \
  --decision-dir docs/decisions \
  --pretrain-steps 500000 \
  --fine-tune-steps 1000000
```

`--force` is global in this milestone. If any phase already has `summary.json`, rerunning without `--force` fails before that phase starts. Rerunning with `--force` permits overwriting completed phases globally; there is no per-phase force selector yet.

## 5. Understand the artifacts

Pilot artifacts live under `results/runs/<pilot_id>/`:

- `preflight.json`
- `pilot.log`
- `pilot_report.json`
- `shared_pretrain/...`
- `seed_<seed>/...`
- `extreme_probe/summary.json`

`pilot_report.json` embeds a stable `environment` block plus `preflight_path`. `pilot.log` appends a timestamped header on each attempt so interrupted reruns are visible in one file.
A durable decision stub is also written to `docs/decisions/YYYY-MM-DD-<run_id>.md` by default. Override the location with `--decision-dir` if the pilot is being orchestrated outside the repo checkout.
If the decision file already exists and has been edited, rerunning with the same date and `run_id` fails; use a different `run_id` or clean up the old note first.

## 6. Interruption semantics

There is no resume in this milestone.

- If a phase crashes or is interrupted, rerunning starts that phase from the beginning.
- The current code does not skip completed phases.
- `summary.json` is written last for each phase and is reserved as the future completion sentinel for a later resume-focused change.

## 7. Decision handling

- `proceed`
  Generate the manifest from the pilot report, then hand the manifest and single-run CLI to the external scheduler. This handoff is only valid for a production-profile pilot run on the target hardware, and the decision note must be recorded before launch.
  ```bash
  python3 scripts/run_sweep.py \
    --from-pilot-report results/runs/pilot_gate/pilot_report.json \
    --output results/sweep_manifest.json
  ```
  The default manifest uses the 2M-step main-study horizon. If you intentionally launch a different horizon, rerun the pilot with matching `--sweep-fine-tune-steps` and pass the same `--fine-tune-steps` override here.
  Generate the representative-cell pretrain-sensitivity manifest separately:
  ```bash
  python3 scripts/run_pretrain_sensitivity.py \
    --from-pilot-report results/runs/pilot_gate/pilot_report.json \
    --output results/pretrain_sensitivity_manifest.json
  ```
- `adjust`
  Change shift strength and/or threshold calibration only, then rerun the pilot.
- `fail`
  Fix infrastructure, runtime, or budget issues before attempting any sweep.

Do not generate either manifest from a pilot that ended in `adjust` or `fail`, and do not reuse an outdated pilot report after changing the intended sweep horizon.

### Adjust matrix

- Majority `drop_fraction < 0.15`
  Increase shift severity only.
- Majority `drop_fraction > 0.50`
  Decrease shift severity only.
- Drop band is acceptable, but `threshold_drop_fraction > 0.50`
  Keep shift fixed and adjust threshold only.
- If multiple conditions hold simultaneously, or different seeds fall on opposite sides of the drop band
  Adjust shift first, rerun, and only then reconsider threshold calibration.

Threshold-only adjustments:

- If `threshold_rule == "sigma"`, reduce `collapse_c` by `0.5`.
- If `threshold_rule == "floor"`, reduce `collapse_rho` by `0.05`.

Do not change `num_envs`, `eval_interval`, or other operational knobs as part of an `adjust` response in this milestone.

## 8. CPU-host validation

On a CPU development machine:

```bash
python3 scripts/run_pilot.py --profile production --preflight-only
```

This should fail fast unless `--allow-cpu` is passed. That behavior is intentional.
