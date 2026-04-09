# Pilot Runbook

This is the canonical operator flow for running the pilot calibration gate on a single host.

## 1. Install the training stack

```bash
./scripts/setup.sh --train
```

The default `jax` dependency installed by `.[train]` is the CPU build so local smoke runs work by default. For a real GPU pilot, replace that install with the accelerator-specific JAX/JAXlib wheel set that matches the target host before proceeding.

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

## 3. Run the production-profile pilot

```bash
python3 scripts/run_pilot.py --profile production
```

Useful overrides:

```bash
python3 scripts/run_pilot.py \
  --profile production \
  --output-dir results/runs/pilot_gate_alt \
  --pretrain-steps 500000 \
  --fine-tune-steps 1000000
```

## 4. Understand the artifacts

Pilot artifacts live under `results/runs/<pilot_id>/`:

- `preflight.json`
- `pilot.log`
- `pilot_report.json`
- `shared_pretrain/...`
- `seed_<seed>/...`
- `extreme_probe/summary.json`

`pilot_report.json` embeds a stable `environment` block plus `preflight_path`. `pilot.log` appends a timestamped header on each attempt so interrupted reruns are visible in one file.

## 5. Interruption semantics

There is no resume in this milestone.

- If a phase crashes or is interrupted, rerunning starts that phase from the beginning.
- The current code does not skip completed phases.
- `summary.json` is written last for each phase and is reserved as the future completion sentinel for a later resume-focused change.

## 6. Decision handling

- `proceed`
  Use `scripts/run_sweep.py` to generate the manifest, then hand the manifest and single-run CLI to the external scheduler.
- `adjust`
  Change shift strength and/or calibration settings, then rerun the pilot.
- `fail`
  Fix infrastructure, runtime, or budget issues before attempting any sweep.

## 7. CPU-host validation

On a CPU development machine:

```bash
python3 scripts/run_pilot.py --profile production --preflight-only
```

This should fail fast unless `--allow-cpu` is passed. That behavior is intentional.
