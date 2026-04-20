# Contributing

Thanks for your interest in `fine-tune-stability`. This repository is the reproducibility
artifact behind the paper; contributions that strengthen the benchmark, harden the
reproducibility story, or extend the warning-signal catalog are all welcome.

## Environment setup

```
./scripts/setup.sh           # CPU-only: enough to run the analysis path and tests
./scripts/setup.sh --train   # Installs the GPU training stack (Brax + MuJoCo Playground)
```

## Running tests

```
python -m unittest discover tests
```

Tests that require the training stack are auto-skipped when JAX/Brax are unavailable,
so the default command is safe to run in a CPU-only checkout. The end-to-end GPU
training smoke is gated behind an environment variable:

```
FINE_TUNE_STABILITY_RUN_TRAINING_SMOKE=1 python -m unittest tests.test_training_smoke
```

## Adding a new warning signal

The signal catalog is instrumented end-to-end across four files. To add a new signal:

1. **Compute the signal** alongside `actor_kl_drift` / `q_magnitude_drift` in
   [atlas_training/signals.py](atlas_training/signals.py). Consume the frozen
   `ProbeContext` if you need pretrained parameters.
2. **Declare the field** on `EvalLogRow` in
   [atlas_training/diagnostics.py](atlas_training/diagnostics.py). Keep it
   `Optional[float]` with a `None` default so the JSONL schema stays
   backward-compatible.
3. **Emit it** from `_run_training_loop` in
   [atlas_training/runtime.py](atlas_training/runtime.py) on each post-warmup eval,
   guarded on `probe_context is not None` where the signal depends on the probe.
4. **Analyze it** with
   `python scripts/run_diagnostic.py --eval-log path/to/eval_log.jsonl --score-field your_signal_name`.
   Pass `--trigger-threshold <value>` only when you have a calibrated threshold.

Benchmark rule: **ROC-AUC is available for any score field; lead-time metrics require
a calibrated trigger threshold.** Unthresholded fields are scored on AUC only.

## Benchmark snapshot

`benchmark/data/` is immutable. If you regenerate a file in that directory, regenerate
`manifest.sha256` as well:

```
cd benchmark/data && rm -f manifest.sha256 && shasum -a 256 * | sort -k2 > manifest.sha256
```

`tests/test_benchmark_data.py` enforces schema stability and manifest integrity, so
any benchmark edit must pass those checks before review.
