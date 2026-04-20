# Benchmark Data Card

This directory is the **frozen, immutable snapshot** of the empirical data behind the
`fine-tune-stability` paper. It is the reproducibility artifact: every number in the
paper is traceable to one of the CSV / JSON files below, and the vendored SVGs are the
canonical rendered figures. Integrity is checked via `manifest.sha256`.

## Contents

| File | Description |
| --- | --- |
| `horizon_final_return_points.csv` | Per-seed final-return results for the n-step horizon comparison (n ∈ {1, 3, 10}). |
| `horizon_final_return_summary.csv` | Per-horizon aggregates (mean / std / min / max / count). |
| `horizon_final_return_summary.json` | Same aggregates in JSON form, consumed by `atlas.paper_plots`. |
| `warning_pilot_runs.csv` | Per-run warning-signal pilot data (v3, v4): first warning eval, first collapse eval, lead time. |
| `warning_pilot_summary.csv` | Per-pilot aggregates including global ROC-AUC and mean lead time. |
| `warning_pilot_summary.json` | Same pilot aggregates in JSON form. |
| `figure1_horizon_final_return.svg` | Canonical rendered Figure 1 (horizon vs. final return). |
| `figure2_warning_pilot_summary.svg` | Canonical rendered Figure 2 (warning-signal pilot comparison). |
| `sample_eval_log.jsonl` | Twelve hand-crafted post-warmup eval rows across two synthetic runs, demonstrating the full `EvalLogRow` schema including `actor_kl_drift` and `q_magnitude_drift`. Feeds the alt-signal demo in `scripts/run_diagnostic.py` without requiring a training run. |
| `manifest.sha256` | SHA-256 hash of every file above. Run `shasum -a 256 -c manifest.sha256` to verify. |

## Schemas

Column definitions are the authoritative contract between this data and
`atlas.paper_analysis` / `atlas.paper_plots`. See `tests/test_paper_analysis.py` for the
authoritative column lists and expected caption strings.

**`horizon_final_return_points.csv`** — `horizon, seed, collapsed, warning_triggered, final_return_mean, final_return_std, steps_per_second, summary_path`.
`summary_path` contains absolute paths from the run that produced this snapshot; it is
provenance metadata, not a reproducibility input.

**`warning_pilot_runs.csv`** — `pilot_id, run_id, evals, first_warning_eval, first_collapse_eval, lead_time_evals`. Empty fields denote that no warning fired or no collapse occurred.

**`warning_pilot_summary.csv`** — pilot-level decision record. `global_roc_auc` and `mean_lead_time_evals` are `None` when insufficient positive/negative labels exist.

## How the data was generated

- **Horizon experiment:** SAC on `Go1JoystickFlatTerrain` (MuJoCo Playground), 1M
  environment-step fine-tune under a friction×payload shift, across n-step horizons
  n ∈ {1, 3, 10}. Seeds: 8 for n=1, 4 each for n=3 and n=10.
- **Warning-signal pilots (v3, v4):** same environment + shift, `n=1`, critic_width=256,
  3 seeds per pilot. v3 used a milder shift; v4 used a more severe shift that induces
  collapse in every seed.
- Per-eval diagnostic rows (`eval_log.jsonl`) were produced by
  `atlas_training.runtime.run_finetune`. This snapshot contains the aggregates, not the
  per-eval rows.

## Confounds and limits (read before citing)

- **Horizon confound:** the horizon sweep compares n-step targets *without holding the
  bootstrap target distribution fixed across horizons*. Differences in final return
  cannot be attributed to n alone.
- **Single environment, single shift:** one locomotion task with one friction×payload
  shift regime. Results do not generalize to other environments or shift types.
- **Small pilot counts:** v3 and v4 are 3-seed pilots at `n=1` only. The v4 ROC-AUC of
  0.298 is a 3-run estimate and not statistically meaningful on its own.
- **Negative warning result:** on v4, the TD-variance warning score fails to
  discriminate collapse runs (AUC < 0.5). This is the honest headline finding.
- **No per-eval time series in this snapshot:** Figure 3 (TD-variance trajectory) is
  therefore not regenerable from `benchmark/data/` alone. If needed, re-run the v4
  seeds via the repo's training CLIs to obtain `eval_log.jsonl`.

## Reproducibility commands

**Verify integrity of the snapshot:**

```
cd benchmark/data && shasum -a 256 -c manifest.sha256
```

**Re-run the warning-signal analysis on any `eval_log.jsonl`:**

```
python scripts/run_diagnostic.py \
    --eval-log path/to/eval_log.jsonl \
    --output results/diagnostic_summary.json
```

**Analyze an alternative signal (e.g. `actor_kl_drift`):**

```
python scripts/run_diagnostic.py \
    --eval-log path/to/eval_log.jsonl \
    --score-field actor_kl_drift \
    --trigger-threshold 0.5 \
    --output results/diag_actor_kl.json
```

**Demo against the vendored sample log (no training required):**

```
python scripts/run_diagnostic.py \
    --eval-log benchmark/data/sample_eval_log.jsonl \
    --score-field actor_kl_drift \
    --trigger-threshold 0.5 \
    --output /tmp/sample_diag.json
```

## Extending the benchmark

The signal catalog is instrumented in `atlas_training/`. To log a new warning signal
end-to-end:

1. Add the signal's computation alongside the existing `actor_kl_drift` /
   `q_magnitude_drift` functions in `atlas_training/signals.py`.
2. Add the field to `EvalLogRow` in `atlas_training/diagnostics.py`.
3. Wire the computation into `_run_training_loop` in `atlas_training/runtime.py` so the
   value is emitted on each post-warmup eval row.
4. Analyze with `python scripts/run_diagnostic.py --eval-log path/to/eval_log.jsonl
   --score-field your_signal_name` to compute ROC-AUC against collapse-horizon labels.
5. Pass `--trigger-threshold <value>` to additionally populate `first_warning_eval` and
   `lead_time_evals`.

Benchmark claim: **AUC for any score field; lead time only for fields with calibrated
trigger thresholds.**
