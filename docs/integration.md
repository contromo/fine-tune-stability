# Brax Integration Notes

The current repo contains reusable core logic but not a bundled Brax SAC fork. When you wire this into the actual training code, use these hook points.

## 1. One-step transition normalization

Before any transition reaches replay:

- ensure `transition.discount` uses bootstrap-multiplier semantics
- call `atlas.time_limit.apply_timeout_bootstrap(transition, gamma)` if the environment signals `extras["state_extras"]["time_out"] == 1`

If your environment wrapper already emits `gamma` on continuing steps and `0` on terminal steps, timeout handling is the only special case.

## 2. n-step replay insertion

Replace direct replay insertion with:

1. Push the one-step transition through `NStepTransitionAggregator`
2. Insert every emitted aggregated transition into the main replay
3. Also append each emitted transition into `RecentTransitionBuffer`

For vectorized actors, use `MultiStreamNStepAggregator` with the environment index as `stream_id`.

## 3. Diagnostic callback

At each eval checkpoint:

1. Sample or snapshot the recent buffer
2. Compute TD errors with the current critic and target value estimate
3. Summarize with `summarize_td_errors`
4. Update `InstabilityTrigger`

Persist at least:

- eval step
- mean eval return
- collapse flag
- TD variance
- 95th percentile absolute TD error
- diagnostic score

## 4. Validation data format

`scripts/run_diagnostic.py` expects JSONL rows with these fields:

- `run_id`
- `eval_index`
- `score`
- `collapsed`

Optional fields:

- `return_mean`
- `variance`
- `q95_abs_td`

This is sufficient to compute per-run warning lead times and global ROC-AUC for collapse prediction.
