# Brax Integration Notes

The repository now has two layers:

- `atlas/` stays pure and holds the reusable scientific logic
- `atlas_training/` contains the current Brax/JAX/MuJoCo Playground vertical slice

If you extend or replace the current training runtime, keep the same boundaries and hook points below.

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

At each eligible eval checkpoint:

1. Require at least `diagnostic_min_transitions` recent transitions
2. Sample the recent buffer only, using the configured diagnostic minibatch workload
3. Compute TD errors with SAC semantics:
   - target value: `min(Q1_target, Q2_target) - alpha * log pi(a' | s')`
   - online estimate: `Q1_online(s, a)`
4. Use the first two eligible evals as warmup variance only and emit no log rows yet
5. On the third eligible eval and later, summarize with `summarize_td_errors`
6. Update `InstabilityTrigger`

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
- `threshold`
- `env_steps`

This is sufficient to compute per-run warning lead times and global ROC-AUC for collapse prediction.
