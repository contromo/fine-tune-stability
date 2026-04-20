# Methodology Snapshot

This repository encodes the preregistered constants from the sprint plan:

- `n in {1, 3, 10}`
- critic width in `{256, 1024}` with fixed depth `3`
- actor width fixed at `256 x 3`
- `gamma = 0.99`
- collapse threshold `min(mu0 - 2 sigma0, mu0 - 0.2 |mu0|)`
- instability trigger `log(var_t) - log(var_warmup) > log(3)` for two consecutive evals

## Transition Convention

The core modules use a single convention throughout:

- one-step continuing transition discount equals `gamma`
- environment termination sets discount to `0.0`
- time-limit truncation also uses `gamma` so value bootstrapping is preserved

This is the same convention expected by the n-step aggregator and diagnostic TD-error helpers.

## n-step Aggregation

`atlas.nstep.NStepTransitionAggregator` performs naive off-policy aggregation:

- reward: `sum_{k=0}^{n-1} gamma^k r_{t+k}`
- discount: product of per-step bootstrap multipliers
- observation/action: from the first transition in the window
- next observation: from the final transition in the window
- extras: copied from the final transition plus atlas metadata

On true terminals, the aggregator flushes the remaining partial windows so the last `n - 1` starts are not silently lost.

## Diagnostics

`atlas.diagnostics` includes:

- collapse threshold computation
- TD-error computation
- variance and 95th percentile summaries
- a hold-based early-warning trigger
- simple evaluation metrics such as ROC-AUC and Pearson correlation

The recent-buffer implementation lives in `atlas.recent_buffer` and is intended to store only freshly collected fine-tuning transitions. In the current vertical slice:

- diagnostic TD errors are computed from recent-buffer samples only
- the first two eligible eval checkpoints are warmup-only and do not emit `eval_log.jsonl` rows
- emitted eval rows begin at `eval_index = 0` on the first post-warmup checkpoint

## Instrumented Warning Signals

Every post-warmup eval row carries up to three warning signals:

| Field | Formula | Source |
| --- | --- | --- |
| `score` | `log(var_recent_TD + eps) - log(var_warmup_TD + eps)` | canonical TD-variance (`atlas/diagnostics.py::summarize_td_errors`) |
| `actor_kl_drift` | diagonal-Gaussian KL from pretrained to current policy on a probe batch, sum over action dims, mean over batch | `atlas_training/signals.py::actor_kl_drift` |
| `q_magnitude_drift` | `log(mean(min(\|Q1\|,\|Q2\|))_current + eps) - log(mean(min(\|Q1_pre\|,\|Q2_pre\|)) + eps)` on a probe batch | `atlas_training/signals.py::q_magnitude_drift` |

Only `score` has a calibrated trigger threshold (`log(3)` for two consecutive evals).
`actor_kl_drift` and `q_magnitude_drift` are emitted unthresholded; downstream analysis
computes ROC-AUC against collapse-horizon labels directly, and lead-time metrics are
only produced when a user-provided trigger threshold is supplied via
`scripts/run_diagnostic.py --trigger-threshold`. KL is computed on the underlying
pre-tanh Normal with `scale = softplus(scale_logit) + min_std` matching
`brax.training.distribution.NormalTanhDistribution`. The `min(|Q1|, |Q2|)` rule is a
conservative twin-critic magnitude heuristic consistent with SAC's twin-critic design
philosophy; it is a benchmark-level pinning, not a theoretical derivation.
