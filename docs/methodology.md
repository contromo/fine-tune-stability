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

The recent-buffer implementation lives in `atlas.recent_buffer` and is intended to store only freshly collected fine-tuning transitions.
