from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

EPS = 1e-8


@dataclass(frozen=True)
class ProbeContext:
    """Frozen pretrained snapshot used for alt-signal drift scoring.

    `probe_obs` is the observation pytree returned by `replay_buffer.sample`
    (identical structure to what the policy/critic consume during training).
    `probe_actions` is the deterministic pretrained mode `tanh(loc)` evaluated
    on `probe_obs`.
    """

    policy_params: Any
    q_params: Any
    normalizer_params: Any
    probe_obs: Any
    probe_actions: jnp.ndarray
    probe_size: int


def _freeze_tree(tree: Any) -> Any:
    return jax.tree_util.tree_map(jnp.asarray, tree)


def _split_loc_scale(runtime: dict[str, Any], dist_params: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Extract (loc, scale) of the underlying Normal from NormalTanhDistribution params.

    Mirrors `NormalTanhDistribution.create_dist` (brax.training.distribution):
    splits logits in half, applies softplus and adds `min_std` to the scale half.
    """
    parametric = runtime["sac_network"].parametric_action_distribution
    loc, scale_logit = jnp.split(dist_params, 2, axis=-1)
    min_std = getattr(parametric, "_min_std", 0.001)
    scale = jax.nn.softplus(scale_logit) + min_std
    return loc, scale


def build_probe_context(
    runtime: dict[str, Any],
    training_state: Any,
    replay_buffer,
    replay_state,
) -> tuple[ProbeContext, Any]:
    """Sample a probe batch and freeze the current (pretrained) params.

    Calls `replay_buffer.sample(replay_state)` exactly once; the returned
    `new_replay_state` MUST be threaded forward by the caller so the sampling
    RNG advance is consistent with downstream training.
    """
    new_replay_state, transitions = replay_buffer.sample(replay_state)
    probe_obs = transitions.observation
    probe_size = int(jax.tree_util.tree_leaves(probe_obs)[0].shape[0])

    dist_params = runtime["sac_network"].policy_network.apply(
        training_state.normalizer_params,
        training_state.policy_params,
        probe_obs,
    )
    loc, _scale = _split_loc_scale(runtime, dist_params)
    probe_actions = jnp.tanh(loc)

    return (
        ProbeContext(
            policy_params=_freeze_tree(training_state.policy_params),
            q_params=_freeze_tree(training_state.q_params),
            normalizer_params=_freeze_tree(training_state.normalizer_params),
            probe_obs=jax.tree_util.tree_map(jnp.asarray, probe_obs),
            probe_actions=jnp.asarray(probe_actions),
            probe_size=probe_size,
        ),
        new_replay_state,
    )


def actor_kl_drift(
    runtime: dict[str, Any],
    probe: ProbeContext,
    current_policy_params: Any,
    current_normalizer_params: Any,
) -> float:
    """Diagonal-Gaussian KL from pretrained to current policy on the probe batch.

    Computed on the underlying (pre-tanh) Normal — the tanh squash is
    deterministic and cancels out of KL. Returned as a Python float.
    """
    policy_network = runtime["sac_network"].policy_network

    cur_params = policy_network.apply(
        current_normalizer_params,
        current_policy_params,
        probe.probe_obs,
    )
    pre_params = policy_network.apply(
        probe.normalizer_params,
        probe.policy_params,
        probe.probe_obs,
    )
    loc_cur, scale_cur = _split_loc_scale(runtime, cur_params)
    loc_pre, scale_pre = _split_loc_scale(runtime, pre_params)

    var_pre = scale_pre * scale_pre
    var_cur = scale_cur * scale_cur
    log_ratio = jnp.log(scale_cur) - jnp.log(scale_pre)
    per_dim = log_ratio + (var_pre + (loc_pre - loc_cur) ** 2) / (2.0 * var_cur) - 0.5
    per_sample = jnp.sum(per_dim, axis=-1)
    return float(jnp.mean(per_sample))


def q_magnitude_drift(
    runtime: dict[str, Any],
    probe: ProbeContext,
    current_q_params: Any,
    current_normalizer_params: Any,
) -> float:
    """log(mean(min(|Q1|,|Q2|))_current) - log(mean(...)_pretrained) on probe batch.

    Uses `min(|Q1|, |Q2|)` as a conservative twin-critic magnitude rule,
    consistent with SAC's twin-critic design philosophy.
    """
    q_network = runtime["sac_network"].q_network
    q_cur = q_network.apply(current_normalizer_params, current_q_params, probe.probe_obs, probe.probe_actions)
    q_pre = q_network.apply(probe.normalizer_params, probe.q_params, probe.probe_obs, probe.probe_actions)
    mag_cur = jnp.mean(jnp.min(jnp.abs(q_cur), axis=-1))
    mag_pre = jnp.mean(jnp.min(jnp.abs(q_pre), axis=-1))
    return float(jnp.log(mag_cur + EPS) - jnp.log(mag_pre + EPS))
