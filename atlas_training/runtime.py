from __future__ import annotations

import argparse
import functools
import json
import random
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import flax
from flax import serialization as flax_serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax
from brax.envs.wrappers import training as brax_training
from brax.training import acting
from brax.training import distribution
from brax.training import gradients
from brax.training import networks as brax_networks
from brax.training import replay_buffers
from brax.training import types as brax_types
from brax.training.acme import running_statistics
from brax.training.acme import specs
from brax.training.agents.sac import networks as sac_networks
from mujoco_playground import registry
from mujoco_playground._src import wrapper as playground_wrapper

from atlas import InstabilityTrigger, MultiStreamNStepAggregator, RecentTransitionBuffer, summarize_td_errors, td_error
from atlas.time_limit import extract_timeout_flag
from atlas.transitions import Transition as AtlasTransition
from atlas_training.config import VerticalSliceConfig, build_run_id, checkpoint_signature, validate_checkpoint_compatibility
from atlas_training.diagnostics import (
    DiagnosticLogState,
    FrozenBaseline,
    advance_next_eval_at,
    current_warmup_variance,
    freeze_baseline,
    make_eval_log_row,
    mark_eval_row_emitted,
    record_warmup_variance,
)

FLOOR_GEOM_ID = 0
TORSO_BODY_ID = 1


@flax.struct.dataclass
class TrainingState:
    policy_optimizer_state: Any
    policy_params: Any
    q_optimizer_state: Any
    q_params: Any
    target_q_params: Any
    alpha_optimizer_state: Any
    alpha_params: Any
    normalizer_params: Any


def run_pretrain_cli(args: argparse.Namespace) -> None:
    config = VerticalSliceConfig(
        stage="pretrain",
        output_dir=args.output_dir,
        run_id=args.run_id or build_run_id("pretrain", args.n_step, args.critic_width, args.seed),
        env_name=args.env_name,
        n_step=args.n_step,
        critic_width=args.critic_width,
        seed=args.seed,
        train_steps=args.train_steps,
        eval_interval=args.eval_interval,
        num_envs=args.num_envs,
        eval_episodes=args.eval_episodes,
        batch_size=args.batch_size,
        replay_capacity=args.replay_capacity,
        min_replay_size=args.min_replay_size,
        diagnostic_min_transitions=args.diagnostic_min_transitions,
        diagnostic_minibatches=args.diagnostic_minibatches,
        diagnostic_batch_size=args.diagnostic_batch_size,
        episode_length=args.episode_length,
        action_repeat=args.action_repeat,
        stop_on_collapse=False,
    )
    summary = run_pretrain(config)
    print(f"Wrote {config.output_dir}")
    print(json.dumps({"run_id": summary["run_id"], "final_env_steps": summary["final_env_steps"]}, indent=2))


def run_finetune_cli(args: argparse.Namespace) -> None:
    config = VerticalSliceConfig(
        stage="finetune",
        output_dir=args.output_dir,
        checkpoint=args.checkpoint,
        run_id=args.run_id or build_run_id("finetune", args.n_step, args.critic_width, args.seed),
        env_name=args.env_name,
        n_step=args.n_step,
        critic_width=args.critic_width,
        seed=args.seed,
        train_steps=args.train_steps,
        eval_interval=args.eval_interval,
        num_envs=args.num_envs,
        eval_episodes=args.eval_episodes,
        batch_size=args.batch_size,
        replay_capacity=args.replay_capacity,
        min_replay_size=args.min_replay_size,
        diagnostic_min_transitions=args.diagnostic_min_transitions,
        diagnostic_minibatches=args.diagnostic_minibatches,
        diagnostic_batch_size=args.diagnostic_batch_size,
        episode_length=args.episode_length,
        action_repeat=args.action_repeat,
        stop_on_collapse=args.stop_on_collapse,
    )
    summary = run_finetune(config)
    print(f"Wrote {config.output_dir}")
    print(json.dumps({"run_id": summary["run_id"], "collapsed": summary["collapsed"]}, indent=2))


def run_pretrain(config: VerticalSliceConfig) -> dict[str, Any]:
    config = config.with_run_id()
    _ensure_output_dir(config.output_dir)
    _write_json(config.config_path(), config.to_dict())

    runtime = _build_runtime(config)
    train_env = _build_env(config, batch_size=config.num_envs, domain="pretrain")
    eval_env = _build_env(config, batch_size=config.eval_episodes, domain="nominal_eval")
    evaluator = _build_evaluator(runtime, eval_env, config.eval_episodes)

    key = jax.random.PRNGKey(config.seed)
    key, env_key, state_key = jax.random.split(key, 3)
    env_state = train_env.reset(jax.random.split(env_key, config.num_envs))
    training_state = _init_training_state(state_key, runtime, config)
    replay_buffer, replay_state = _init_replay_buffer(runtime, config, key)

    training_state, replay_state, env_state, final_metrics, env_steps, replay_size, warning_triggered, _collapsed = _run_training_loop(
        config=config,
        runtime=runtime,
        train_env=train_env,
        eval_callback=lambda state, _: _evaluate_policy(runtime, evaluator, state),
        training_state=training_state,
        replay_buffer=replay_buffer,
        replay_state=replay_state,
        env_state=env_state,
        restore_replay_size=0,
        enable_diagnostics=False,
    )

    nominal_metrics = _evaluate_policy(runtime, evaluator, training_state)
    _save_checkpoint(
        config.checkpoint_dir(),
        config,
        runtime["obs_size"],
        training_state,
        replay_state,
        replay_size=replay_size,
    )

    summary = {
        "stage": "pretrain",
        "created_at": _utc_now(),
        "run_id": config.run_id,
        "checkpoint_dir": str(config.checkpoint_dir()),
        "final_env_steps": env_steps,
        "final_eval_return_mean": nominal_metrics["return_mean"],
        "final_eval_return_std": nominal_metrics["return_std"],
        "warning_triggered": warning_triggered,
        "training_metrics": final_metrics,
    }
    _write_json(config.summary_path(), summary)
    return summary


def run_finetune(config: VerticalSliceConfig) -> dict[str, Any]:
    config = config.with_run_id()
    if config.checkpoint is None:
        raise ValueError("--checkpoint is required for fine-tuning")
    _ensure_output_dir(config.output_dir)
    checkpoint_payload = _load_checkpoint(config.checkpoint)
    runtime = _build_runtime(config)
    validate_checkpoint_compatibility(
        config,
        checkpoint_payload["metadata"],
        observation_spec=runtime["obs_size"],
        observation_dtype="float32",
    )
    _write_json(config.config_path(), config.to_dict())
    train_env = _build_env(config, batch_size=config.num_envs, domain="finetune")
    eval_env = _build_env(config, batch_size=config.eval_episodes, domain="finetune")
    evaluator = _build_evaluator(runtime, eval_env, config.eval_episodes)

    key = jax.random.PRNGKey(config.seed)
    key, env_key, state_key = jax.random.split(key, 3)
    env_state = train_env.reset(jax.random.split(env_key, config.num_envs))
    training_state = _init_training_state(state_key, runtime, config)
    replay_buffer, replay_state = _init_replay_buffer(runtime, config, key)
    training_state = _tree_to_jax(
        flax_serialization.from_state_dict(training_state, checkpoint_payload["training_state"])
    )
    replay_state = _tree_to_jax(
        flax_serialization.from_state_dict(replay_state, checkpoint_payload["replay_state"])
    )

    baseline_returns = _evaluate_policy(runtime, evaluator, training_state)["episode_returns"]
    baseline = freeze_baseline(baseline_returns)
    _write_json(
        config.pretrain_baseline_path(),
        {
            "created_at": _utc_now(),
            "run_id": config.run_id,
            "mu0": baseline.mu0,
            "sigma0": baseline.sigma0,
            "threshold": baseline.threshold,
        },
    )

    def eval_callback(state: TrainingState, _: int) -> dict[str, Any]:
        return _evaluate_policy(runtime, evaluator, state)

    training_state, replay_state, env_state, final_metrics, env_steps, _replay_size, warning_triggered, collapsed = _run_training_loop(
        config=config,
        runtime=runtime,
        train_env=train_env,
        eval_callback=eval_callback,
        training_state=training_state,
        replay_buffer=replay_buffer,
        replay_state=replay_state,
        env_state=env_state,
        restore_replay_size=int(checkpoint_payload["replay_size"]),
        enable_diagnostics=True,
        baseline=baseline,
        eval_log_path=config.eval_log_path(),
    )

    summary = {
        "stage": "finetune",
        "created_at": _utc_now(),
        "run_id": config.run_id,
        "checkpoint": str(config.checkpoint),
        "final_env_steps": env_steps,
        "collapsed": collapsed,
        "threshold": baseline.threshold,
        "warning_triggered": warning_triggered,
        "training_metrics": final_metrics,
    }
    _write_json(config.summary_path(), summary)
    return summary


def _build_runtime(config: VerticalSliceConfig) -> dict[str, Any]:
    probe_env = _build_env(config, batch_size=config.num_envs, domain="pretrain")
    obs_size = probe_env.observation_size
    policy_obs_size = _obs_dim(obs_size["state"] if isinstance(obs_size, dict) else obs_size)
    critic_obs_size = _obs_dim(
        obs_size.get("privileged_state", obs_size["state"]) if isinstance(obs_size, dict) else obs_size
    )
    action_size = probe_env.action_size

    def policy_preprocess(observations, mean_std):
        if config.normalize_observations:
            return running_statistics.normalize(observations, mean_std)
        return observations

    def critic_preprocess(observations, _mean_std):
        if isinstance(observations, dict):
            return observations.get("privileged_state", observations["state"])
        return observations

    parametric_action_distribution = distribution.NormalTanhDistribution(event_size=action_size)
    policy_network = brax_networks.make_policy_network(
        parametric_action_distribution.param_size,
        policy_obs_size,
        preprocess_observations_fn=policy_preprocess,
        hidden_layer_sizes=config.actor_layers(),
        distribution_type="tanh_normal",
    )
    q_network = brax_networks.make_q_network(
        critic_obs_size,
        action_size,
        preprocess_observations_fn=critic_preprocess,
        hidden_layer_sizes=config.critic_layers(),
        n_critics=2,
    )
    sac_network = sac_networks.SACNetworks(
        policy_network=policy_network,
        q_network=q_network,
        parametric_action_distribution=parametric_action_distribution,
    )
    make_policy = sac_networks.make_inference_fn(sac_network)

    alpha_optimizer = optax.adam(config.learning_rate)
    policy_optimizer = optax.adam(config.learning_rate)
    q_optimizer = optax.adam(config.learning_rate)

    alpha_loss, critic_loss, actor_loss = _make_losses(
        sac_network=sac_network,
        reward_scaling=config.reward_scaling,
        discounting=config.gamma,
        action_size=action_size,
    )

    alpha_update = gradients.gradient_update_fn(alpha_loss, alpha_optimizer, pmap_axis_name=None)
    critic_update = gradients.gradient_update_fn(critic_loss, q_optimizer, pmap_axis_name=None)
    actor_update = gradients.gradient_update_fn(actor_loss, policy_optimizer, pmap_axis_name=None)

    return {
        "obs_size": obs_size,
        "policy_obs_size": policy_obs_size,
        "critic_obs_size": critic_obs_size,
        "action_size": action_size,
        "sac_network": sac_network,
        "make_policy": make_policy,
        "alpha_optimizer": alpha_optimizer,
        "policy_optimizer": policy_optimizer,
        "q_optimizer": q_optimizer,
        "alpha_update": alpha_update,
        "critic_update": critic_update,
        "actor_update": actor_update,
        "actor_step_fn": _build_actor_step_fn(make_policy),
        "td_error_fn": _build_td_error_batch_fn(sac_network),
    }


def _init_training_state(key: jax.Array, runtime: dict[str, Any], config: VerticalSliceConfig) -> TrainingState:
    key_policy, key_q = jax.random.split(key)
    policy_params = runtime["sac_network"].policy_network.init(key_policy)
    q_params = runtime["sac_network"].q_network.init(key_q)
    alpha_params = jnp.asarray(0.0, dtype=jnp.float32)
    policy_obs_spec: Any = specs.Array((runtime["policy_obs_size"],), jnp.dtype("float32"))
    if isinstance(runtime["obs_size"], dict):
        policy_obs_spec = {"state": policy_obs_spec}
    normalizer_params = running_statistics.init_state(policy_obs_spec)
    return TrainingState(
        policy_optimizer_state=runtime["policy_optimizer"].init(policy_params),
        policy_params=policy_params,
        q_optimizer_state=runtime["q_optimizer"].init(q_params),
        q_params=q_params,
        target_q_params=q_params,
        alpha_optimizer_state=runtime["alpha_optimizer"].init(alpha_params),
        alpha_params=alpha_params,
        normalizer_params=normalizer_params,
    )


def _init_replay_buffer(runtime: dict[str, Any], config: VerticalSliceConfig, key: jax.Array) -> tuple[replay_buffers.UniformSamplingQueue, Any]:
    dummy_obs = _dummy_observation(runtime["obs_size"])
    dummy_action = jnp.zeros((runtime["action_size"],), dtype=jnp.float32)
    dummy_transition = brax_types.Transition(
        observation=dummy_obs,
        action=dummy_action,
        reward=0.0,
        discount=0.0,
        next_observation=dummy_obs,
        extras={"state_extras": {"truncation": 0.0, "time_out": 0.0}, "policy_extras": {}},
    )
    replay_buffer = replay_buffers.UniformSamplingQueue(
        max_replay_size=config.replay_capacity,
        dummy_data_sample=dummy_transition,
        sample_batch_size=config.batch_size,
    )
    replay_state = replay_buffer.init(key)
    return replay_buffer, replay_state


def _build_env(config: VerticalSliceConfig, batch_size: int, domain: str):
    env_config = registry.get_default_config(config.env_name)
    env_config.impl = "jax"
    env_config.episode_length = config.episode_length
    env_config.action_repeat = config.action_repeat
    env = registry.load(config.env_name, config=env_config)
    if domain == "pretrain":
        randomization_fn = _make_randomization_fn(
            batch_size=batch_size,
            seed=config.seed,
            friction_range=config.shift.train_friction_range,
            payload_range=config.shift.train_payload_range,
        )
    elif domain == "finetune":
        randomization_fn = _make_fixed_randomization_fn(
            batch_size=batch_size,
            friction=config.shift.fine_tune_friction,
            payload=config.shift.fine_tune_payload,
        )
    elif domain == "nominal_eval":
        randomization_fn = _make_fixed_randomization_fn(
            batch_size=batch_size,
            friction=statistics.mean(config.shift.train_friction_range),
            payload=statistics.mean(config.shift.train_payload_range),
        )
    else:
        raise ValueError(f"Unsupported domain: {domain}")

    env = playground_wrapper.BraxDomainRandomizationVmapWrapper(env, randomization_fn)
    env = brax_training.EpisodeWrapper(env, config.episode_length, config.action_repeat)
    return env


def _make_randomization_fn(
    *,
    batch_size: int,
    seed: int,
    friction_range: Sequence[float],
    payload_range: Sequence[float],
):
    def randomize(model):
        rng = jax.random.split(jax.random.PRNGKey(seed), batch_size)

        @jax.vmap
        def apply(single_rng):
            key_friction, key_payload = jax.random.split(single_rng)
            friction = jax.random.uniform(
                key_friction,
                minval=friction_range[0],
                maxval=friction_range[1],
            )
            payload = jax.random.uniform(
                key_payload,
                minval=payload_range[0],
                maxval=payload_range[1],
            )
            geom_friction = model.geom_friction.at[FLOOR_GEOM_ID, 0].set(friction)
            body_mass = model.body_mass.at[TORSO_BODY_ID].set(model.body_mass[TORSO_BODY_ID] * payload)
            return geom_friction, body_mass

        geom_friction, body_mass = apply(rng)
        in_axes = jax.tree_util.tree_map(lambda _: None, model)
        in_axes = in_axes.tree_replace({"geom_friction": 0, "body_mass": 0})
        model = model.tree_replace({"geom_friction": geom_friction, "body_mass": body_mass})
        return model, in_axes

    return randomize


def _make_fixed_randomization_fn(*, batch_size: int, friction: float, payload: float):
    def randomize(model):
        geom_friction = jnp.repeat(model.geom_friction[jnp.newaxis, ...], batch_size, axis=0)
        body_mass = jnp.repeat(model.body_mass[jnp.newaxis, ...], batch_size, axis=0)
        geom_friction = geom_friction.at[:, FLOOR_GEOM_ID, 0].set(friction)
        body_mass = body_mass.at[:, TORSO_BODY_ID].set(model.body_mass[TORSO_BODY_ID] * payload)
        in_axes = jax.tree_util.tree_map(lambda _: None, model)
        in_axes = in_axes.tree_replace({"geom_friction": 0, "body_mass": 0})
        model = model.tree_replace({"geom_friction": geom_friction, "body_mass": body_mass})
        return model, in_axes

    return randomize


def _build_actor_step_fn(make_policy):
    def actor_step(normalizer_params, policy_params, env, env_state, key):
        policy = make_policy((normalizer_params, policy_params))
        return acting.actor_step(env, env_state, policy, key, extra_fields=("truncation",))

    return actor_step


def _build_manual_reset_fn(env):
    def reset_done_envs(env_state, key):
        done = env_state.done
        reset_keys = jax.random.split(key, done.shape[0])
        reset_state = env.reset(reset_keys)

        def where_done(reset_value, current_value):
            if not hasattr(current_value, "shape"):
                return current_value
            if done.shape and done.shape[0] == current_value.shape[0]:
                mask = done.reshape((done.shape[0],) + (1,) * (current_value.ndim - 1))
                return jnp.where(mask, reset_value, current_value)
            return current_value

        return jax.tree_util.tree_map(where_done, reset_state, env_state)

    def maybe_reset(env_state, key):
        return jax.lax.cond(
            jnp.any(env_state.done),
            lambda carry: reset_done_envs(*carry),
            lambda carry: carry[0],
            (env_state, key),
        )

    return jax.jit(maybe_reset)


def _make_losses(
    *,
    sac_network: sac_networks.SACNetworks,
    reward_scaling: float,
    discounting: float,
    action_size: int,
):
    target_entropy = -0.5 * action_size
    policy_network = sac_network.policy_network
    q_network = sac_network.q_network
    parametric_action_distribution = sac_network.parametric_action_distribution

    def alpha_loss(log_alpha, policy_params, normalizer_params, transitions, key):
        dist_params = policy_network.apply(normalizer_params, policy_params, transitions.observation)
        action = parametric_action_distribution.sample_no_postprocessing(dist_params, key)
        log_prob = parametric_action_distribution.log_prob(dist_params, action)
        alpha = jnp.exp(log_alpha)
        return jnp.mean(alpha * jax.lax.stop_gradient(-log_prob - target_entropy))

    def critic_loss(q_params, policy_params, normalizer_params, target_q_params, alpha, transitions, key):
        q_old_action = q_network.apply(normalizer_params, q_params, transitions.observation, transitions.action)
        next_dist_params = policy_network.apply(normalizer_params, policy_params, transitions.next_observation)
        next_action = parametric_action_distribution.sample_no_postprocessing(next_dist_params, key)
        next_log_prob = parametric_action_distribution.log_prob(next_dist_params, next_action)
        next_action = parametric_action_distribution.postprocess(next_action)
        next_q = q_network.apply(normalizer_params, target_q_params, transitions.next_observation, next_action)
        next_v = jnp.min(next_q, axis=-1) - alpha * next_log_prob
        target_q = jax.lax.stop_gradient(
            transitions.reward * reward_scaling + transitions.discount * discounting * next_v
        )
        q_error = q_old_action - jnp.expand_dims(target_q, -1)
        return 0.5 * jnp.mean(jnp.square(q_error))

    def actor_loss(policy_params, normalizer_params, q_params, alpha, transitions, key):
        dist_params = policy_network.apply(normalizer_params, policy_params, transitions.observation)
        action = parametric_action_distribution.sample_no_postprocessing(dist_params, key)
        log_prob = parametric_action_distribution.log_prob(dist_params, action)
        action = parametric_action_distribution.postprocess(action)
        q_action = q_network.apply(normalizer_params, q_params, transitions.observation, action)
        min_q = jnp.min(q_action, axis=-1)
        return jnp.mean(alpha * log_prob - min_q)

    return alpha_loss, critic_loss, actor_loss


def _run_training_loop(
    *,
    config: VerticalSliceConfig,
    runtime: dict[str, Any],
    train_env,
    eval_callback,
    training_state: TrainingState,
    replay_buffer,
    replay_state,
    env_state,
    restore_replay_size: int,
    enable_diagnostics: bool,
    baseline: FrozenBaseline | None = None,
    eval_log_path: Path | None = None,
):
    key = jax.random.PRNGKey(config.seed + 17)
    aggregator = MultiStreamNStepAggregator(n_step=config.n_step, gamma=config.gamma)
    recent_buffer: RecentTransitionBuffer[AtlasTransition] = RecentTransitionBuffer(config.recent_buffer_capacity)
    trigger = InstabilityTrigger()
    diagnostic_state = DiagnosticLogState()
    replay_size = restore_replay_size
    metrics: dict[str, Any] = {}
    collapsed = False
    manual_reset_fn = _build_manual_reset_fn(train_env)

    update_step_fn = _build_update_step_fn(runtime, replay_buffer, config)
    env_steps = 0
    next_eval_at = config.eval_interval
    eval_log_handle = None

    if eval_log_path is not None:
        eval_log_path.parent.mkdir(parents=True, exist_ok=True)
        eval_log_path.write_text("", encoding="utf-8")
        eval_log_handle = eval_log_path.open("a", encoding="utf-8", buffering=1)

    try:
        for _ in range(config.train_steps // max(config.num_envs, 1)):
            key, actor_key, reset_key = jax.random.split(key, 3)
            env_state, one_step_transition = runtime["actor_step_fn"](
                training_state.normalizer_params,
                training_state.policy_params,
                train_env,
                env_state,
                actor_key,
            )
            training_state = training_state.replace(
                normalizer_params=running_statistics.update(
                    training_state.normalizer_params,
                    {"state": _policy_observation(one_step_transition.observation)}
                    if isinstance(runtime["obs_size"], dict)
                    else _policy_observation(one_step_transition.observation),
                    pmap_axis_name=None,
                )
            )

            emitted = _aggregate_transitions(one_step_transition, config.gamma, aggregator)
            if emitted:
                replay_state = replay_buffer.insert(replay_state, _to_brax_transition_batch(emitted, config.gamma))
                recent_buffer.extend(emitted)
                replay_size = min(config.replay_capacity, replay_size + len(emitted))
            env_state = manual_reset_fn(env_state, reset_key)
            env_steps += config.num_envs * config.action_repeat

            if replay_size >= config.min_replay_size:
                for _update in range(config.grad_updates_per_step):
                    key, update_key = jax.random.split(key)
                    training_state, replay_state, metrics = update_step_fn(training_state, replay_state, update_key)

            if env_steps < next_eval_at:
                continue

            eval_metrics = eval_callback(training_state, env_steps)
            next_eval_at = advance_next_eval_at(next_eval_at, env_steps, config.eval_interval)
            metrics = {**metrics, **eval_metrics}

            if not enable_diagnostics or baseline is None or eval_log_path is None:
                continue

            if len(recent_buffer) < config.diagnostic_min_transitions:
                continue

            td_errors = _sample_td_errors(runtime, training_state, recent_buffer, config, seed=config.seed + env_steps)
            raw_variance = statistics.pvariance(td_errors)
            warmup_variance = current_warmup_variance(diagnostic_state)
            if warmup_variance is None:
                # Warmup evals intentionally do not emit rows, which also means collapse
                # can only be observed in persisted diagnostics after warmup completes.
                diagnostic_state = record_warmup_variance(diagnostic_state, raw_variance)
                continue

            snapshot = summarize_td_errors(td_errors, warmup_variance)
            return_mean = float(eval_metrics["return_mean"])
            collapsed = return_mean < baseline.threshold
            trigger.update(snapshot.score)
            row = make_eval_log_row(
                run_id=config.run_id,
                eval_index=diagnostic_state.emitted_rows,
                score=snapshot.score,
                collapsed=collapsed,
                return_mean=return_mean,
                variance=snapshot.variance,
                q95_abs_td=snapshot.q95_abs_td,
                threshold=baseline.threshold,
                env_steps=env_steps,
            )
            diagnostic_state = mark_eval_row_emitted(diagnostic_state)
            if eval_log_handle is not None:
                eval_log_handle.write(json.dumps(row.to_dict()) + "\n")

            if collapsed and config.stop_on_collapse:
                break
    finally:
        if eval_log_handle is not None:
            eval_log_handle.close()

    return training_state, replay_state, env_state, metrics, env_steps, replay_size, trigger.ever_triggered, collapsed


def _build_update_step_fn(runtime: dict[str, Any], replay_buffer, config: VerticalSliceConfig):
    def update_step(training_state: TrainingState, replay_state, key):
        replay_state, sampled_transitions = replay_buffer.sample(replay_state)
        key_alpha, key_critic, key_actor = jax.random.split(key, 3)
        alpha_loss, alpha_params, alpha_optimizer_state = runtime["alpha_update"](
            training_state.alpha_params,
            training_state.policy_params,
            training_state.normalizer_params,
            sampled_transitions,
            key_alpha,
            optimizer_state=training_state.alpha_optimizer_state,
        )
        alpha = jnp.exp(training_state.alpha_params)
        critic_loss, q_params, q_optimizer_state = runtime["critic_update"](
            training_state.q_params,
            training_state.policy_params,
            training_state.normalizer_params,
            training_state.target_q_params,
            alpha,
            sampled_transitions,
            key_critic,
            optimizer_state=training_state.q_optimizer_state,
        )
        actor_loss, policy_params, policy_optimizer_state = runtime["actor_update"](
            training_state.policy_params,
            training_state.normalizer_params,
            training_state.q_params,
            alpha,
            sampled_transitions,
            key_actor,
            optimizer_state=training_state.policy_optimizer_state,
        )
        target_q_params = jax.tree_util.tree_map(
            lambda old, new: old * (1.0 - config.tau) + new * config.tau,
            training_state.target_q_params,
            q_params,
        )
        next_state = TrainingState(
            policy_optimizer_state=policy_optimizer_state,
            policy_params=policy_params,
            q_optimizer_state=q_optimizer_state,
            q_params=q_params,
            target_q_params=target_q_params,
            alpha_optimizer_state=alpha_optimizer_state,
            alpha_params=alpha_params,
            normalizer_params=training_state.normalizer_params,
        )
        metrics = {
            "critic_loss": float(critic_loss),
            "actor_loss": float(actor_loss),
            "alpha_loss": float(alpha_loss),
            "alpha": float(jnp.exp(alpha_params)),
        }
        return next_state, replay_state, metrics

    return update_step


def _aggregate_transitions(batch_transition, gamma: float, aggregator: MultiStreamNStepAggregator) -> list[AtlasTransition]:
    batch_size = int(np.asarray(batch_transition.reward).shape[0])
    emitted: list[AtlasTransition] = []
    for env_index in range(batch_size):
        truncation = float(np.asarray(batch_transition.extras["state_extras"]["truncation"])[env_index])
        brax_discount = float(np.asarray(batch_transition.discount)[env_index])
        discount = gamma if (brax_discount == 1.0 or truncation == 1.0) else 0.0
        emitted.extend(
            aggregator.push(
                env_index,
                AtlasTransition(
                    observation=_tree_to_numpy(_tree_index(batch_transition.observation, env_index)),
                    action=_tree_to_numpy(_tree_index(batch_transition.action, env_index)),
                    reward=float(np.asarray(batch_transition.reward)[env_index]),
                    discount=discount,
                    next_observation=_tree_to_numpy(_tree_index(batch_transition.next_observation, env_index)),
                    extras={"state_extras": {"truncation": truncation, "time_out": truncation}},
                ),
            )
        )
    return emitted


def _tree_index(tree: Any, index: int) -> Any:
    return jax.tree_util.tree_map(lambda leaf: leaf[index], tree)


def _stack_tree(items: Sequence[Any]) -> Any:
    return jax.tree_util.tree_map(lambda *leaves: jnp.stack(leaves), *items)


def _to_brax_transition_batch(transitions: Sequence[AtlasTransition], gamma: float):
    discounts = []
    time_outs = []
    for transition in transitions:
        if transition.discount == 0.0:
            discounts.append(0.0)
        else:
            # Atlas stores the full n-step bootstrap multiplier (for example gamma**n).
            # Brax SAC multiplies sampled transition.discount by `discounting=gamma`
            # inside critic_loss, so replay stores gamma**(n-1) here to round-trip
            # back to the original n-step multiplier in the Bellman target.
            discounts.append(float(transition.discount) / gamma)
        time_outs.append(1.0 if extract_timeout_flag(transition.extras) else 0.0)
    return brax_types.Transition(
        observation=_stack_tree([transition.observation for transition in transitions]),
        action=_stack_tree([transition.action for transition in transitions]),
        reward=jnp.asarray([transition.reward for transition in transitions], dtype=jnp.float32),
        discount=jnp.asarray(discounts, dtype=jnp.float32),
        next_observation=_stack_tree([transition.next_observation for transition in transitions]),
        extras={
            "state_extras": {
                "truncation": jnp.zeros((len(transitions),), dtype=jnp.float32),
                "time_out": jnp.asarray(time_outs, dtype=jnp.float32),
            },
            "policy_extras": {},
        },
    )


def _sample_td_errors(
    runtime: dict[str, Any],
    training_state: TrainingState,
    recent_buffer: RecentTransitionBuffer[AtlasTransition],
    config: VerticalSliceConfig,
    *,
    seed: int,
) -> list[float]:
    snapshot = recent_buffer.snapshot()
    rng = random.Random(seed)
    batches = [
        [snapshot[rng.randrange(len(snapshot))] for _ in range(config.diagnostic_batch_size)]
        for _ in range(config.diagnostic_minibatches)
    ]
    observations = [_stack_tree([transition.observation for transition in batch]) for batch in batches]
    actions = [_stack_tree([transition.action for transition in batch]) for batch in batches]
    next_observations = [_stack_tree([transition.next_observation for transition in batch]) for batch in batches]
    rewards = [
        jnp.asarray([transition.reward for transition in batch], dtype=jnp.float32)
        for batch in batches
    ]
    discounts = [
        jnp.asarray([transition.discount for transition in batch], dtype=jnp.float32)
        for batch in batches
    ]

    errors: list[float] = []
    for batch_index in range(config.diagnostic_minibatches):
        batch_errors = runtime["td_error_fn"](
            training_state,
            observations[batch_index],
            actions[batch_index],
            next_observations[batch_index],
            rewards[batch_index],
            discounts[batch_index],
            jax.random.PRNGKey(seed + batch_index),
        )
        errors.extend(np.asarray(batch_errors).tolist())
    return errors


def _build_td_error_batch_fn(sac_network: sac_networks.SACNetworks):
    def batch_td_errors(training_state: TrainingState, observations, actions, next_observations, rewards, discounts, key):
        dist_params = sac_network.policy_network.apply(
            training_state.normalizer_params,
            training_state.policy_params,
            next_observations,
        )
        raw_action = sac_network.parametric_action_distribution.sample_no_postprocessing(dist_params, key)
        log_prob = sac_network.parametric_action_distribution.log_prob(dist_params, raw_action)
        next_action = sac_network.parametric_action_distribution.postprocess(raw_action)
        next_q = sac_network.q_network.apply(
            training_state.normalizer_params,
            training_state.target_q_params,
            next_observations,
            next_action,
        )
        next_v = jnp.min(next_q, axis=-1) - jnp.exp(training_state.alpha_params) * log_prob
        current_q = sac_network.q_network.apply(
            training_state.normalizer_params,
            training_state.q_params,
            observations,
            actions,
        )[..., 0]
        return td_error(rewards, discounts, next_v, current_q)

    return jax.jit(batch_td_errors)


def _obs_dim(spec: Any) -> int:
    if isinstance(spec, int):
        return spec
    if isinstance(spec, tuple):
        size = 1
        for value in spec:
            size *= value
        return size
    raise TypeError(f"Unsupported observation spec: {spec!r}")


def _dummy_observation(obs_spec: Any) -> Any:
    if isinstance(obs_spec, dict):
        return {key: _dummy_observation(value) for key, value in obs_spec.items()}
    return jnp.zeros((_obs_dim(obs_spec),), dtype=jnp.float32)


def _policy_observation(observation: Any) -> Any:
    if isinstance(observation, dict):
        return observation["state"]
    return observation


def _build_evaluator(runtime: dict[str, Any], eval_env, num_eval_envs: int) -> acting.Evaluator:
    return acting.Evaluator(
        eval_env,
        functools.partial(runtime["make_policy"], deterministic=True),
        num_eval_envs=num_eval_envs,
        episode_length=eval_env.episode_length,
        action_repeat=1,
        key=jax.random.PRNGKey(0),
    )


def _evaluate_policy(runtime: dict[str, Any], evaluator: acting.Evaluator, training_state: TrainingState) -> dict[str, Any]:
    # Brax's public Evaluator API only exposes aggregated metrics; this vertical slice
    # needs per-episode returns, so it relies on 0.14.x private evaluator internals.
    evaluator._key, unroll_key = jax.random.split(evaluator._key)
    eval_state = evaluator._generate_eval_unroll(
        evaluator._eval_state_to_donate,
        (training_state.normalizer_params, training_state.policy_params),
        unroll_key,
    )
    evaluator._eval_state_to_donate = eval_state

    eval_metrics = eval_state.info["eval_metrics"]
    eval_metrics.active_episodes.block_until_ready()
    eval_metrics = jax.tree.map(np.asarray, eval_metrics)
    episode_returns = np.asarray(eval_metrics.episode_metrics["reward"], dtype=float)
    return_mean = float(np.mean(episode_returns))
    return_std = float(np.std(episode_returns))
    return {
        "return_mean": return_mean,
        "return_std": return_std,
        "episode_returns": episode_returns.tolist(),
    }


def _save_checkpoint(
    checkpoint_dir: Path,
    config: VerticalSliceConfig,
    observation_spec: Any,
    training_state: TrainingState,
    replay_state,
    replay_size: int,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": _utc_now(),
        "metadata": {
            "run_id": config.run_id,
            "signature": checkpoint_signature(
                config,
                observation_spec=observation_spec,
                observation_dtype="float32",
            ),
        },
        "training_state": flax_serialization.to_state_dict(training_state),
        "replay_state": flax_serialization.to_state_dict(replay_state),
        "replay_size": replay_size,
    }
    (_checkpoint_path(checkpoint_dir)).write_bytes(flax_serialization.msgpack_serialize(payload))


def _load_checkpoint(checkpoint_dir: Path) -> dict[str, Any]:
    payload = flax_serialization.msgpack_restore(_checkpoint_path(checkpoint_dir).read_bytes())
    return {
        "metadata": payload["metadata"],
        "training_state": payload["training_state"],
        "replay_state": payload["replay_state"],
        "replay_size": int(payload["replay_size"]),
    }


def _checkpoint_path(checkpoint_dir: Path) -> Path:
    return checkpoint_dir / "checkpoint.msgpack"


def _tree_to_numpy(tree: Any) -> Any:
    return jax.tree_util.tree_map(lambda value: np.asarray(value), tree)


def _tree_to_jax(tree: Any) -> Any:
    return jax.tree_util.tree_map(lambda value: jnp.asarray(value), tree)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2) + "\n", encoding="utf-8")


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


def _ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
