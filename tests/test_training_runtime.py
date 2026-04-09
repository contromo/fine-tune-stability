from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest
from unittest import mock
from types import SimpleNamespace

from atlas.transitions import Transition as AtlasTransition


def _training_stack_available() -> bool:
    required = ("brax", "flax", "jax", "mujoco_playground")
    return all(importlib.util.find_spec(name) is not None for name in required)


class TrainingRuntimeTest(unittest.TestCase):
    def _require_runtime(self):
        if not _training_stack_available():
            self.skipTest("Training runtime dependencies are not installed in this interpreter.")
        import jax.numpy as jnp
        import numpy as np
        from atlas import MultiStreamNStepAggregator
        from atlas_training.runtime import _aggregate_transitions, _to_brax_transition_batch

        return jnp, np, MultiStreamNStepAggregator, _aggregate_transitions, _to_brax_transition_batch

    def test_to_brax_transition_batch_round_trips_n_step_discount(self) -> None:
        _jnp, np, _aggregator_cls, _aggregate_transitions, to_brax_transition_batch = self._require_runtime()
        gamma = 0.99
        n_step_discount = gamma**3
        transition = AtlasTransition(
            observation=np.asarray([1.0, 2.0], dtype=np.float32),
            action=np.asarray([0.5], dtype=np.float32),
            reward=1.0,
            discount=n_step_discount,
            next_observation=np.asarray([3.0, 4.0], dtype=np.float32),
            extras={"state_extras": {"time_out": 0.0}},
        )

        batch = to_brax_transition_batch([transition], gamma)
        stored_discount = float(np.asarray(batch.discount)[0])
        self.assertAlmostEqual(stored_discount * gamma, n_step_discount)

        terminal = transition.with_updates(discount=0.0)
        terminal_batch = to_brax_transition_batch([terminal], gamma)
        self.assertEqual(float(np.asarray(terminal_batch.discount)[0]), 0.0)

    def test_aggregate_transitions_converts_recent_buffer_payloads_to_numpy(self) -> None:
        jnp, np, aggregator_cls, aggregate_transitions, _to_brax_transition_batch = self._require_runtime()
        gamma = 0.99
        batched = SimpleNamespace(
            observation={
                "state": jnp.asarray([[1.0, 2.0]], dtype=jnp.float32),
                "privileged_state": jnp.asarray([[3.0, 4.0, 5.0]], dtype=jnp.float32),
            },
            action=jnp.asarray([[0.1]], dtype=jnp.float32),
            reward=jnp.asarray([1.0], dtype=jnp.float32),
            discount=jnp.asarray([1.0], dtype=jnp.float32),
            next_observation={
                "state": jnp.asarray([[6.0, 7.0]], dtype=jnp.float32),
                "privileged_state": jnp.asarray([[8.0, 9.0, 10.0]], dtype=jnp.float32),
            },
            extras={"state_extras": {"truncation": jnp.asarray([0.0], dtype=jnp.float32)}},
        )

        emitted = aggregate_transitions(batched, gamma, aggregator_cls(n_step=1, gamma=gamma))
        self.assertEqual(len(emitted), 1)
        self.assertIsInstance(emitted[0].observation["state"], np.ndarray)
        self.assertIsInstance(emitted[0].observation["privileged_state"], np.ndarray)
        self.assertIsInstance(emitted[0].action, np.ndarray)
        self.assertIsInstance(emitted[0].next_observation["state"], np.ndarray)
        self.assertNotIn("time_out", emitted[0].extras["state_extras"])

    def test_aggregate_transitions_marks_timeout_only_on_truncation(self) -> None:
        jnp, _np, aggregator_cls, aggregate_transitions, _to_brax_transition_batch = self._require_runtime()
        gamma = 0.99
        batched = SimpleNamespace(
            observation={"state": jnp.asarray([[1.0]], dtype=jnp.float32)},
            action=jnp.asarray([[0.1]], dtype=jnp.float32),
            reward=jnp.asarray([1.0], dtype=jnp.float32),
            discount=jnp.asarray([0.0], dtype=jnp.float32),
            next_observation={"state": jnp.asarray([[2.0]], dtype=jnp.float32)},
            extras={"state_extras": {"truncation": jnp.asarray([1.0], dtype=jnp.float32)}},
        )

        emitted = aggregate_transitions(batched, gamma, aggregator_cls(n_step=1, gamma=gamma))
        self.assertEqual(emitted[0].extras["state_extras"]["time_out"], 1.0)

    def test_run_pretrain_writes_summary_last(self) -> None:
        self._require_runtime()
        from atlas_training.config import VerticalSliceConfig
        from atlas_training import runtime as runtime_module

        class _FakeEnv:
            def reset(self, _keys):
                return object()

        events: list[str] = []

        def spy_write_json(path, payload):
            events.append(Path(path).name)

        config = VerticalSliceConfig(stage="pretrain", output_dir=Path("results/runtime_pretrain_order"))
        with mock.patch.object(runtime_module, "_ensure_output_dir"), mock.patch.object(
            runtime_module, "write_json", side_effect=spy_write_json
        ), mock.patch.object(
            runtime_module, "_build_runtime", return_value={"obs_size": {"state": 1}}
        ), mock.patch.object(
            runtime_module, "_build_env", return_value=_FakeEnv()
        ), mock.patch.object(
            runtime_module, "_build_evaluator", return_value=object()
        ), mock.patch.object(
            runtime_module, "_init_training_state", return_value=object()
        ), mock.patch.object(
            runtime_module, "_init_replay_buffer", return_value=(object(), object())
        ), mock.patch.object(
            runtime_module,
            "_run_training_loop",
            return_value=(object(), object(), object(), {"loss": 1.0}, 32, 1, False, False),
        ), mock.patch.object(
            runtime_module, "_evaluate_policy", return_value={"return_mean": 1.0, "return_std": 0.1}
        ), mock.patch.object(
            runtime_module, "_save_checkpoint", side_effect=lambda *args, **kwargs: events.append("checkpoint")
        ), mock.patch.object(
            runtime_module, "_steps_per_second", return_value=1.0
        ):
            runtime_module.run_pretrain(config)

        self.assertEqual(events[-1], "summary.json")

    def test_run_throughput_probe_writes_summary_last(self) -> None:
        _jnp, np, _aggregator_cls, _aggregate_transitions, _to_brax_transition_batch = self._require_runtime()
        from atlas_training.config import VerticalSliceConfig
        from atlas_training import runtime as runtime_module

        class _FakeEnv:
            def reset(self, _keys):
                return object()

        class _FakeReplayBuffer:
            def insert(self, replay_state, _batch):
                return replay_state

        def fake_actor_step_fn(_normalizer_params, _policy_params, _train_env, env_state, _actor_key):
            transition = SimpleNamespace(
                observation={"state": np.asarray([[1.0]], dtype=np.float32)},
                action=np.asarray([[0.0]], dtype=np.float32),
                reward=np.asarray([1.0], dtype=np.float32),
                discount=np.asarray([1.0], dtype=np.float32),
                next_observation={"state": np.asarray([[2.0]], dtype=np.float32)},
                extras={"state_extras": {"truncation": np.asarray([0.0], dtype=np.float32)}},
            )
            return env_state, transition

        events: list[str] = []

        def spy_write_json(path, payload):
            events.append(Path(path).name)

        config = VerticalSliceConfig(
            stage="throughput_probe",
            output_dir=Path("results/runtime_probe_order"),
            min_replay_size=1,
        )
        fake_training_state = SimpleNamespace(normalizer_params=None, policy_params=None)
        with mock.patch.object(runtime_module, "_ensure_output_dir"), mock.patch.object(
            runtime_module, "write_json", side_effect=spy_write_json
        ), mock.patch.object(
            runtime_module, "_build_runtime", return_value={"actor_step_fn": fake_actor_step_fn}
        ), mock.patch.object(
            runtime_module, "_build_env", return_value=_FakeEnv()
        ), mock.patch.object(
            runtime_module, "_init_training_state", return_value=fake_training_state
        ), mock.patch.object(
            runtime_module, "_init_replay_buffer", return_value=(_FakeReplayBuffer(), object())
        ), mock.patch.object(
            runtime_module, "_update_normalizer_from_transition", side_effect=lambda state, *_args: state
        ), mock.patch.object(
            runtime_module,
            "_aggregate_transitions",
            return_value=[
                AtlasTransition(
                    observation=np.asarray([1.0], dtype=np.float32),
                    action=np.asarray([0.0], dtype=np.float32),
                    reward=1.0,
                    discount=0.99,
                    next_observation=np.asarray([2.0], dtype=np.float32),
                    extras={"state_extras": {}},
                )
            ],
        ), mock.patch.object(
            runtime_module, "_to_brax_transition_batch", return_value=object()
        ), mock.patch.object(
            runtime_module, "_build_manual_reset_fn", return_value=lambda env_state, _reset_key: env_state
        ), mock.patch.object(
            runtime_module,
            "_build_update_step_fn",
            return_value=lambda training_state, replay_state, _update_key: (training_state, replay_state, {"loss": 1.0}),
        ):
            runtime_module.run_throughput_probe(config, updates_per_window=1, timed_update_windows=1)

        self.assertEqual(events[-1], "summary.json")


if __name__ == "__main__":
    unittest.main()
