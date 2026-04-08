from __future__ import annotations

import importlib.util
import unittest
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


if __name__ == "__main__":
    unittest.main()
