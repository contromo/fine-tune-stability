from __future__ import annotations

import math
import unittest

from atlas.nstep import MultiStreamNStepAggregator, NStepTransitionAggregator
from atlas.transitions import Transition


class NStepAggregatorTest(unittest.TestCase):
    def test_aggregates_three_continuing_steps(self) -> None:
        gamma = 0.99
        aggregator = NStepTransitionAggregator(n_step=3, gamma=gamma)
        steps = [
            Transition("s0", "a0", 1.0, gamma, "s1"),
            Transition("s1", "a1", 2.0, gamma, "s2"),
            Transition("s2", "a2", 3.0, gamma, "s3"),
        ]

        outputs = []
        for step in steps:
            outputs.extend(aggregator.push(step))

        self.assertEqual(len(outputs), 1)
        self.assertAlmostEqual(outputs[0].reward, 1.0 + gamma * 2.0 + (gamma ** 2) * 3.0)
        self.assertAlmostEqual(outputs[0].discount, gamma ** 3)
        self.assertEqual(outputs[0].observation, "s0")
        self.assertEqual(outputs[0].next_observation, "s3")
        self.assertEqual(outputs[0].extras["atlas"]["n_step"], 3)

    def test_true_terminal_flushes_partial_windows(self) -> None:
        gamma = 0.99
        aggregator = NStepTransitionAggregator(n_step=3, gamma=gamma)
        steps = [
            Transition("s0", "a0", 1.0, gamma, "s1"),
            Transition("s1", "a1", 2.0, gamma, "s2"),
            Transition("s2", "a2", 3.0, 0.0, "s3"),
        ]

        outputs = []
        for step in steps:
            outputs.extend(aggregator.push(step))

        self.assertEqual(len(outputs), 3)
        self.assertAlmostEqual(outputs[0].reward, 1.0 + gamma * 2.0 + (gamma ** 2) * 3.0)
        self.assertEqual(outputs[0].discount, 0.0)
        self.assertAlmostEqual(outputs[1].reward, 2.0 + gamma * 3.0)
        self.assertEqual(outputs[1].discount, 0.0)
        self.assertAlmostEqual(outputs[2].reward, 3.0)
        self.assertEqual(outputs[2].discount, 0.0)

    def test_timeout_is_bootstrapped_and_tagged(self) -> None:
        gamma = 0.99
        aggregator = NStepTransitionAggregator(n_step=2, gamma=gamma)
        outputs = []
        outputs.extend(
            aggregator.push(
                Transition(
                    "s0",
                    "a0",
                    1.0,
                    gamma,
                    "s1",
                    extras={"state_extras": {"time_out": 1.0}},
                )
            )
        )
        outputs.extend(aggregator.push(Transition("s1", "a1", 2.0, gamma, "s2")))

        self.assertEqual(len(outputs), 1)
        self.assertAlmostEqual(outputs[0].discount, gamma ** 2)
        self.assertTrue(outputs[0].extras["atlas"]["window_timeout"])
        self.assertTrue(outputs[0].extras["atlas"]["timeout_bootstrapped"])

    def test_multi_stream_keeps_independent_state(self) -> None:
        gamma = 0.99
        aggregator = MultiStreamNStepAggregator(n_step=2, gamma=gamma)

        first = aggregator.push("env0", Transition("s0", "a0", 1.0, gamma, "s1"))
        second = aggregator.push("env1", Transition("t0", "b0", 5.0, gamma, "t1"))
        third = aggregator.push("env0", Transition("s1", "a1", 2.0, gamma, "s2"))

        self.assertEqual(first, [])
        self.assertEqual(second, [])
        self.assertEqual(len(third), 1)
        self.assertEqual(third[0].observation, "s0")
        self.assertEqual(third[0].next_observation, "s2")


if __name__ == "__main__":
    unittest.main()
