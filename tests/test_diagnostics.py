from __future__ import annotations

import math
import unittest

from atlas.diagnostics import (
    InstabilityTrigger,
    collapse_horizon_labels,
    collapse_threshold,
    pearson_correlation,
    roc_auc,
    summarize_td_errors,
    td_error,
)


class DiagnosticsTest(unittest.TestCase):
    def test_collapse_threshold_uses_more_conservative_rule(self) -> None:
        self.assertEqual(collapse_threshold(mu0=100.0, sigma0=5.0, c=2.0, rho=0.2), 80.0)

    def test_td_summary_computes_variance_quantile_and_score(self) -> None:
        snapshot = summarize_td_errors([1.0, -1.0, 3.0, -3.0], warmup_variance=1.0)
        self.assertAlmostEqual(snapshot.variance, 5.0)
        self.assertAlmostEqual(snapshot.q95_abs_td, 3.0)
        self.assertAlmostEqual(snapshot.score, math.log(5.0))

    def test_trigger_requires_hold_evals(self) -> None:
        trigger = InstabilityTrigger(threshold=1.0, hold_evals=2)
        self.assertFalse(trigger.update(1.1))
        self.assertTrue(trigger.update(1.2))
        self.assertTrue(trigger.ever_triggered)
        self.assertFalse(trigger.update(0.5))

    def test_collapse_horizon_labels(self) -> None:
        labels = collapse_horizon_labels([False, False, True, False], horizon=2)
        self.assertEqual(labels, [1, 1, 0, 0])

    def test_roc_auc_perfect_separation(self) -> None:
        auc = roc_auc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1])
        self.assertAlmostEqual(auc, 1.0)

    def test_pearson_correlation(self) -> None:
        corr = pearson_correlation([1.0, 2.0, 3.0], [10.0, 20.0, 30.0])
        self.assertAlmostEqual(corr, 1.0)

    def test_td_error(self) -> None:
        self.assertAlmostEqual(td_error(reward=2.0, discount=0.9, bootstrap_value=5.0, q_value=4.0), 2.5)


if __name__ == "__main__":
    unittest.main()
