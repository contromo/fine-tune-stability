from __future__ import annotations

import math
import unittest

from atlas.diagnostics import (
    DEFAULT_TRIGGER_THRESHOLD,
    InstabilityTrigger,
    collapse_horizon_labels,
    collapse_threshold,
    gaussian_kl_diagonal,
    pearson_correlation,
    roc_auc,
    summarize_td_errors,
    td_error,
)


class DiagnosticsTest(unittest.TestCase):
    def test_default_trigger_threshold_matches_instability_trigger(self) -> None:
        self.assertAlmostEqual(DEFAULT_TRIGGER_THRESHOLD, math.log(3.0))
        self.assertAlmostEqual(InstabilityTrigger().threshold, DEFAULT_TRIGGER_THRESHOLD)

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

    def test_gaussian_kl_self_is_zero(self) -> None:
        mu = [0.1, -0.2, 0.3]
        log_std = [0.0, -0.5, 0.25]
        self.assertAlmostEqual(gaussian_kl_diagonal(mu, log_std, mu, log_std), 0.0)

    def test_gaussian_kl_closed_form_scalar(self) -> None:
        # KL(N(0,1) || N(1, 2^2)) = log(2) + (1 + 1) / (2 * 4) - 0.5
        expected = math.log(2.0) + (1.0 + 1.0) / 8.0 - 0.5
        value = gaussian_kl_diagonal([0.0], [0.0], [1.0], [math.log(2.0)])
        self.assertAlmostEqual(value, expected)

    def test_gaussian_kl_is_asymmetric(self) -> None:
        mu_a, ls_a = [0.0, 0.0], [0.0, 0.0]
        mu_b, ls_b = [1.0, -1.0], [0.5, -0.5]
        forward = gaussian_kl_diagonal(mu_a, ls_a, mu_b, ls_b)
        reverse = gaussian_kl_diagonal(mu_b, ls_b, mu_a, ls_a)
        self.assertNotAlmostEqual(forward, reverse)
        self.assertGreaterEqual(forward, 0.0)
        self.assertGreaterEqual(reverse, 0.0)

    def test_gaussian_kl_rejects_mismatched_lengths(self) -> None:
        with self.assertRaises(ValueError):
            gaussian_kl_diagonal([0.0], [0.0], [0.0, 0.0], [0.0, 0.0])

    def test_td_error(self) -> None:
        self.assertAlmostEqual(td_error(reward=2.0, discount=0.9, bootstrap_value=5.0, q_value=4.0), 2.5)


if __name__ == "__main__":
    unittest.main()
