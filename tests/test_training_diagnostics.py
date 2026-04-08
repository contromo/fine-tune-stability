from __future__ import annotations

import unittest

from atlas_training.diagnostics import (
    DiagnosticLogState,
    advance_next_eval_at,
    current_warmup_variance,
    freeze_baseline,
    make_eval_log_row,
    mark_eval_row_emitted,
    record_warmup_variance,
)


class TrainingDiagnosticsTest(unittest.TestCase):
    def test_freeze_baseline(self) -> None:
        baseline = freeze_baseline([10.0, 12.0, 8.0, 10.0])
        self.assertAlmostEqual(baseline.mu0, 10.0)
        self.assertAlmostEqual(baseline.sigma0, 1.4142135623730951)
        self.assertLess(baseline.threshold, baseline.mu0)

    def test_warmup_variance_skips_first_two_rows(self) -> None:
        state = DiagnosticLogState()
        self.assertIsNone(current_warmup_variance(state))

        state = record_warmup_variance(state, 4.0)
        self.assertIsNone(current_warmup_variance(state))

        state = record_warmup_variance(state, 16.0)
        self.assertEqual(current_warmup_variance(state), 10.0)

    def test_eval_indices_are_contiguous_from_first_emitted_row(self) -> None:
        state = DiagnosticLogState(warmup_variances=(4.0, 16.0), emitted_rows=0)
        row0 = make_eval_log_row(
            run_id="run",
            eval_index=state.emitted_rows,
            score=0.5,
            collapsed=False,
            return_mean=1.0,
            variance=2.0,
            q95_abs_td=3.0,
            threshold=0.0,
            env_steps=100,
        )
        state = mark_eval_row_emitted(state)
        row1 = make_eval_log_row(
            run_id="run",
            eval_index=state.emitted_rows,
            score=0.7,
            collapsed=False,
            return_mean=2.0,
            variance=4.0,
            q95_abs_td=5.0,
            threshold=0.0,
            env_steps=200,
        )

        self.assertEqual(row0.eval_index, 0)
        self.assertEqual(row1.eval_index, 1)

    def test_advance_next_eval_at_handles_non_divisible_step_sizes(self) -> None:
        self.assertEqual(advance_next_eval_at(10, 12, 10), 20)
        self.assertEqual(advance_next_eval_at(20, 24, 10), 30)
        self.assertEqual(advance_next_eval_at(10, 30, 10), 40)

    def test_advance_next_eval_at_rejects_non_positive_interval(self) -> None:
        with self.assertRaises(ValueError):
            advance_next_eval_at(10, 12, 0)


if __name__ == "__main__":
    unittest.main()
