from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from atlas_training.diagnostics import (
    DiagnosticLogState,
    advance_next_eval_at,
    current_warmup_variance,
    freeze_baseline,
    load_eval_log,
    make_eval_log_row,
    mark_eval_row_emitted,
    record_warmup_variance,
    summarize_eval_groups,
    write_diagnostic_summary,
)


class TrainingDiagnosticsTest(unittest.TestCase):
    def test_freeze_baseline(self) -> None:
        baseline = freeze_baseline([10.0, 12.0, 8.0, 10.0])
        self.assertAlmostEqual(baseline.mu0, 10.0)
        self.assertAlmostEqual(baseline.sigma0, 1.4142135623730951)
        self.assertLess(baseline.threshold, baseline.mu0)
        self.assertEqual(baseline.collapse_c, 2.0)
        self.assertEqual(baseline.collapse_rho, 0.2)
        self.assertEqual(baseline.threshold_rule, "sigma")

    def test_freeze_baseline_records_floor_rule_when_more_conservative(self) -> None:
        baseline = freeze_baseline([100.0, 100.0, 100.0], c=0.1, rho=0.2)
        self.assertEqual(baseline.threshold_rule, "floor")
        self.assertEqual(baseline.threshold, 80.0)

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

    def test_load_eval_log_allows_missing_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            missing = Path(tmpdir) / "missing_eval_log.jsonl"
            grouped = load_eval_log(missing, allow_missing=True)
            self.assertEqual(grouped, {})

    def test_summarize_eval_groups_matches_expected_schema(self) -> None:
        grouped = {
            "run_a": [
                {"run_id": "run_a", "eval_index": 0, "score": 0.1, "collapsed": False},
                {"run_id": "run_a", "eval_index": 1, "score": 1.5, "collapsed": True},
            ],
            "run_b": [
                {"run_id": "run_b", "eval_index": 0, "score": 0.0, "collapsed": False},
                {"run_id": "run_b", "eval_index": 1, "score": 0.2, "collapsed": False},
            ],
        }
        summary = summarize_eval_groups(grouped, prediction_horizon=2)
        self.assertEqual(summary["prediction_horizon_evals"], 2)
        self.assertEqual(len(summary["runs"]), 2)
        self.assertIsNotNone(summary["global_roc_auc"])

    def test_write_diagnostic_summary_round_trips_shared_logic(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            eval_log = Path(tmpdir) / "eval_log.jsonl"
            output = Path(tmpdir) / "summary.json"
            eval_log.write_text(
                "\n".join(
                    [
                        json.dumps({"run_id": "run", "eval_index": 0, "score": 0.1, "collapsed": False}),
                        json.dumps({"run_id": "run", "eval_index": 1, "score": 1.4, "collapsed": True}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            summary = write_diagnostic_summary(eval_log, output, prediction_horizon=3)
            persisted = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(persisted["prediction_horizon_evals"], 3)
            self.assertEqual(summary["runs"][0]["run_id"], "run")


if __name__ == "__main__":
    unittest.main()
