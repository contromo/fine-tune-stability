from __future__ import annotations

import unittest
from pathlib import Path

from atlas_training.config import VerticalSliceConfig
from atlas_training.pilot import (
    DEFAULT_FINE_TUNE_STEPS,
    ProbePhaseResult,
    PretrainPhaseResult,
    _build_pilot_report,
    build_pilot_layout,
    build_budget_summary,
    classify_pilot_gate,
    drop_fraction,
    hours_per_100m,
    minimum_finetune_steps,
    parse_args,
    parse_seed_list,
    realized_env_steps,
    required_eval_env_steps,
    threshold_drop_fraction,
)


class PilotTest(unittest.TestCase):
    def test_parse_seed_list(self) -> None:
        self.assertEqual(parse_seed_list("0,1,2"), (0, 1, 2))
        self.assertEqual(parse_seed_list("7"), (7,))
        with self.assertRaises(ValueError):
            parse_seed_list("")

    def test_build_pilot_layout_uses_shared_pretrain_and_per_seed_dirs(self) -> None:
        layout = build_pilot_layout(Path("results/runs/pilot_gate"), (0, 2))
        self.assertEqual(layout.pretrain_dir, Path("results/runs/pilot_gate/shared_pretrain"))
        self.assertEqual(layout.probe_dir, Path("results/runs/pilot_gate/extreme_probe"))
        self.assertEqual(layout.seed_dirs[0], Path("results/runs/pilot_gate/seed_0"))
        self.assertEqual(layout.seed_dirs[2], Path("results/runs/pilot_gate/seed_2"))
        self.assertEqual(layout.report_path, Path("results/runs/pilot_gate/pilot_report.json"))

    def test_parse_args_requires_pretrain_steps(self) -> None:
        with self.assertRaises(SystemExit):
            parse_args([])

    def test_parse_args_uses_pilot_defaults(self) -> None:
        args = parse_args(["--pretrain-steps", "1000"])
        self.assertEqual(args.seed_values, (0, 1, 2))
        self.assertEqual(args.fine_tune_steps, DEFAULT_FINE_TUNE_STEPS)
        self.assertEqual(args.baseline_eval_episodes, 50)

    def test_parse_args_rejects_too_short_finetune_horizon(self) -> None:
        with self.assertRaises(ValueError):
            parse_args(
                [
                    "--pretrain-steps",
                    "1000",
                    "--eval-interval",
                    "100",
                    "--fine-tune-steps",
                    str(
                        minimum_finetune_steps(
                            100,
                            num_envs=32,
                            action_repeat=1,
                        )
                        - 1
                    ),
                ]
            )

    def test_minimum_finetune_steps_accounts_for_action_repeat(self) -> None:
        minimum_steps = minimum_finetune_steps(100, num_envs=32, action_repeat=2)
        self.assertEqual(minimum_steps, 256)
        self.assertGreaterEqual(realized_env_steps(minimum_steps, 32, 2), required_eval_env_steps(100))
        self.assertLess(realized_env_steps(minimum_steps - 1, 32, 2), required_eval_env_steps(100))

    def test_parse_args_accepts_shorter_train_budget_when_action_repeat_increases_env_steps(self) -> None:
        args = parse_args(
            [
                "--pretrain-steps",
                "1000",
                "--eval-interval",
                "100",
                "--num-envs",
                "32",
                "--action-repeat",
                "2",
                "--fine-tune-steps",
                "256",
            ]
        )
        self.assertEqual(args.fine_tune_steps, 256)

    def test_parse_args_rejects_non_divisible_budget_with_insufficient_realized_env_steps(self) -> None:
        with self.assertRaises(ValueError):
            parse_args(
                [
                    "--pretrain-steps",
                    "1000",
                    "--eval-interval",
                    "100",
                    "--num-envs",
                    "32",
                    "--action-repeat",
                    "2",
                    "--fine-tune-steps",
                    "255",
                ]
            )

    def test_hours_per_100m_budget_math(self) -> None:
        self.assertAlmostEqual(hours_per_100m(10_000.0), 2.7777777777777777)
        self.assertGreater(hours_per_100m(1_000.0), hours_per_100m(10_000.0))

    def test_build_budget_summary_assembles_optimistic_and_conservative_bounds(self) -> None:
        budget = build_budget_summary([10.0, 20.0], 30.0)
        self.assertEqual(budget["hours_per_100m_small"], {"mean": 15.0, "min": 10.0, "max": 20.0})
        self.assertEqual(budget["hours_per_100m_extreme"], 30.0)
        self.assertEqual(budget["sweep_hours_optimistic"], 720.0)
        self.assertEqual(budget["sweep_hours_conservative"], 1440.0)

    def test_build_budget_summary_handles_probe_failure_with_infinite_conservative_bound(self) -> None:
        budget = build_budget_summary([10.0], float("inf"))
        self.assertEqual(budget["hours_per_100m_small"], {"mean": 10.0, "min": 10.0, "max": 10.0})
        self.assertIsNone(budget["hours_per_100m_extreme"])
        self.assertEqual(budget["sweep_hours_optimistic"], 480.0)
        self.assertEqual(budget["sweep_hours_conservative"], float("inf"))

    def test_build_pilot_report_treats_missing_probe_as_zero_throughput(self) -> None:
        args = parse_args(["--pretrain-steps", "1000", "--run-id", "pilot_gate"])
        layout = build_pilot_layout(Path("results/runs/pilot_gate"), args.seed_values)
        shift = args.shift if hasattr(args, "shift") else None
        if shift is None:
            from atlas_training.config import shift_from_args

            shift = shift_from_args(args)
        pretrain = PretrainPhaseResult(
            config=VerticalSliceConfig(stage="pretrain", output_dir=layout.pretrain_dir),
            summary={"wallclock_seconds": 1.0, "steps_per_second": 2.0},
            nominal_mean=10.0,
            nominal_std=1.0,
        )
        seed_results = [
            {
                "seed": 0,
                "status": "ok",
                "usable": True,
                "drop_fraction": 0.2,
                "threshold_drop_fraction": 0.3,
                "steps_per_second": 100.0,
                "hours_per_100m": hours_per_100m(100.0),
                "has_nonfinite_metrics": False,
            },
            {
                "seed": 1,
                "status": "ok",
                "usable": True,
                "drop_fraction": 0.3,
                "threshold_drop_fraction": 0.4,
                "steps_per_second": 120.0,
                "hours_per_100m": hours_per_100m(120.0),
                "has_nonfinite_metrics": False,
            },
        ]
        probe = ProbePhaseResult(
            config=VerticalSliceConfig(stage="throughput_probe", output_dir=layout.probe_dir),
            summary=None,
            error="probe failed",
        )

        report = _build_pilot_report(args, layout, shift, pretrain, seed_results, probe)

        self.assertEqual(report["extreme_probe"]["steps_per_second"], None)
        self.assertEqual(report["budget"]["sweep_hours_conservative"], float("inf"))

    def test_drop_fraction_handles_negative_nominal_mean(self) -> None:
        self.assertAlmostEqual(drop_fraction(-10.0, -5.0), -0.5)

    def test_drop_fraction_uses_epsilon_guard_near_zero(self) -> None:
        self.assertAlmostEqual(drop_fraction(0.0, -1e-9), 0.1)

    def test_threshold_drop_fraction_uses_epsilon_guard_near_zero(self) -> None:
        self.assertAlmostEqual(threshold_drop_fraction(0.0, -1e-9), 0.1)

    def test_baseline_evaluator_selection_is_config_driven(self) -> None:
        shared = VerticalSliceConfig(stage="finetune", output_dir=Path("results"), eval_episodes=10)
        separate = VerticalSliceConfig(
            stage="finetune",
            output_dir=Path("results"),
            eval_episodes=10,
            baseline_eval_episodes=50,
        )
        self.assertFalse(shared.uses_separate_baseline_evaluator())
        self.assertEqual(shared.effective_baseline_eval_episodes(), 10)
        self.assertTrue(separate.uses_separate_baseline_evaluator())
        self.assertEqual(separate.effective_baseline_eval_episodes(), 50)

    def test_gate_classification_proceed(self) -> None:
        decision, reasons = classify_pilot_gate(
            [
                {"usable": True, "drop_fraction": 0.2, "threshold_drop_fraction": 0.3, "has_nonfinite_metrics": False},
                {"usable": True, "drop_fraction": 0.35, "threshold_drop_fraction": 0.4, "has_nonfinite_metrics": False},
                {"usable": True, "drop_fraction": 0.6, "threshold_drop_fraction": 0.45, "has_nonfinite_metrics": False},
            ],
            100.0,
        )
        self.assertEqual(decision, "proceed")
        self.assertTrue(reasons)

    def test_gate_classification_proceed_with_one_error_seed(self) -> None:
        decision, reasons = classify_pilot_gate(
            [
                {"usable": True, "drop_fraction": 0.2, "threshold_drop_fraction": 0.3, "has_nonfinite_metrics": False},
                {"usable": True, "drop_fraction": 0.35, "threshold_drop_fraction": 0.4, "has_nonfinite_metrics": False},
                {"usable": False, "status": "error", "has_nonfinite_metrics": True},
            ],
            100.0,
        )
        self.assertEqual(decision, "proceed")
        self.assertTrue(reasons)

    def test_gate_classification_adjust(self) -> None:
        decision, reasons = classify_pilot_gate(
            [
                {"usable": True, "drop_fraction": 0.1, "threshold_drop_fraction": 0.3, "has_nonfinite_metrics": False},
                {"usable": True, "drop_fraction": 0.12, "threshold_drop_fraction": 0.3, "has_nonfinite_metrics": False},
                {"usable": True, "drop_fraction": 0.2, "threshold_drop_fraction": 0.55, "has_nonfinite_metrics": False},
            ],
            100.0,
        )
        self.assertEqual(decision, "adjust")
        self.assertTrue(reasons)

    def test_gate_classification_fail(self) -> None:
        decision, reasons = classify_pilot_gate(
            [
                {"usable": True, "drop_fraction": 0.2, "threshold_drop_fraction": 0.3, "has_nonfinite_metrics": False},
                {"usable": False, "has_nonfinite_metrics": False},
                {"usable": False, "has_nonfinite_metrics": True},
            ],
            140.0,
        )
        self.assertEqual(decision, "fail")
        self.assertTrue(reasons)


if __name__ == "__main__":
    unittest.main()
