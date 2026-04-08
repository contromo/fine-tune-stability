from __future__ import annotations

import unittest
from pathlib import Path

from atlas_training.config import VerticalSliceConfig
from atlas_training.pilot import (
    DEFAULT_FINE_TUNE_STEPS,
    build_pilot_layout,
    classify_pilot_gate,
    hours_per_100m,
    minimum_finetune_steps,
    parse_args,
    parse_seed_list,
    realized_env_steps,
    required_eval_env_steps,
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
