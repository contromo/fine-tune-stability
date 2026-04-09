from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from unittest import mock
from pathlib import Path

from atlas_training.config import VerticalSliceConfig, shift_from_args
from atlas_training.pilot import (
    DECISION_NOTE_MARKER,
    DECISION_NOTE_PLACEHOLDER,
    DEFAULT_FINE_TUNE_STEPS,
    DEFAULT_PRODUCTION_PRETRAIN_STEPS,
    ProbePhaseResult,
    PretrainPhaseResult,
    _build_pilot_report,
    _decision_note_path,
    _ensure_phase_can_run,
    _run_finetune_seed,
    _write_decision_note,
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
from atlas_training.util import write_json as util_write_json


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

    def test_parse_args_uses_production_profile_defaults(self) -> None:
        args = parse_args(["--profile", "production"])
        self.assertEqual(args.output_dir, Path("results/runs/pilot_gate"))
        self.assertEqual(args.pretrain_steps, DEFAULT_PRODUCTION_PRETRAIN_STEPS)
        self.assertEqual(args.num_envs, 32)
        self.assertTrue(args.stop_on_collapse)
        self.assertFalse(args.force)

    def test_parse_args_preflight_only_uses_production_output_dir(self) -> None:
        args = parse_args(["--profile", "production", "--preflight-only"])
        self.assertTrue(args.preflight_only)
        self.assertEqual(args.output_dir, Path("results/runs/pilot_gate"))

    def test_run_pilot_cli_preflight_only_uses_production_preflight_path(self) -> None:
        args = parse_args(["--profile", "production", "--preflight-only"])
        payload = {
            "status": "ok",
            "environment": {
                "hostname": "host",
                "platform": "platform",
                "python_version": "3.11.0",
                "git_commit": None,
                "jax_backend": "cpu",
                "jax_devices": ["cpu:0:cpu"],
                "packages": {
                    "brax": "0.14.2",
                    "jax": "0.9.2",
                    "mujoco": "3.6.0",
                    "mujoco_mjx": "3.6.0",
                    "playground": "0.2.0",
                },
            },
        }
        with mock.patch("atlas_training.pilot.run_preflight", return_value=payload) as run_preflight_mock:
            from atlas_training.pilot import run_pilot_cli

            result = run_pilot_cli(args)

        self.assertEqual(result["preflight_path"], Path("results/runs/pilot_gate/preflight.json"))
        run_preflight_mock.assert_called_once()
        self.assertEqual(run_preflight_mock.call_args.kwargs["output_dir"], Path("results/runs/pilot_gate"))
        self.assertEqual(run_preflight_mock.call_args.kwargs["preflight_path"], Path("results/runs/pilot_gate/preflight.json"))

    def test_parse_args_profile_defaults_can_be_overridden_explicitly(self) -> None:
        args = parse_args(
            [
                "--profile",
                "production",
                "--pretrain-steps",
                "123",
                "--num-envs",
                "8",
                "--collapse-c",
                "1.5",
                "--collapse-rho",
                "0.1",
            ]
        )
        self.assertEqual(args.pretrain_steps, 123)
        self.assertEqual(args.num_envs, 8)
        self.assertEqual(args.collapse_c, 1.5)
        self.assertEqual(args.collapse_rho, 0.1)

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

    def test_parse_args_accepts_documented_gpu_smoke_command(self) -> None:
        args = parse_args(
            [
                "--profile",
                "production",
                "--run-id",
                "pilot_gpu_smoke",
                "--output-dir",
                "results/runs/pilot_gpu_smoke",
                "--seeds",
                "0,1",
                "--pretrain-steps",
                "50000",
                "--eval-interval",
                "10000",
                "--fine-tune-steps",
                "50016",
                "--baseline-eval-episodes",
                "10",
                "--throughput-probe-updates",
                "50",
            ]
        )
        self.assertEqual(args.seed_values, (0, 1))
        self.assertEqual(args.fine_tune_steps, 50016)

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
        preflight = {
            "environment": {
                "hostname": "host",
                "platform": "platform",
                "python_version": "3.11.0",
                "git_commit": None,
                "jax_backend": "gpu",
                "jax_devices": ["gpu:0:Test GPU"],
                "packages": {
                    "brax": "0.14.2",
                    "jax": "0.9.2",
                    "mujoco": "3.6.0",
                    "mujoco_mjx": "3.6.0",
                    "playground": "0.2.0",
                },
            }
        }

        report = _build_pilot_report(
            args,
            layout,
            shift,
            layout.output_dir / "preflight.json",
            preflight,
            Path(__file__).resolve().parents[1] / "docs" / "decisions" / "2026-04-09-pilot_gate.md",
            datetime(2026, 4, 9, 0, 0, tzinfo=timezone.utc),
            pretrain,
            seed_results,
            probe,
        )

        self.assertEqual(report["extreme_probe"]["steps_per_second"], None)
        self.assertEqual(report["budget"]["sweep_hours_conservative"], float("inf"))
        self.assertEqual(report["created_at"], "2026-04-09T00:00:00+00:00")
        self.assertEqual(report["preflight_path"], layout.output_dir / "preflight.json")
        self.assertEqual(
            report["artifacts"]["decision_note"],
            Path(__file__).resolve().parents[1] / "docs" / "decisions" / "2026-04-09-pilot_gate.md",
        )
        self.assertEqual(
            set(report["environment"].keys()),
            {"hostname", "platform", "python_version", "git_commit", "jax_backend", "jax_devices", "packages"},
        )
        self.assertEqual(report["environment"]["jax_backend"], "gpu")
        self.assertEqual(report["environment"]["packages"]["mujoco_mjx"], "3.6.0")
        self.assertEqual(report["threshold_calibration"], {"collapse_c": 2.0, "collapse_rho": 0.2})

    def test_run_finetune_seed_rewrites_summary_after_diagnostic_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            args = parse_args(["--pretrain-steps", "1000", "--output-dir", str(output_dir)])
            layout = build_pilot_layout(output_dir, (0,))
            shift = shift_from_args(args)
            pretrain = PretrainPhaseResult(
                config=VerticalSliceConfig(stage="pretrain", output_dir=layout.pretrain_dir),
                summary={"wallclock_seconds": 1.0, "steps_per_second": 1.0},
                nominal_mean=10.0,
                nominal_std=1.0,
            )
            events: list[str] = []

            def fake_run_finetune(config: VerticalSliceConfig) -> dict[str, float | bool]:
                util_write_json(
                    config.pretrain_baseline_path(),
                    {
                        "mu0": 8.0,
                        "sigma0": 1.0,
                        "threshold": 6.0,
                        "collapse_c": 2.0,
                        "collapse_rho": 0.2,
                        "threshold_rule": "sigma",
                    },
                )
                config.eval_log_path().parent.mkdir(parents=True, exist_ok=True)
                config.eval_log_path().write_text(
                    '{"run_id":"seed0","eval_index":0,"score":1.2,"collapsed":false,"env_steps":32}\n',
                    encoding="utf-8",
                )
                util_write_json(
                    config.summary_path(),
                    {
                        "run_id": "seed0",
                        "collapsed": False,
                        "warning_triggered": False,
                        "wallclock_seconds": 1.0,
                        "steps_per_second": 100.0,
                    },
                )
                return {
                    "collapsed": False,
                    "warning_triggered": False,
                    "wallclock_seconds": 1.0,
                    "steps_per_second": 100.0,
                }

            def fake_diagnostic(eval_log_path: Path, output_path: Path, *, prediction_horizon: int, allow_missing: bool):
                self.assertEqual(prediction_horizon, 10)
                self.assertTrue(allow_missing)
                events.append(output_path.name)
                util_write_json(output_path, {"status": "ok"})
                return {"status": "ok"}

            def spy_write_json(path: Path, payload) -> None:
                events.append(path.name)
                util_write_json(path, payload)

            with mock.patch("atlas_training.pilot.write_diagnostic_summary", side_effect=fake_diagnostic), mock.patch(
                "atlas_training.pilot.write_json",
                side_effect=spy_write_json,
            ):
                result = _run_finetune_seed(
                    args,
                    shift,
                    seed=0,
                    layout=layout,
                    pretrain=pretrain,
                    run_finetune=fake_run_finetune,
                )

            self.assertEqual(result["status"], "ok")
            self.assertEqual(events[-2:], ["diagnostic_summary.json", "summary.json"])
            self.assertEqual(result["collapse_c"], 2.0)
            self.assertEqual(result["collapse_rho"], 0.2)
            self.assertEqual(result["threshold_rule"], "sigma")

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

    def test_phase_guard_refuses_completed_phase_without_force(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "summary.json"
            summary_path.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "shared_pretrain is already complete"):
                _ensure_phase_can_run("shared_pretrain", summary_path, force=False)

    def test_phase_guard_allows_force_for_completed_phase(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "summary.json"
            summary_path.write_text("{}", encoding="utf-8")
            _ensure_phase_can_run("shared_pretrain", summary_path, force=True)

    def test_decision_note_path_uses_utc_date_and_run_id(self) -> None:
        note_path = _decision_note_path("pilot_gate")
        self.assertEqual(note_path.parent, Path(__file__).resolve().parents[1] / "docs" / "decisions")
        self.assertTrue(note_path.name.endswith("-pilot_gate.md"))

    def test_decision_note_creation_and_template_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            note_path = Path(tmpdir) / "2026-04-09-pilot_gate.md"
            report = {
                "created_at": "2026-04-09T00:00:00+00:00",
                "pilot_id": "pilot_gate",
                "decision": "adjust",
                "artifacts": {"report": Path("results/runs/pilot_gate/pilot_report.json")},
                "budget": {"sweep_hours_conservative": 120.0, "sweep_hours_optimistic": 80.0},
                "representative_cell": {
                    "drop_fraction_stats": {"mean": 0.2, "min": 0.1, "max": 0.3},
                    "threshold_drop_fraction_stats": {"mean": 0.3, "min": 0.2, "max": 0.4},
                },
                "threshold_calibration": {"collapse_c": 2.0, "collapse_rho": 0.2},
            }
            _write_decision_note(report, note_path)
            contents = note_path.read_text(encoding="utf-8")
            self.assertIn(DECISION_NOTE_MARKER, contents)
            self.assertIn(DECISION_NOTE_PLACEHOLDER, contents)

            _write_decision_note(report, note_path)
            overwritten = note_path.read_text(encoding="utf-8")
            self.assertEqual(contents, overwritten)

    def test_decision_note_refuses_to_overwrite_human_edits(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            note_path = Path(tmpdir) / "2026-04-09-pilot_gate.md"
            report = {
                "created_at": "2026-04-09T00:00:00+00:00",
                "pilot_id": "pilot_gate",
                "decision": "proceed",
                "artifacts": {"report": Path("results/runs/pilot_gate/pilot_report.json")},
                "budget": {"sweep_hours_conservative": 100.0, "sweep_hours_optimistic": 80.0},
                "representative_cell": {
                    "drop_fraction_stats": {"mean": 0.2, "min": 0.1, "max": 0.3},
                    "threshold_drop_fraction_stats": {"mean": 0.3, "min": 0.2, "max": 0.4},
                },
                "threshold_calibration": {"collapse_c": 2.0, "collapse_rho": 0.2},
            }
            _write_decision_note(report, note_path)
            note_path.write_text(
                note_path.read_text(encoding="utf-8").replace(
                    DECISION_NOTE_PLACEHOLDER,
                    "- Ship the sweep manifest to the external scheduler.",
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(FileExistsError, "decision note already exists and appears to be edited"):
                _write_decision_note(report, note_path)


if __name__ == "__main__":
    unittest.main()
