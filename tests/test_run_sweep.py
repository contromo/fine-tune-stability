from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from atlas.config import ShiftSpec


ROOT = Path(__file__).resolve().parents[1]


def _load_script(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RunSweepTest(unittest.TestCase):
    def test_resolve_budget_source_scales_budget_from_pilot_report_to_default_horizon(self) -> None:
        module = _load_script(ROOT / "scripts" / "run_sweep.py")
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "pilot_report.json"
            report_path.write_text(
                json.dumps({"budget": {"hours_per_100m_extreme": 12.5}}) + "\n",
                encoding="utf-8",
            )
            pilot_hours, source = module._resolve_budget_source(
                SimpleNamespace(from_pilot_report=report_path, pilot_hours=None),
                2_000_000,
            )
        self.assertEqual(pilot_hours, 0.25)
        self.assertEqual(source["mode"], "pilot_report")
        self.assertEqual(source["pilot_report"], str(report_path))
        self.assertEqual(source["pilot_hours_per_100m"], 12.5)
        self.assertEqual(source["target_fine_tune_steps"], 2_000_000)
        self.assertIn("2000000-step sweep run", source["assumption"])

    def test_pilot_hours_from_report_supports_explicit_nondefault_horizon(self) -> None:
        module = _load_script(ROOT / "scripts" / "run_sweep.py")
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "pilot_report.json"
            report_path.write_text(
                json.dumps({"budget": {"hours_per_100m_extreme": 12.5}}) + "\n",
                encoding="utf-8",
            )
            pilot_hours, source = module._pilot_hours_from_report(report_path, 50_000_000)
        self.assertEqual(pilot_hours, 6.25)
        self.assertEqual(source["pilot_hours_per_100m"], 12.5)
        self.assertEqual(source["target_fine_tune_steps"], 50_000_000)

    def test_main_writes_manifest_with_pilot_report_budget_source(self) -> None:
        module = _load_script(ROOT / "scripts" / "run_sweep.py")
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            report_path = tmpdir_path / "pilot_report.json"
            output_path = tmpdir_path / "sweep_manifest.json"
            report_path.write_text(
                json.dumps(
                    {
                        "budget": {"hours_per_100m_extreme": 8.25},
                        "shift": {
                            "train_friction_range": [0.8, 1.2],
                            "train_payload_range": [0.8, 1.2],
                            "fine_tune_friction": 0.2,
                            "fine_tune_payload": 1.8,
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            args = SimpleNamespace(
                output=output_path,
                pilot_hours=None,
                from_pilot_report=report_path,
                fine_tune_steps=2_000_000,
                shift_spec=None,
            )
            with mock.patch.object(module, "parse_args", return_value=args):
                module.main()

            manifest = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["budget_source"]["mode"], "pilot_report")
            self.assertEqual(manifest["hyperparameters"]["total_fine_tune_steps"], 2_000_000)
            self.assertEqual(manifest["budget_source"]["pilot_hours_per_run"], 0.165)
            self.assertEqual(manifest["budget_table"][-1]["pilot_hours_per_run"], 0.165)
            self.assertEqual(manifest["shift"]["fine_tune_friction"], 0.2)
            self.assertEqual(manifest["runs"][0]["shift"]["fine_tune_payload"], 1.8)

    def test_main_honors_explicit_fine_tune_step_override(self) -> None:
        module = _load_script(ROOT / "scripts" / "run_sweep.py")
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "sweep_manifest.json"
            args = SimpleNamespace(
                output=output_path,
                pilot_hours=1.5,
                from_pilot_report=None,
                fine_tune_steps=5_000_000,
                shift_spec=ShiftSpec(
                    train_friction_range=(0.8, 1.2),
                    train_payload_range=(0.8, 1.2),
                    fine_tune_friction=0.3,
                    fine_tune_payload=1.5,
                ),
            )
            with mock.patch.object(module, "parse_args", return_value=args):
                module.main()

            manifest = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["hyperparameters"]["total_fine_tune_steps"], 5_000_000)
            self.assertEqual(manifest["runs"][0]["train_steps"], 5_000_000)


if __name__ == "__main__":
    unittest.main()
