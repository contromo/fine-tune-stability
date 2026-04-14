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


class RunPretrainSensitivityTest(unittest.TestCase):
    def test_main_writes_representative_cell_manifest(self) -> None:
        module = _load_script(ROOT / "scripts" / "run_pretrain_sensitivity.py")
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "pretrain_sensitivity_manifest.json"
            args = SimpleNamespace(
                output=output_path,
                pilot_hours=1.5,
                from_pilot_report=None,
                fine_tune_steps=2_000_000,
                pretrain_seed_values=(0, 1, 2),
                finetune_seed_values=(0, 1),
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
            self.assertEqual(manifest["experiment"], "pretrain_sensitivity")
            self.assertEqual(manifest["representative_cell"]["n_step"], 3)
            self.assertEqual(manifest["representative_cell"]["critic_width"], 256)
            self.assertEqual(manifest["run_count"], 6)
            self.assertEqual(manifest["pretrain_seed_values"], [0, 1, 2])
            self.assertEqual(manifest["fine_tune_seed_values"], [0, 1])
            self.assertEqual(manifest["runs"][0]["pretrain_seed"], 0)
            self.assertEqual(manifest["runs"][0]["finetune_args"]["seed"], 0)

    def test_main_uses_shift_from_pilot_report(self) -> None:
        module = _load_script(ROOT / "scripts" / "run_pretrain_sensitivity.py")
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            output_path = tmpdir_path / "pretrain_sensitivity_manifest.json"
            report_path = tmpdir_path / "pilot_report.json"
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
                pretrain_seed_values=(1, 2),
                finetune_seed_values=(0, 1),
                shift_spec=None,
            )
            with mock.patch.object(module, "parse_args", return_value=args):
                module.main()

            manifest = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["shift"]["fine_tune_friction"], 0.2)
            self.assertEqual(manifest["runs"][0]["shift"]["fine_tune_payload"], 1.8)


if __name__ == "__main__":
    unittest.main()
