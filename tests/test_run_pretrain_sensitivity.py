from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


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


if __name__ == "__main__":
    unittest.main()
