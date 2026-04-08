from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_script(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ImportBoundaryTest(unittest.TestCase):
    def test_run_pretrain_imports_without_training_dependencies(self) -> None:
        module = _load_script(ROOT / "scripts" / "run_pretrain.py")
        self.assertTrue(callable(module.parse_args))

    def test_run_finetune_imports_without_training_dependencies(self) -> None:
        module = _load_script(ROOT / "scripts" / "run_finetune.py")
        self.assertTrue(callable(module.parse_args))


if __name__ == "__main__":
    unittest.main()
