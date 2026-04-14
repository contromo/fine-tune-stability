from __future__ import annotations

import ast
import importlib.util
import io
import tempfile
import types
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock


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

    def test_run_pilot_imports_without_training_dependencies(self) -> None:
        module = _load_script(ROOT / "scripts" / "run_pilot.py")
        self.assertTrue(callable(module.parse_args))

    def test_preflight_pilot_imports_without_training_dependencies(self) -> None:
        module = _load_script(ROOT / "scripts" / "preflight_pilot.py")
        self.assertTrue(callable(module.parse_args))

    def test_training_runtime_imports_playground_registry_from_internal_module(self) -> None:
        source = (ROOT / "atlas_training" / "runtime.py").read_text(encoding="utf-8")
        tree = ast.parse(source)

        registry_imports = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            and any(alias.name == "registry" for alias in node.names)
        ]

        self.assertTrue(
            any(node.module == "mujoco_playground._src" for node in registry_imports),
            "runtime.py should import registry from mujoco_playground._src",
        )
        self.assertFalse(
            any(node.module == "mujoco_playground" for node in registry_imports),
            "runtime.py should avoid top-level mujoco_playground imports",
        )

    def test_runtime_cli_builders_apply_shift_from_args(self) -> None:
        source = (ROOT / "atlas_training" / "runtime.py").read_text(encoding="utf-8")

        self.assertIn("shift=shift_from_args(args)", source)

    def test_standalone_training_scripts_accept_shift_args(self) -> None:
        pretrain_source = (ROOT / "scripts" / "run_pretrain.py").read_text(encoding="utf-8")
        finetune_source = (ROOT / "scripts" / "run_finetune.py").read_text(encoding="utf-8")

        self.assertIn("add_shift_cli_args", pretrain_source)
        self.assertIn("add_shift_cli_args", finetune_source)

    def test_run_pilot_teestream_delegates_unknown_attributes(self) -> None:
        module = _load_script(ROOT / "scripts" / "run_pilot.py")

        class _DummyStream:
            encoding = "utf-8"
            errors = "strict"

            def __init__(self) -> None:
                self.buffer = []

            def write(self, data: str) -> int:
                self.buffer.append(data)
                return len(data)

            def flush(self) -> None:
                return None

            def isatty(self) -> bool:
                return True

            def fileno(self) -> int:
                return 7

        original = _DummyStream()
        tee = module._TeeStream(original, io.StringIO())

        self.assertTrue(tee.isatty())
        self.assertEqual(tee.fileno(), 7)
        self.assertEqual(tee.encoding, "utf-8")
        self.assertEqual(tee.errors, "strict")

    def test_run_pilot_opens_pilot_log_line_buffered(self) -> None:
        module = _load_script(ROOT / "scripts" / "run_pilot.py")

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            args = Namespace(
                output_dir=output_dir,
                run_id="pilot_gate",
                profile="production",
                preflight_only=True,
            )
            fake_pilot = types.SimpleNamespace(
                run_pilot_cli=lambda parsed_args: {
                    "pilot_id": parsed_args.run_id,
                    "preflight_path": output_dir / "preflight.json",
                }
            )
            original_open = Path.open
            with mock.patch.object(module, "parse_args", return_value=args), mock.patch.dict(
                "sys.modules",
                {"atlas_training.pilot": fake_pilot},
            ), mock.patch("pathlib.Path.open", autospec=True, wraps=original_open) as open_mock:
                module.main()

            self.assertEqual(open_mock.call_args.kwargs["buffering"], 1)


if __name__ == "__main__":
    unittest.main()
