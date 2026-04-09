from __future__ import annotations

from collections import namedtuple
import importlib.util
import tempfile
import unittest
from unittest import mock
from pathlib import Path

from atlas_training.preflight import PreflightError, collect_preflight, environment_from_preflight, run_preflight


class _FakeDevice:
    def __init__(self, *, platform: str, device_kind: str, device_id: int = 0, memory_supported: bool = False) -> None:
        self.platform = platform
        self.device_kind = device_kind
        self.id = device_id
        self._memory_supported = memory_supported

    def memory_stats(self):
        if not self._memory_supported:
            return None
        return {"bytes_limit": 1024, "bytes_in_use": 128}


class _FakeJax:
    def __init__(self, *, backend: str, devices: list[_FakeDevice]) -> None:
        self._backend = backend
        self._devices = devices

    def default_backend(self) -> str:
        return self._backend

    def devices(self):
        return self._devices


def _load_script(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PreflightPilotTest(unittest.TestCase):
    def _base_versions(self) -> tuple[dict[str, str | None], list[str]]:
        return (
            {
                "brax": "0.14.2",
                "jax": "0.9.2",
                "mujoco": "3.6.0",
                "mujoco_mjx": "3.6.0",
                "playground": "0.2.0",
            },
            [],
        )

    def test_collect_preflight_fails_on_cpu_backend_without_override(self) -> None:
        fake_jax = _FakeJax(backend="cpu", devices=[_FakeDevice(platform="cpu", device_kind="cpu")])
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch(
            "atlas_training.preflight._package_versions",
            return_value=self._base_versions(),
        ), mock.patch("atlas_training.preflight.importlib.import_module", return_value=fake_jax):
            payload = collect_preflight(
                output_dir=Path(tmpdir),
                preflight_path=Path(tmpdir) / "preflight.json",
                allow_cpu=False,
            )
        self.assertEqual(payload["status"], "error")
        self.assertTrue(any("CPU-only" in message for message in payload["errors"]))

    def test_collect_preflight_accepts_cpu_backend_with_override(self) -> None:
        fake_jax = _FakeJax(backend="cpu", devices=[_FakeDevice(platform="cpu", device_kind="cpu")])
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch(
            "atlas_training.preflight._package_versions",
            return_value=self._base_versions(),
        ), mock.patch("atlas_training.preflight.importlib.import_module", return_value=fake_jax):
            payload = collect_preflight(
                output_dir=Path(tmpdir),
                preflight_path=Path(tmpdir) / "preflight.json",
                allow_cpu=True,
            )
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["environment"]["jax_backend"], "cpu")
        self.assertTrue(any("override" in message for message in payload["warnings"]))

    def test_collect_preflight_fails_on_insufficient_disk(self) -> None:
        fake_jax = _FakeJax(backend="gpu", devices=[_FakeDevice(platform="gpu", device_kind="Test GPU")])
        disk_usage = namedtuple("DiskUsage", "total used free")(10, 9, 1024)
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch(
            "atlas_training.preflight._package_versions",
            return_value=self._base_versions(),
        ), mock.patch("atlas_training.preflight.importlib.import_module", return_value=fake_jax), mock.patch(
            "atlas_training.preflight.shutil.disk_usage",
            return_value=disk_usage,
        ):
            payload = collect_preflight(
                output_dir=Path(tmpdir),
                preflight_path=Path(tmpdir) / "preflight.json",
                min_free_disk_gb=5.0,
            )
        self.assertEqual(payload["status"], "error")
        self.assertFalse(payload["checks"]["disk_ok"])

    def test_collect_preflight_fails_when_training_dependencies_are_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch(
            "atlas_training.preflight._package_versions",
            return_value=({"brax": None}, ["brax"]),
        ):
            payload = collect_preflight(
                output_dir=Path(tmpdir),
                preflight_path=Path(tmpdir) / "preflight.json",
            )
        self.assertEqual(payload["status"], "error")
        self.assertTrue(any("missing training dependencies" in message for message in payload["errors"]))

    def test_collect_preflight_records_null_git_commit_when_unavailable(self) -> None:
        fake_jax = _FakeJax(
            backend="gpu",
            devices=[_FakeDevice(platform="gpu", device_kind="Test GPU", memory_supported=True)],
        )
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch(
            "atlas_training.preflight._package_versions",
            return_value=self._base_versions(),
        ), mock.patch("atlas_training.preflight.importlib.import_module", return_value=fake_jax), mock.patch(
            "atlas_training.preflight._git_commit",
            return_value=None,
        ):
            payload = collect_preflight(
                output_dir=Path(tmpdir),
                preflight_path=Path(tmpdir) / "preflight.json",
            )
        self.assertIsNone(payload["environment"]["git_commit"])
        self.assertEqual(payload["memory"]["status"], "available")

    def test_collect_preflight_warns_when_memory_report_errors(self) -> None:
        fake_jax = _FakeJax(backend="gpu", devices=[_FakeDevice(platform="gpu", device_kind="Test GPU")])
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch(
            "atlas_training.preflight._package_versions",
            return_value=self._base_versions(),
        ), mock.patch("atlas_training.preflight.importlib.import_module", return_value=fake_jax), mock.patch(
            "atlas_training.preflight._memory_report",
            return_value={"status": "error", "message": "boom"},
        ):
            payload = collect_preflight(
                output_dir=Path(tmpdir),
                preflight_path=Path(tmpdir) / "preflight.json",
            )
        self.assertEqual(payload["status"], "ok")
        self.assertTrue(any("memory inspection failed" in message for message in payload["warnings"]))

    def test_collect_preflight_fails_when_output_dir_is_not_writable(self) -> None:
        fake_jax = _FakeJax(backend="gpu", devices=[_FakeDevice(platform="gpu", device_kind="Test GPU")])
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch(
            "atlas_training.preflight._package_versions",
            return_value=self._base_versions(),
        ), mock.patch("atlas_training.preflight.importlib.import_module", return_value=fake_jax), mock.patch(
            "atlas_training.preflight._ensure_output_dir_writable",
            side_effect=PermissionError("read-only"),
        ):
            payload = collect_preflight(
                output_dir=Path(tmpdir),
                preflight_path=Path(tmpdir) / "preflight.json",
            )
        self.assertEqual(payload["status"], "error")
        self.assertFalse(payload["checks"]["output_dir_writable"])

    def test_environment_from_preflight_has_stable_schema(self) -> None:
        environment = environment_from_preflight(
            {
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
        )
        self.assertEqual(
            environment,
            {
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
            },
        )

    def test_run_preflight_raises_preflight_error_when_checks_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch(
            "atlas_training.preflight.collect_preflight",
            return_value={
                "status": "error",
                "errors": ["bad host"],
                "warnings": [],
            },
        ), mock.patch("atlas_training.preflight.write_json"):
            with self.assertRaises(PreflightError):
                run_preflight(
                    output_dir=Path(tmpdir),
                    preflight_path=Path(tmpdir) / "preflight.json",
                )

    def test_preflight_script_main_exits_cleanly_on_preflight_error(self) -> None:
        script = _load_script(Path(__file__).resolve().parents[1] / "scripts" / "preflight_pilot.py")
        with mock.patch.object(
            script,
            "parse_args",
            return_value=mock.Mock(),
        ), mock.patch(
            "atlas_training.pilot.run_pilot_cli",
            side_effect=PreflightError("cpu only"),
        ), mock.patch("sys.stderr") as stderr:
            with self.assertRaises(SystemExit) as exc_info:
                script.main()
        self.assertEqual(exc_info.exception.code, 1)
        stderr.write.assert_called()

    def test_preflight_script_parse_args_forces_preflight_only(self) -> None:
        script = _load_script(Path(__file__).resolve().parents[1] / "scripts" / "preflight_pilot.py")
        with mock.patch("sys.argv", ["preflight_pilot.py", "--profile", "production"]):
            args = script.parse_args()
        self.assertTrue(args.preflight_only)
        self.assertEqual(args.profile, "production")


if __name__ == "__main__":
    unittest.main()
