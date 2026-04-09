from __future__ import annotations

import importlib
from importlib import metadata as importlib_metadata
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from atlas_training.util import write_json

DEFAULT_MIN_FREE_DISK_GB = 5.0
PACKAGE_DISTRIBUTIONS = {
    "brax": "brax",
    "jax": "jax",
    "mujoco": "mujoco",
    "mujoco_mjx": "mujoco-mjx",
    "playground": "playground",
}
ENVIRONMENT_KEYS = (
    "hostname",
    "platform",
    "python_version",
    "git_commit",
    "jax_backend",
    "jax_devices",
    "packages",
)


class PreflightError(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_preflight_path(output_dir: Path, explicit_path: Path | None) -> Path:
    return explicit_path if explicit_path is not None else output_dir / "preflight.json"


def _resolve_existing_path(path: Path) -> Path:
    current = path
    while not current.exists():
        if current.parent == current:
            return Path.cwd()
        current = current.parent
    return current


def _git_commit(cwd: Path | None = None) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=str(cwd or Path.cwd()),
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    value = completed.stdout.strip()
    return value or None


def _package_versions() -> tuple[dict[str, str | None], list[str]]:
    versions: dict[str, str | None] = {}
    missing: list[str] = []
    for key, distribution_name in PACKAGE_DISTRIBUTIONS.items():
        try:
            versions[key] = importlib_metadata.version(distribution_name)
        except importlib_metadata.PackageNotFoundError:
            versions[key] = None
            missing.append(distribution_name)
    return versions, missing


def _device_descriptors(devices: list[Any]) -> list[str]:
    descriptors: list[str] = []
    for index, device in enumerate(devices):
        platform_name = getattr(device, "platform", "unknown")
        device_id = getattr(device, "id", index)
        device_kind = getattr(device, "device_kind", platform_name)
        descriptors.append(f"{platform_name}:{device_id}:{device_kind}")
    return descriptors


def _memory_report(devices: list[Any]) -> dict[str, Any]:
    if not devices:
        return {"status": "no_devices"}
    primary_device = devices[0]
    memory_stats = getattr(primary_device, "memory_stats", None)
    if not callable(memory_stats):
        return {"status": "unsupported"}
    try:
        stats = memory_stats()
    except Exception as exc:  # pragma: no cover - depends on backend support
        return {"status": "error", "message": str(exc)}
    if stats is None:
        return {"status": "unsupported"}
    return {"status": "available", "stats": dict(stats)}


def _ensure_output_dir_writable(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output_dir, delete=True):
        pass


def environment_from_preflight(preflight_payload: dict[str, Any]) -> dict[str, Any]:
    environment = dict(preflight_payload.get("environment", {}))
    packages = dict(environment.get("packages", {}))
    return {
        "hostname": environment.get("hostname"),
        "platform": environment.get("platform"),
        "python_version": environment.get("python_version"),
        "git_commit": environment.get("git_commit"),
        "jax_backend": environment.get("jax_backend"),
        "jax_devices": list(environment.get("jax_devices", [])),
        "packages": {
            "brax": packages.get("brax"),
            "jax": packages.get("jax"),
            "mujoco": packages.get("mujoco"),
            "mujoco_mjx": packages.get("mujoco_mjx"),
            "playground": packages.get("playground"),
        },
    }


def collect_preflight(
    *,
    output_dir: Path,
    preflight_path: Path,
    allow_cpu: bool = False,
    min_free_disk_gb: float = DEFAULT_MIN_FREE_DISK_GB,
    cwd: Path | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []

    output_dir_writable = False
    try:
        _ensure_output_dir_writable(output_dir)
        output_dir_writable = True
    except Exception as exc:
        errors.append(f"output directory is not writable: {exc}")

    disk_usage_path = _resolve_existing_path(output_dir)
    free_bytes = shutil.disk_usage(disk_usage_path).free
    free_disk_gb = free_bytes / (1024.0**3)
    disk_ok = free_disk_gb >= min_free_disk_gb
    if not disk_ok:
        errors.append(
            f"free disk {free_disk_gb:.3f} GiB is below required minimum {min_free_disk_gb:.3f} GiB"
        )

    package_versions, missing_packages = _package_versions()
    if missing_packages:
        errors.append("missing training dependencies: " + ", ".join(sorted(missing_packages)))

    environment = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "git_commit": _git_commit(cwd),
        "jax_backend": None,
        "jax_devices": [],
        "packages": package_versions,
    }
    memory = {"status": "not_checked"}

    if not missing_packages:
        try:
            jax = importlib.import_module("jax")
            devices = list(jax.devices())
            environment["jax_backend"] = str(jax.default_backend())
            environment["jax_devices"] = _device_descriptors(devices)
            if environment["jax_backend"] == "cpu" and not allow_cpu:
                errors.append("JAX backend is CPU-only; rerun with --allow-cpu to override")
            if environment["jax_backend"] == "cpu" and allow_cpu:
                warnings.append("CPU-only JAX backend allowed by operator override")
            memory = _memory_report(devices)
        except Exception as exc:
            errors.append(f"failed to initialize JAX runtime: {exc}")
            memory = {"status": "error", "message": str(exc)}

    if memory.get("status") == "unsupported":
        warnings.append("GPU memory inspection is unsupported on this backend")
    if memory.get("status") == "error":
        warnings.append("GPU memory inspection failed on this backend")

    payload = {
        "created_at": _utc_now(),
        "status": "error" if errors else "ok",
        "output_dir": output_dir,
        "preflight_path": preflight_path,
        "environment": environment,
        "checks": {
            "output_dir_writable": output_dir_writable,
            "disk_usage_path": disk_usage_path,
            "free_disk_bytes": free_bytes,
            "free_disk_gb": round(free_disk_gb, 6),
            "min_free_disk_gb": float(min_free_disk_gb),
            "disk_ok": disk_ok,
            "allow_cpu": bool(allow_cpu),
        },
        "memory": memory,
        "warnings": warnings,
        "errors": errors,
    }
    return payload


def run_preflight(
    *,
    output_dir: Path,
    preflight_path: Path,
    allow_cpu: bool = False,
    min_free_disk_gb: float = DEFAULT_MIN_FREE_DISK_GB,
    cwd: Path | None = None,
) -> dict[str, Any]:
    payload = collect_preflight(
        output_dir=output_dir,
        preflight_path=preflight_path,
        allow_cpu=allow_cpu,
        min_free_disk_gb=min_free_disk_gb,
        cwd=cwd,
    )
    write_json(preflight_path, payload)
    if payload["errors"]:
        raise PreflightError("; ".join(str(error) for error in payload["errors"]))
    return payload
