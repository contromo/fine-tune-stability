from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from atlas.config import ShiftSpec, estimate_run_hours


def positive_int(raw_value: str) -> int:
    value = int(raw_value)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def parse_seed_csv(raw_value: str) -> tuple[int, ...]:
    seeds = tuple(int(part.strip()) for part in raw_value.split(",") if part.strip())
    if not seeds:
        raise argparse.ArgumentTypeError("seed list must not be empty")
    return seeds


def parse_positive_int_csv(raw_value: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in raw_value.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("value list must not be empty")
    if any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("all values must be positive")
    return values


def pilot_hours_from_report(
    path: Path,
    total_fine_tune_steps: int,
    *,
    run_label: str,
) -> tuple[float, dict[str, object]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    budget = report.get("budget", {})
    pilot_hours = budget.get("hours_per_100m_extreme")
    if pilot_hours is None or not math.isfinite(pilot_hours):
        raise ValueError("pilot report is missing a finite budget.hours_per_100m_extreme")
    pilot_hours_per_100m = float(pilot_hours)
    pilot_hours_value = estimate_run_hours(pilot_hours_per_100m, total_fine_tune_steps)
    return pilot_hours_value, {
        "mode": "pilot_report",
        "pilot_report": str(path),
        "pilot_hours_per_100m": pilot_hours_per_100m,
        "pilot_hours_per_run": pilot_hours_value,
        "target_fine_tune_steps": total_fine_tune_steps,
        "assumption": (
            "budget.hours_per_100m_extreme is scaled linearly from the report's 100M-step normalization "
            f"to the requested {total_fine_tune_steps}-step {run_label}"
        ),
    }


def shift_from_pilot_report(path: Path) -> ShiftSpec:
    report = json.loads(path.read_text(encoding="utf-8"))
    raw_shift = report.get("shift")
    if not isinstance(raw_shift, dict):
        raise ValueError("pilot report is missing shift metadata")
    return ShiftSpec(
        train_friction_range=tuple(raw_shift["train_friction_range"]),
        train_payload_range=tuple(raw_shift["train_payload_range"]),
        fine_tune_friction=float(raw_shift["fine_tune_friction"]),
        fine_tune_payload=float(raw_shift["fine_tune_payload"]),
    )
