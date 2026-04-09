from __future__ import annotations

import json
import math
import statistics
import tempfile
from pathlib import Path
from typing import Any, Sequence


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(json_ready(payload), indent=2) + "\n"
    with tempfile.TemporaryDirectory(dir=path.parent) as tempdir:
        temp_path = Path(tempdir) / path.name
        temp_path.write_text(encoded, encoding="utf-8")
        temp_path.replace(path)


def hours_per_100m(steps_per_second: float) -> float:
    if steps_per_second <= 0 or not math.isfinite(steps_per_second):
        return math.inf
    return 100_000_000.0 / steps_per_second / 3600.0


def summarize_throughput_rates(window_rates: Sequence[float]) -> dict[str, float]:
    if not window_rates:
        raise ValueError("window_rates must not be empty")
    window_hours = [hours_per_100m(rate) for rate in window_rates]
    return {
        "steps_per_second_mean": statistics.mean(window_rates),
        "steps_per_second_min": min(window_rates),
        "steps_per_second_max": max(window_rates),
        "hours_per_100m_mean": statistics.mean(window_hours),
        "hours_per_100m_min": min(window_hours),
        "hours_per_100m_max": max(window_hours),
    }
