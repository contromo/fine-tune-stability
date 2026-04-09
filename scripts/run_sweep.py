#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from atlas.config import build_budget_table, default_hyperparameters, generate_sweep

NON_100M_SWEEP_ERROR = "sweep model assumes 100M steps per run; rerun with --pilot-hours"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the atlas sweep manifest.")
    parser.add_argument("--output", type=Path, default=Path("results/sweep_manifest.json"))
    budget_group = parser.add_mutually_exclusive_group()
    budget_group.add_argument("--pilot-hours", type=float, default=None)
    budget_group.add_argument("--from-pilot-report", type=Path, default=None)
    return parser.parse_args()


def _pilot_hours_from_report(path: Path, total_fine_tune_steps: int) -> tuple[float, dict[str, object]]:
    if total_fine_tune_steps != 100_000_000:
        raise ValueError(NON_100M_SWEEP_ERROR)
    report = json.loads(path.read_text(encoding="utf-8"))
    budget = report.get("budget", {})
    pilot_hours = budget.get("hours_per_100m_extreme")
    if pilot_hours is None or not math.isfinite(float(pilot_hours)):
        raise ValueError("pilot report is missing a finite budget.hours_per_100m_extreme")
    return float(pilot_hours), {
        "mode": "pilot_report",
        "pilot_report": str(path),
        "pilot_hours_per_run": float(pilot_hours),
        "assumption": "budget.hours_per_100m_extreme is treated as hours per full 100M-step sweep run",
    }


def _resolve_budget_source(args: argparse.Namespace, total_fine_tune_steps: int) -> tuple[float, dict[str, object]]:
    if args.from_pilot_report is not None:
        return _pilot_hours_from_report(args.from_pilot_report, total_fine_tune_steps)
    pilot_hours = 1.0 if args.pilot_hours is None else float(args.pilot_hours)
    return pilot_hours, {
        "mode": "manual",
        "pilot_report": None,
        "pilot_hours_per_run": pilot_hours,
        "assumption": "manual pilot-hours input is treated as hours per full sweep run",
    }


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    hyperparameters = default_hyperparameters()
    pilot_hours, budget_source = _resolve_budget_source(args, hyperparameters.total_fine_tune_steps)
    sweep = generate_sweep(hyperparameters=hyperparameters)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hyperparameters": hyperparameters.to_dict(),
        "run_count": len(sweep),
        "budget_source": budget_source,
        "budget_table": build_budget_table(pilot_hours, sweep),
        "runs": [
            {
                **cell.to_dict(),
                "critic_width": cell.critic.width,
                "train_steps": cell.total_steps,
                "eval_interval": cell.eval_interval_steps,
                "finetune_args": {
                    "n_step": cell.n_step,
                    "critic_width": cell.critic.width,
                    "seed": cell.seed,
                    "train_steps": cell.total_steps,
                    "eval_interval": cell.eval_interval_steps,
                    "stop_on_collapse": True,
                },
            }
            for cell in sweep
        ],
    }

    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output} with {len(sweep)} runs")


if __name__ == "__main__":
    main()
