#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from atlas.config import (
    DEFAULT_SWEEP_FINE_TUNE_STEPS,
    HOURS_PER_100M_NORMALIZATION_STEPS,
    build_budget_table,
    default_hyperparameters,
    generate_sweep,
)
from atlas.manifest_utils import pilot_hours_from_report, positive_int, shift_from_pilot_report


def parse_args() -> argparse.Namespace:
    from atlas_training.config import add_shift_cli_args, shift_from_args

    parser = argparse.ArgumentParser(description="Generate the atlas sweep manifest.")
    parser.add_argument("--output", type=Path, default=Path("results/sweep_manifest.json"))
    parser.add_argument(
        "--fine-tune-steps",
        type=positive_int,
        default=DEFAULT_SWEEP_FINE_TUNE_STEPS,
        help="per-run fine-tune horizon for the main sweep manifest; defaults to the 2M-step main study plan",
    )
    budget_group = parser.add_mutually_exclusive_group()
    budget_group.add_argument("--pilot-hours", type=float, default=None)
    budget_group.add_argument("--from-pilot-report", type=Path, default=None)
    add_shift_cli_args(parser)
    args = parser.parse_args()
    args.shift_spec = shift_from_args(args)
    return args


def _pilot_hours_from_report(path: Path, total_fine_tune_steps: int) -> tuple[float, dict[str, object]]:
    return pilot_hours_from_report(path, total_fine_tune_steps, run_label="sweep run")


def _resolve_budget_source(args: argparse.Namespace, total_fine_tune_steps: int) -> tuple[float, dict[str, object]]:
    if args.from_pilot_report is not None:
        return _pilot_hours_from_report(args.from_pilot_report, total_fine_tune_steps)
    pilot_hours = 1.0 if args.pilot_hours is None else args.pilot_hours
    return pilot_hours, {
        "mode": "manual",
        "pilot_report": None,
        "pilot_hours_per_run": pilot_hours,
        "assumption": "manual pilot-hours input is treated as hours per full sweep run",
    }


def _resolve_shift(args: argparse.Namespace):
    if args.from_pilot_report is not None:
        return shift_from_pilot_report(args.from_pilot_report)
    return args.shift_spec


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    hyperparameters = replace(default_hyperparameters(), total_fine_tune_steps=args.fine_tune_steps)
    pilot_hours, budget_source = _resolve_budget_source(args, hyperparameters.total_fine_tune_steps)
    shift = _resolve_shift(args)
    sweep = generate_sweep(hyperparameters=hyperparameters, shift=shift)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hyperparameters": hyperparameters.to_dict(),
        "shift": {
            "train_friction_range": list(shift.train_friction_range),
            "train_payload_range": list(shift.train_payload_range),
            "fine_tune_friction": shift.fine_tune_friction,
            "fine_tune_payload": shift.fine_tune_payload,
        },
        "hours_per_100m_normalization_steps": HOURS_PER_100M_NORMALIZATION_STEPS,
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
