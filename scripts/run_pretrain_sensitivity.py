#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
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
    REPRESENTATIVE_PRETRAIN_SENSITIVITY_CRITIC_WIDTH,
    REPRESENTATIVE_PRETRAIN_SENSITIVITY_N_STEP,
    build_budget_table,
    default_hyperparameters,
    default_pretrain_sensitivity_pretrain_seeds,
    estimate_run_hours,
    generate_pretrain_sensitivity_sweep,
)


def _positive_int(raw_value: str) -> int:
    value = int(raw_value)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def _parse_seed_list(raw_value: str) -> tuple[int, ...]:
    seeds = tuple(int(part.strip()) for part in raw_value.split(",") if part.strip())
    if not seeds:
        raise argparse.ArgumentTypeError("seed list must not be empty")
    return seeds


def _pilot_hours_from_report(path: Path, total_fine_tune_steps: int) -> tuple[float, dict[str, object]]:
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
            f"to the requested {total_fine_tune_steps}-step representative-cell sensitivity run"
        ),
    }


def _resolve_budget_source(args: argparse.Namespace, total_fine_tune_steps: int) -> tuple[float, dict[str, object]]:
    if args.from_pilot_report is not None:
        return _pilot_hours_from_report(args.from_pilot_report, total_fine_tune_steps)
    pilot_hours = 1.0 if args.pilot_hours is None else args.pilot_hours
    return pilot_hours, {
        "mode": "manual",
        "pilot_report": None,
        "pilot_hours_per_run": pilot_hours,
        "assumption": "manual pilot-hours input is treated as hours per full sensitivity run",
    }


def parse_args() -> argparse.Namespace:
    defaults = default_hyperparameters()
    default_pretrain_seeds = ",".join(str(seed) for seed in default_pretrain_sensitivity_pretrain_seeds())
    default_finetune_seeds = ",".join(str(seed) for seed in range(defaults.seeds_per_cell))

    parser = argparse.ArgumentParser(description="Generate the representative-cell pretrain-sensitivity manifest.")
    parser.add_argument("--output", type=Path, default=Path("results/pretrain_sensitivity_manifest.json"))
    parser.add_argument(
        "--fine-tune-steps",
        type=_positive_int,
        default=DEFAULT_SWEEP_FINE_TUNE_STEPS,
        help="per-run fine-tune horizon for the representative-cell sensitivity bundle",
    )
    parser.add_argument("--pretrain-seeds", type=str, default=default_pretrain_seeds)
    parser.add_argument("--fine-tune-seeds", type=str, default=default_finetune_seeds)
    budget_group = parser.add_mutually_exclusive_group()
    budget_group.add_argument("--pilot-hours", type=float, default=None)
    budget_group.add_argument("--from-pilot-report", type=Path, default=None)
    args = parser.parse_args()
    args.pretrain_seed_values = _parse_seed_list(args.pretrain_seeds)
    args.finetune_seed_values = _parse_seed_list(args.fine_tune_seeds)
    return args


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    hyperparameters = replace(default_hyperparameters(), total_fine_tune_steps=args.fine_tune_steps)
    pilot_hours, budget_source = _resolve_budget_source(args, hyperparameters.total_fine_tune_steps)
    sweep = generate_pretrain_sensitivity_sweep(
        hyperparameters=hyperparameters,
        pretrain_seed_values=args.pretrain_seed_values,
        finetune_seed_values=args.finetune_seed_values,
    )
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment": "pretrain_sensitivity",
        "description": "Representative-cell pretrain-seed sensitivity for the shared-pretrain main sweep design",
        "representative_cell": {
            "n_step": REPRESENTATIVE_PRETRAIN_SENSITIVITY_N_STEP,
            "critic_width": REPRESENTATIVE_PRETRAIN_SENSITIVITY_CRITIC_WIDTH,
        },
        "hyperparameters": hyperparameters.to_dict(),
        "hours_per_100m_normalization_steps": HOURS_PER_100M_NORMALIZATION_STEPS,
        "pretrain_seed_values": list(args.pretrain_seed_values),
        "fine_tune_seed_values": list(args.finetune_seed_values),
        "run_count": len(sweep),
        "budget_source": budget_source,
        "budget_table": build_budget_table(pilot_hours, sweep),
        "runs": [
            {
                **cell.to_dict(),
                "critic_width": cell.critic.width,
                "train_steps": cell.total_steps,
                "eval_interval": cell.eval_interval_steps,
                "pretrain_run_id": f"representative_pretrain_seed{cell.pretrain_seed}",
                "pretrain_args": {
                    "seed": cell.pretrain_seed,
                    "purpose": "materialize an alternate pretrain checkpoint for representative-cell sensitivity",
                },
                "finetune_args": {
                    "n_step": cell.n_step,
                    "critic_width": cell.critic.width,
                    "seed": cell.finetune_seed,
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
