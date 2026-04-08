#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from atlas.config import build_budget_table, default_hyperparameters, generate_sweep


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the atlas sweep manifest.")
    parser.add_argument("--output", type=Path, default=Path("results/sweep_manifest.json"))
    parser.add_argument("--pilot-hours", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    hyperparameters = default_hyperparameters()
    sweep = generate_sweep(hyperparameters=hyperparameters)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hyperparameters": hyperparameters.to_dict(),
        "run_count": len(sweep),
        "budget_table": build_budget_table(args.pilot_hours, sweep),
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
