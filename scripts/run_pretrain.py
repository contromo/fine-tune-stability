#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from atlas.config import default_hyperparameters, default_shift_spec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a scaffold pretrain manifest.")
    parser.add_argument("--output", type=Path, default=Path("results/pretrain_manifest.json"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("results/checkpoints/pretrain"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "stage": "pretrain",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "scaffold_only",
        "checkpoint_dir": str(args.checkpoint_dir),
        "hyperparameters": default_hyperparameters().to_dict(),
        "shift_spec": asdict(default_shift_spec()),
        "next_step": "Replace this manifest writer with the actual offline or sim pretraining job.",
    }

    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
