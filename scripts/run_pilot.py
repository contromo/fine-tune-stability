#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args():
    from atlas_training.pilot import parse_args as parse_pilot_args

    return parse_pilot_args()


def main() -> None:
    from atlas_training.pilot import run_pilot_cli

    args = parse_args()
    report = run_pilot_cli(args)
    print(f"Wrote {args.output_dir}")
    print(
        json.dumps(
            {
                "pilot_id": report["pilot_id"],
                "decision": report["decision"],
                "report": str(report["artifacts"]["report"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
