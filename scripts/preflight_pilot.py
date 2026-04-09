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

    forwarded = ["--preflight-only", *sys.argv[1:]]
    return parse_pilot_args(forwarded)


def main() -> None:
    from atlas_training.pilot import run_pilot_cli

    args = parse_args()
    result = run_pilot_cli(args)
    print(f"Wrote {result['preflight_path']}")
    print(
        json.dumps(
            {
                "pilot_id": result["pilot_id"],
                "preflight_path": str(result["preflight_path"]),
                "status": result["preflight"]["status"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
