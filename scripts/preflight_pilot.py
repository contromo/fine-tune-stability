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

    forwarded = list(sys.argv[1:])
    if "--preflight-only" not in forwarded:
        forwarded = ["--preflight-only", *forwarded]
    return parse_pilot_args(forwarded)


def main() -> None:
    from atlas_training.pilot import run_pilot_cli
    from atlas_training.preflight import PreflightError

    args = parse_args()
    try:
        result = run_pilot_cli(args)
    except PreflightError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
    print(f"Wrote {str(result['preflight_path'])}")
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
