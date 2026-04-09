#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args():
    from atlas_training.pilot import parse_args as parse_pilot_args

    return parse_pilot_args()


class _TeeStream:
    def __init__(self, original, log_handle) -> None:
        self._original = original
        self._log_handle = log_handle

    def write(self, data: str) -> int:
        self._original.write(data)
        self._log_handle.write(data)
        return len(data)

    def flush(self) -> None:
        self._original.flush()
        self._log_handle.flush()

    def __getattr__(self, name: str):
        return getattr(self._original, name)


def main() -> None:
    from atlas_training.pilot import run_pilot_cli

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "pilot.log"
    exit_code = 0
    with log_path.open("a", encoding="utf-8", buffering=1) as log_handle:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = _TeeStream(original_stdout, log_handle)
        sys.stderr = _TeeStream(original_stderr, log_handle)
        try:
            print(f"=== Pilot Run Started: {datetime.now(timezone.utc).isoformat()} | {args.run_id} ===")
            print(
                json.dumps(
                    {
                        "output_dir": str(args.output_dir),
                        "profile": getattr(args, "profile", "default"),
                        "preflight_only": bool(getattr(args, "preflight_only", False)),
                    },
                    indent=2,
                )
            )
            try:
                report = run_pilot_cli(args)
            except Exception:
                traceback.print_exc()
                exit_code = 1
            else:
                print(f"Wrote {args.output_dir}")
                if args.preflight_only:
                    print(
                        json.dumps(
                            {
                                "pilot_id": report["pilot_id"],
                                "preflight_path": str(report["preflight_path"]),
                            },
                            indent=2,
                        )
                    )
                else:
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
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
