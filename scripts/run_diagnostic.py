#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from atlas_training.diagnostics import load_eval_log, summarize_eval_groups


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize diagnostic eval logs.")
    parser.add_argument("--eval-log", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("results/diagnostic_summary.json"))
    parser.add_argument("--write-sample", type=Path, default=None)
    parser.add_argument("--prediction-horizon", type=int, default=10)
    return parser.parse_args()


def write_sample(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"run_id": "n1_small_seed0", "eval_index": 0, "score": 0.1, "collapsed": False, "return_mean": 123.0},
        {"run_id": "n1_small_seed0", "eval_index": 1, "score": 0.2, "collapsed": False, "return_mean": 120.0},
        {"run_id": "n1_small_seed0", "eval_index": 2, "score": 1.3, "collapsed": True, "return_mean": 70.0},
        {"run_id": "n3_large_seed0", "eval_index": 0, "score": -0.1, "collapsed": False, "return_mean": 130.0},
        {"run_id": "n3_large_seed0", "eval_index": 1, "score": 0.0, "collapsed": False, "return_mean": 131.0},
        {"run_id": "n3_large_seed0", "eval_index": 2, "score": 0.4, "collapsed": False, "return_mean": 129.0},
    ]
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    print(f"Wrote sample log to {path}")


def main() -> None:
    args = parse_args()

    if args.write_sample is not None:
        write_sample(args.write_sample)

    if args.eval_log is None:
        return

    grouped_rows = load_eval_log(args.eval_log)
    summary = summarize_eval_groups(grouped_rows, args.prediction_horizon)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
