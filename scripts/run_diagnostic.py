#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from atlas.diagnostics import collapse_horizon_labels, roc_auc


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


def load_eval_log(path: Path) -> Dict[str, List[dict]]:
    grouped: Dict[str, List[dict]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            row = json.loads(line)
            required = {"run_id", "eval_index", "score", "collapsed"}
            missing = required - row.keys()
            if missing:
                raise ValueError(f"{path}:{line_number} missing required keys: {sorted(missing)}")
            grouped[str(row["run_id"])].append(row)

    for rows in grouped.values():
        rows.sort(key=lambda row: int(row["eval_index"]))
    return grouped


def summarize(grouped_rows: Dict[str, List[dict]], prediction_horizon: int) -> dict:
    scores: List[float] = []
    labels: List[int] = []
    lead_times: List[int] = []
    per_run = []

    for run_id, rows in sorted(grouped_rows.items()):
        run_scores = [float(row["score"]) for row in rows]
        collapse_flags = [bool(row["collapsed"]) for row in rows]
        run_labels = collapse_horizon_labels(collapse_flags, prediction_horizon)
        scores.extend(run_scores)
        labels.extend(run_labels)

        warning_index = next((index for index, score in enumerate(run_scores) if score > 1.0986122886681098), None)
        collapse_index = next((index for index, flag in enumerate(collapse_flags) if flag), None)
        lead_time = None
        if warning_index is not None and collapse_index is not None and warning_index < collapse_index:
            lead_time = collapse_index - warning_index
            lead_times.append(lead_time)

        per_run.append(
            {
                "run_id": run_id,
                "evals": len(rows),
                "first_warning_eval": warning_index,
                "first_collapse_eval": collapse_index,
                "lead_time_evals": lead_time,
            }
        )

    auc = None
    if labels and 0 < sum(labels) < len(labels):
        auc = roc_auc(scores, labels)

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runs": per_run,
        "global_roc_auc": auc,
        "mean_lead_time_evals": (sum(lead_times) / len(lead_times)) if lead_times else None,
        "prediction_horizon_evals": prediction_horizon,
    }


def main() -> None:
    args = parse_args()

    if args.write_sample is not None:
        write_sample(args.write_sample)

    if args.eval_log is None:
        return

    grouped_rows = load_eval_log(args.eval_log)
    summary = summarize(grouped_rows, args.prediction_horizon)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
