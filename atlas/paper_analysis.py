from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

DEFAULT_HORIZON_SEEDS: dict[int, tuple[int, ...]] = {
    1: tuple(range(8)),
    3: tuple(range(4)),
    10: tuple(range(4)),
}

DEFAULT_HORIZON_RUN_DIRS: dict[int, str] = {
    1: "horizon_only_1m",
    3: "horizon_confirm_v1",
    10: "horizon_confirm_v1",
}

DEFAULT_PILOT_SPECS: tuple[dict[str, str], ...] = (
    {
        "pilot_id": "pilot_gate_1m_v3",
        "pilot_report_relpath": "runs/pilot_gate_1m_v3/pilot_report.json",
        "diagnostic_relpath": "tmp/pilot_v3_diag_summary.json",
    },
    {
        "pilot_id": "pilot_gate_1m_v4",
        "pilot_report_relpath": "runs/pilot_gate_1m_v4/pilot_report.json",
        "diagnostic_relpath": "tmp/pilot_v4_diag_summary.json",
    },
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _summary_path(results_root: Path, run_dir: str, horizon: int, seed: int) -> Path:
    return results_root / "runs" / run_dir / f"n{horizon}_c256_seed{seed}" / "summary.json"


def _stdev(values: Sequence[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def collect_horizon_points(
    results_root: Path,
    *,
    horizon_seeds: Mapping[int, Sequence[int]] | None = None,
    horizon_run_dirs: Mapping[int, str] | None = None,
) -> list[dict[str, Any]]:
    selected_seeds = DEFAULT_HORIZON_SEEDS if horizon_seeds is None else horizon_seeds
    selected_dirs = DEFAULT_HORIZON_RUN_DIRS if horizon_run_dirs is None else horizon_run_dirs
    rows: list[dict[str, Any]] = []

    for horizon in sorted(selected_seeds):
        run_dir = selected_dirs[horizon]
        for seed in selected_seeds[horizon]:
            summary_path = _summary_path(results_root, run_dir, horizon, int(seed))
            summary = _load_json(summary_path)
            rows.append(
                {
                    "horizon": int(horizon),
                    "seed": int(seed),
                    "summary_path": str(summary_path),
                    "collapsed": bool(summary["collapsed"]),
                    "warning_triggered": bool(summary["warning_triggered"]),
                    "final_return_mean": float(summary["training_metrics"]["return_mean"]),
                    "final_return_std": float(summary["training_metrics"]["return_std"]),
                    "steps_per_second": float(summary["steps_per_second"]),
                }
            )
    return rows


def summarize_horizon_points(points: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[Mapping[str, Any]]] = {}
    for row in points:
        grouped.setdefault(int(row["horizon"]), []).append(row)

    summaries: list[dict[str, Any]] = []
    for horizon in sorted(grouped):
        rows = grouped[horizon]
        final_returns = [float(row["final_return_mean"]) for row in rows]
        sps_values = [float(row["steps_per_second"]) for row in rows]
        summaries.append(
            {
                "horizon": horizon,
                "n": len(rows),
                "mean_final_return": statistics.mean(final_returns),
                "stdev_final_return": _stdev(final_returns),
                "min_final_return": min(final_returns),
                "max_final_return": max(final_returns),
                "mean_steps_per_second": statistics.mean(sps_values),
                "collapsed_count": sum(1 for row in rows if bool(row["collapsed"])),
                "warning_count": sum(1 for row in rows if bool(row["warning_triggered"])),
            }
        )
    return summaries


def collect_warning_pilot_data(
    results_root: Path,
    *,
    pilot_specs: Sequence[Mapping[str, str]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    specs = DEFAULT_PILOT_SPECS if pilot_specs is None else pilot_specs
    summary_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []

    for spec in specs:
        pilot_id = spec["pilot_id"]
        report = _load_json(results_root / spec["pilot_report_relpath"])
        diagnostic = _load_json(results_root / spec["diagnostic_relpath"])

        summary_rows.append(
            {
                "pilot_id": pilot_id,
                "decision": str(report["decision"]),
                "reasons": " | ".join(str(reason) for reason in report["reasons"]),
                "drop_fraction_mean": float(report["representative_cell"]["drop_fraction_stats"]["mean"]),
                "drop_fraction_min": float(report["representative_cell"]["drop_fraction_stats"]["min"]),
                "drop_fraction_max": float(report["representative_cell"]["drop_fraction_stats"]["max"]),
                "threshold_drop_fraction_mean": float(
                    report["representative_cell"]["threshold_drop_fraction_stats"]["mean"]
                ),
                "threshold_drop_fraction_min": float(
                    report["representative_cell"]["threshold_drop_fraction_stats"]["min"]
                ),
                "threshold_drop_fraction_max": float(
                    report["representative_cell"]["threshold_drop_fraction_stats"]["max"]
                ),
                "budget_conservative_gpu_hours": float(report["budget"]["sweep_hours_conservative"]),
                "budget_optimistic_gpu_hours": float(report["budget"]["sweep_hours_optimistic"]),
                "global_roc_auc": diagnostic["global_roc_auc"],
                "mean_lead_time_evals": diagnostic["mean_lead_time_evals"],
                "run_count": len(diagnostic["runs"]),
                "runs_with_warning": sum(
                    1 for run in diagnostic["runs"] if run["first_warning_eval"] is not None
                ),
                "runs_with_collapse": sum(
                    1 for run in diagnostic["runs"] if run["first_collapse_eval"] is not None
                ),
            }
        )

        for run in diagnostic["runs"]:
            run_rows.append(
                {
                    "pilot_id": pilot_id,
                    "run_id": str(run["run_id"]),
                    "evals": int(run["evals"]),
                    "first_warning_eval": run["first_warning_eval"],
                    "first_collapse_eval": run["first_collapse_eval"],
                    "lead_time_evals": run["lead_time_evals"],
                }
            )

    return summary_rows, run_rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def export_paper_artifacts(
    results_root: Path,
    output_dir: Path,
    *,
    horizon_seeds: Mapping[int, Sequence[int]] | None = None,
    horizon_run_dirs: Mapping[int, str] | None = None,
    pilot_specs: Sequence[Mapping[str, str]] | None = None,
) -> dict[str, str]:
    points = collect_horizon_points(
        results_root,
        horizon_seeds=horizon_seeds,
        horizon_run_dirs=horizon_run_dirs,
    )
    summaries = summarize_horizon_points(points)
    warning_summaries, warning_runs = collect_warning_pilot_data(results_root, pilot_specs=pilot_specs)

    paths = {
        "horizon_points_csv": output_dir / "horizon_final_return_points.csv",
        "horizon_summary_csv": output_dir / "horizon_final_return_summary.csv",
        "horizon_summary_json": output_dir / "horizon_final_return_summary.json",
        "warning_summary_csv": output_dir / "warning_pilot_summary.csv",
        "warning_runs_csv": output_dir / "warning_pilot_runs.csv",
        "warning_summary_json": output_dir / "warning_pilot_summary.json",
    }

    _write_csv(
        paths["horizon_points_csv"],
        points,
        fieldnames=[
            "horizon",
            "seed",
            "collapsed",
            "warning_triggered",
            "final_return_mean",
            "final_return_std",
            "steps_per_second",
            "summary_path",
        ],
    )
    _write_csv(
        paths["horizon_summary_csv"],
        summaries,
        fieldnames=[
            "horizon",
            "n",
            "mean_final_return",
            "stdev_final_return",
            "min_final_return",
            "max_final_return",
            "mean_steps_per_second",
            "collapsed_count",
            "warning_count",
        ],
    )
    _write_json(paths["horizon_summary_json"], summaries)
    _write_csv(
        paths["warning_summary_csv"],
        warning_summaries,
        fieldnames=[
            "pilot_id",
            "decision",
            "reasons",
            "drop_fraction_mean",
            "drop_fraction_min",
            "drop_fraction_max",
            "threshold_drop_fraction_mean",
            "threshold_drop_fraction_min",
            "threshold_drop_fraction_max",
            "budget_conservative_gpu_hours",
            "budget_optimistic_gpu_hours",
            "global_roc_auc",
            "mean_lead_time_evals",
            "run_count",
            "runs_with_warning",
            "runs_with_collapse",
        ],
    )
    _write_csv(
        paths["warning_runs_csv"],
        warning_runs,
        fieldnames=[
            "pilot_id",
            "run_id",
            "evals",
            "first_warning_eval",
            "first_collapse_eval",
            "lead_time_evals",
        ],
    )
    _write_json(
        paths["warning_summary_json"],
        {
            "pilot_summaries": warning_summaries,
            "pilot_runs": warning_runs,
        },
    )
    return {name: str(path) for name, path in paths.items()}
