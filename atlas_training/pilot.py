from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

from atlas_training.config import VerticalSliceConfig, add_shift_cli_args, shift_from_args
from atlas_training.diagnostics import write_diagnostic_summary
from atlas_training.util import write_json

REPRESENTATIVE_N_STEP = 1
REPRESENTATIVE_CRITIC_WIDTH = 256
EXTREMAL_N_STEP = 10
EXTREMAL_CRITIC_WIDTH = 1024
DEFAULT_FINE_TUNE_STEPS = 2_000_000
DEFAULT_BASELINE_EVAL_EPISODES = 50
DEFAULT_THROUGHPUT_PROBE_UPDATES = 500
DEFAULT_PREDICTION_HORIZON = 10
POST_WARMUP_ROWS_REQUIRED = 3
WARMUP_EVALS = 2
SWEEP_RUN_COUNT = 48
CONSERVATIVE_SWEEP_HOURS_CEILING = 120.0
DROP_FRACTION_MIN = 0.15
DROP_FRACTION_MAX = 0.50
THRESHOLD_DROP_FRACTION_MAX = 0.50
EPSILON = 1e-8


@dataclass(frozen=True)
class PilotLayout:
    output_dir: Path
    pretrain_dir: Path
    probe_dir: Path
    report_path: Path
    seed_dirs: Dict[int, Path]


def parse_seed_list(raw_value: str) -> tuple[int, ...]:
    seeds = tuple(int(part.strip()) for part in raw_value.split(",") if part.strip())
    if not seeds:
        raise ValueError("seed list must not be empty")
    return seeds


def build_pilot_layout(output_dir: Path, seeds: Sequence[int]) -> PilotLayout:
    return PilotLayout(
        output_dir=output_dir,
        pretrain_dir=output_dir / "shared_pretrain",
        probe_dir=output_dir / "extreme_probe",
        report_path=output_dir / "pilot_report.json",
        seed_dirs={seed: output_dir / f"seed_{seed}" for seed in seeds},
    )


def required_eval_env_steps(eval_interval: int, required_rows: int = POST_WARMUP_ROWS_REQUIRED) -> int:
    return (WARMUP_EVALS + required_rows) * eval_interval


def realized_env_steps(train_steps: int, num_envs: int, action_repeat: int) -> int:
    if num_envs <= 0:
        raise ValueError("num_envs must be positive")
    if action_repeat <= 0:
        raise ValueError("action_repeat must be positive")
    return (train_steps // num_envs) * num_envs * action_repeat


def minimum_finetune_steps(
    eval_interval: int,
    *,
    num_envs: int,
    action_repeat: int,
    required_rows: int = POST_WARMUP_ROWS_REQUIRED,
) -> int:
    if num_envs <= 0:
        raise ValueError("num_envs must be positive")
    if action_repeat <= 0:
        raise ValueError("action_repeat must be positive")
    minimum_env_steps = required_eval_env_steps(eval_interval, required_rows=required_rows)
    env_steps_per_iteration = num_envs * action_repeat
    iterations_needed = math.ceil(minimum_env_steps / env_steps_per_iteration)
    return iterations_needed * num_envs


def hours_per_100m(steps_per_second: float) -> float:
    if steps_per_second <= 0 or not math.isfinite(steps_per_second):
        return math.inf
    return 100_000_000.0 / steps_per_second / 3600.0


def summarize_numeric(values: Iterable[float]) -> dict[str, float] | None:
    cleaned = [float(value) for value in values if math.isfinite(value)]
    if not cleaned:
        return None
    return {
        "mean": round(statistics.mean(cleaned), 6),
        "min": round(min(cleaned), 6),
        "max": round(max(cleaned), 6),
    }


def drop_fraction(nominal_mean: float, shifted_mean: float) -> float:
    return (nominal_mean - shifted_mean) / max(abs(nominal_mean), EPSILON)


def threshold_drop_fraction(mu0: float, threshold: float) -> float:
    return (mu0 - threshold) / max(abs(mu0), EPSILON)


def classify_pilot_gate(seed_results: Sequence[dict[str, Any]], sweep_hours_conservative: float) -> tuple[str, list[str]]:
    reasons: list[str] = []
    usable = [result for result in seed_results if bool(result.get("usable"))]

    if len(usable) < 2:
        reasons.append("fewer than 2 fine-tune seeds produced at least 3 post-warmup eval rows")
    if any(bool(result.get("has_nonfinite_metrics")) for result in usable):
        reasons.append("non-finite baseline or diagnostic metrics were observed")
    if not math.isfinite(sweep_hours_conservative) or sweep_hours_conservative > CONSERVATIVE_SWEEP_HOURS_CEILING:
        reasons.append(
            f"conservative 48-run budget {sweep_hours_conservative:.3f} exceeds {CONSERVATIVE_SWEEP_HOURS_CEILING:.0f} GPU-hours"
        )
    if reasons:
        return "fail", reasons

    in_band = [
        result
        for result in usable
        if DROP_FRACTION_MIN <= float(result["drop_fraction"]) <= DROP_FRACTION_MAX
    ]
    threshold_ok = all(float(result["threshold_drop_fraction"]) <= THRESHOLD_DROP_FRACTION_MAX for result in usable)

    if len(in_band) >= 2 and threshold_ok:
        return "proceed", [
            "at least 2 fine-tune seeds landed in the target drop-fraction band",
            "all usable seeds had non-degenerate collapse thresholds",
            f"conservative 48-run budget {sweep_hours_conservative:.3f} is within ceiling",
        ]

    adjust_reasons: list[str] = []
    if len(in_band) < 2:
        adjust_reasons.append(
            f"target drop-fraction band [{DROP_FRACTION_MIN:.2f}, {DROP_FRACTION_MAX:.2f}] was not met by at least 2 seeds"
        )
    if not threshold_ok:
        adjust_reasons.append(
            f"one or more usable seeds exceeded threshold_drop_fraction {THRESHOLD_DROP_FRACTION_MAX:.2f}"
        )
    return "adjust", adjust_reasons


def count_eval_rows(eval_log_path: Path) -> int:
    if not eval_log_path.exists():
        return 0
    with eval_log_path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Stability Atlas pilot calibration gate.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/runs/pilot_gate"))
    parser.add_argument("--run-id", type=str, default="pilot_gate")
    parser.add_argument("--env-name", type=str, default="Go1JoystickFlatTerrain")
    parser.add_argument("--pretrain-seed", type=int, default=0)
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--pretrain-steps", type=int, required=True)
    parser.add_argument("--fine-tune-steps", type=int, default=DEFAULT_FINE_TUNE_STEPS)
    parser.add_argument("--eval-interval", type=int, default=100_000)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--baseline-eval-episodes", type=int, default=DEFAULT_BASELINE_EVAL_EPISODES)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--replay-capacity", type=int, default=100_000)
    parser.add_argument("--min-replay-size", type=int, default=1024)
    parser.add_argument("--diagnostic-min-transitions", type=int, default=1024)
    parser.add_argument("--diagnostic-minibatches", type=int, default=100)
    parser.add_argument("--diagnostic-batch-size", type=int, default=256)
    parser.add_argument("--episode-length", type=int, default=1000)
    parser.add_argument("--action-repeat", type=int, default=1)
    parser.add_argument("--throughput-probe-updates", type=int, default=DEFAULT_THROUGHPUT_PROBE_UPDATES)
    parser.add_argument("--stop-on-collapse", action=argparse.BooleanOptionalAction, default=True)
    add_shift_cli_args(parser)
    args = parser.parse_args(argv)
    args.seed_values = parse_seed_list(args.seeds)
    minimum_steps = minimum_finetune_steps(
        args.eval_interval,
        num_envs=args.num_envs,
        action_repeat=args.action_repeat,
    )
    reachable_env_steps = realized_env_steps(args.fine_tune_steps, args.num_envs, args.action_repeat)
    minimum_env_steps = required_eval_env_steps(args.eval_interval)
    if reachable_env_steps < minimum_env_steps:
        raise ValueError(
            "--fine-tune-steps must be at least "
            f"{minimum_steps} to permit {POST_WARMUP_ROWS_REQUIRED} post-warmup eval rows "
            f"(reachable_env_steps={reachable_env_steps}, required_env_steps={minimum_env_steps})"
        )
    return args


def _is_finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def run_pilot_cli(args: argparse.Namespace) -> dict[str, Any]:
    from atlas_training.runtime import run_finetune, run_pretrain, run_throughput_probe

    seed_values = tuple(int(seed) for seed in args.seed_values)
    layout = build_pilot_layout(args.output_dir, seed_values)
    shift = shift_from_args(args)
    layout.output_dir.mkdir(parents=True, exist_ok=True)

    pretrain_config = VerticalSliceConfig(
        stage="pretrain",
        output_dir=layout.pretrain_dir,
        env_name=args.env_name,
        n_step=REPRESENTATIVE_N_STEP,
        critic_width=REPRESENTATIVE_CRITIC_WIDTH,
        seed=args.pretrain_seed,
        train_steps=args.pretrain_steps,
        eval_interval=args.eval_interval,
        num_envs=args.num_envs,
        eval_episodes=args.eval_episodes,
        batch_size=args.batch_size,
        replay_capacity=args.replay_capacity,
        min_replay_size=args.min_replay_size,
        diagnostic_min_transitions=args.diagnostic_min_transitions,
        diagnostic_minibatches=args.diagnostic_minibatches,
        diagnostic_batch_size=args.diagnostic_batch_size,
        episode_length=args.episode_length,
        action_repeat=args.action_repeat,
        stop_on_collapse=False,
        shift=shift,
    )
    pretrain_summary = run_pretrain(pretrain_config)
    nominal_mean = float(pretrain_summary["final_eval_return_mean"])
    nominal_std = float(pretrain_summary["final_eval_return_std"])

    seed_results: list[dict[str, Any]] = []
    for seed in seed_values:
        finetune_config = VerticalSliceConfig(
            stage="finetune",
            output_dir=layout.seed_dirs[seed],
            checkpoint=pretrain_config.checkpoint_dir(),
            env_name=args.env_name,
            n_step=REPRESENTATIVE_N_STEP,
            critic_width=REPRESENTATIVE_CRITIC_WIDTH,
            seed=seed,
            train_steps=args.fine_tune_steps,
            eval_interval=args.eval_interval,
            num_envs=args.num_envs,
            eval_episodes=args.eval_episodes,
            baseline_eval_episodes=args.baseline_eval_episodes,
            batch_size=args.batch_size,
            replay_capacity=args.replay_capacity,
            min_replay_size=args.min_replay_size,
            diagnostic_min_transitions=args.diagnostic_min_transitions,
            diagnostic_minibatches=args.diagnostic_minibatches,
            diagnostic_batch_size=args.diagnostic_batch_size,
            episode_length=args.episode_length,
            action_repeat=args.action_repeat,
            stop_on_collapse=args.stop_on_collapse,
            shift=shift,
        )
        diagnostic_summary_path = finetune_config.output_dir / "diagnostic_summary.json"
        try:
            finetune_summary = run_finetune(finetune_config)
            baseline_payload = json.loads(finetune_config.pretrain_baseline_path().read_text(encoding="utf-8"))
            diagnostic_summary = write_diagnostic_summary(
                finetune_config.eval_log_path(),
                diagnostic_summary_path,
                prediction_horizon=DEFAULT_PREDICTION_HORIZON,
                allow_missing=True,
            )
            emitted_rows = count_eval_rows(finetune_config.eval_log_path())
            mu0 = float(baseline_payload["mu0"])
            sigma0 = float(baseline_payload["sigma0"])
            threshold = float(baseline_payload["threshold"])
            has_nonfinite_metrics = not all(
                _is_finite_number(value)
                for value in (
                    mu0,
                    sigma0,
                    threshold,
                    finetune_summary["steps_per_second"],
                    finetune_summary["wallclock_seconds"],
                )
            )
            seed_results.append(
                {
                    "seed": seed,
                    "status": "ok",
                    "usable": emitted_rows >= POST_WARMUP_ROWS_REQUIRED,
                    "emitted_eval_rows": emitted_rows,
                    "mu0": mu0,
                    "sigma0": sigma0,
                    "threshold": threshold,
                    "drop_fraction": round(drop_fraction(nominal_mean, mu0), 6),
                    "threshold_drop_fraction": round(threshold_drop_fraction(mu0, threshold), 6),
                    "wallclock_seconds": float(finetune_summary["wallclock_seconds"]),
                    "steps_per_second": float(finetune_summary["steps_per_second"]),
                    "hours_per_100m": round(hours_per_100m(float(finetune_summary["steps_per_second"])), 6),
                    "throughput_scope": finetune_summary.get("throughput_scope", "wallclock_inclusive"),
                    "throughput_notes": list(finetune_summary.get("throughput_notes", [])),
                    "collapsed": bool(finetune_summary["collapsed"]),
                    "warning_triggered": bool(finetune_summary["warning_triggered"]),
                    "has_nonfinite_metrics": has_nonfinite_metrics,
                    "artifacts": {
                        "summary": finetune_config.summary_path(),
                        "baseline": finetune_config.pretrain_baseline_path(),
                        "eval_log": finetune_config.eval_log_path(),
                        "diagnostic_summary": diagnostic_summary_path,
                    },
                    "diagnostic_summary": diagnostic_summary,
                }
            )
        except Exception as exc:  # pragma: no cover - exercised in integration path
            seed_results.append(
                {
                    "seed": seed,
                    "status": "error",
                    "usable": False,
                    "error": str(exc),
                    "has_nonfinite_metrics": True,
                    "artifacts": {
                        "summary": finetune_config.summary_path(),
                        "baseline": finetune_config.pretrain_baseline_path(),
                        "eval_log": finetune_config.eval_log_path(),
                        "diagnostic_summary": diagnostic_summary_path,
                    },
                }
            )

    probe_config = VerticalSliceConfig(
        stage="throughput_probe",
        output_dir=layout.probe_dir,
        env_name=args.env_name,
        n_step=EXTREMAL_N_STEP,
        critic_width=EXTREMAL_CRITIC_WIDTH,
        seed=args.pretrain_seed,
        train_steps=args.fine_tune_steps,
        eval_interval=args.eval_interval,
        num_envs=args.num_envs,
        eval_episodes=args.eval_episodes,
        batch_size=args.batch_size,
        replay_capacity=args.replay_capacity,
        min_replay_size=args.min_replay_size,
        diagnostic_min_transitions=args.diagnostic_min_transitions,
        diagnostic_minibatches=args.diagnostic_minibatches,
        diagnostic_batch_size=args.diagnostic_batch_size,
        episode_length=args.episode_length,
        action_repeat=args.action_repeat,
        stop_on_collapse=False,
        shift=shift,
    )
    probe_error = None
    probe_summary: dict[str, Any] | None = None
    try:
        probe_summary = run_throughput_probe(
            probe_config,
            updates_per_window=args.throughput_probe_updates,
            timed_update_windows=3,
        )
    except Exception as exc:  # pragma: no cover - exercised in integration path
        probe_error = str(exc)

    usable_results = [result for result in seed_results if bool(result.get("usable"))]
    representative_hours = [
        float(result["hours_per_100m"])
        for result in usable_results
        if result.get("status") == "ok" and _is_finite_number(result.get("hours_per_100m"))
    ]
    representative_speeds = [
        float(result["steps_per_second"])
        for result in usable_results
        if result.get("status") == "ok" and _is_finite_number(result.get("steps_per_second"))
    ]
    drop_stats = summarize_numeric(
        float(result["drop_fraction"])
        for result in usable_results
        if _is_finite_number(result.get("drop_fraction"))
    )
    threshold_drop_stats = summarize_numeric(
        float(result["threshold_drop_fraction"])
        for result in usable_results
        if _is_finite_number(result.get("threshold_drop_fraction"))
    )
    representative_throughput_stats = summarize_numeric(representative_speeds)
    representative_hours_stats = summarize_numeric(representative_hours)
    extreme_steps_per_second = float(probe_summary["steps_per_second"]) if probe_summary is not None else math.inf
    extreme_hours_per_100m = hours_per_100m(extreme_steps_per_second)
    budget = {
        "hours_per_100m_small": representative_hours_stats,
        "hours_per_100m_extreme": None if not math.isfinite(extreme_hours_per_100m) else round(extreme_hours_per_100m, 6),
        "sweep_hours_optimistic": None,
        "sweep_hours_conservative": math.inf,
    }
    if representative_hours_stats is not None:
        budget["sweep_hours_optimistic"] = round(SWEEP_RUN_COUNT * representative_hours_stats["mean"], 6)
    if math.isfinite(extreme_hours_per_100m):
        budget["sweep_hours_conservative"] = round(SWEEP_RUN_COUNT * extreme_hours_per_100m, 6)

    decision, reasons = classify_pilot_gate(seed_results, float(budget["sweep_hours_conservative"]))

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pilot_id": args.run_id,
        "decision": decision,
        "reasons": reasons,
        "caveats": [
            "Fine-tune seed variance is measured from a shared pretrain checkpoint; pretrain-seed variance is not captured.",
            "The conservative budget bound is derived from a single extremal throughput probe on n=10, critic_width=1024.",
            "Representative throughput is derived from full fine-tune wall-clock and includes baseline evaluation, eval callbacks, and artifact overhead, so it is a pessimistic end-to-end estimate.",
        ],
        "pretrain": {
            "seed": args.pretrain_seed,
            "steps": args.pretrain_steps,
            "checkpoint_dir": pretrain_config.checkpoint_dir(),
            "summary": pretrain_config.summary_path(),
            "nominal_return_mean": nominal_mean,
            "nominal_return_std": nominal_std,
            "wallclock_seconds": float(pretrain_summary["wallclock_seconds"]),
            "steps_per_second": float(pretrain_summary["steps_per_second"]),
            "throughput_scope": pretrain_summary.get("throughput_scope", "wallclock_inclusive"),
            "throughput_notes": list(pretrain_summary.get("throughput_notes", [])),
        },
        "representative_cell": {
            "n_step": REPRESENTATIVE_N_STEP,
            "critic_width": REPRESENTATIVE_CRITIC_WIDTH,
            "fine_tune_steps": args.fine_tune_steps,
            "baseline_eval_episodes": args.baseline_eval_episodes,
            "throughput_scope": "wallclock_inclusive",
            "throughput_notes": [
                "Representative per-seed throughput is taken from full fine-tune wall-clock summaries and includes eval overhead.",
            ],
            "seed_results": seed_results,
            "drop_fraction_stats": drop_stats,
            "threshold_drop_fraction_stats": threshold_drop_stats,
            "throughput_steps_per_second_stats": representative_throughput_stats,
        },
        "extreme_probe": {
            "n_step": EXTREMAL_N_STEP,
            "critic_width": EXTREMAL_CRITIC_WIDTH,
            "summary": probe_config.summary_path(),
            "error": probe_error,
            "throughput_scope": None if probe_summary is None else probe_summary.get("throughput_scope"),
            "throughput_notes": [] if probe_summary is None else list(probe_summary.get("throughput_notes", [])),
            "timed_update_windows": None if probe_summary is None else probe_summary["timed_update_windows"],
            "throughput_window_stats": None if probe_summary is None else probe_summary["throughput_window_stats"],
            "steps_per_second": None if probe_summary is None else probe_summary["steps_per_second"],
        },
        "budget": budget,
        "artifacts": {
            "report": layout.report_path,
            "pretrain_summary": pretrain_config.summary_path(),
            "pretrain_checkpoint": pretrain_config.checkpoint_dir(),
            "extreme_probe_summary": probe_config.summary_path(),
        },
        "shift": {
            "train_friction_range": shift.train_friction_range,
            "train_payload_range": shift.train_payload_range,
            "fine_tune_friction": shift.fine_tune_friction,
            "fine_tune_payload": shift.fine_tune_payload,
        },
    }
    write_json(layout.report_path, report)
    return report
