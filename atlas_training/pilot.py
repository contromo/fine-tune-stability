from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

from atlas_training.config import VerticalSliceConfig, add_collapse_cli_args, add_shift_cli_args, shift_from_args
from atlas_training.diagnostics import write_diagnostic_summary
from atlas_training.preflight import (
    DEFAULT_MIN_FREE_DISK_GB,
    environment_from_preflight,
    resolve_preflight_path,
    run_preflight,
)
from atlas_training.util import hours_per_100m, write_json

DEFAULT_PROFILE = "default"
PRODUCTION_PROFILE = "production"
DEFAULT_PRODUCTION_PRETRAIN_STEPS = 1_000_000
REPRESENTATIVE_N_STEP = 1
REPRESENTATIVE_CRITIC_WIDTH = 256
EXTREMAL_N_STEP = 10
EXTREMAL_CRITIC_WIDTH = 1024
DEFAULT_FINE_TUNE_STEPS = 2_000_000
DEFAULT_BASELINE_EVAL_EPISODES = 50
DEFAULT_THROUGHPUT_PROBE_UPDATES = 500
DEFAULT_PREDICTION_HORIZON = 10
POST_WARMUP_ROWS_REQUIRED = 3
MIN_USABLE_SEEDS = 2
WARMUP_EVALS = 2
SWEEP_RUN_COUNT = 48
CONSERVATIVE_SWEEP_HOURS_CEILING = 120.0
DROP_FRACTION_MIN = 0.15
DROP_FRACTION_MAX = 0.50
THRESHOLD_DROP_FRACTION_MAX = 0.50
EPSILON = 1e-8
DECISION_NOTE_MARKER = "<!-- AUTO-GENERATED PILOT DECISION STUB: SAFE TO OVERWRITE UNTIL NEXT ACTION IS EDITED -->"
DECISION_NOTE_PLACEHOLDER = "- TODO: replace with the chosen next action before committing."
DEFAULT_DECISION_DIR = Path(__file__).resolve().parents[1] / "docs" / "decisions"


@dataclass(frozen=True)
class PilotLayout:
    output_dir: Path
    pretrain_dir: Path
    probe_dir: Path
    report_path: Path
    seed_dirs: Dict[int, Path]


@dataclass(frozen=True)
class PretrainPhaseResult:
    config: VerticalSliceConfig
    summary: dict[str, Any]
    nominal_mean: float
    nominal_std: float


@dataclass(frozen=True)
class ProbePhaseResult:
    config: VerticalSliceConfig
    summary: dict[str, Any] | None
    error: str | None


def _profile_defaults(profile: str) -> dict[str, Any]:
    if profile == PRODUCTION_PROFILE:
        return {
            "output_dir": Path("results/runs/pilot_gate"),
            "run_id": "pilot_gate",
            "pretrain_steps": DEFAULT_PRODUCTION_PRETRAIN_STEPS,
            "fine_tune_steps": DEFAULT_FINE_TUNE_STEPS,
            "seeds": "0,1,2",
            "baseline_eval_episodes": DEFAULT_BASELINE_EVAL_EPISODES,
            "eval_interval": 100_000,
            "num_envs": 32,
            "stop_on_collapse": True,
        }
    return {}


def _pilot_config_kwargs(args: argparse.Namespace, shift: Any) -> dict[str, Any]:
    return {
        "env_name": args.env_name,
        "eval_interval": args.eval_interval,
        "num_envs": args.num_envs,
        "eval_episodes": args.eval_episodes,
        "batch_size": args.batch_size,
        "replay_capacity": args.replay_capacity,
        "min_replay_size": args.min_replay_size,
        "collapse_c": args.collapse_c,
        "collapse_rho": args.collapse_rho,
        "diagnostic_min_transitions": args.diagnostic_min_transitions,
        "diagnostic_minibatches": args.diagnostic_minibatches,
        "diagnostic_batch_size": args.diagnostic_batch_size,
        "episode_length": args.episode_length,
        "action_repeat": args.action_repeat,
        "shift": shift,
    }


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


def build_budget_summary(
    representative_hours_per_100m: Sequence[float],
    extreme_hours_per_100m: float,
) -> dict[str, Any]:
    representative_hours_stats = summarize_numeric(representative_hours_per_100m)
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
    return budget


def classify_pilot_gate(seed_results: Sequence[dict[str, Any]], sweep_hours_conservative: float) -> tuple[str, list[str]]:
    reasons: list[str] = []
    usable = [result for result in seed_results if bool(result.get("usable"))]

    if len(usable) < MIN_USABLE_SEEDS:
        reasons.append(
            f"fewer than {MIN_USABLE_SEEDS} fine-tune seeds produced at least {POST_WARMUP_ROWS_REQUIRED} post-warmup eval rows"
        )
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

    if len(in_band) >= MIN_USABLE_SEEDS and threshold_ok:
        return "proceed", [
            f"at least {MIN_USABLE_SEEDS} fine-tune seeds landed in the target drop-fraction band",
            "all usable seeds had non-degenerate collapse thresholds",
            f"conservative 48-run budget {sweep_hours_conservative:.3f} is within ceiling",
        ]

    adjust_reasons: list[str] = []
    if len(in_band) < MIN_USABLE_SEEDS:
        adjust_reasons.append(
            f"target drop-fraction band [{DROP_FRACTION_MIN:.2f}, {DROP_FRACTION_MAX:.2f}] was not met by at least {MIN_USABLE_SEEDS} seeds"
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


def _decision_note_path(run_id: str, decision_dir: Path, created_at: datetime | None = None) -> Path:
    timestamp = datetime.now(timezone.utc) if created_at is None else created_at
    return decision_dir / f"{timestamp.date().isoformat()}-{run_id}.md"


def _can_overwrite_decision_note(path: Path) -> bool:
    if not path.exists():
        return True
    contents = path.read_text(encoding="utf-8")
    # Only auto-overwrite the untouched stub. If either sentinel is missing,
    # assume a human has taken ownership of the file.
    return DECISION_NOTE_MARKER in contents and DECISION_NOTE_PLACEHOLDER in contents


def _ensure_decision_note_can_be_written(path: Path) -> None:
    if _can_overwrite_decision_note(path):
        return
    raise FileExistsError(
        f"decision note already exists and appears to be edited: {path}; use a different --run-id or remove it manually"
    )


def _ensure_phase_can_run(phase_name: str, summary_path: Path, *, force: bool) -> None:
    if force or not summary_path.exists():
        return
    raise FileExistsError(
        f"{phase_name} is already complete at {summary_path}; rerun with --force to overwrite completed phases"
    )


def _decision_note_template(report: dict[str, Any]) -> str:
    representative = report["representative_cell"]
    budget = report["budget"]
    drop_stats = representative.get("drop_fraction_stats")
    threshold_stats = representative.get("threshold_drop_fraction_stats")
    lines = [
        DECISION_NOTE_MARKER,
        f"# Pilot Decision: {report['pilot_id']}",
        "",
        f"- Created at: {report['created_at']}",
        f"- Decision: {report['decision']}",
        f"- Pilot report: {report['artifacts']['report']}",
        "",
        "## Key Metrics",
        f"- Conservative sweep budget (48 runs): {budget['sweep_hours_conservative']}",
        f"- Optimistic sweep budget (48 runs): {budget['sweep_hours_optimistic']}",
        f"- Representative drop-fraction stats: {drop_stats}",
        f"- Threshold-drop-fraction stats: {threshold_stats}",
        f"- Threshold calibration: c={report['threshold_calibration']['collapse_c']}, rho={report['threshold_calibration']['collapse_rho']}",
        "",
        "## Next Action",
        DECISION_NOTE_PLACEHOLDER,
        "",
    ]
    return "\n".join(lines)


def _write_decision_note(report: dict[str, Any], note_path: Path) -> None:
    _ensure_decision_note_can_be_written(note_path)
    note_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text(_decision_note_template(report), encoding="utf-8")


def _preparse_profile(argv: Sequence[str] | None) -> argparse.Namespace:
    # Parse just enough to choose profile-scoped defaults before the full CLI parse.
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--profile", choices=(DEFAULT_PROFILE, PRODUCTION_PROFILE), default=DEFAULT_PROFILE)
    parser.add_argument("--preflight-only", action="store_true")
    parsed, _unknown = parser.parse_known_args(argv)
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    pre_parsed = _preparse_profile(argv)
    profile_defaults = _profile_defaults(pre_parsed.profile)
    require_pretrain_steps = not pre_parsed.preflight_only and profile_defaults.get("pretrain_steps") is None
    parser = argparse.ArgumentParser(description="Run the Stability Atlas pilot calibration gate.")
    parser.add_argument("--profile", choices=(DEFAULT_PROFILE, PRODUCTION_PROFILE), default=pre_parsed.profile)
    parser.add_argument("--output-dir", type=Path, default=profile_defaults.get("output_dir", Path("results/runs/pilot_gate")))
    parser.add_argument("--run-id", type=str, default=profile_defaults.get("run_id", "pilot_gate"))
    parser.add_argument(
        "--decision-dir",
        type=Path,
        default=DEFAULT_DECISION_DIR,
        help="directory for the generated decision note (defaults to the repo's docs/decisions directory)",
    )
    parser.add_argument("--env-name", type=str, default="Go1JoystickFlatTerrain")
    parser.add_argument("--pretrain-seed", type=int, default=0)
    parser.add_argument("--seeds", type=str, default=profile_defaults.get("seeds", "0,1,2"))
    parser.add_argument(
        "--pretrain-steps",
        type=int,
        default=profile_defaults.get("pretrain_steps"),
        required=require_pretrain_steps,
    )
    parser.add_argument("--fine-tune-steps", type=int, default=profile_defaults.get("fine_tune_steps", DEFAULT_FINE_TUNE_STEPS))
    parser.add_argument("--eval-interval", type=int, default=profile_defaults.get("eval_interval", 100_000))
    parser.add_argument("--num-envs", type=int, default=profile_defaults.get("num_envs", 32))
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument(
        "--baseline-eval-episodes",
        type=int,
        default=profile_defaults.get("baseline_eval_episodes", DEFAULT_BASELINE_EVAL_EPISODES),
    )
    add_collapse_cli_args(parser)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--replay-capacity", type=int, default=100_000)
    parser.add_argument("--min-replay-size", type=int, default=1024)
    parser.add_argument("--diagnostic-min-transitions", type=int, default=1024)
    parser.add_argument("--diagnostic-minibatches", type=int, default=100)
    parser.add_argument("--diagnostic-batch-size", type=int, default=256)
    parser.add_argument("--episode-length", type=int, default=1000)
    parser.add_argument("--action-repeat", type=int, default=1)
    parser.add_argument("--throughput-probe-updates", type=int, default=DEFAULT_THROUGHPUT_PROBE_UPDATES)
    parser.add_argument(
        "--stop-on-collapse",
        action=argparse.BooleanOptionalAction,
        default=profile_defaults.get("stop_on_collapse", True),
    )
    parser.add_argument("--preflight-only", action="store_true", default=pre_parsed.preflight_only)
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="overwrite all completed pilot phases for this run; this is a global rerun flag, not a per-phase selector",
    )
    parser.add_argument("--allow-cpu", action="store_true", default=False)
    parser.add_argument("--preflight-json", type=Path, default=None)
    parser.add_argument("--min-free-disk-gb", type=float, default=DEFAULT_MIN_FREE_DISK_GB)
    add_shift_cli_args(parser)
    args = parser.parse_args(argv)
    args.seed_values = parse_seed_list(args.seeds)
    if args.preflight_only:
        return args
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


def _build_pretrain_config(
    args: argparse.Namespace,
    layout: PilotLayout,
    shift: Any,
) -> VerticalSliceConfig:
    return VerticalSliceConfig(
        stage="pretrain",
        output_dir=layout.pretrain_dir,
        n_step=REPRESENTATIVE_N_STEP,
        critic_width=REPRESENTATIVE_CRITIC_WIDTH,
        seed=args.pretrain_seed,
        train_steps=args.pretrain_steps,
        stop_on_collapse=False,
        **_pilot_config_kwargs(args, shift),
    )


def _run_pretrain_phase(
    args: argparse.Namespace,
    layout: PilotLayout,
    shift: Any,
    run_pretrain: Any,
) -> PretrainPhaseResult:
    pretrain_config = _build_pretrain_config(args, layout, shift)
    _ensure_phase_can_run("shared_pretrain", pretrain_config.summary_path(), force=args.force)
    pretrain_summary = run_pretrain(pretrain_config)
    return PretrainPhaseResult(
        config=pretrain_config,
        summary=pretrain_summary,
        nominal_mean=float(pretrain_summary["final_eval_return_mean"]),
        nominal_std=float(pretrain_summary["final_eval_return_std"]),
    )


def _run_preflight_phase(args: argparse.Namespace, layout: PilotLayout) -> tuple[Path, dict[str, Any]]:
    preflight_path = resolve_preflight_path(layout.output_dir, args.preflight_json)
    payload = run_preflight(
        output_dir=layout.output_dir,
        preflight_path=preflight_path,
        allow_cpu=args.allow_cpu,
        min_free_disk_gb=args.min_free_disk_gb,
        cwd=Path.cwd(),
    )
    return preflight_path, payload


def _build_finetune_config(
    args: argparse.Namespace,
    layout: PilotLayout,
    shift: Any,
    *,
    seed: int,
    checkpoint_dir: Path,
) -> VerticalSliceConfig:
    return VerticalSliceConfig(
        stage="finetune",
        output_dir=layout.seed_dirs[seed],
        checkpoint=checkpoint_dir,
        n_step=REPRESENTATIVE_N_STEP,
        critic_width=REPRESENTATIVE_CRITIC_WIDTH,
        seed=seed,
        train_steps=args.fine_tune_steps,
        baseline_eval_episodes=args.baseline_eval_episodes,
        stop_on_collapse=args.stop_on_collapse,
        **_pilot_config_kwargs(args, shift),
    )


def _run_finetune_seed(
    args: argparse.Namespace,
    shift: Any,
    *,
    seed: int,
    layout: PilotLayout,
    pretrain: PretrainPhaseResult,
    run_finetune: Any,
) -> dict[str, Any]:
    finetune_config = _build_finetune_config(
        args,
        layout,
        shift,
        seed=seed,
        checkpoint_dir=pretrain.config.checkpoint_dir(),
    )
    _ensure_phase_can_run(f"seed_{seed}", finetune_config.summary_path(), force=args.force)
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
        write_json(finetune_config.summary_path(), finetune_summary)
        emitted_rows = count_eval_rows(finetune_config.eval_log_path())
        mu0 = float(baseline_payload["mu0"])
        sigma0 = float(baseline_payload["sigma0"])
        threshold = float(baseline_payload["threshold"])
        collapse_c = float(baseline_payload.get("collapse_c", finetune_config.collapse_c))
        collapse_rho = float(baseline_payload.get("collapse_rho", finetune_config.collapse_rho))
        threshold_rule = str(baseline_payload.get("threshold_rule", "sigma"))
        has_nonfinite_metrics = not all(
            _is_finite_number(value)
            for value in (
                mu0,
                sigma0,
                threshold,
                collapse_c,
                collapse_rho,
                finetune_summary["steps_per_second"],
                finetune_summary["wallclock_seconds"],
            )
        )
        return {
            "seed": seed,
            "status": "ok",
            "usable": emitted_rows >= POST_WARMUP_ROWS_REQUIRED,
            "emitted_eval_rows": emitted_rows,
            "mu0": mu0,
            "sigma0": sigma0,
            "threshold": threshold,
            "collapse_c": collapse_c,
            "collapse_rho": collapse_rho,
            "threshold_rule": threshold_rule,
            "drop_fraction": round(drop_fraction(pretrain.nominal_mean, mu0), 6),
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
    except Exception as exc:  # pragma: no cover - exercised in integration path
        return {
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


def _run_finetune_phase(
    args: argparse.Namespace,
    layout: PilotLayout,
    shift: Any,
    pretrain: PretrainPhaseResult,
    run_finetune: Any,
) -> list[dict[str, Any]]:
    return [
        _run_finetune_seed(
            args,
            shift,
            seed=seed,
            layout=layout,
            pretrain=pretrain,
            run_finetune=run_finetune,
        )
        for seed in args.seed_values
    ]


def _build_probe_config(
    args: argparse.Namespace,
    layout: PilotLayout,
    shift: Any,
) -> VerticalSliceConfig:
    return VerticalSliceConfig(
        stage="throughput_probe",
        output_dir=layout.probe_dir,
        n_step=EXTREMAL_N_STEP,
        critic_width=EXTREMAL_CRITIC_WIDTH,
        seed=args.pretrain_seed,
        train_steps=args.fine_tune_steps,
        stop_on_collapse=False,
        **_pilot_config_kwargs(args, shift),
    )


def _run_probe_phase(
    args: argparse.Namespace,
    layout: PilotLayout,
    shift: Any,
    run_throughput_probe: Any,
) -> ProbePhaseResult:
    probe_config = _build_probe_config(args, layout, shift)
    _ensure_phase_can_run("extreme_probe", probe_config.summary_path(), force=args.force)
    try:
        return ProbePhaseResult(
            config=probe_config,
            summary=run_throughput_probe(
                probe_config,
                updates_per_window=args.throughput_probe_updates,
                timed_update_windows=3,
            ),
            error=None,
        )
    except Exception as exc:  # pragma: no cover - exercised in integration path
        return ProbePhaseResult(config=probe_config, summary=None, error=str(exc))


def _build_pilot_report(
    args: argparse.Namespace,
    layout: PilotLayout,
    shift: Any,
    preflight_path: Path,
    preflight: dict[str, Any],
    decision_note_path: Path,
    created_at: datetime,
    pretrain: PretrainPhaseResult,
    seed_results: Sequence[dict[str, Any]],
    probe: ProbePhaseResult,
) -> dict[str, Any]:
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
    extreme_steps_per_second = float(probe.summary["steps_per_second"]) if probe.summary is not None else 0.0
    extreme_hours_per_100m = hours_per_100m(extreme_steps_per_second)
    budget = build_budget_summary(representative_hours, extreme_hours_per_100m)

    decision, reasons = classify_pilot_gate(seed_results, float(budget["sweep_hours_conservative"]))

    return {
        "created_at": created_at.isoformat(),
        "pilot_id": args.run_id,
        "preflight_path": preflight_path,
        "environment": environment_from_preflight(preflight),
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
            "checkpoint_dir": pretrain.config.checkpoint_dir(),
            "summary": pretrain.config.summary_path(),
            "nominal_return_mean": pretrain.nominal_mean,
            "nominal_return_std": pretrain.nominal_std,
            "wallclock_seconds": float(pretrain.summary["wallclock_seconds"]),
            "steps_per_second": float(pretrain.summary["steps_per_second"]),
            "throughput_scope": pretrain.summary.get("throughput_scope", "wallclock_inclusive"),
            "throughput_notes": list(pretrain.summary.get("throughput_notes", [])),
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
        "threshold_calibration": {
            "collapse_c": args.collapse_c,
            "collapse_rho": args.collapse_rho,
        },
        "extreme_probe": {
            "n_step": EXTREMAL_N_STEP,
            "critic_width": EXTREMAL_CRITIC_WIDTH,
            "summary": probe.config.summary_path(),
            "error": probe.error,
            "throughput_scope": None if probe.summary is None else probe.summary.get("throughput_scope"),
            "throughput_notes": [] if probe.summary is None else list(probe.summary.get("throughput_notes", [])),
            "timed_update_windows": None if probe.summary is None else probe.summary["timed_update_windows"],
            "throughput_window_stats": None if probe.summary is None else probe.summary["throughput_window_stats"],
            "steps_per_second": None if probe.summary is None else probe.summary["steps_per_second"],
        },
        "budget": budget,
        "artifacts": {
            "report": layout.report_path,
            "preflight": preflight_path,
            "decision_note": decision_note_path,
            "pretrain_summary": pretrain.config.summary_path(),
            "pretrain_checkpoint": pretrain.config.checkpoint_dir(),
            "extreme_probe_summary": probe.config.summary_path(),
        },
        "shift": {
            "train_friction_range": shift.train_friction_range,
            "train_payload_range": shift.train_payload_range,
            "fine_tune_friction": shift.fine_tune_friction,
            "fine_tune_payload": shift.fine_tune_payload,
        },
    }


def run_pilot_cli(args: argparse.Namespace) -> dict[str, Any]:
    seed_values = args.seed_values
    layout = build_pilot_layout(args.output_dir, seed_values)
    shift = shift_from_args(args)
    created_at = datetime.now(timezone.utc)
    decision_note_path = _decision_note_path(args.run_id, args.decision_dir, created_at=created_at)

    if not args.preflight_only:
        _ensure_decision_note_can_be_written(decision_note_path)

    preflight_path, preflight = _run_preflight_phase(args, layout)
    if args.preflight_only:
        return {
            "mode": "preflight",
            "pilot_id": args.run_id,
            "output_dir": layout.output_dir,
            "preflight_path": preflight_path,
            "preflight": preflight,
        }

    from atlas_training.runtime import run_finetune, run_pretrain, run_throughput_probe

    pretrain = _run_pretrain_phase(args, layout, shift, run_pretrain)
    seed_results = _run_finetune_phase(args, layout, shift, pretrain, run_finetune)
    probe = _run_probe_phase(args, layout, shift, run_throughput_probe)
    report = _build_pilot_report(
        args,
        layout,
        shift,
        preflight_path,
        preflight,
        decision_note_path,
        created_at,
        pretrain,
        seed_results,
        probe,
    )
    write_json(layout.report_path, report)
    _write_decision_note(report, decision_note_path)
    return report
