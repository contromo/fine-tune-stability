from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Sequence

from atlas.diagnostics import (
    DEFAULT_TRIGGER_HOLD_EVALS,
    DEFAULT_TRIGGER_THRESHOLD,
    InstabilityTrigger,
    collapse_horizon_labels,
    collapse_threshold,
    roc_auc,
)


@dataclass(frozen=True)
class FrozenBaseline:
    mu0: float
    sigma0: float
    threshold: float
    collapse_c: float
    collapse_rho: float
    threshold_rule: str


@dataclass(frozen=True)
class DiagnosticLogState:
    warmup_variances: tuple[float, ...] = ()
    emitted_rows: int = 0


@dataclass(frozen=True)
class EvalLogRow:
    run_id: str
    eval_index: int
    score: float
    collapsed: bool
    return_mean: float
    variance: float
    q95_abs_td: float
    threshold: float
    env_steps: int
    actor_kl_drift: float | None = None
    q_magnitude_drift: float | None = None

    def to_dict(self) -> Dict[str, float | int | bool | str]:
        row: Dict[str, float | int | bool | str] = {
            "run_id": self.run_id,
            "eval_index": self.eval_index,
            "score": self.score,
            "collapsed": self.collapsed,
            "return_mean": self.return_mean,
            "variance": self.variance,
            "q95_abs_td": self.q95_abs_td,
            "threshold": self.threshold,
            "env_steps": self.env_steps,
        }
        if self.actor_kl_drift is not None:
            row["actor_kl_drift"] = self.actor_kl_drift
        if self.q_magnitude_drift is not None:
            row["q_magnitude_drift"] = self.q_magnitude_drift
        return row


def freeze_baseline(
    returns: Sequence[float],
    c: float = 2.0,
    rho: float = 0.2,
) -> FrozenBaseline:
    if not returns:
        raise ValueError("returns must not be empty")
    mu0 = statistics.mean(returns)
    sigma0 = statistics.pstdev(returns)
    sigma_rule = mu0 - (c * sigma0)
    floor_rule = mu0 - (rho * abs(mu0))
    threshold = collapse_threshold(mu0, sigma0, c=c, rho=rho)
    threshold_rule = "sigma" if sigma_rule <= floor_rule else "floor"
    return FrozenBaseline(
        mu0=mu0,
        sigma0=sigma0,
        threshold=threshold,
        collapse_c=c,
        collapse_rho=rho,
        threshold_rule=threshold_rule,
    )


def current_warmup_variance(state: DiagnosticLogState) -> float | None:
    if len(state.warmup_variances) < 2:
        return None
    return statistics.mean(state.warmup_variances[:2])


def record_warmup_variance(state: DiagnosticLogState, variance: float) -> DiagnosticLogState:
    if len(state.warmup_variances) >= 2:
        return state
    return DiagnosticLogState(
        warmup_variances=state.warmup_variances + (variance,),
        emitted_rows=state.emitted_rows,
    )


def mark_eval_row_emitted(state: DiagnosticLogState) -> DiagnosticLogState:
    return DiagnosticLogState(
        warmup_variances=state.warmup_variances,
        emitted_rows=state.emitted_rows + 1,
    )


def advance_next_eval_at(next_eval_at: int, env_steps: int, eval_interval: int) -> int:
    if eval_interval <= 0:
        raise ValueError("eval_interval must be positive")
    if env_steps < next_eval_at:
        return next_eval_at
    skipped_intervals = (env_steps - next_eval_at) // eval_interval
    return next_eval_at + (skipped_intervals + 1) * eval_interval


def make_eval_log_row(
    *,
    run_id: str,
    eval_index: int,
    score: float,
    collapsed: bool,
    return_mean: float,
    variance: float,
    q95_abs_td: float,
    threshold: float,
    env_steps: int,
    actor_kl_drift: float | None = None,
    q_magnitude_drift: float | None = None,
) -> EvalLogRow:
    return EvalLogRow(
        run_id=run_id,
        eval_index=eval_index,
        score=score,
        collapsed=collapsed,
        return_mean=return_mean,
        variance=variance,
        q95_abs_td=q95_abs_td,
        threshold=threshold,
        env_steps=env_steps,
        actor_kl_drift=actor_kl_drift,
        q_magnitude_drift=q_magnitude_drift,
    )


def load_eval_log(path: Path, *, allow_missing: bool = False) -> Dict[str, list[dict[str, Any]]]:
    grouped: Dict[str, list[dict[str, Any]]] = {}
    if allow_missing and not path.exists():
        return grouped
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
            grouped.setdefault(str(row["run_id"]), []).append(row)

    for rows in grouped.values():
        rows.sort(key=lambda row: int(row["eval_index"]))
    return grouped


def _first_warning_index(scores: Sequence[float], threshold: float, hold_evals: int) -> int | None:
    trigger = InstabilityTrigger(threshold=threshold, hold_evals=hold_evals)
    for index, score in enumerate(scores):
        if trigger.update(float(score)):
            return index - (hold_evals - 1)
    return None


def summarize_eval_groups(
    grouped_rows: Dict[str, list[dict[str, Any]]],
    prediction_horizon: int,
    *,
    score_field: str = "score",
    trigger_threshold: float | None = None,
    trigger_hold_evals: int = DEFAULT_TRIGGER_HOLD_EVALS,
) -> dict[str, Any]:
    """Summarize grouped eval rows into AUC and (optional) lead-time metrics.

    AUC is rank-based and threshold-free — always computed when labels permit.
    Warning / lead-time metrics require a trigger threshold. For the canonical
    TD-variance score field, `DEFAULT_TRIGGER_THRESHOLD` is used implicitly.
    For alt score fields, pass `trigger_threshold` explicitly; otherwise
    `first_warning_eval` and `lead_time_evals` are `None`.
    """
    effective_threshold: float | None
    if trigger_threshold is not None:
        effective_threshold = trigger_threshold
    elif score_field == "score":
        effective_threshold = DEFAULT_TRIGGER_THRESHOLD
    else:
        effective_threshold = None

    scores: list[float] = []
    labels: list[int] = []
    lead_times: list[int] = []
    per_run: list[dict[str, Any]] = []

    for run_id, rows in sorted(grouped_rows.items()):
        missing = [index for index, row in enumerate(rows) if score_field not in row]
        if missing:
            raise ValueError(
                f"run {run_id}: rows {missing} missing score_field '{score_field}'"
            )
        run_scores = [float(row[score_field]) for row in rows]
        collapse_flags = [bool(row["collapsed"]) for row in rows]
        run_labels = collapse_horizon_labels(collapse_flags, prediction_horizon)
        scores.extend(run_scores)
        labels.extend(run_labels)

        if effective_threshold is None:
            warning_index = None
        else:
            warning_index = _first_warning_index(run_scores, effective_threshold, trigger_hold_evals)
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
        "score_field": score_field,
        "trigger_threshold": effective_threshold,
        "trigger_calibrated": score_field == "score" and trigger_threshold is None,
        "runs": per_run,
        "global_roc_auc": auc,
        "mean_lead_time_evals": (sum(lead_times) / len(lead_times)) if lead_times else None,
        "prediction_horizon_evals": prediction_horizon,
    }


def write_diagnostic_summary(
    eval_log_path: Path,
    output_path: Path,
    *,
    prediction_horizon: int,
    allow_missing: bool = False,
    score_field: str = "score",
    trigger_threshold: float | None = None,
) -> dict[str, Any]:
    summary = summarize_eval_groups(
        load_eval_log(eval_log_path, allow_missing=allow_missing),
        prediction_horizon,
        score_field=score_field,
        trigger_threshold=trigger_threshold,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
