from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Dict, Sequence

from atlas.diagnostics import collapse_threshold


@dataclass(frozen=True)
class FrozenBaseline:
    mu0: float
    sigma0: float
    threshold: float


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

    def to_dict(self) -> Dict[str, float | int | bool | str]:
        return {
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


def freeze_baseline(
    returns: Sequence[float],
    c: float = 2.0,
    rho: float = 0.2,
) -> FrozenBaseline:
    if not returns:
        raise ValueError("returns must not be empty")
    mu0 = statistics.mean(returns)
    sigma0 = statistics.pstdev(returns)
    return FrozenBaseline(mu0=mu0, sigma0=sigma0, threshold=collapse_threshold(mu0, sigma0, c=c, rho=rho))


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
    )
