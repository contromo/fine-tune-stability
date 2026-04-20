from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Iterable, List, Sequence

DEFAULT_TRIGGER_THRESHOLD = math.log(3.0)
DEFAULT_TRIGGER_HOLD_EVALS = 2


def collapse_threshold(mu0: float, sigma0: float, c: float = 2.0, rho: float = 0.2) -> float:
    sigma_rule = mu0 - (c * sigma0)
    floor_rule = mu0 - (rho * abs(mu0))
    return min(sigma_rule, floor_rule)


def td_error(reward: float, discount: float, bootstrap_value: float, q_value: float) -> float:
    return reward + (discount * bootstrap_value) - q_value


def _quantile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("values must not be empty")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]

    index = (len(ordered) - 1) * quantile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


@dataclass(frozen=True)
class DiagnosticSnapshot:
    variance: float
    q95_abs_td: float
    score: float
    sample_size: int


def summarize_td_errors(
    errors: Sequence[float],
    warmup_variance: float,
    epsilon: float = 1e-8,
) -> DiagnosticSnapshot:
    if not errors:
        raise ValueError("errors must not be empty")
    variance = statistics.pvariance(errors)
    q95 = _quantile([abs(error) for error in errors], 0.95)
    score = math.log(variance + epsilon) - math.log(warmup_variance + epsilon)
    return DiagnosticSnapshot(
        variance=variance,
        q95_abs_td=q95,
        score=score,
        sample_size=len(errors),
    )


@dataclass
class InstabilityTrigger:
    threshold: float = DEFAULT_TRIGGER_THRESHOLD
    hold_evals: int = DEFAULT_TRIGGER_HOLD_EVALS
    _consecutive_hits: int = field(init=False, default=0)
    _ever_triggered: bool = field(init=False, default=False)

    def update(self, score: float) -> bool:
        if score > self.threshold:
            self._consecutive_hits += 1
        else:
            self._consecutive_hits = 0

        if self._consecutive_hits >= self.hold_evals:
            self._ever_triggered = True
            return True
        return False

    @property
    def ever_triggered(self) -> bool:
        return self._ever_triggered


def collapse_horizon_labels(collapse_flags: Sequence[bool], horizon: int) -> List[int]:
    labels: List[int] = []
    for index in range(len(collapse_flags)):
        upcoming = collapse_flags[index + 1 : index + 1 + horizon]
        labels.append(1 if any(upcoming) else 0)
    return labels


def roc_auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have equal length")
    positives = [(score, index) for index, (score, label) in enumerate(zip(scores, labels)) if label == 1]
    negatives = [(score, index) for index, (score, label) in enumerate(zip(scores, labels)) if label == 0]
    if not positives or not negatives:
        raise ValueError("roc_auc requires at least one positive and one negative label")

    ranked = sorted(enumerate(scores), key=lambda item: item[1])
    ranks = [0.0] * len(scores)
    cursor = 0

    while cursor < len(ranked):
        next_cursor = cursor + 1
        while next_cursor < len(ranked) and ranked[next_cursor][1] == ranked[cursor][1]:
            next_cursor += 1
        average_rank = (cursor + 1 + next_cursor) / 2.0
        for offset in range(cursor, next_cursor):
            original_index = ranked[offset][0]
            ranks[original_index] = average_rank
        cursor = next_cursor

    positive_rank_sum = sum(ranks[index] for index, label in enumerate(labels) if label == 1)
    positive_count = sum(labels)
    negative_count = len(labels) - positive_count
    return (positive_rank_sum - (positive_count * (positive_count + 1) / 2.0)) / (positive_count * negative_count)


def gaussian_kl_diagonal(
    mu_a: Sequence[float],
    log_std_a: Sequence[float],
    mu_b: Sequence[float],
    log_std_b: Sequence[float],
) -> float:
    """Closed-form KL(N(mu_a, diag(exp(log_std_a)^2)) || N(mu_b, diag(exp(log_std_b)^2))).

    Parameterized on log-std so callers pass the logarithm directly, avoiding a
    log() of a near-zero scale. Returns the sum over dimensions (not the mean).
    """
    if not (len(mu_a) == len(log_std_a) == len(mu_b) == len(log_std_b)):
        raise ValueError("all inputs must have equal length")
    if not mu_a:
        raise ValueError("inputs must be non-empty")
    total = 0.0
    for m_a, ls_a, m_b, ls_b in zip(mu_a, log_std_a, mu_b, log_std_b):
        var_a = math.exp(2.0 * ls_a)
        var_b = math.exp(2.0 * ls_b)
        total += (ls_b - ls_a) + (var_a + (m_a - m_b) ** 2) / (2.0 * var_b) - 0.5
    return total


def pearson_correlation(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys):
        raise ValueError("xs and ys must have equal length")
    if len(xs) < 2:
        raise ValueError("at least two points are required")
    mean_x = statistics.mean(xs)
    mean_y = statistics.mean(ys)
    centered_products = [(x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)]
    sum_sq_x = sum((x - mean_x) ** 2 for x in xs)
    sum_sq_y = sum((y - mean_y) ** 2 for y in ys)
    if sum_sq_x == 0.0 or sum_sq_y == 0.0:
        return 0.0
    return sum(centered_products) / math.sqrt(sum_sq_x * sum_sq_y)
