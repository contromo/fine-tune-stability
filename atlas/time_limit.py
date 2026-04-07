from __future__ import annotations

from typing import Dict, Iterable, Tuple

from .transitions import Transition, clone_extras, nested_get, nested_set

TIMEOUT_PATHS: Tuple[Tuple[str, ...], ...] = (
    ("state_extras", "time_out"),
    ("time_out",),
)


def extract_timeout_flag(extras: Dict[str, object], paths: Iterable[Tuple[str, ...]] = TIMEOUT_PATHS) -> bool:
    for path in paths:
        value = nested_get(extras, path)
        if value is None:
            continue
        try:
            return float(value) == 1.0
        except (TypeError, ValueError):
            continue
    return False


def apply_timeout_bootstrap(transition: Transition, gamma: float) -> Transition:
    """Converts time-limit truncations into bootstrapable transitions."""

    if not extract_timeout_flag(transition.extras):
        return transition

    extras = clone_extras(transition.extras)
    nested_set(extras, ("atlas", "timeout_bootstrapped"), True)
    return transition.with_updates(discount=gamma, extras=extras)
