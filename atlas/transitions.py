from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Iterable, Tuple


@dataclass(frozen=True)
class Transition:
    """A minimal replay transition with bootstrap-multiplier semantics."""

    observation: Any
    action: Any
    reward: float
    discount: float
    next_observation: Any
    extras: Dict[str, Any] = field(default_factory=dict)

    def with_updates(self, **changes: Any) -> "Transition":
        return replace(self, **changes)


def clone_extras(extras: Dict[str, Any]) -> Dict[str, Any]:
    return deepcopy(extras)


def nested_get(mapping: Dict[str, Any], path: Iterable[str], default: Any = None) -> Any:
    current: Any = mapping
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def nested_set(mapping: Dict[str, Any], path: Tuple[str, ...], value: Any) -> None:
    current = mapping
    for key in path[:-1]:
        next_value = current.get(key)
        if not isinstance(next_value, dict):
            next_value = {}
            current[key] = next_value
        current = next_value
    current[path[-1]] = value
