from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Hashable, List

from .time_limit import apply_timeout_bootstrap, extract_timeout_flag
from .transitions import Transition


@dataclass
class NStepTransitionAggregator:
    """Aggregates one-step transitions into naive off-policy n-step replay items.

    The input transition discount is assumed to be the one-step bootstrap
    multiplier used by the critic target:

    - continuing step: gamma
    - true terminal: 0.0
    - time-limit truncation: gamma
    """

    n_step: int
    gamma: float
    _buffer: Deque[Transition] = field(default_factory=deque)

    def __post_init__(self) -> None:
        if self.n_step < 1:
            raise ValueError("n_step must be at least 1")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")

    def push(self, transition: Transition) -> List[Transition]:
        processed = apply_timeout_bootstrap(transition, self.gamma)
        self._buffer.append(processed)
        emitted: List[Transition] = []

        if len(self._buffer) >= self.n_step:
            emitted.append(self._aggregate_window(self.n_step))
            self._buffer.popleft()

        if self._ends_episode(processed):
            emitted.extend(self.flush())

        return emitted

    def flush(self) -> List[Transition]:
        emitted: List[Transition] = []
        while self._buffer:
            emitted.append(self._aggregate_window(len(self._buffer)))
            self._buffer.popleft()
        return emitted

    def __len__(self) -> int:
        return len(self._buffer)

    def _aggregate_window(self, window_length: int) -> Transition:
        window = list(self._buffer)[:window_length]
        reward = 0.0
        discount = 1.0
        timeout_seen = False
        timeout_bootstrapped = False

        for index, step in enumerate(window):
            reward += (self.gamma ** index) * float(step.reward)
            discount *= float(step.discount)
            timeout_seen = timeout_seen or extract_timeout_flag(step.extras)
            timeout_bootstrapped = timeout_bootstrapped or bool(
                step.extras.get("atlas", {}).get("timeout_bootstrapped", False)
            )

        start = window[0]
        end = window[-1]
        extras = deepcopy(end.extras)
        atlas_meta = extras.setdefault("atlas", {})
        atlas_meta["n_step"] = window_length
        atlas_meta["window_timeout"] = timeout_seen
        if timeout_bootstrapped:
            atlas_meta["timeout_bootstrapped"] = True

        return Transition(
            observation=start.observation,
            action=start.action,
            reward=reward,
            discount=discount,
            next_observation=end.next_observation,
            extras=extras,
        )

    @staticmethod
    def _is_true_terminal(transition: Transition) -> bool:
        return float(transition.discount) == 0.0 and not extract_timeout_flag(transition.extras)

    @classmethod
    def _ends_episode(cls, transition: Transition) -> bool:
        return cls._is_true_terminal(transition) or extract_timeout_flag(transition.extras)


@dataclass
class MultiStreamNStepAggregator:
    """Maintains one n-step queue per stream or environment id."""

    n_step: int
    gamma: float
    _streams: Dict[Hashable, NStepTransitionAggregator] = field(default_factory=dict)

    def push(self, stream_id: Hashable, transition: Transition) -> List[Transition]:
        return self._get_stream(stream_id).push(transition)

    def flush_stream(self, stream_id: Hashable) -> List[Transition]:
        aggregator = self._streams.get(stream_id)
        if aggregator is None:
            return []
        return aggregator.flush()

    def flush_all(self) -> Dict[Hashable, List[Transition]]:
        flushed: Dict[Hashable, List[Transition]] = {}
        for stream_id, aggregator in self._streams.items():
            items = aggregator.flush()
            if items:
                flushed[stream_id] = items
        return flushed

    def _get_stream(self, stream_id: Hashable) -> NStepTransitionAggregator:
        aggregator = self._streams.get(stream_id)
        if aggregator is None:
            aggregator = NStepTransitionAggregator(n_step=self.n_step, gamma=self.gamma)
            self._streams[stream_id] = aggregator
        return aggregator
