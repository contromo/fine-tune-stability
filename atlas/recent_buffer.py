from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Generic, Iterable, List, Optional, Sequence, TypeVar

T = TypeVar("T")


@dataclass
class RecentTransitionBuffer(Generic[T]):
    """Fixed-capacity cyclic buffer for recent diagnostics."""

    capacity: int
    _items: List[Optional[T]] = field(init=False)
    _cursor: int = field(init=False, default=0)
    _size: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        if self.capacity < 1:
            raise ValueError("capacity must be at least 1")
        self._items = [None] * self.capacity

    def append(self, item: T) -> None:
        self._items[self._cursor] = item
        self._cursor = (self._cursor + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def extend(self, items: Iterable[T]) -> None:
        for item in items:
            self.append(item)

    def snapshot(self) -> List[T]:
        if self._size == 0:
            return []
        if self._size < self.capacity:
            return [item for item in self._items[: self._size] if item is not None]

        ordered = self._items[self._cursor :] + self._items[: self._cursor]
        return [item for item in ordered if item is not None]

    def sample(self, batch_size: int, rng: Optional[random.Random] = None) -> List[T]:
        items = self.snapshot()
        if not items:
            return []
        if batch_size >= len(items):
            return items
        generator = rng or random
        indices = generator.sample(range(len(items)), batch_size)
        return [items[index] for index in indices]

    def __len__(self) -> int:
        return self._size

    def is_full(self) -> bool:
        return self._size == self.capacity
