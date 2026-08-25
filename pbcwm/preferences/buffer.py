"""Lifetime preference replay, intentionally separate from dynamics memory."""

from collections.abc import Iterator

import numpy as np

from .types import PreferenceExample


class PreferenceBuffer:
    """Uniform preference replay with optional FIFO capacity."""

    def __init__(self, capacity: int | None = None, seed: int | None = None) -> None:
        if capacity is not None and capacity <= 0:
            raise ValueError("capacity must be positive when provided")
        self.capacity = capacity
        self._storage: list[PreferenceExample] = []
        self._rng = np.random.default_rng(seed)

    def add(self, example: PreferenceExample) -> None:
        stored = PreferenceExample(
            traj_a=example.traj_a.detached(),
            traj_b=example.traj_b.detached(),
            label=example.label,
        )
        if self.capacity is not None and len(self._storage) >= self.capacity:
            self._storage.pop(0)
        self._storage.append(stored)

    def sample(self, batch_size: int) -> list[PreferenceExample]:
        if not 0 < batch_size <= len(self):
            raise ValueError("batch_size must be in [1, len(buffer)]")
        indices = self._rng.choice(len(self), size=batch_size, replace=False)
        return [self._storage[int(index)] for index in indices]

    def __len__(self) -> int:
        return len(self._storage)

    def __iter__(self) -> Iterator[PreferenceExample]:
        return iter(self._storage)

    def state_dict(self) -> dict:
        return {"capacity": self.capacity, "storage": list(self._storage), "rng_state": self._rng.bit_generator.state}

    def load_state_dict(self, state: dict) -> None:
        self.capacity = state.get("capacity")
        self._storage = list(state["storage"])
        self._rng = np.random.default_rng()
        self._rng.bit_generator.state = state["rng_state"]
