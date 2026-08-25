"""A small task-agnostic FIFO replay buffer."""

from collections.abc import Iterator

import numpy as np
import torch

from .types import Transition, TransitionBatch


class ReplayBuffer:
    """Fixed-capacity ring buffer with uniform sampling."""

    def __init__(self, capacity: int, seed: int | None = None) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self._storage: list[Transition | None] = [None] * self.capacity
        self._size = 0
        self._next_index = 0
        self._rng = np.random.default_rng(seed)

    def add(self, transition: Transition) -> None:
        self._storage[self._next_index] = Transition(
            obs=np.asarray(transition.obs, dtype=np.float32).copy(),
            action=np.asarray(transition.action, dtype=np.float32).copy(),
            next_obs=np.asarray(transition.next_obs, dtype=np.float32).copy(),
            reward=float(transition.reward),
            terminated=bool(transition.terminated),
            truncated=bool(transition.truncated),
        )
        self._next_index = (self._next_index + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> TransitionBatch:
        if not 0 < batch_size <= len(self):
            raise ValueError("batch_size must be in [1, len(buffer)]")
        indices = self._rng.choice(len(self), size=batch_size, replace=False)
        transitions = [self._storage[int(index)] for index in indices]
        valid = [transition for transition in transitions if transition is not None]
        return TransitionBatch(
            obs=torch.as_tensor(np.stack([t.obs for t in valid]), dtype=torch.float32),
            action=torch.as_tensor(np.stack([t.action for t in valid]), dtype=torch.float32),
            next_obs=torch.as_tensor(np.stack([t.next_obs for t in valid]), dtype=torch.float32),
        )

    def __len__(self) -> int:
        return self._size

    def __iter__(self) -> Iterator[Transition]:
        for transition in self._storage:
            if transition is not None:
                yield transition
