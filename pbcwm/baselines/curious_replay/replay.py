"""Lifetime FIFO replay with count- and model-loss curiosity priorities."""

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from pbcwm.core.types import Transition


def count_priority(replay_count: int, beta: float, count_weight_c: float) -> float:
    """Compute ``c * beta**v`` for one replay entry."""

    if replay_count < 0 or not 0.0 < beta < 1.0 or count_weight_c < 0:
        raise ValueError("invalid count-priority parameters")
    return float(count_weight_c * beta**int(replay_count))


def loss_priority(model_loss: float, alpha: float, epsilon: float) -> float:
    """Compute ``(L + epsilon)**alpha`` for a non-negative model loss."""

    if model_loss < 0 or alpha < 0 or epsilon <= 0:
        raise ValueError("invalid loss-priority parameters")
    return float((float(model_loss) + epsilon) ** alpha)


def combined_priority(
    replay_count: int,
    model_loss: float,
    beta: float,
    alpha: float,
    epsilon: float,
    count_weight_c: float,
) -> float:
    """Compute the Curious Replay priority ``c*beta**v + (L+eps)**alpha``."""

    priority = count_priority(replay_count, beta, count_weight_c) + loss_priority(
        model_loss, alpha, epsilon
    )
    if not np.isfinite(priority) or priority <= 0:
        raise FloatingPointError("Curious Replay priority is non-finite")
    return float(priority)


@dataclass
class CuriousReplayEntry:
    """One transition and its cached replay statistics."""

    transition: Transition
    replay_count: int = 0
    priority: float = 1.0
    last_model_loss: float = 0.0


class CuriousReplayBuffer:
    """Bounded FIFO storage with weighted sampling and mutable slot statistics."""

    def __init__(
        self,
        capacity: int,
        beta: float = 0.7,
        alpha: float = 0.6,
        epsilon: float = 1e-6,
        count_weight_c: float = 1.0,
        initial_priority: float = 1.0,
        seed: int | None = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if not 0.0 < beta < 1.0:
            raise ValueError("beta must be in (0, 1)")
        if alpha < 0 or epsilon <= 0 or count_weight_c < 0 or initial_priority <= 0:
            raise ValueError("invalid Curious Replay priority configuration")
        self.capacity = int(capacity)
        self.beta = float(beta)
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.count_weight_c = float(count_weight_c)
        self.initial_priority = float(initial_priority)
        self._storage: list[CuriousReplayEntry | None] = [None] * self.capacity
        self._size = 0
        self._next_index = 0
        self._seen = 0
        self._rng = np.random.default_rng(seed)

    def add(self, transition: Transition) -> int:
        """Insert a copied transition and return its stable physical slot."""

        priority = max(self.initial_priority, self.max_priority)
        entry = CuriousReplayEntry(
            transition=self._copy_transition(transition),
            replay_count=0,
            priority=float(priority),
            last_model_loss=0.0,
        )
        index = self._next_index
        self._storage[index] = entry
        self._next_index = (index + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)
        self._seen += 1
        return index

    def sample(self, batch_size: int) -> tuple[np.ndarray, list[CuriousReplayEntry]]:
        """Sample weighted slots, replacing only when the request exceeds storage."""

        if not 0 < batch_size:
            raise ValueError("batch_size must be positive")
        if not self._size:
            raise ValueError("cannot sample an empty Curious Replay buffer")
        probabilities = self._probabilities()
        replace = batch_size > self._size
        indices = self._rng.choice(
            self._size_indices(), size=batch_size, replace=replace, p=probabilities
        )
        entries = [self._storage[int(index)] for index in indices]
        if any(entry is None for entry in entries):
            raise RuntimeError("Curious Replay storage contains an invalid slot")
        return indices.astype(np.int64), [entry for entry in entries if entry is not None]

    def update_priorities(self, indices: np.ndarray, model_losses: np.ndarray) -> None:
        """Update each sampled occurrence with its own post-update loss."""

        indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        model_losses = np.asarray(model_losses, dtype=np.float64).reshape(-1)
        if len(indices) != len(model_losses):
            raise ValueError("indices and model_losses must have equal length")
        for index, loss in zip(indices, model_losses):
            if not 0 <= int(index) < self._size:
                raise IndexError("replay slot is out of range")
            if not np.isfinite(loss) or loss < 0:
                raise FloatingPointError("sampled model loss is non-finite")
            entry = self._storage[int(index)]
            if entry is None:
                raise RuntimeError("cannot update an empty replay slot")
            entry.replay_count += 1
            entry.last_model_loss = float(loss)
            entry.priority = combined_priority(
                replay_count=entry.replay_count,
                model_loss=entry.last_model_loss,
                beta=self.beta,
                alpha=self.alpha,
                epsilon=self.epsilon,
                count_weight_c=self.count_weight_c,
            )

    def get(self, index: int) -> CuriousReplayEntry:
        if not 0 <= int(index) < self._size:
            raise IndexError("replay slot is out of range")
        entry = self._storage[int(index)]
        if entry is None:
            raise RuntimeError("replay slot is empty")
        return entry

    def __len__(self) -> int:
        return self._size

    def __iter__(self) -> Iterator[CuriousReplayEntry]:
        for index in self._size_indices():
            entry = self._storage[int(index)]
            if entry is not None:
                yield entry

    @property
    def max_priority(self) -> float:
        priorities = [entry.priority for entry in self if entry is not None]
        return max(priorities, default=0.0)

    @property
    def min_priority(self) -> float:
        priorities = [entry.priority for entry in self if entry is not None]
        return min(priorities, default=0.0)

    def statistics(self) -> dict[str, float]:
        entries = list(self)
        if not entries:
            return {
                "buffer_size": 0.0,
                "mean_priority": 0.0,
                "max_priority": 0.0,
                "min_priority": 0.0,
                "mean_replay_count": 0.0,
                "median_replay_count": 0.0,
                "mean_cached_model_loss": 0.0,
                "count_priority_mean": 0.0,
                "loss_priority_mean": 0.0,
            }
        count_values = np.asarray([entry.replay_count for entry in entries], dtype=np.float64)
        loss_values = np.asarray([entry.last_model_loss for entry in entries], dtype=np.float64)
        count_terms = np.asarray(
            [count_priority(int(value), self.beta, self.count_weight_c) for value in count_values]
        )
        loss_terms = np.asarray(
            [loss_priority(float(value), self.alpha, self.epsilon) for value in loss_values]
        )
        priorities = np.asarray([entry.priority for entry in entries], dtype=np.float64)
        return {
            "buffer_size": float(len(entries)),
            "mean_priority": float(priorities.mean()),
            "max_priority": float(priorities.max()),
            "min_priority": float(priorities.min()),
            "mean_replay_count": float(count_values.mean()),
            "median_replay_count": float(np.median(count_values)),
            "mean_cached_model_loss": float(loss_values.mean()),
            "count_priority_mean": float(count_terms.mean()),
            "loss_priority_mean": float(loss_terms.mean()),
        }

    def state_dict(self) -> dict:
        return {
            "storage": self._storage,
            "size": self._size,
            "next_index": self._next_index,
            "seen": self._seen,
            "rng_state": self._rng.bit_generator.state,
        }

    def load_state_dict(self, state: dict) -> None:
        storage = list(state["storage"])
        if len(storage) != self.capacity:
            raise ValueError("saved Curious Replay capacity does not match learner")
        self._storage = storage
        self._size = int(state["size"])
        self._next_index = int(state["next_index"])
        self._seen = int(state["seen"])
        self._rng.bit_generator.state = state["rng_state"]

    def _probabilities(self) -> np.ndarray:
        priorities = np.asarray([self.get(index).priority for index in self._size_indices()])
        total = float(priorities.sum())
        if not np.isfinite(total) or total <= 0:
            raise FloatingPointError("Curious Replay priority mass is invalid")
        return priorities / total

    def _size_indices(self) -> np.ndarray:
        if self._size < self.capacity:
            return np.arange(self._size, dtype=np.int64)
        return np.arange(self.capacity, dtype=np.int64)

    @staticmethod
    def _copy_transition(transition: Transition) -> Transition:
        # Curious Replay is reward-free: keep no environment reward in the
        # dynamics buffer, even though the shared Transition type contains it.
        return Transition(
            obs=np.asarray(transition.obs, dtype=np.float32).copy(),
            action=np.asarray(transition.action, dtype=np.float32).copy(),
            next_obs=np.asarray(transition.next_obs, dtype=np.float32).copy(),
            reward=0.0,
            terminated=bool(transition.terminated),
            truncated=bool(transition.truncated),
        )
