from collections import deque

import numpy as np


class AnchorMemory:
    """Bounded reservoir of learner-observed anchors per prototype."""

    def __init__(self, capacity: int, seed: int | None = None):
        if capacity <= 0:
            raise ValueError("anchor capacity must be positive")
        self.capacity = int(capacity)
        self.rng = np.random.default_rng(seed)
        self.storage: dict[int, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
        self.counts: dict[int, int] = {}

    def add(self, prototype_id: int, obs: np.ndarray, action: np.ndarray, context: np.ndarray) -> None:
        entries = self.storage.setdefault(prototype_id, [])
        count = self.counts.get(prototype_id, 0) + 1
        self.counts[prototype_id] = count
        item = (np.asarray(obs, dtype=np.float32).copy(), np.asarray(action, dtype=np.float32).copy(), np.asarray(context, dtype=np.float32).copy())
        if len(entries) < self.capacity:
            entries.append(item)
        else:
            index = int(self.rng.integers(count))
            if index < self.capacity:
                entries[index] = item

    def remove(self, prototype_id: int) -> None:
        self.storage.pop(prototype_id, None)
        self.counts.pop(prototype_id, None)

    def state_dict(self) -> dict:
        return {"capacity": self.capacity, "storage": self.storage, "counts": self.counts, "rng_state": self.rng.bit_generator.state}

    def load_state_dict(self, state: dict) -> None:
        capacity = int(state["capacity"])
        if capacity <= 0:
            raise ValueError("anchor capacity must be positive")
        self.capacity = capacity
        self.storage = state["storage"]
        self.counts = state["counts"]
        self.rng = np.random.default_rng()
        self.rng.bit_generator.state = state["rng_state"]
