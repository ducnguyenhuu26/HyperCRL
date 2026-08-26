from __future__ import annotations

import numpy as np
import torch

from pbcwm.core.device import move_batch

from ..types import RadiusReplayItem


class RadiusReplayBuffer:
    """Replay with historical context snapshots and zero-padding after expansion."""

    def __init__(self, capacity: int, seed: int | None = None):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self.storage: list[RadiusReplayItem] = []
        self.next_index = 0
        self.rng = np.random.default_rng(seed)

    def add(self, item: RadiusReplayItem) -> None:
        copied = RadiusReplayItem(
            item.obs.detach().cpu().float().clone(),
            item.action.detach().cpu().float().clone(),
            item.next_obs.detach().cpu().float().clone(),
            item.context_mean.detach().cpu().float().clone(),
            item.prototype_id,
        )
        if len(self.storage) < self.capacity:
            self.storage.append(copied)
        else:
            self.storage[self.next_index] = copied
        self.next_index = (self.next_index + 1) % self.capacity

    def sample(self, batch_size: int, rank: int, device: torch.device, prototype_means: dict[int, torch.Tensor] | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not 0 < batch_size <= len(self.storage):
            raise ValueError("batch_size must be in [1, len(replay)]")
        target_device = torch.device(device)
        indices = self.rng.choice(len(self.storage), size=batch_size, replace=False)
        entries = [self.storage[int(index)] for index in indices]
        context_values = []
        for entry in entries:
            context = prototype_means.get(entry.prototype_id, entry.context_mean) if prototype_means is not None and entry.prototype_id is not None else entry.context_mean
            # Replay entries are intentionally stored on CPU, while current
            # prototype means live on the learner device. Normalize the
            # selected context before padding so a mixed batch can be stacked.
            context = context.detach().to(device=target_device, dtype=torch.float32)
            context_values.append(torch.nn.functional.pad(context, (0, max(0, rank - context.numel())))[:rank])
        return (
            move_batch(torch.stack([entry.obs for entry in entries]), device),
            move_batch(torch.stack([entry.action for entry in entries]), device),
            move_batch(torch.stack([entry.next_obs for entry in entries]), device),
            move_batch(torch.stack(context_values), device),
        )

    def __len__(self) -> int:
        return len(self.storage)

    def state_dict(self) -> dict:
        return {"capacity": self.capacity, "storage": self.storage, "next_index": self.next_index, "rng_state": self.rng.bit_generator.state}

    def load_state_dict(self, state: dict) -> None:
        capacity = int(state["capacity"])
        storage = list(state["storage"])
        next_index = int(state["next_index"])
        if capacity <= 0 or len(storage) > capacity or not 0 <= next_index < capacity:
            raise ValueError("invalid replay checkpoint")
        self.capacity = capacity
        self.storage = storage
        self.next_index = next_index
        self.rng = np.random.default_rng()
        self.rng.bit_generator.state = state["rng_state"]
