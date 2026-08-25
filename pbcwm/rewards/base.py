"""Reward function protocol used by planners."""

from typing import Protocol

import torch


class RewardFunction(Protocol):
    def __call__(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        next_obs: torch.Tensor,
    ) -> torch.Tensor:
        """Return one batched reward per imagined transition."""
