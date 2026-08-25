"""Plain MLP for predicting continuous state differences."""

from collections.abc import Sequence

import torch
from torch import nn


class MLPDynamicsModel(nn.Module):
    """Predict ``next_obs - obs`` from the current observation and action."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
    ) -> None:
        super().__init__()
        if obs_dim <= 0 or action_dim <= 0:
            raise ValueError("obs_dim and action_dim must be positive")
        layers: list[nn.Module] = []
        input_dim = obs_dim + action_dim
        for hidden_dim in hidden_dims:
            if hidden_dim <= 0:
                raise ValueError("hidden dimensions must be positive")
            layers.extend([nn.Linear(input_dim, hidden_dim), nn.ReLU()])
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, obs_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat((obs, action), dim=-1))
