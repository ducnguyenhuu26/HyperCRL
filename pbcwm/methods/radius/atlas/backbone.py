import torch
from torch import nn


class SharedDynamicsBackbone(nn.Module):
    """Shared reward-free base delta predictor."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, obs_dim),
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat((obs, action), dim=-1))
