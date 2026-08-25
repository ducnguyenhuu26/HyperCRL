import torch
from torch import nn


class VariationAtomBank(nn.Module):
    """Shared feature trunk with one output head per variation atom."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int = 256, rank: int = 2):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.hidden_size = int(hidden_size)
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
        )
        self.heads = nn.ModuleList([nn.Linear(hidden_size, obs_dim) for _ in range(rank)])

    @property
    def rank(self) -> int:
        return len(self.heads)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        features = self.trunk(torch.cat((obs, action), dim=-1))
        return torch.stack([head(features) for head in self.heads], dim=-1)

    def append_atom(self) -> nn.Module:
        head = nn.Linear(self.hidden_size, self.obs_dim).to(next(self.parameters()).device)
        self.heads.append(head)
        return head
