import torch
from torch import nn

from .atoms import VariationAtomBank
from .backbone import SharedDynamicsBackbone


class FactorizedDynamicsAtlas(nn.Module):
    """FDA: base delta plus a context-weighted low-rank atom correction."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int = 256, rank: int = 2):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.backbone = SharedDynamicsBackbone(obs_dim, action_dim, hidden_size)
        self.atom_bank = VariationAtomBank(obs_dim, action_dim, hidden_size, rank)

    @property
    def rank(self) -> int:
        return self.atom_bank.rank

    def basis_outputs(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.atom_bank(obs, action)

    def predict_delta(self, obs: torch.Tensor, action: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if context.ndim != 2 or context.shape[0] != obs.shape[0] or context.shape[1] != self.rank:
            raise ValueError("context must have shape [batch, atlas_rank]")
        base = self.backbone(obs, action)
        basis = self.basis_outputs(obs, action)
        return base + torch.einsum("bdr,br->bd", basis, context)

    def predict_next(self, obs: torch.Tensor, action: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return obs + self.predict_delta(obs, action, context)

    def append_atom(self) -> nn.Module:
        return self.atom_bank.append_atom()
