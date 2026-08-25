import torch

from .model import FactorizedDynamicsAtlas


def atom_orthogonality_loss(basis: torch.Tensor) -> torch.Tensor:
    if basis.shape[-1] <= 1:
        return basis.new_zeros(())
    flattened = basis.permute(0, 2, 1).reshape(-1, basis.shape[-1])
    normalized = flattened / flattened.norm(dim=0, keepdim=True).clamp_min(1e-8)
    gram = normalized.T @ normalized / max(1, normalized.shape[0])
    identity = torch.eye(basis.shape[-1], device=basis.device, dtype=basis.dtype)
    return (gram - identity).square().mean()


def atlas_loss(
    atlas: FactorizedDynamicsAtlas,
    obs: torch.Tensor,
    action: torch.Tensor,
    target_delta: torch.Tensor,
    context: torch.Tensor,
    *,
    context_l2: float = 1e-4,
    atom_orthogonality: float = 1e-3,
) -> tuple[torch.Tensor, dict[str, float]]:
    prediction = atlas.predict_delta(obs, action, context)
    dynamics = (prediction - target_delta).square().mean()
    context_penalty = context.square().mean()
    orth_penalty = atom_orthogonality_loss(atlas.basis_outputs(obs, action))
    total = dynamics + context_l2 * context_penalty + atom_orthogonality * orth_penalty
    return total, {"loss": float(total.detach()), "dynamics_loss": float(dynamics.detach()), "context_l2": float(context_penalty.detach()), "atom_orthogonality": float(orth_penalty.detach())}
