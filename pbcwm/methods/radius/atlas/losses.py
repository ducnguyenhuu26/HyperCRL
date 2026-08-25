import torch

from .model import FactorizedDynamicsAtlas


def atom_orthogonality_loss(basis: torch.Tensor) -> torch.Tensor:
    """Control atom decorrelation and RMS scale in normalized delta space."""
    if basis.shape[-1] <= 1:
        return basis.new_zeros(())
    flattened = basis.reshape(-1, basis.shape[-1])
    gram = flattened.T @ flattened / max(1, flattened.shape[0])
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
    context_energy = context.square().mean()
    orth_penalty = atom_orthogonality_loss(atlas.basis_outputs(obs, action))
    # Replay contexts are historical snapshots and are detached by design;
    # adding context_l2 here would create a scalar term with no atlas gradient.
    total = dynamics + atom_orthogonality * orth_penalty
    return total, {"loss": float(total.detach()), "dynamics_loss": float(dynamics.detach()), "mean_context_energy": float(context_energy.detach()), "atom_gram_loss": float(orth_penalty.detach())}
