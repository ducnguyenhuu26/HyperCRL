from __future__ import annotations

import torch

from ..types import NoveltyState


def explained_residual(target_delta: torch.Tensor, base_delta: torch.Tensor, basis: torch.Tensor, context: torch.Tensor, sigma: float) -> torch.Tensor:
    prediction = base_delta + torch.einsum("bdr,br->bd", basis, context)
    residual = target_delta - prediction
    return residual.square().mean() / max(float(sigma**2), 1e-12)


def orthogonal_residual(atom_outputs: torch.Tensor, target: torch.Tensor, ridge: float = 1e-4) -> torch.Tensor:
    """Project target away from the existing flattened atom-output span."""

    if target.ndim != 1 or atom_outputs.ndim != 2:
        raise ValueError("atom_outputs and target must be [N,R] and [N]")
    if atom_outputs.shape[1] == 0:
        return target.clone()
    gram = atom_outputs.T @ atom_outputs + ridge * torch.eye(atom_outputs.shape[1], device=atom_outputs.device, dtype=atom_outputs.dtype)
    coefficients = torch.linalg.solve(gram, atom_outputs.T @ target)
    return target - atom_outputs @ coefficients


class ResidualNoveltyMonitor:
    def __init__(self, residual_threshold: float, new_threshold: float, persistence_steps: int, cooldown_steps: int):
        self.residual_threshold = float(residual_threshold)
        self.new_threshold = float(new_threshold)
        self.persistence_steps = int(persistence_steps)
        self.cooldown_steps = int(cooldown_steps)
        self.consecutive_trigger_count = 0
        self.cooldown_until = -1
        self.expansion_count = 0
        self.last_state = NoveltyState(0.0, 0.0, 0, False)

    def update(self, standardized_residual: float, new_probability: float, step: int, *, allow_trigger: bool = True) -> NoveltyState:
        if not allow_trigger:
            self.consecutive_trigger_count = 0
            self.last_state = NoveltyState(float(standardized_residual), float(new_probability), 0, False)
            return self.last_state
        high = standardized_residual >= self.residual_threshold and new_probability >= self.new_threshold and step >= self.cooldown_until
        self.consecutive_trigger_count = self.consecutive_trigger_count + 1 if high else 0
        should_expand = self.consecutive_trigger_count >= self.persistence_steps and step >= self.cooldown_until
        self.last_state = NoveltyState(float(standardized_residual), float(new_probability), self.consecutive_trigger_count, should_expand)
        return self.last_state

    def mark_expanded(self, step: int) -> None:
        self.expansion_count += 1
        self.consecutive_trigger_count = 0
        self.cooldown_until = int(step) + self.cooldown_steps
        self.last_state = NoveltyState(self.last_state.standardized_residual, self.last_state.new_hypothesis_probability, 0, False)

    def state_dict(self) -> dict:
        return {"consecutive_trigger_count": self.consecutive_trigger_count, "cooldown_until": self.cooldown_until, "expansion_count": self.expansion_count, "last_state": self.last_state.__dict__}

    def load_state_dict(self, state: dict) -> None:
        self.consecutive_trigger_count = int(state["consecutive_trigger_count"])
        self.cooldown_until = int(state["cooldown_until"])
        self.expansion_count = int(state["expansion_count"])
        self.last_state = NoveltyState(**state["last_state"])
