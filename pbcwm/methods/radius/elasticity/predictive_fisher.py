from __future__ import annotations

import torch


def woodbury_solve(sketch: torch.Tensor, damping: float, vector: torch.Tensor) -> torch.Tensor:
    """Apply ``(U U^T + damping I)^-1`` without forming a dense Fisher."""

    if damping <= 0.0:
        raise ValueError("damping must be positive")
    if sketch.numel() == 0:
        return vector / damping
    if sketch.ndim != 2 or vector.ndim != 1 or sketch.shape[0] != vector.numel():
        raise ValueError("sketch must be [parameter_dim, sketch_rank] and vector [parameter_dim]")
    rank_matrix = torch.eye(sketch.shape[1], device=sketch.device, dtype=sketch.dtype) + (sketch.T @ sketch) / damping
    correction = sketch @ torch.linalg.solve(rank_matrix, sketch.T @ vector)
    return (vector - correction / damping) / damping


class LowRankPredictiveFisher:
    """Low-rank operator ``F_old ~= U U^T`` used by PEC."""

    def __init__(self, parameter_dim: int, damping: float = 1e-3, max_rank: int = 32):
        self.parameter_dim = int(parameter_dim)
        self.damping = float(damping)
        self.max_rank = int(max_rank)
        if self.parameter_dim <= 0 or self.max_rank < 0 or self.damping <= 0.0:
            raise ValueError("invalid low-rank predictive Fisher configuration")
        self.sketch = torch.empty(self.parameter_dim, 0)

    @property
    def rank(self) -> int:
        return int(self.sketch.shape[1])

    def set_sketch(self, sketch: torch.Tensor) -> None:
        if sketch.ndim != 2 or sketch.shape[0] != self.parameter_dim:
            raise ValueError("sketch shape does not match parameter dimension")
        self.sketch = sketch.detach().clone()[:, : self.max_rank]

    def solve(self, vector: torch.Tensor) -> torch.Tensor:
        return woodbury_solve(self.sketch.to(vector), self.damping, vector)

    def elasticity(self, gradient: torch.Tensor) -> float:
        return float(gradient @ self.solve(gradient))

    def protected_energy(self, direction: torch.Tensor) -> float:
        if self.rank == 0:
            return 0.0
        return float((self.sketch.to(direction).T @ direction).square().sum())

    def quadratic_form(self, vector: torch.Tensor) -> torch.Tensor:
        sketch = self.sketch.to(vector)
        fisher_vector = sketch @ (sketch.T @ vector) if self.rank else torch.zeros_like(vector)
        return vector @ (fisher_vector + self.damping * vector)

    def state_dict(self) -> dict:
        return {"parameter_dim": self.parameter_dim, "damping": self.damping, "max_rank": self.max_rank, "sketch": self.sketch}

    def load_state_dict(self, state: dict) -> None:
        self.parameter_dim = int(state["parameter_dim"])
        self.damping = float(state["damping"])
        self.max_rank = int(state["max_rank"])
        if self.parameter_dim <= 0 or self.max_rank < 0 or self.damping <= 0.0:
            raise ValueError("invalid low-rank predictive Fisher checkpoint")
        self.set_sketch(state["sketch"])
