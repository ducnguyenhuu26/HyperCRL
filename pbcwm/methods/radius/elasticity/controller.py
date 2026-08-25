from __future__ import annotations

import math

import torch

from .predictive_fisher import LowRankPredictiveFisher


class PredictiveElasticityController:
    """Trust-region gradient transform backed by a low-rank predictive Fisher."""

    def __init__(self, parameter_dim: int, config):
        self.enabled = bool(config.enabled)
        self.mode = str(config.mode)
        if self.mode != "trust_region":
            raise ValueError("RADIUS v1 PEC mode must be trust_region")
        self.forgetting_budget = float(config.forgetting_budget)
        self.fisher = LowRankPredictiveFisher(parameter_dim, float(config.fisher_damping), int(config.fisher_sketch_rank))
        self.last_scale = 1.0

    @property
    def rank(self) -> int:
        return self.fisher.rank

    def transform_gradient(self, gradient: torch.Tensor) -> torch.Tensor:
        if not self.enabled or gradient.numel() == 0:
            return gradient
        natural = self.fisher.solve(gradient)
        curvature = max(float(gradient @ natural), 1e-12)
        scale = math.sqrt(2.0 * self.forgetting_budget / curvature)
        self.last_scale = scale
        # Optimizers subtract gradients, so expose the positive transformed
        # gradient corresponding to the bounded descent direction.
        return scale * natural

    def diagnostics(self, gradient: torch.Tensor | None = None) -> dict[str, float]:
        elasticity = None if gradient is None else self.fisher.elasticity(gradient)
        return {
            "radius/pec_continual_elasticity": 0.0 if elasticity is None else elasticity,
            "radius/pec_fisher_rank": float(self.rank),
            "radius/pec_update_scale": float(self.last_scale),
        }

    def refresh_from_sketch(self, sketch: torch.Tensor) -> None:
        self.fisher.set_sketch(sketch)

    def state_dict(self) -> dict:
        return {"enabled": self.enabled, "mode": self.mode, "forgetting_budget": self.forgetting_budget, "last_scale": self.last_scale, "fisher": self.fisher.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        self.enabled = bool(state["enabled"])
        self.mode = str(state["mode"])
        self.forgetting_budget = float(state["forgetting_budget"])
        self.last_scale = float(state["last_scale"])
        self.fisher.load_state_dict(state["fisher"])
