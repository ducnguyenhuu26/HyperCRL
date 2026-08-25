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
        self.max_step_norm = float(getattr(config, "max_step_norm", 1.0))
        self.last_scale = 1.0
        self.last_continual_elasticity = 0.0
        self.last_step_norm = 0.0
        self.last_predicted_forgetting_cost = 0.0
        self.last_step_capped = False

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
        self.last_continual_elasticity = float(gradient @ natural)
        # Optimizers subtract gradients, so expose the positive transformed
        # gradient corresponding to the bounded descent direction.
        return scale * natural

    def direct_step(self, gradient: torch.Tensor) -> torch.Tensor:
        """Return a direct local trust-region parameter delta."""

        if not self.enabled or gradient.numel() == 0:
            self.last_step_norm = 0.0
            self.last_predicted_forgetting_cost = 0.0
            self.last_step_capped = False
            return torch.zeros_like(gradient)
        if not torch.isfinite(gradient).all():
            raise FloatingPointError("non-finite PEC gradient")
        natural = self.fisher.solve(gradient)
        elasticity = max(float(gradient @ natural), 1e-12)
        scale = math.sqrt(2.0 * self.forgetting_budget / elasticity)
        delta = -scale * natural
        self.last_continual_elasticity = elasticity
        self.last_scale = scale
        self.last_step_capped = False
        norm = float(delta.norm())
        if norm > self.max_step_norm:
            delta = delta * (self.max_step_norm / max(norm, 1e-12))
            self.last_step_capped = True
        self.last_step_norm = float(delta.norm())
        self.last_predicted_forgetting_cost = float(0.5 * self.fisher.quadratic_form(delta))
        if not torch.isfinite(delta).all():
            raise FloatingPointError("non-finite PEC step")
        return delta

    def diagnostics(self, gradient: torch.Tensor | None = None) -> dict[str, float]:
        elasticity = None if gradient is None else self.fisher.elasticity(gradient)
        return {
            "radius/pec_continual_elasticity": self.last_continual_elasticity if elasticity is None else elasticity,
            "radius/pec_fisher_rank": float(self.rank),
            "radius/pec_update_scale": float(self.last_scale),
            "radius/pec_step_scale": float(self.last_scale),
            "radius/pec_step_norm": float(self.last_step_norm),
            "radius/pec_predicted_forgetting_cost": float(self.last_predicted_forgetting_cost),
            "radius/pec_step_capped": float(self.last_step_capped),
        }

    def refresh_from_sketch(self, sketch: torch.Tensor) -> None:
        self.fisher.set_sketch(sketch)

    def state_dict(self) -> dict:
        return {"enabled": self.enabled, "mode": self.mode, "forgetting_budget": self.forgetting_budget, "max_step_norm": self.max_step_norm, "last_scale": self.last_scale, "last_continual_elasticity": self.last_continual_elasticity, "last_step_norm": self.last_step_norm, "last_predicted_forgetting_cost": self.last_predicted_forgetting_cost, "last_step_capped": self.last_step_capped, "fisher": self.fisher.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        self.enabled = bool(state["enabled"])
        self.mode = str(state["mode"])
        self.forgetting_budget = float(state["forgetting_budget"])
        self.max_step_norm = float(state.get("max_step_norm", self.max_step_norm))
        self.last_scale = float(state["last_scale"])
        self.last_continual_elasticity = float(state.get("last_continual_elasticity", 0.0))
        self.last_step_norm = float(state.get("last_step_norm", 0.0))
        self.last_predicted_forgetting_cost = float(state.get("last_predicted_forgetting_cost", 0.0))
        self.last_step_capped = bool(state.get("last_step_capped", False))
        self.fisher.load_state_dict(state["fisher"])
