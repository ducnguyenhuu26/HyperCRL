from __future__ import annotations

from dataclasses import dataclass

import torch

from ..types import ContextPosterior, ContextPrototype


@dataclass(frozen=True)
class ConsolidationResult:
    event: str
    prototype_id: int | None
    evicted_prototype_id: int | None

    def __iter__(self):
        yield self.event
        yield self.prototype_id

    def __getitem__(self, index: int):
        return (self.event, self.prototype_id)[index]


class ContextMemory:
    """Bounded frozen retrieval prototypes; repeated retrieval is not fusion."""

    def __init__(self, max_prototypes: int, merge_mahalanobis: float, fusion_weight: float):
        self.max_prototypes = int(max_prototypes)
        self.merge_mahalanobis = float(merge_mahalanobis)
        self.fusion_weight = float(fusion_weight)
        if self.max_prototypes <= 0 or self.merge_mahalanobis < 0.0 or not 0.0 < self.fusion_weight <= 1.0:
            raise ValueError("invalid context memory configuration")
        self.prototypes: list[ContextPrototype] = []
        self._next_id = 0

    def _distance(self, posterior: ContextPosterior, prototype: ContextPrototype) -> torch.Tensor:
        prototype_mean = prototype.mean.to(device=posterior.mean.device, dtype=posterior.mean.dtype)
        prototype_covariance = prototype.covariance.to(device=posterior.mean.device, dtype=posterior.mean.dtype)
        covariance = 0.5 * (posterior.covariance + posterior.covariance.T) + 0.5 * (prototype_covariance + prototype_covariance.T)
        covariance = covariance + 1e-6 * torch.eye(posterior.mean.numel(), device=posterior.mean.device, dtype=posterior.mean.dtype)
        delta = posterior.mean - prototype_mean
        return delta @ torch.linalg.solve(covariance, delta)

    def nearest(self, posterior: ContextPosterior) -> tuple[ContextPrototype | None, float]:
        if not self.prototypes:
            return None, float("inf")
        distances = [(prototype, float(self._distance(posterior, prototype))) for prototype in self.prototypes]
        return min(distances, key=lambda item: item[1])

    def consolidate(self, posterior: ContextPosterior, step: int) -> ConsolidationResult:
        nearest, distance = self.nearest(posterior)
        if nearest is not None and distance <= self.merge_mahalanobis**2:
            self.mark_reused(nearest.prototype_id, step)
            return ConsolidationResult("CONTEXT_PROTOTYPE_TOUCHED", nearest.prototype_id, None)
        prototype = ContextPrototype(self._next_id, posterior.mean.detach().clone(), posterior.covariance.detach().clone(), 1, int(step), int(step), 1, 0)
        self._next_id += 1
        evicted_id = None
        if len(self.prototypes) >= self.max_prototypes:
            victim = min(self.prototypes, key=lambda item: (item.reuse_count, item.last_active_step, item.creation_step))
            self.prototypes.remove(victim)
            evicted_id = victim.prototype_id
        self.prototypes.append(prototype)
        return ConsolidationResult("CONTEXT_PROTOTYPE_CREATED", prototype.prototype_id, evicted_id)

    def touch(self, prototype_id: int, step: int) -> None:
        for prototype in self.prototypes:
            if prototype.prototype_id == prototype_id:
                prototype.last_active_step = int(step)
                return

    def mark_reused(self, prototype_id: int, step: int) -> None:
        for prototype in self.prototypes:
            if prototype.prototype_id == prototype_id:
                prototype.last_active_step = int(step)
                prototype.reuse_count += 1
                return

    def state_dict(self) -> dict:
        return {"next_id": self._next_id, "prototypes": [prototype.__dict__ for prototype in self.prototypes]}

    def load_state_dict(self, state: dict) -> None:
        self._next_id = int(state["next_id"])
        self.prototypes = [ContextPrototype(**item) for item in state["prototypes"]]
        ids = [prototype.prototype_id for prototype in self.prototypes]
        if len(ids) != len(set(ids)) or len(self.prototypes) > self.max_prototypes:
            raise ValueError("invalid or over-capacity context memory checkpoint")
        for prototype in self.prototypes:
            covariance = 0.5 * (prototype.covariance + prototype.covariance.T)
            if not torch.isfinite(prototype.mean).all() or not torch.isfinite(covariance).all() or torch.linalg.eigvalsh(covariance).min() < -1e-6:
                raise ValueError("invalid context prototype covariance")
            prototype.covariance = covariance
