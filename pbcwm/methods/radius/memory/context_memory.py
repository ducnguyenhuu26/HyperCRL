from __future__ import annotations

import torch

from ..types import ContextPosterior, ContextPrototype


class ContextMemory:
    """Bounded learner-derived prototype memory with precision fusion."""

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
        covariance = posterior.covariance + prototype_covariance + 1e-6 * torch.eye(posterior.mean.numel(), device=posterior.mean.device, dtype=posterior.mean.dtype)
        delta = posterior.mean - prototype_mean
        return delta @ torch.linalg.solve(covariance, delta)

    def nearest(self, posterior: ContextPosterior) -> tuple[ContextPrototype | None, float]:
        if not self.prototypes:
            return None, float("inf")
        distances = [(prototype, float(self._distance(posterior, prototype))) for prototype in self.prototypes]
        return min(distances, key=lambda item: item[1])

    def consolidate(self, posterior: ContextPosterior, step: int) -> tuple[str, int | None]:
        nearest, distance = self.nearest(posterior)
        if nearest is not None and distance <= self.merge_mahalanobis**2:
            self._fuse(nearest, posterior, step)
            return "CONTEXT_PROTOTYPE_MERGED", nearest.prototype_id
        prototype = ContextPrototype(self._next_id, posterior.mean.detach().clone(), posterior.covariance.detach().clone(), 1, int(step), int(step), 1)
        self._next_id += 1
        self.prototypes.append(prototype)
        if len(self.prototypes) > self.max_prototypes:
            self.prototypes.sort(key=lambda item: (item.usage_count, item.last_active_step))
            self.prototypes.pop(0)
        return "CONTEXT_PROTOTYPE_CREATED", prototype.prototype_id

    def touch(self, prototype_id: int, step: int) -> None:
        for prototype in self.prototypes:
            if prototype.prototype_id == prototype_id:
                prototype.last_active_step = int(step)
                prototype.usage_count += 1
                return

    def _fuse(self, prototype: ContextPrototype, posterior: ContextPosterior, step: int) -> None:
        prototype.mean = prototype.mean.to(device=posterior.mean.device, dtype=posterior.mean.dtype)
        prototype.covariance = prototype.covariance.to(device=posterior.mean.device, dtype=posterior.mean.dtype)
        identity = torch.eye(prototype.mean.numel(), device=posterior.mean.device, dtype=posterior.mean.dtype)
        precision_old = torch.linalg.solve(prototype.covariance, identity)
        precision_new = torch.linalg.solve(posterior.covariance, identity)
        combined_precision = precision_old + self.fusion_weight * precision_new
        covariance = torch.linalg.solve(combined_precision, identity)
        mean = covariance @ (precision_old @ prototype.mean + self.fusion_weight * precision_new @ posterior.mean)
        prototype.mean = mean.detach()
        prototype.covariance = (0.5 * (covariance + covariance.T)).detach()
        prototype.num_consolidations += 1
        prototype.last_active_step = int(step)
        prototype.usage_count += 1

    def state_dict(self) -> dict:
        return {"next_id": self._next_id, "prototypes": [prototype.__dict__ for prototype in self.prototypes]}

    def load_state_dict(self, state: dict) -> None:
        self._next_id = int(state["next_id"])
        self.prototypes = [ContextPrototype(**item) for item in state["prototypes"]]
