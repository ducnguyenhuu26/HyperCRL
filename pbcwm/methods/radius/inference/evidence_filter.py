from __future__ import annotations

from collections.abc import Sequence

import torch

from ..memory.context_memory import ContextMemory
from ..types import ContextPosterior
from .gaussian_context import gaussian_context_posterior, log_marginal_evidence, moment_match


class RecurrentEvidenceFilter:
    """Soft active/memory/new Gaussian hypothesis filter."""

    def __init__(self, rank: int, sigma: float, config, memory: ContextMemory, device: torch.device, *, disable_memory: bool = False, hard_routing: bool = False):
        self.rank = int(rank)
        self.sigma = float(sigma)
        self.config = config
        self.memory = memory
        self.device = device
        self.disable_memory = bool(disable_memory)
        self.hard_routing = bool(hard_routing)

    def infer(
        self,
        basis: torch.Tensor,
        residual: torch.Tensor,
        active: ContextPosterior,
    ) -> ContextPosterior:
        if basis.shape[-1] != self.rank:
            raise ValueError("REF rank does not match atlas rank")
        prior_covariances: list[torch.Tensor] = []
        prior_means: list[torch.Tensor] = []
        names: list[str] = []
        prototype_ids: list[int | None] = []
        prior_means.append(active.mean.detach())
        prior_covariances.append(active.covariance.detach())
        names.append("active")
        prototype_ids.append(active.prototype_id)
        if not self.disable_memory:
            for prototype in self.memory.prototypes:
                prior_means.append(prototype.mean.to(self.device))
                prior_covariances.append(prototype.covariance.to(self.device))
                names.append("memory")
                prototype_ids.append(prototype.prototype_id)
        prior_means.append(torch.zeros(self.rank, device=self.device, dtype=basis.dtype))
        prior_covariances.append(torch.eye(self.rank, device=self.device, dtype=basis.dtype) * float(self.config.new_prior_std**2))
        names.append("new")
        prototype_ids.append(None)
        evidences: list[float] = []
        means: list[torch.Tensor] = []
        covariances: list[torch.Tensor] = []
        log_masses: list[float] = []
        for name, mean, covariance in zip(names, prior_means, prior_covariances):
            means_i, covariance_i = gaussian_context_posterior(basis, residual, mean, covariance, self.sigma, self.config.numerical_jitter)
            means.append(means_i)
            covariances.append(covariance_i)
            evidences.append(log_marginal_evidence(basis, residual, mean, covariance, self.sigma, self.config.numerical_jitter))
            if name == "active":
                log_masses.append(float(self.config.active_prior_bonus))
            elif name == "memory":
                log_masses.append(float(self.config.memory_prior_mass))
            else:
                log_masses.append(float(self.config.new_prior_mass))
        if any(mass <= 0.0 for mass in log_masses):
            raise ValueError("REF prior masses must be positive")
        scores = torch.tensor(evidences, device=self.device) + torch.tensor(log_masses, device=self.device).log()
        weights = torch.softmax(scores, dim=0)
        if self.hard_routing:
            index = int(torch.argmax(weights))
            mean, covariance = means[index], covariances[index]
        else:
            mean, covariance = moment_match(torch.stack(means), torch.stack(covariances), weights)
            index = int(torch.argmax(weights))
        probabilities = {f"{name}:{idx}" if name == "memory" else name: float(weight) for idx, (name, weight) in enumerate(zip(names, weights))}
        new_probability = float(weights[names.index("new")])
        return ContextPosterior(mean, covariance, float(torch.logsumexp(scores, dim=0)), names[index], prototype_ids[index], probabilities, new_probability)
