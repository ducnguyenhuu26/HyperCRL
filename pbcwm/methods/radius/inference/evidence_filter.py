from __future__ import annotations

from dataclasses import dataclass

import torch

from ..memory.context_memory import ContextMemory
from ..types import ContextPosterior
from .gaussian_context import gaussian_context_posterior, log_marginal_evidence, moment_match


@dataclass
class RoutingResult:
    """Window evidence over stored prototypes and the diffuse new route."""

    candidates: list[ContextPosterior]
    weights: torch.Tensor
    new_probability: float


class RecurrentEvidenceFilter:
    """Sequential active tracking plus non-duplicated window routing evidence."""

    def __init__(self, rank: int, sigma: float, config, memory: ContextMemory, device: torch.device, *, disable_memory: bool = False, hard_routing: bool = False):
        self.rank = int(rank)
        self.sigma = float(sigma)
        self.config = config
        self.memory = memory
        self.device = device
        self.disable_memory = bool(disable_memory)
        self.hard_routing = bool(hard_routing)

    def _check(self, basis: torch.Tensor, residual: torch.Tensor) -> None:
        if basis.ndim != 3 or residual.ndim != 2 or basis.shape[:2] != residual.shape or basis.shape[-1] != self.rank:
            raise ValueError("REF inputs must have shapes [N,state_dim,rank] and [N,state_dim]")

    def update_active(self, basis_t: torch.Tensor, residual_t: torch.Tensor, active: ContextPosterior) -> ContextPosterior:
        """Update the carried posterior with exactly one newly observed sample."""

        self._check(basis_t, residual_t)
        if basis_t.shape[0] != 1:
            raise ValueError("update_active accepts exactly one transition")
        eye = torch.eye(self.rank, device=basis_t.device, dtype=basis_t.dtype)
        prior_covariance = active.covariance.to(basis_t) + float(self.config.context_process_noise) * eye
        mean, covariance = gaussian_context_posterior(basis_t, residual_t, active.mean.to(basis_t), prior_covariance, self.sigma, self.config.numerical_jitter)
        return ContextPosterior(mean, covariance, active.log_evidence, "active", active.prototype_id, dict(active.hypothesis_probabilities), active.new_hypothesis_probability)

    def evaluate_hypotheses(self, basis_window: torch.Tensor, residual_window: torch.Tensor, active: ContextPosterior) -> RoutingResult:
        """Score memory/new routes from independent priors over the window."""

        self._check(basis_window, residual_window)
        candidates: list[ContextPosterior] = []
        log_masses: list[float] = []
        if not self.disable_memory:
            for prototype in self.memory.prototypes:
                if active.prototype_id == prototype.prototype_id:
                    continue
                mean = prototype.mean.to(self.device)
                covariance = prototype.covariance.to(self.device)
                posterior_mean, posterior_covariance = gaussian_context_posterior(basis_window, residual_window, mean, covariance, self.sigma, self.config.numerical_jitter)
                evidence = log_marginal_evidence(basis_window, residual_window, mean, covariance, self.sigma, self.config.numerical_jitter)
                candidates.append(ContextPosterior(posterior_mean, posterior_covariance, evidence, "memory", prototype.prototype_id))
                log_masses.append(float(self.config.memory_prior_mass))
        new_mean = torch.zeros(self.rank, device=self.device, dtype=basis_window.dtype)
        new_covariance = torch.eye(self.rank, device=self.device, dtype=basis_window.dtype) * float(self.config.new_prior_std**2)
        posterior_mean, posterior_covariance = gaussian_context_posterior(basis_window, residual_window, new_mean, new_covariance, self.sigma, self.config.numerical_jitter)
        evidence = log_marginal_evidence(basis_window, residual_window, new_mean, new_covariance, self.sigma, self.config.numerical_jitter)
        candidates.append(ContextPosterior(posterior_mean, posterior_covariance, evidence, "new", None))
        log_masses.append(float(self.config.new_prior_mass))
        if any(mass <= 0.0 for mass in log_masses):
            raise ValueError("REF prior masses must be positive")
        scores = torch.tensor([item.log_evidence for item in candidates], device=self.device) + torch.tensor(log_masses, device=self.device).log()
        weights = torch.softmax(scores, dim=0)
        # With the active prototype deduplicated, a lone new candidate is not
        # evidence against the active route; assigning it probability one
        # would manufacture novelty from the absence of a second hypothesis.
        new_probability = float(weights[-1]) if active.prototype_id is None or len(candidates) > 1 else 0.0
        return RoutingResult(candidates, weights, new_probability)

    def resolve_context(self, active: ContextPosterior, routing_result: RoutingResult) -> ContextPosterior:
        """Choose/merge an independent recurrence route, otherwise keep active."""

        if not routing_result.candidates:
            return active
        best_index = int(torch.argmax(routing_result.weights))
        best_weight = float(routing_result.weights[best_index])
        best = routing_result.candidates[best_index]
        if best.source == "new" and (len(routing_result.candidates) == 1 or best_weight <= 0.5):
            active.hypothesis_probabilities = {"active": 1.0 - routing_result.new_probability, "new": routing_result.new_probability}
            active.new_hypothesis_probability = routing_result.new_probability
            return active
        if self.hard_routing:
            selected = best
        else:
            selected_mean, selected_covariance = moment_match(
                torch.stack([item.mean for item in routing_result.candidates]),
                torch.stack([item.covariance for item in routing_result.candidates]),
                routing_result.weights,
            )
            selected = ContextPosterior(selected_mean, selected_covariance, best.log_evidence, best.source, best.prototype_id)
        selected.hypothesis_probabilities = {f"{item.source}:{item.prototype_id}" if item.source == "memory" else item.source: float(weight) for item, weight in zip(routing_result.candidates, routing_result.weights)}
        selected.new_hypothesis_probability = routing_result.new_probability
        return selected

    def infer(self, basis: torch.Tensor, residual: torch.Tensor, active: ContextPosterior) -> ContextPosterior:
        """Compatibility wrapper that processes a batch once per sample."""

        current = active
        for index in range(basis.shape[0]):
            current = self.update_active(basis[index:index + 1], residual[index:index + 1], current)
        return current
