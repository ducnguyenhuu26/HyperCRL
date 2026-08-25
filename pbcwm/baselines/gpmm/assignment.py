"""Latent expert assignment from transition prior and predictive likelihood."""

from dataclasses import dataclass
import math

import torch

from .expert import GPExpert


@dataclass
class AssignmentResult:
    selected_index: int
    log_scores: torch.Tensor
    posterior: torch.Tensor
    entropy: float
    margin: float
    new_component_posterior: float


def transition_log_priors(
    num_experts: int,
    previous_expert: int | None,
    transition_counts: torch.Tensor,
    alpha: float,
    sticky_bonus: float,
    base_count: float,
    allow_new: bool = True,
) -> torch.Tensor:
    if num_experts == 0:
        return torch.tensor([math.log(alpha)], dtype=torch.float64) if allow_new else torch.empty(0)
    if min(alpha, base_count) <= 0 or sticky_bonus < 0:
        raise ValueError("alpha and base_count must be positive; sticky_bonus cannot be negative")
    if previous_expert is None:
        old_counts = torch.full((num_experts,), base_count, dtype=torch.float64)
    else:
        old_counts = transition_counts[previous_expert, :num_experts].double() + base_count
        old_counts[previous_expert] += sticky_bonus
    old_log_priors = old_counts.log()
    if not allow_new:
        return old_log_priors - torch.logsumexp(old_log_priors, dim=0)
    new_log_prior = torch.tensor([math.log(alpha)], dtype=torch.float64)
    all_log_priors = torch.cat((old_log_priors, new_log_prior))
    return all_log_priors - torch.logsumexp(all_log_priors, dim=0)


def assign_transition(
    experts: list[GPExpert],
    obs: torch.Tensor,
    action: torch.Tensor,
    next_obs: torch.Tensor,
    previous_expert: int | None,
    transition_counts: torch.Tensor,
    alpha: float,
    sticky_bonus: float,
    base_count: float,
    allow_new: bool = True,
) -> AssignmentResult:
    """Compute hard assignment using log-space posterior scores."""

    obs_batch = torch.as_tensor(obs, dtype=torch.float64).reshape(1, -1)
    action_batch = torch.as_tensor(action, dtype=torch.float64).reshape(1, -1)
    next_batch = torch.as_tensor(next_obs, dtype=torch.float64).reshape(1, -1)
    priors = transition_log_priors(
        len(experts), previous_expert, transition_counts, alpha, sticky_bonus, base_count, allow_new
    )
    likelihoods = [expert.log_likelihood(obs_batch, action_batch, next_batch)[0] for expert in experts]
    if allow_new:
        new_expert = GPExpert(
            obs_dim=obs_batch.shape[-1],
            action_dim=action_batch.shape[-1],
            max_points=1,
            prior_variance=1.0,
            observation_noise=0.05,
        )
        likelihoods.append(new_expert.prior_predictive_log_likelihood(obs_batch, action_batch, next_batch)[0])
    log_likelihoods = torch.stack(likelihoods) if likelihoods else torch.empty(0, dtype=torch.float64)
    log_scores = priors + log_likelihoods
    posterior = torch.softmax(log_scores, dim=0)
    selected = int(torch.argmax(log_scores).item())
    sorted_scores = torch.sort(log_scores, descending=True).values
    margin = (
        float((sorted_scores[0] - sorted_scores[1]).detach().cpu())
        if len(sorted_scores) > 1
        else float("inf")
    )
    entropy = float((-(posterior * posterior.clamp_min(1e-12).log()).sum()).detach().cpu())
    new_probability = float(posterior[-1].detach().cpu()) if allow_new else 0.0
    return AssignmentResult(selected, log_scores, posterior, entropy, margin, new_probability)
