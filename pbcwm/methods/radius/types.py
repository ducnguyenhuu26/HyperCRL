"""Typed state owned by RADIUS; evaluator metadata is intentionally absent."""

from dataclasses import dataclass, field

import torch


@dataclass
class ContextPosterior:
    mean: torch.Tensor
    covariance: torch.Tensor
    log_evidence: float
    source: str
    prototype_id: int | None = None
    hypothesis_probabilities: dict[str, float] = field(default_factory=dict)
    new_hypothesis_probability: float = 0.0


@dataclass
class ContextPrototype:
    prototype_id: int
    mean: torch.Tensor
    covariance: torch.Tensor
    num_consolidations: int
    last_active_step: int
    creation_step: int
    usage_count: int
    reuse_count: int = 0


@dataclass
class NoveltyState:
    standardized_residual: float
    new_hypothesis_probability: float
    consecutive_trigger_count: int
    should_expand: bool


@dataclass
class ElasticityState:
    old_fisher_rank: int
    generalized_eigenvalues: torch.Tensor | None
    continual_elasticity: float | None
    protected_energy: float | None


@dataclass
class RadiusPrediction:
    next_obs_mean: torch.Tensor
    delta_mean: torch.Tensor
    context_mean: torch.Tensor
    context_covariance: torch.Tensor


@dataclass(frozen=True)
class RadiusReplayItem:
    obs: torch.Tensor
    action: torch.Tensor
    next_obs: torch.Tensor
    context_mean: torch.Tensor
    prototype_id: int | None = None


@dataclass(frozen=True)
class RadiusRecentItem:
    """Raw physical transition retained for one-coordinate REF windows."""

    obs: torch.Tensor
    action: torch.Tensor
    next_obs: torch.Tensor


@dataclass(frozen=True)
class ActivePriorSnapshot:
    """Active posterior immediately before the aligned recent transition."""

    mean: torch.Tensor
    covariance: torch.Tensor
    prototype_id: int | None


@dataclass(frozen=True)
class RadiusEvent:
    name: str
    global_step: int
    diagnostics: dict[str, float | int | str]
