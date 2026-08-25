"""Shared metric data contracts and invalid-result semantics."""

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

import torch


@dataclass(frozen=True)
class BenchmarkEvalMetadata:
    """Evaluator-only context; never pass this object to a learner or planner."""

    global_step: int
    segment_id: int | None = None
    dynamics_id: int | str | None = None
    visit_id: int | None = None
    change_event: bool = False
    parameter_vector: dict[str, float] | None = None


@dataclass(frozen=True)
class RolloutProbeBatch:
    """Frozen multi-step probes used only for model evaluation."""

    initial_obs: torch.Tensor
    actions: torch.Tensor
    true_states: torch.Tensor

    def __post_init__(self) -> None:
        if not all(torch.is_tensor(value) for value in (self.initial_obs, self.actions, self.true_states)):
            raise TypeError("rollout probe fields must be torch tensors")
        if self.initial_obs.ndim != 2 or self.actions.ndim != 3 or self.true_states.ndim != 3:
            raise ValueError("probe shapes must be [B,D], [B,H,A], and [B,H+1,D]")
        batch, obs_dim = self.initial_obs.shape
        if self.actions.shape[0] != batch or self.true_states.shape[0] != batch:
            raise ValueError("probe fields must share batch size")
        if self.true_states.shape[1] != self.actions.shape[1] + 1 or self.true_states.shape[2] != obs_dim:
            raise ValueError("true_states must have shape [B,H+1,obs_dim]")
        if not torch.equal(self.true_states[:, 0], self.initial_obs):
            raise ValueError("true_states[:, 0] must equal initial_obs")

    @property
    def horizon(self) -> int:
        return int(self.actions.shape[1])


@dataclass(frozen=True)
class PreferenceEvalBatch:
    """Held-out pairwise labels; label -1 is an explicit skipped/tied pair."""

    traj_a: Sequence[Any]
    traj_b: Sequence[Any]
    labels: torch.Tensor
    true_returns_a: torch.Tensor | None = None
    true_returns_b: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if not torch.is_tensor(self.labels):
            raise TypeError("labels must be a torch tensor")
        if self.labels.ndim != 1:
            raise ValueError("labels must be a one-dimensional tensor")
        if len(self.traj_a) != len(self.traj_b) or len(self.traj_a) != int(self.labels.numel()):
            raise ValueError("preference pair fields must have equal length")
        allowed = set(self.labels.detach().cpu().tolist())
        if not allowed.issubset({-1, 0, 1}):
            raise ValueError("preference labels must be 0, 1, or -1 for an explicit skip")


@dataclass(frozen=True)
class CandidateTrajectoryBank:
    """Fixed action candidates shared by true and imagined rollouts."""

    initial_obs: torch.Tensor
    action_sequences: torch.Tensor

    def __post_init__(self) -> None:
        if self.initial_obs.ndim != 2 or self.action_sequences.ndim != 4:
            raise ValueError("candidate bank shapes must be [B,D] and [B,M,H,A]")
        if self.initial_obs.shape[0] != self.action_sequences.shape[0]:
            raise ValueError("candidate bank batch sizes must match")


@dataclass(frozen=True)
class PlannerEvalResult:
    return_mean: float
    return_std: float
    episode_returns: list[float]
    num_episodes: int


@dataclass
class EvaluationSummary:
    world_model: dict[str, float] = field(default_factory=dict)
    continual: dict[str, float] = field(default_factory=dict)
    reward: dict[str, float] = field(default_factory=dict)
    coupling: dict[str, float] = field(default_factory=dict)
    planning: dict[str, float] = field(default_factory=dict)
    oracle: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class MetricResult:
    name: str
    value: float | None
    higher_is_better: bool
    valid: bool = True
    reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.valid and self.value is None:
            raise ValueError("a valid metric must have a numeric value")
        if self.value is not None and not isinstance(self.value, (int, float)):
            raise TypeError("metric values must be numeric or None")

    def to_record(self, **context: Any) -> dict[str, Any]:
        record = {
            "metric_name": self.name,
            "value": None if self.value is None else float(self.value),
            "higher_is_better": self.higher_is_better,
            "valid": self.valid,
            "reason": self.reason,
            **context,
        }
        record.update(self.metadata)
        return record


class Metric(Protocol):
    name: str
    higher_is_better: bool

    def compute(self, *args: Any, **kwargs: Any) -> MetricResult:
        ...


def invalid_result(name: str, higher_is_better: bool, reason: str, **metadata: Any) -> MetricResult:
    """Create an explicit undefined metric result; never coerce it to zero."""

    return MetricResult(
        name=name,
        value=None,
        higher_is_better=higher_is_better,
        valid=False,
        reason=reason,
        metadata=metadata,
    )
