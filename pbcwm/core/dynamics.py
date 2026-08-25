"""Neutral interface for replaceable dynamics learners."""

from abc import ABC, abstractmethod

import torch

from .types import Transition


class DynamicsLearner(ABC):
    """Minimal contract shared by all future dynamics baselines."""

    @abstractmethod
    def observe(self, transition: Transition) -> None:
        """Consume one transition from the online stream."""

    @abstractmethod
    def update(self, num_steps: int = 1) -> dict[str, float]:
        """Perform learner updates and return scalar diagnostics."""

    @abstractmethod
    def predict(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Return predicted next observations for batched inputs."""

    @abstractmethod
    def state_dict(self) -> dict:
        """Return serializable learner state."""

    @abstractmethod
    def load_state_dict(self, state: dict) -> None:
        """Restore learner state."""


class StochasticDynamicsLearner(DynamicsLearner):
    """Dynamics contract for planners that propagate posterior particles."""

    @abstractmethod
    def sample_next(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        num_samples: int,
    ) -> torch.Tensor:
        """Return sampled next observations as ``[samples, batch, obs_dim]``."""
