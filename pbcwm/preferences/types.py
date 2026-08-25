"""Data types shared by preference collection and reward learning."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TrajectorySegment:
    """An imagined rollout with one action per transition."""

    obs: torch.Tensor
    actions: torch.Tensor
    next_obs: torch.Tensor

    def __post_init__(self) -> None:
        if not all(torch.is_tensor(value) for value in (self.obs, self.actions, self.next_obs)):
            raise TypeError("trajectory fields must be torch tensors")
        if self.obs.ndim != 2 or self.actions.ndim != 2 or self.next_obs.ndim != 2:
            raise ValueError("trajectory fields must have shape [horizon, feature]")
        if self.obs.shape[0] != self.actions.shape[0] or self.obs.shape[0] != self.next_obs.shape[0]:
            raise ValueError("obs, actions, and next_obs must share the horizon")
        if self.obs.shape[0] == 0:
            raise ValueError("trajectory segments must be non-empty")

    def detached(self) -> "TrajectorySegment":
        """Return an immutable, gradient-free copy suitable for replay."""

        return TrajectorySegment(
            obs=self.obs.detach().clone(),
            actions=self.actions.detach().clone(),
            next_obs=self.next_obs.detach().clone(),
        )


@dataclass(frozen=True)
class PreferenceExample:
    """Pairwise label: ``0`` means A preferred, ``1`` means B preferred."""

    traj_a: TrajectorySegment
    traj_b: TrajectorySegment
    label: int

    def __post_init__(self) -> None:
        if self.label not in (0, 1):
            raise ValueError("preference label must be 0 (A) or 1 (B)")
