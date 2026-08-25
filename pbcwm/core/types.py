"""Shared transition types."""

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class Transition:
    """One environment transition without hidden stage metadata."""

    obs: np.ndarray
    action: np.ndarray
    next_obs: np.ndarray
    reward: float
    terminated: bool
    truncated: bool


@dataclass
class TransitionBatch:
    """Learner-ready tensors with shape ``[batch, feature]``."""

    obs: torch.Tensor
    action: torch.Tensor
    next_obs: torch.Tensor
