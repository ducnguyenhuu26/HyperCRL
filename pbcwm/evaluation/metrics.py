"""Small, reusable model and rollout metrics."""

from collections.abc import Iterable, Sequence

import numpy as np
import torch

from pbcwm.core.dynamics import DynamicsLearner
from pbcwm.core.types import Transition


def episode_return(rewards: Iterable[float]) -> float:
    return float(sum(float(reward) for reward in rewards))


def prediction_mse(dynamics: DynamicsLearner, transitions: Sequence[Transition]) -> float:
    if not transitions:
        return float("nan")
    obs = torch.as_tensor(np.stack([transition.obs for transition in transitions]), dtype=torch.float32)
    action = torch.as_tensor(np.stack([transition.action for transition in transitions]), dtype=torch.float32)
    next_obs = torch.as_tensor(np.stack([transition.next_obs for transition in transitions]), dtype=torch.float32)
    with torch.no_grad():
        predicted = dynamics.predict(obs, action).to(next_obs.device)
        error = torch.mean((predicted - next_obs).square())
    return float(error.cpu())


def evaluate_dynamics(
    dynamics: DynamicsLearner,
    transitions: Sequence[Transition],
) -> dict[str, float]:
    return {
        "prediction_mse": prediction_mse(dynamics, transitions),
        "num_transitions": float(len(transitions)),
    }
