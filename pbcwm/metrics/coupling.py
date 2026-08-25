"""Reward ranking and learned-world/true-outcome coupling metrics."""

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from scipy.stats import kendalltau

from .common import CandidateTrajectoryBank, MetricResult, invalid_result


@dataclass(frozen=True)
class CandidateScoreSet:
    true_reward_on_true_trajectory: torch.Tensor
    learned_reward_on_true_trajectory: torch.Tensor
    learned_reward_on_model_trajectory: torch.Tensor
    true_reward_on_model_trajectory: torch.Tensor | None = None


def _kendall(name: str, left: torch.Tensor, right: torch.Tensor) -> MetricResult:
    left_np = left.detach().cpu().reshape(-1).numpy()
    right_np = right.detach().cpu().reshape(-1).numpy()
    if left_np.size < 2:
        return invalid_result(name, True, "at least two candidates are required", candidate_count=int(left_np.size))
    result = kendalltau(left_np, right_np, variant="b", nan_policy="omit")
    if result.statistic is None or not np.isfinite(result.statistic):
        return invalid_result(name, True, "Kendall tau is undefined for the supplied ties", candidate_count=int(left_np.size), ties=True)
    return MetricResult(name, float(result.statistic), True, metadata={"candidate_count": int(left_np.size), "ties": bool(len(np.unique(left_np)) < left_np.size or len(np.unique(right_np)) < right_np.size), "variant": "tau_b"})


def reward_kendall_true(scores: CandidateScoreSet) -> MetricResult:
    return _kendall("reward/kendall_true", scores.learned_reward_on_true_trajectory, scores.true_reward_on_true_trajectory)


def reward_kendall_imagined(scores: CandidateScoreSet) -> MetricResult:
    if scores.true_reward_on_model_trajectory is None:
        return invalid_result("reward/kendall_imagined", True, "true reward cannot validly score imagined states")
    return _kendall("reward/kendall_imagined", scores.learned_reward_on_model_trajectory, scores.true_reward_on_model_trajectory)


def world_reward_kendall(scores: CandidateScoreSet) -> MetricResult:
    return _kendall("coupling/world_reward_kendall", scores.learned_reward_on_model_trajectory, scores.true_reward_on_true_trajectory)


def selection_regret(true_returns: torch.Tensor, system_scores: torch.Tensor) -> MetricResult:
    true_returns = true_returns.reshape(-1)
    system_scores = system_scores.reshape(-1)
    if true_returns.numel() < 2 or true_returns.numel() != system_scores.numel():
        return invalid_result("coupling/selection_regret", False, "at least two aligned candidates are required")
    oracle = int(torch.argmax(true_returns))
    selected = int(torch.argmax(system_scores))
    return MetricResult("coupling/selection_regret", float(true_returns[oracle] - true_returns[selected]), False, metadata={"oracle_index": oracle, "system_index": selected, "candidate_count": int(true_returns.numel())})


def normalized_selection_regret(true_returns: torch.Tensor, system_scores: torch.Tensor, *, eps: float = 1e-8) -> MetricResult:
    result = selection_regret(true_returns, system_scores)
    if not result.valid:
        return MetricResult("coupling/normalized_selection_regret", None, False, False, result.reason, result.metadata)
    spread = float(torch.max(true_returns) - torch.min(true_returns))
    if spread <= eps:
        return invalid_result("coupling/normalized_selection_regret", False, "true candidate-return spread is degenerate")
    return MetricResult("coupling/normalized_selection_regret", float(result.value) / (spread + eps), False, metadata=result.metadata)


def rollout_candidate_bank(predictor: object, bank: CandidateTrajectoryBank) -> torch.Tensor:
    """Roll out all candidates recursively; output is [B,M,H+1,D]."""

    batch, candidates, horizon, action_dim = bank.action_sequences.shape
    del action_dim
    state = bank.initial_obs[:, None, :].expand(batch, candidates, -1).clone()
    states = [state]
    with torch.no_grad():
        for step in range(horizon):
            flat_state = state.reshape(batch * candidates, -1)
            flat_action = bank.action_sequences[:, :, step].reshape(batch * candidates, -1)
            next_state = predictor.predict(flat_state, flat_action) if hasattr(predictor, "predict") else predictor(flat_state, flat_action)
            state = next_state.reshape(batch, candidates, -1)
            states.append(state)
    return torch.stack(states, dim=2)


def score_candidate_trajectories(states: torch.Tensor, actions: torch.Tensor, reward_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]) -> torch.Tensor:
    if states.ndim != 4 or actions.ndim != 4 or states.shape[:3] != (actions.shape[0], actions.shape[1], actions.shape[2]):
        raise ValueError("states/actions must be [B,M,H+1,D] and [B,M,H,A]")
    batch, candidates, horizon = actions.shape[:3]
    rewards = reward_fn(states[:, :, :-1].reshape(-1, states.shape[-1]), actions.reshape(-1, actions.shape[-1]), states[:, :, 1:].reshape(-1, states.shape[-1]))
    return rewards.reshape(batch, candidates, horizon).sum(dim=-1)
