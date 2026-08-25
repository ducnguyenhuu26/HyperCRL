from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from pbcwm.preferences.types import TrajectorySegment


def pair_entropy(probability: torch.Tensor | float) -> torch.Tensor:
    p = torch.as_tensor(probability, dtype=torch.float32).clamp(1e-7, 1.0 - 1e-7)
    return -(p * p.log() + (1.0 - p) * (1.0 - p).log())


@dataclass(frozen=True)
class PFPASelection:
    pairs: list[tuple[int, int]]
    frontier_pairs: int
    coverage_pairs: int
    mean_entropy: float
    mean_frontier_score: float


class PFPASelector:
    """Select uncertain, frontier-relevant pairs without teacher access."""

    def __init__(self, frontier_fraction: float = 0.8, max_pair_action_similarity: float = 0.98, seed: int | None = None):
        self.frontier_fraction = float(frontier_fraction)
        self.max_pair_action_similarity = float(max_pair_action_similarity)
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def score_from_samples(candidate_scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return pair probabilities, entropies, and frontier scores.

        ``candidate_scores`` is ``[sample, candidate]``. Samples can combine
        reward-model members and context posterior samples.
        """

        if candidate_scores.ndim != 2 or candidate_scores.shape[1] < 2:
            raise ValueError("candidate_scores must have shape [sample, candidate>=2]")
        candidate_count = candidate_scores.shape[1]
        elite_count = max(1, round(candidate_count * 0.10))
        elite = torch.zeros_like(candidate_scores, dtype=torch.bool)
        elite.scatter_(1, torch.topk(candidate_scores, elite_count, dim=1).indices, True)
        elite_probability = elite.float().mean(dim=0)
        pair_indices = torch.triu_indices(candidate_count, candidate_count, offset=1, device=candidate_scores.device)
        pair_probabilities = torch.sigmoid(
            candidate_scores[:, pair_indices[0]] - candidate_scores[:, pair_indices[1]]
        ).mean(dim=0)
        entropies = pair_entropy(pair_probabilities)
        frontier_scores = (
            elite_probability[pair_indices[0]] + elite_probability[pair_indices[1]]
        ) * entropies
        return pair_probabilities, entropies, frontier_scores

    def _pair_list(self, candidate_count: int) -> list[tuple[int, int]]:
        return [(i, j) for i in range(candidate_count) for j in range(i + 1, candidate_count)]

    def select_from_scores(self, candidate_scores: torch.Tensor, actions: torch.Tensor | None, num_queries: int) -> PFPASelection:
        if num_queries <= 0 or candidate_scores.shape[-1] < 2:
            return PFPASelection([], 0, 0, 0.0, 0.0)
        probabilities, entropies, frontier_scores = self.score_from_samples(candidate_scores)
        pairs = self._pair_list(candidate_scores.shape[-1])
        frontier_count = min(num_queries, max(0, round(num_queries * self.frontier_fraction)))
        ordering = torch.argsort(frontier_scores, descending=True).tolist()
        selected: list[int] = []
        for index in ordering:
            if len(selected) >= frontier_count:
                break
            if self._diverse(index, pairs, actions, selected):
                selected.append(index)
        # Fallback only on candidate shortage; every requested pair still comes
        # from model scores and never from true reward.
        for index in ordering:
            if len(selected) >= frontier_count:
                break
            if index not in selected:
                selected.append(index)
        coverage_count = max(0, min(num_queries - len(selected), num_queries - frontier_count))
        remaining = [index for index in torch.argsort(entropies, descending=True).tolist() if index not in selected]
        if actions is not None:
            remaining.sort(key=lambda index: self._coverage_key(index, pairs, actions), reverse=True)
        selected.extend(remaining[:coverage_count])
        selected = selected[:num_queries]
        selected_pairs = [pairs[index] for index in selected]
        selected_entropy = [float(entropies[index]) for index in selected]
        selected_frontier = [float(frontier_scores[index]) for index in selected[: min(frontier_count, len(selected))]]
        return PFPASelection(selected_pairs, min(frontier_count, len(selected)), max(0, len(selected) - min(frontier_count, len(selected))), float(np.mean(selected_entropy)) if selected_entropy else 0.0, float(np.mean(selected_frontier)) if selected_frontier else 0.0)

    def select(self, candidates: list[TrajectorySegment], reward_model: Any, num_queries: int, *, context_samples: int = 4) -> PFPASelection:
        if len(candidates) < 2:
            return PFPASelection([], 0, 0, 0.0, 0.0)
        models = getattr(reward_model, "models", None)
        if models is None:
            raise TypeError("PFPA requires the shared reward ensemble")
        scores = []
        with torch.no_grad():
            for model in models:
                scores.append(torch.stack([model(traj.obs, traj.actions).sum() for traj in candidates]))
        candidate_scores = torch.stack(scores)
        if context_samples > 1:
            candidate_scores = candidate_scores.repeat((context_samples, 1))
        actions = torch.stack([traj.actions.flatten() for traj in candidates])
        return self.select_from_scores(candidate_scores, actions, num_queries)

    def _diverse(self, pair_index: int, pairs: list[tuple[int, int]], actions: torch.Tensor | None, selected: list[int]) -> bool:
        if actions is None or not selected:
            return True
        pair = pairs[pair_index]
        vector = (actions[pair[0]] - actions[pair[1]]).flatten()
        for selected_index in selected:
            other = pairs[selected_index]
            other_vector = (actions[other[0]] - actions[other[1]]).flatten()
            cosine = torch.nn.functional.cosine_similarity(vector[None], other_vector[None]).item()
            if cosine > self.max_pair_action_similarity:
                return False
        return True

    @staticmethod
    def _coverage_key(pair_index: int, pairs: list[tuple[int, int]], actions: torch.Tensor) -> float:
        pair = pairs[pair_index]
        return float((actions[pair[0]] - actions[pair[1]]).square().mean())

    def state_dict(self) -> dict:
        return {"frontier_fraction": self.frontier_fraction, "max_pair_action_similarity": self.max_pair_action_similarity, "rng_state": self.rng.bit_generator.state}

    def load_state_dict(self, state: dict) -> None:
        self.frontier_fraction = float(state["frontier_fraction"])
        self.max_pair_action_similarity = float(state["max_pair_action_similarity"])
        self.rng = np.random.default_rng()
        self.rng.bit_generator.state = state["rng_state"]
