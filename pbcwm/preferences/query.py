"""Disagreement-based active preference query selection."""

from collections.abc import Sequence

import numpy as np

from .reward_model import PreferenceRewardEnsemble
from .types import TrajectorySegment


class DisagreementQuerySelector:
    """Select candidate pairs with high ensemble variance in P(A preferred)."""

    def __init__(self, pair_pool_size: int = 256, seed: int | None = None) -> None:
        if pair_pool_size <= 0:
            raise ValueError("pair_pool_size must be positive")
        self.pair_pool_size = int(pair_pool_size)
        self._rng = np.random.default_rng(seed)

    def score_pairs_with_ensemble(
        self,
        candidates: Sequence[TrajectorySegment],
        reward_ensemble: PreferenceRewardEnsemble,
    ) -> list[tuple[tuple[int, int], float]]:
        pairs = self._candidate_pairs(len(candidates))
        scored = []
        for pair in pairs:
            probabilities = reward_ensemble.preference_probabilities(
                candidates[pair[0]], candidates[pair[1]]
            )
            disagreement = float(probabilities.var(unbiased=False).cpu())
            scored.append((pair, disagreement))
        return sorted(scored, key=lambda item: item[1], reverse=True)

    def select(
        self,
        candidates: Sequence[TrajectorySegment],
        reward_ensemble: PreferenceRewardEnsemble,
        num_queries: int,
    ) -> list[tuple[int, int]]:
        if num_queries <= 0 or len(candidates) < 2:
            return []
        scored = self.score_pairs_with_ensemble(candidates, reward_ensemble)
        return [pair for pair, _ in scored[:num_queries]]

    def _candidate_pairs(self, candidate_count: int) -> list[tuple[int, int]]:
        if candidate_count < 2:
            return []
        total_pairs = candidate_count * (candidate_count - 1) // 2
        if total_pairs <= self.pair_pool_size:
            return [(i, j) for i in range(candidate_count) for j in range(i + 1, candidate_count)]

        pairs: set[tuple[int, int]] = set()
        while len(pairs) < self.pair_pool_size:
            i, j = self._rng.choice(candidate_count, size=2, replace=False)
            pairs.add((min(int(i), int(j)), max(int(i), int(j))))
        return list(pairs)

    def state_dict(self) -> dict:
        return {"pair_pool_size": self.pair_pool_size, "rng_state": self._rng.bit_generator.state}

    def load_state_dict(self, state: dict) -> None:
        if int(state["pair_pool_size"]) != self.pair_pool_size:
            raise ValueError("query selector configuration mismatch")
        self._rng = np.random.default_rng()
        self._rng.bit_generator.state = state["rng_state"]
