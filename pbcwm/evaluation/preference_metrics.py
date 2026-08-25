"""Evaluation-only metrics for learned pairwise preferences."""

from collections.abc import Sequence

import numpy as np

from pbcwm.preferences.reward_model import PreferenceRewardEnsemble
from pbcwm.preferences.types import PreferenceExample, TrajectorySegment


def preference_accuracy(
    reward_ensemble: PreferenceRewardEnsemble,
    examples: Sequence[PreferenceExample],
) -> float:
    """Measure held-out pairwise accuracy without updating the ensemble."""

    return reward_ensemble.predict_preference_accuracy(examples)


def ensemble_disagreement(
    reward_ensemble: PreferenceRewardEnsemble,
    traj_a: TrajectorySegment,
    traj_b: TrajectorySegment,
) -> float:
    probabilities = reward_ensemble.preference_probabilities(traj_a, traj_b)
    return float(probabilities.var(unbiased=False).cpu())


def mean_query_disagreement(
    reward_ensemble: PreferenceRewardEnsemble,
    candidates: Sequence[TrajectorySegment],
    pairs: Sequence[tuple[int, int]],
) -> float:
    if not pairs:
        return 0.0
    scores = [
        ensemble_disagreement(reward_ensemble, candidates[index_a], candidates[index_b])
        for index_a, index_b in pairs
    ]
    return float(np.mean(scores))
