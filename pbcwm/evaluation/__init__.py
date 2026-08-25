"""Evaluation utilities independent of the training loop."""

from .metrics import episode_return, evaluate_dynamics, prediction_mse
from .preference_metrics import ensemble_disagreement, mean_query_disagreement, preference_accuracy
from .gpmm_metrics import assignment_contingency, assignment_purity, expert_reuse_rate
from .hypercrl_metrics import embedding_contingency, embedding_purity, embedding_reuse_rate
from .vblrl_metrics import posterior_contingency, posterior_purity, posterior_reuse_rate
from .curious_replay_metrics import (
    first_recovery_step,
    first_return_recovery_step,
    replay_stage_share,
    sampled_stage_counts,
)

__all__ = [
    "ensemble_disagreement",
    "episode_return",
    "evaluate_dynamics",
    "mean_query_disagreement",
    "preference_accuracy",
    "prediction_mse",
    "assignment_contingency",
    "assignment_purity",
    "expert_reuse_rate",
    "embedding_contingency",
    "embedding_purity",
    "embedding_reuse_rate",
    "posterior_contingency",
    "posterior_purity",
    "posterior_reuse_rate",
    "first_recovery_step",
    "first_return_recovery_step",
    "replay_stage_share",
    "sampled_stage_counts",
]
