"""Stable machine-readable metric registry."""

from typing import Any

from .common import Metric, MetricResult
from .continual import acquisition_auc, few_shot_quality, reacquisition_auc, reuse_advantage
from .coupling import normalized_selection_regret, reward_kendall_imagined, reward_kendall_true, selection_regret, world_reward_kendall
from .oracle import oracle_diagnostics
from .planning import adaptation_cost, dynamic_regret, planning_deficit, planning_reacquisition_cost, planning_reuse_advantage, return_metrics
from .reward import bradley_terry_nll, pairwise_preference_accuracy
from .world_model import nrmse_at_H, nrmse_at_h, one_step_raw_errors, r2_at_H, r2_at_h


class RegisteredMetric:
    def __init__(self, name: str, higher_is_better: bool, function: Any):
        self.name = name
        self.higher_is_better = higher_is_better
        self._function = function

    def compute(self, *args: Any, **kwargs: Any) -> MetricResult | dict[str, MetricResult]:
        return self._function(*args, **kwargs)


def _return_metric(key: str):
    return lambda episode_returns: return_metrics(episode_returns)[key]


def _dynamic_metric(key: str):
    return lambda checkpoint_times, j_ll, j_oo: dynamic_regret(checkpoint_times, j_ll, j_oo)[key]


def _oracle_metric(key: str):
    return lambda scores: oracle_diagnostics(scores)[key]


METRIC_REGISTRY: dict[str, Metric] = {
    "wm/r2_h1": RegisteredMetric("wm/r2_h1", True, lambda predictor, probe, **kwargs: r2_at_h(predictor, probe, 1, **kwargs)),
    "wm/r2_h": RegisteredMetric("wm/r2_h", True, r2_at_H),
    "wm/nrmse_h": RegisteredMetric("wm/nrmse_h", False, nrmse_at_H),
    "wm/mse_h1": RegisteredMetric("wm/mse_h1", False, lambda predictor, probe: one_step_raw_errors(predictor, probe)["wm/mse_h1"]),
    "wm/rmse_h1": RegisteredMetric("wm/rmse_h1", False, lambda predictor, probe: one_step_raw_errors(predictor, probe)["wm/rmse_h1"]),
    "continual/wm_acq_auc": RegisteredMetric("continual/wm_acq_auc", True, acquisition_auc),
    "continual/wm_reacq_auc": RegisteredMetric("continual/wm_reacq_auc", True, reacquisition_auc),
    "continual/wm_reuse_advantage": RegisteredMetric("continual/wm_reuse_advantage", True, reuse_advantage),
    "continual/wm_quality_at_interactions": RegisteredMetric("continual/wm_quality_at_interactions", True, few_shot_quality),
    "reward/pairwise_accuracy": RegisteredMetric("reward/pairwise_accuracy", True, pairwise_preference_accuracy),
    "reward/bt_nll": RegisteredMetric("reward/bt_nll", False, bradley_terry_nll),
    "reward/kendall_true": RegisteredMetric("reward/kendall_true", True, reward_kendall_true),
    "reward/kendall_imagined": RegisteredMetric("reward/kendall_imagined", True, reward_kendall_imagined),
    "coupling/world_reward_kendall": RegisteredMetric("coupling/world_reward_kendall", True, world_reward_kendall),
    "coupling/selection_regret": RegisteredMetric("coupling/selection_regret", False, selection_regret),
    "coupling/normalized_selection_regret": RegisteredMetric("coupling/normalized_selection_regret", False, normalized_selection_regret),
    "planning/return_mean": RegisteredMetric("planning/return_mean", True, _return_metric("planning/return_mean")),
    "planning/return_std": RegisteredMetric("planning/return_std", False, _return_metric("planning/return_std")),
    "planning/planning_deficit": RegisteredMetric("planning/planning_deficit", False, planning_deficit),
    "planning/dynamic_regret": RegisteredMetric("planning/dynamic_regret", False, _dynamic_metric("planning/dynamic_regret")),
    "planning/mean_dynamic_regret": RegisteredMetric("planning/mean_dynamic_regret", False, _dynamic_metric("planning/mean_dynamic_regret")),
    "planning/adaptation_cost": RegisteredMetric("planning/adaptation_cost", False, adaptation_cost),
    "planning/reacquisition_cost": RegisteredMetric("planning/reacquisition_cost", False, planning_reacquisition_cost),
    "planning/reuse_advantage": RegisteredMetric("planning/reuse_advantage", True, planning_reuse_advantage),
    "oracle/j_ll": RegisteredMetric("oracle/j_ll", True, _oracle_metric("oracle/j_ll")),
    "oracle/j_ol": RegisteredMetric("oracle/j_ol", True, _oracle_metric("oracle/j_ol")),
    "oracle/j_lo": RegisteredMetric("oracle/j_lo", True, _oracle_metric("oracle/j_lo")),
    "oracle/j_oo": RegisteredMetric("oracle/j_oo", True, _oracle_metric("oracle/j_oo")),
    "oracle/world_side_gap": RegisteredMetric("oracle/world_side_gap", False, _oracle_metric("oracle/world_side_gap")),
    "oracle/reward_side_gap": RegisteredMetric("oracle/reward_side_gap", False, _oracle_metric("oracle/reward_side_gap")),
    "oracle/full_system_gap": RegisteredMetric("oracle/full_system_gap", False, _oracle_metric("oracle/full_system_gap")),
    "oracle/world_reward_interaction": RegisteredMetric("oracle/world_reward_interaction", False, _oracle_metric("oracle/world_reward_interaction")),
}


def get_metric(name: str) -> Metric:
    try:
        return METRIC_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"unknown metric: {name}") from exc
