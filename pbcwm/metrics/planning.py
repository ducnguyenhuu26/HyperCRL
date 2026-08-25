"""Fixed-planner return and adaptation-cost metrics."""

from collections.abc import Sequence

import numpy as np

from .aggregation import normalized_auc
from .common import MetricResult, PlannerEvalResult, invalid_result


def planner_returns(episode_returns: Sequence[float]) -> PlannerEvalResult:
    values = [float(value) for value in episode_returns]
    if not values:
        raise ValueError("planner evaluation needs at least one episode")
    return PlannerEvalResult(float(np.mean(values)), float(np.std(values)), values, len(values))


def return_metrics(episode_returns: Sequence[float]) -> dict[str, MetricResult]:
    result = planner_returns(episode_returns)
    return {
        "planning/return_mean": MetricResult("planning/return_mean", result.return_mean, True, metadata={"num_episodes": result.num_episodes}),
        "planning/return_std": MetricResult("planning/return_std", result.return_std, False, metadata={"num_episodes": result.num_episodes}),
    }


def planning_deficit(j_ll: float, j_oo: float) -> MetricResult:
    return MetricResult("planning/planning_deficit", float(j_oo - j_ll), False)


def dynamic_regret(checkpoint_times: Sequence[float], j_ll: Sequence[float], j_oo: Sequence[float]) -> dict[str, MetricResult]:
    if len(checkpoint_times) != len(j_ll) or len(j_ll) != len(j_oo) or len(checkpoint_times) < 2:
        invalid = invalid_result("planning/dynamic_regret", False, "dynamic regret requires paired checkpoints")
        return {"planning/dynamic_regret": invalid, "planning/mean_dynamic_regret": invalid_result("planning/mean_dynamic_regret", False, invalid.reason or "invalid")}
    try:
        area = normalized_auc(checkpoint_times, np.asarray(j_oo) - np.asarray(j_ll), normalize=False)
        span = float(checkpoint_times[-1] - checkpoint_times[0])
    except ValueError as exc:
        invalid = invalid_result("planning/dynamic_regret", False, str(exc))
        return {"planning/dynamic_regret": invalid, "planning/mean_dynamic_regret": invalid_result("planning/mean_dynamic_regret", False, str(exc))}
    return {"planning/dynamic_regret": MetricResult("planning/dynamic_regret", area, False), "planning/mean_dynamic_regret": MetricResult("planning/mean_dynamic_regret", area / span, False)}


def adaptation_cost(checkpoint_times: Sequence[float], j_ll: Sequence[float], j_oo: Sequence[float], *, window: float | None = None, name: str = "planning/adaptation_cost") -> MetricResult:
    gaps = np.asarray(j_oo, dtype=float) - np.asarray(j_ll, dtype=float)
    end = float(checkpoint_times[-1] if window is None else window)
    try:
        value = normalized_auc(checkpoint_times, gaps, start=float(checkpoint_times[0]), end=end)
    except ValueError as exc:
        return invalid_result(name, False, str(exc))
    return MetricResult(name, value, False)


def planning_reacquisition_cost(checkpoint_times: Sequence[float], j_ll: Sequence[float], j_oo: Sequence[float], *, window: float | None = None) -> MetricResult:
    return adaptation_cost(checkpoint_times, j_ll, j_oo, window=window, name="planning/reacquisition_cost")


def planning_reuse_advantage(first_cost: MetricResult, recurrence_cost: MetricResult) -> MetricResult:
    if not first_cost.valid or not recurrence_cost.valid or first_cost.value is None or recurrence_cost.value is None:
        return invalid_result("planning/reuse_advantage", True, "both planning costs must be valid")
    return MetricResult("planning/reuse_advantage", first_cost.value - recurrence_cost.value, True)
