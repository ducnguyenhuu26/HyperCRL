"""Visit-level acquisition, recurrence, and reuse metrics."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from .aggregation import metric_auc
from .common import MetricResult, invalid_result


@dataclass(frozen=True)
class VisitMetricSeries:
    dynamics_id: int | str
    visit_id: int
    checkpoints_since_visit_start: list[int]
    values: list[float]

    def __post_init__(self) -> None:
        if len(self.checkpoints_since_visit_start) != len(self.values) or len(self.values) == 0:
            raise ValueError("visit series needs paired, non-empty checkpoints and values")
        if any(b <= a for a, b in zip(self.checkpoints_since_visit_start, self.checkpoints_since_visit_start[1:])):
            raise ValueError("visit checkpoints must be strictly increasing")


def visit_auc(series: VisitMetricSeries, *, window: float | None = None, higher_is_better: bool = True, name: str = "continual/visit_auc") -> MetricResult:
    end = float(series.checkpoints_since_visit_start[-1] if window is None else window)
    return metric_auc(name, series.checkpoints_since_visit_start, series.values, higher_is_better=higher_is_better, start=float(series.checkpoints_since_visit_start[0]), end=end)


def acquisition_auc(series: VisitMetricSeries, *, window: float | None = None) -> MetricResult:
    return visit_auc(series, window=window, name="continual/wm_acq_auc")


def reacquisition_auc(series: VisitMetricSeries, *, window: float | None = None) -> MetricResult:
    return visit_auc(series, window=window, name="continual/wm_reacq_auc")


def reuse_advantage(first_auc: MetricResult, recurrence_auc: MetricResult) -> MetricResult:
    name = "continual/wm_reuse_advantage"
    if not first_auc.valid or not recurrence_auc.valid or first_auc.value is None or recurrence_auc.value is None:
        return invalid_result(name, True, "first and recurrence AUC must both be valid")
    return MetricResult(name, recurrence_auc.value - first_auc.value, True)


def few_shot_quality(series: VisitMetricSeries, interactions: int, *, name: str = "continual/wm_quality_at_interactions") -> MetricResult:
    if interactions < 0:
        return invalid_result(name, True, "interaction count must be non-negative")
    for checkpoint, value in zip(series.checkpoints_since_visit_start, series.values):
        if checkpoint >= interactions:
            return MetricResult(name, float(value), True, metadata={"interactions": interactions})
    return invalid_result(name, True, "series does not cover the requested interaction budget", interactions=interactions)


def stable_quality(series: VisitMetricSeries, *, start: int, end: int) -> MetricResult:
    return visit_auc(series, window=end, name="wm/stable_r2_h") if start == series.checkpoints_since_visit_start[0] else metric_auc("wm/stable_r2_h", series.checkpoints_since_visit_start, series.values, higher_is_better=True, start=start, end=end)


def group_visits(records: list[VisitMetricSeries]) -> dict[int | str, dict[int, VisitMetricSeries]]:
    grouped: dict[int | str, dict[int, VisitMetricSeries]] = {}
    for record in records:
        grouped.setdefault(record.dynamics_id, {})[record.visit_id] = record
    return grouped


def match_first_and_return_visits(records: list[VisitMetricSeries]) -> list[tuple[VisitMetricSeries, list[VisitMetricSeries]]]:
    return [(visits[0], [visits[visit_id] for visit_id in sorted(visits) if visit_id > 0]) for visits in group_visits(records).values() if 0 in visits]


def recurrent_aggregation(records: list[VisitMetricSeries], *, window: float | None = None) -> dict[str, Any]:
    pairs = match_first_and_return_visits(records)
    per_dynamics: dict[str, list[float]] = {}
    for first, returns in pairs:
        first_auc = acquisition_auc(first, window=window)
        values = []
        for recurrence in returns:
            reuse = reuse_advantage(first_auc, reacquisition_auc(recurrence, window=window))
            if reuse.valid and reuse.value is not None:
                values.append(float(reuse.value))
        if values:
            per_dynamics[str(first.dynamics_id)] = values
    all_values = [value for values in per_dynamics.values() for value in values]
    return {
        "per_dynamics": per_dynamics,
        "mean_reuse_advantage": float(np.mean(all_values)) if all_values else None,
        "median_reuse_advantage": float(np.median(all_values)) if all_values else None,
        "recurrence_count": len(all_values),
    }
