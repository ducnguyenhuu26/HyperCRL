"""Irregular-checkpoint AUC and seed/visit aggregation helpers."""

from collections import defaultdict
from collections.abc import Sequence
from statistics import NormalDist

import numpy as np

from .common import MetricResult, invalid_result


def normalized_auc(
    x: Sequence[float],
    y: Sequence[float],
    *,
    start: float | None = None,
    end: float | None = None,
    normalize: bool = True,
) -> float:
    """Integrate irregular checkpoints with trapezoids, optionally normalizing."""

    if len(x) != len(y) or len(x) < 2:
        raise ValueError("AUC requires at least two paired checkpoints")
    times = np.asarray(x, dtype=float)
    values = np.asarray(y, dtype=float)
    if not np.all(np.isfinite(times)) or not np.all(np.isfinite(values)):
        raise ValueError("AUC inputs must be finite")
    if np.any(np.diff(times) <= 0):
        raise ValueError("AUC checkpoint times must be strictly increasing")
    left = float(times[0] if start is None else start)
    right = float(times[-1] if end is None else end)
    if right <= left or left < times[0] or right > times[-1]:
        raise ValueError("AUC interval must be ordered and inside the checkpoint range")
    grid = np.concatenate(([left], times[(times > left) & (times < right)], [right]))
    curve = np.interp(grid, times, values)
    area = float(np.trapezoid(curve, grid))
    return area / (right - left) if normalize else area


def metric_auc(
    name: str,
    x: Sequence[float],
    y: Sequence[float],
    *,
    higher_is_better: bool,
    start: float | None = None,
    end: float | None = None,
) -> MetricResult:
    try:
        value = normalized_auc(x, y, start=start, end=end)
    except ValueError as exc:
        return invalid_result(name, higher_is_better, str(exc), checkpoint_count=len(x))
    return MetricResult(name, value, higher_is_better, metadata={"checkpoint_count": len(x)})


def aggregate_seed_values(
    values: Sequence[float], *, name: str, higher_is_better: bool, confidence: float = 0.95
) -> dict[str, float | int | None]:
    """Return mean, SD, SE, and a normal-approximation confidence interval."""

    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if finite.size == 0:
        return {"metric_name": name, "count": 0, "mean": None, "std": None, "stderr": None, "ci_low": None, "ci_high": None}
    mean = float(finite.mean())
    std = float(finite.std(ddof=1)) if finite.size > 1 else 0.0
    stderr = std / np.sqrt(finite.size)
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    z = NormalDist().inv_cdf((1.0 + confidence) / 2.0)
    return {
        "metric_name": name,
        "higher_is_better": higher_is_better,
        "count": int(finite.size),
        "mean": mean,
        "std": std,
        "stderr": float(stderr),
        "ci_low": float(mean - z * stderr),
        "ci_high": float(mean + z * stderr),
    }


def aggregate_visit_metrics(records: Sequence[dict[str, object]]) -> dict[str, object]:
    """Aggregate only dimensionless metric records across visits/environments."""

    grouped: dict[str, list[float]] = defaultdict(list)
    for record in records:
        name = str(record["metric_name"])
        value = record.get("value")
        if record.get("valid", True) and isinstance(value, (int, float)) and np.isfinite(value):
            grouped[name].append(float(value))
    return {name: {"mean": float(np.mean(values)), "count": len(values)} for name, values in grouped.items()}


_CROSS_ENV_DIMENSIONLESS_PREFIXES = (
    "wm/r2_",
    "wm/nrmse_",
    "continual/",
    "reward/kendall_",
    "coupling/world_reward_kendall",
    "coupling/normalized_selection_regret",
)


def aggregate_cross_environment(records: Sequence[dict[str, object]]) -> dict[str, object]:
    """Aggregate only dimensionless metrics across environments.

    Raw returns/MSE/RMSE are intentionally rejected because their physical
    scales are environment-dependent.
    """

    names = {str(record["metric_name"]) for record in records}
    invalid_names = {name for name in names if not name.startswith(_CROSS_ENV_DIMENSIONLESS_PREFIXES)}
    if invalid_names:
        raise ValueError(f"raw or non-normalized metrics cannot be cross-environment averaged: {sorted(invalid_names)}")
    return aggregate_visit_metrics(records)
