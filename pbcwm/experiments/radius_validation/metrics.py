"""Small, fail-closed metrics used by the Hopper development runners."""

from __future__ import annotations

import numpy as np


def auc(checkpoints: list[int], values: list[float]) -> float:
    if len(checkpoints) != len(values) or len(values) < 2:
        raise ValueError("AUC needs at least two paired checkpoints")
    x = np.asarray(checkpoints, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    if np.any(~np.isfinite(y)) or np.any(np.diff(x) <= 0):
        raise ValueError("metric series must be finite and strictly increasing")
    return float(np.trapezoid(y, x))


def first_visit_auc(checkpoints: list[int], r2_values: list[float]) -> float:
    return auc(checkpoints, r2_values)


def return_visit_auc(checkpoints: list[int], r2_values: list[float]) -> float:
    return auc(checkpoints, r2_values)


def wm_reuse_advantage(first_auc: float, return_auc: float) -> float:
    return float(return_auc - first_auc)


def t90(checkpoints: list[int], values: list[float], *, stage_length: int) -> int:
    """First post-change checkpoint reaching 90% of the stable-tail target."""

    if len(checkpoints) != len(values) or not values:
        raise ValueError("T90 needs a non-empty paired series")
    tail = np.asarray(values[-max(1, len(values) // 5):], dtype=np.float64)
    target = 0.9 * float(np.mean(tail))
    for checkpoint, value in zip(checkpoints, values):
        if value >= target:
            return int(checkpoint)
    return int(stage_length)
