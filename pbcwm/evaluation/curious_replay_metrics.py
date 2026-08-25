"""Evaluator-only replay allocation and recovery metrics."""

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence

import numpy as np


def replay_stage_share(stage_counts: Mapping[int, int]) -> dict[int, float]:
    """Normalize evaluator-maintained sampled-slot counts by stage."""

    total = sum(int(value) for value in stage_counts.values())
    if total <= 0:
        return {}
    return {
        int(stage): float(count / total)
        for stage, count in sorted(stage_counts.items())
    }


def first_recovery_step(
    records: Sequence[Mapping[str, object]],
    shift_step: int,
    metric_key: str,
    pre_shift_value: float,
    tolerance: float = 0.25,
) -> float:
    """Return first logged step whose non-negative metric recovers within tolerance."""

    if not np.isfinite(pre_shift_value) or pre_shift_value < 0 or tolerance < 0:
        return float("nan")
    threshold = pre_shift_value * (1.0 + tolerance)
    for record in records:
        step = int(record.get("global_step", -1))
        value = record.get(metric_key)
        if step >= shift_step and isinstance(value, (float, int)):
            value_float = float(value)
            if np.isfinite(value_float) and value_float <= threshold:
                return float(step - shift_step)
    return float("nan")


def first_return_recovery_step(
    episode_records: Sequence[Mapping[str, object]],
    shift_step: int,
    pre_shift_return: float,
) -> float:
    """Return first post-shift episode whose return reaches the pre-shift return."""

    if not np.isfinite(pre_shift_return):
        return float("nan")
    for record in episode_records:
        step = int(record.get("global_step", -1))
        value = record.get("episode_return")
        if step >= shift_step and isinstance(value, (float, int)):
            if np.isfinite(float(value)) and float(value) >= pre_shift_return:
                return float(step - shift_step)
    return float("nan")


def sampled_stage_counts(stage_by_slot: Mapping[int, int], sampled_slots: Iterable[int]) -> dict[int, int]:
    """Map evaluator-only slot labels onto learner-exposed sampled slot IDs."""

    counts: Counter[int] = Counter()
    for slot in sampled_slots:
        if int(slot) in stage_by_slot:
            counts[int(stage_by_slot[int(slot)])] += 1
    return dict(counts)
