"""Evaluator-only metrics for VBLRL-Adapt posterior assignments."""

from collections import Counter, defaultdict
from collections.abc import Iterable

import numpy as np


def posterior_contingency(
    true_stages: Iterable[int],
    posterior_ids: Iterable[int],
) -> dict[int, dict[int, int]]:
    """Count inferred posterior IDs per evaluator-known environment stage."""

    table: dict[int, Counter] = defaultdict(Counter)
    for stage, posterior_id in zip(true_stages, posterior_ids):
        table[int(stage)][int(posterior_id)] += 1
    return {stage: dict(counts) for stage, counts in table.items()}


def posterior_purity(
    true_stages: Iterable[int],
    posterior_ids: Iterable[int],
) -> float:
    """Return the evaluator-only majority-stage purity of posterior assignments."""

    table = posterior_contingency(true_stages, posterior_ids)
    total = sum(sum(counts.values()) for counts in table.values())
    if total == 0:
        return float("nan")
    return float(sum(max(counts.values()) for counts in table.values()) / total)


def posterior_reuse_rate(
    assignments: Iterable[int],
    return_stage_start: int,
) -> float:
    """Measure how often the return regime uses any pre-return posterior."""

    values = list(assignments)
    if return_stage_start >= len(values):
        return float("nan")
    before = values[:return_stage_start]
    after = values[return_stage_start:]
    if not before or not after:
        return float("nan")
    old_posteriors = set(before)
    return float(np.mean([value in old_posteriors for value in after]))
