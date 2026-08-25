"""Evaluation-only diagnostics for latent GPMM assignments."""

from collections import Counter, defaultdict
from collections.abc import Iterable

import numpy as np


def assignment_contingency(
    true_stages: Iterable[int],
    inferred_experts: Iterable[int],
) -> dict[int, dict[int, int]]:
    table: dict[int, Counter] = defaultdict(Counter)
    for stage, expert in zip(true_stages, inferred_experts):
        table[int(stage)][int(expert)] += 1
    return {stage: dict(counts) for stage, counts in table.items()}


def assignment_purity(
    true_stages: Iterable[int],
    inferred_experts: Iterable[int],
) -> float:
    table = assignment_contingency(true_stages, inferred_experts)
    total = sum(sum(counts.values()) for counts in table.values())
    if total == 0:
        return float("nan")
    return sum(max(counts.values()) for counts in table.values()) / total


def expert_reuse_rate(assignments: Iterable[int], return_stage_start: int) -> float:
    values = list(assignments)
    if return_stage_start >= len(values):
        return float("nan")
    before = values[:return_stage_start]
    after = values[return_stage_start:]
    if not before or not after:
        return float("nan")
    old_experts = set(before)
    return float(np.mean([value in old_experts for value in after]))
