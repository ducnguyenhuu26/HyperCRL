"""Evaluator-only metrics for HyperCRL-Adapt embedding assignments."""

from collections import Counter, defaultdict
from collections.abc import Iterable

import numpy as np


def embedding_contingency(
    true_stages: Iterable[int],
    inferred_embeddings: Iterable[int],
) -> dict[int, dict[int, int]]:
    table: dict[int, Counter] = defaultdict(Counter)
    for stage, embedding in zip(true_stages, inferred_embeddings):
        table[int(stage)][int(embedding)] += 1
    return {stage: dict(counts) for stage, counts in table.items()}


def embedding_purity(
    true_stages: Iterable[int],
    inferred_embeddings: Iterable[int],
) -> float:
    table = embedding_contingency(true_stages, inferred_embeddings)
    total = sum(sum(counts.values()) for counts in table.values())
    if total == 0:
        return float("nan")
    return float(sum(max(counts.values()) for counts in table.values()) / total)


def embedding_reuse_rate(
    assignments: Iterable[int],
    return_stage_start: int,
) -> float:
    values = list(assignments)
    if return_stage_start >= len(values):
        return float("nan")
    old_embeddings = set(values[:return_stage_start])
    returned = values[return_stage_start:]
    if not old_embeddings or not returned:
        return float("nan")
    return float(np.mean([value in old_embeddings for value in returned]))
