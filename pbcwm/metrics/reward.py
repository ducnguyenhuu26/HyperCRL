"""Held-out preference accuracy and Bradley--Terry likelihood."""

from collections.abc import Sequence
from typing import Any

import torch

from pbcwm.preferences.types import PreferenceExample

from .common import MetricResult, PreferenceEvalBatch, invalid_result


def _examples(batch: PreferenceEvalBatch | Sequence[PreferenceExample]) -> tuple[list[Any], torch.Tensor]:
    if isinstance(batch, PreferenceEvalBatch):
        mask = batch.labels != -1
        return [(a, b) for a, b, keep in zip(batch.traj_a, batch.traj_b, mask.tolist()) if keep], batch.labels[mask].to(torch.float32)
    examples = list(batch)
    return [(example.traj_a, example.traj_b) for example in examples], torch.tensor([example.label for example in examples], dtype=torch.float32)


def _logits(reward_model: Any, pairs: Sequence[tuple[Any, Any]]) -> torch.Tensor:
    logits = []
    for traj_a, traj_b in pairs:
        if hasattr(reward_model, "preference_probabilities"):
            probability = reward_model.preference_probabilities(traj_a, traj_b).mean().clamp(1e-7, 1 - 1e-7)
            # NS-Gym/PB-CWM preference probabilities are P(A preferred),
            # while PreferenceExample labels use 1 == B preferred.
            logits.append(-torch.logit(probability))
        else:
            score_a = reward_model(traj_a)
            score_b = reward_model(traj_b)
            logits.append(torch.as_tensor(score_b - score_a, dtype=torch.float32))
    return torch.stack(logits) if logits else torch.empty(0)


def pairwise_preference_accuracy(reward_model: Any, batch: PreferenceEvalBatch | Sequence[PreferenceExample]) -> MetricResult:
    pairs, labels = _examples(batch)
    if not pairs:
        return invalid_result("reward/pairwise_accuracy", True, "no non-skipped preference pairs")
    logits = _logits(reward_model, pairs)
    value = float(((logits >= 0).to(torch.float32) == labels).to(torch.float32).mean())
    return MetricResult("reward/pairwise_accuracy", value, True, metadata={"sample_count": len(pairs), "skipped_count": len(batch.labels) - len(pairs) if isinstance(batch, PreferenceEvalBatch) else 0})


def bradley_terry_nll(reward_model: Any, batch: PreferenceEvalBatch | Sequence[PreferenceExample]) -> MetricResult:
    pairs, labels = _examples(batch)
    if not pairs:
        return invalid_result("reward/bt_nll", False, "no non-skipped preference pairs")
    logits = _logits(reward_model, pairs)
    value = float(torch.nn.functional.binary_cross_entropy_with_logits(logits, labels))
    return MetricResult("reward/bt_nll", value, False, metadata={"sample_count": len(pairs)})
