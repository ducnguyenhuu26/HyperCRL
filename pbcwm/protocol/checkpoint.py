"""Checkpoint writer with a stable protocol metadata envelope."""

from pathlib import Path
from typing import Any, Mapping

import torch

REQUIRED_CHECKPOINT_FIELDS = frozenset({
    "world_model_state",
    "reward_model_state",
    "replay_or_memory_state",
    "method_state",
    "normalizer_state",
    "rng_states",
    "global_lifetime_step",
})


def save_checkpoint(path: str | Path, state: Mapping[str, Any], metadata: Mapping[str, Any]) -> None:
    missing = REQUIRED_CHECKPOINT_FIELDS.difference(state)
    if missing:
        raise ValueError(f"checkpoint is missing required fields: {sorted(missing)}")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"metadata": dict(metadata), "state": dict(state)}, target)
