"""Canonical W0-W4 component variants; no source edits or hidden switches."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pbcwm.baselines.static import StaticDynamicsLearner
from pbcwm.methods.radius import RadiusPbCWM, load_radius_config
from pbcwm.methods.radius.config import AblationConfig, AtlasConfig, RadiusConfig

VARIANT_NAMES = ("W0", "W1", "W2", "W3", "W4")


def variant_config(name: str, base: RadiusConfig | None = None) -> RadiusConfig | None:
    """Return the exact component switch configuration for W1-W4."""

    if name == "W0":
        return None
    if name not in VARIANT_NAMES:
        raise ValueError(f"unknown fixed-stream variant: {name}")
    config = base or load_radius_config(Path(__file__).parent / ".." / ".." / "configs" / "methods" / "radius.yaml")
    fixed_rank = name in {"W1", "W2"}
    max_rank = 3 if fixed_rank else 8
    return replace(
        config,
        atlas=replace(config.atlas, initial_rank=3, max_rank=max_rank),
        ablations=AblationConfig(
            disable_recurrent_memory=name == "W1",
            disable_rne=name in {"W1", "W2"},
            disable_pec=name in {"W1", "W2", "W3"},
            disable_pfpa=True,
            hard_context_routing=False,
            fixed_atlas_rank=fixed_rank,
        ),
    )


def build_variant(
    name: str,
    obs_dim: int,
    action_dim: int,
    *,
    action_low: np.ndarray | None = None,
    action_high: np.ndarray | None = None,
    device: str | torch.device = "cpu",
    seed: int = 0,
    base_config: RadiusConfig | None = None,
):
    """Build a learner from a named ablation, keeping W0 plain and shared."""

    if name not in VARIANT_NAMES:
        raise ValueError(f"unknown fixed-stream variant: {name}")
    if name == "W0":
        return StaticDynamicsLearner(
            obs_dim,
            action_dim,
            hidden_dims=(256, 256),
            learning_rate=3e-4,
            replay_capacity=50_000,
            batch_size=256,
            device=device,
            seed=seed,
        )
    scale = None
    if action_low is not None and action_high is not None:
        scale = np.maximum(np.abs(action_low), np.abs(action_high))
    return RadiusPbCWM(obs_dim, action_dim, variant_config(name, base_config), device=device, seed=seed, action_scale=scale)
