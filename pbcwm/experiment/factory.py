"""Single method factory used by the protocol runner."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

from pbcwm.baselines.static import StaticDynamicsLearner
from pbcwm.methods.radius import RadiusPbCWM
from pbcwm.methods.radius.config import radius_config_from_mapping

CANONICAL_METHODS = (
    "static",
    "moprl_online_ft",
    "gpmm",
    "hypercrl_adapt",
    "vblrl_adapt",
    "curious_replay_adapt",
    "radius_pb_cwm",
)


def _planner_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    raw = dict(config or {})
    result = {
        "horizon": int(raw.get("horizon", 8)),
        "population_size": int(raw.get("population_size", raw.get("population", 64))),
        "elite_size": int(raw.get("elite_size", raw.get("elite_count", 8))),
        "num_iterations": int(raw.get("num_iterations", raw.get("iterations", 2))),
        "initial_std": raw.get("initial_std", 1.0),
        "discount": float(raw.get("discount", 1.0)),
    }
    if "seed" in raw:
        result["seed"] = int(raw["seed"])
    return result


def _preference_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    raw = dict(config or {})
    return {
        "ensemble_size": int(raw.get("ensemble_size", 3)),
        "hidden_dims": tuple(raw.get("hidden_dims", (64, 64))),
        "learning_rate": float(raw.get("learning_rate", 3e-4)),
        "reward_batch_size": int(raw.get("reward_batch_size", 8)),
        "min_preferences_before_planning": int(raw.get("min_preferences_before_planning", 0)),
        "pair_pool_size": int(raw.get("pair_pool_size", 64)),
        "teacher_skip_margin": float(raw.get("teacher_skip_margin", 0.0)),
        "candidate_keep_per_iteration": int(raw.get("candidate_keep_per_iteration", 8)),
        "candidate_keep_final_elites": int(raw.get("candidate_keep_final_elites", 8)),
    }


def build_method(
    name: str,
    *,
    obs_dim: int,
    action_dim: int,
    action_low: np.ndarray | None = None,
    action_high: np.ndarray | None = None,
    config: Mapping[str, Any] | None = None,
    device: str | torch.device = "cpu",
    seed: int | None = None,
    teacher_reward: Any | None = None,
):
    """Instantiate one canonical method name through a shared entrypoint."""

    canonical = str(name).lower()
    if canonical not in CANONICAL_METHODS:
        raise ValueError(f"unknown canonical method: {name}")
    raw = dict(config or {})
    if canonical == "radius_pb_cwm":
        radius_config = raw.get("radius", raw.get("method", raw))
        scale = None
        if action_low is not None and action_high is not None:
            scale = np.maximum(np.abs(np.asarray(action_low)), np.abs(np.asarray(action_high)))
        return RadiusPbCWM(obs_dim, action_dim, radius_config_from_mapping(radius_config), device=device, seed=seed, action_scale=scale)
    if canonical == "static":
        model = dict(raw.get("model", {}))
        training = dict(raw.get("training", {}))
        return StaticDynamicsLearner(obs_dim, action_dim, hidden_dims=tuple(model.get("hidden_dims", (128, 128))), learning_rate=float(model.get("learning_rate", 1e-3)), replay_capacity=int(training.get("replay_capacity", 100000)), batch_size=int(training.get("batch_size", 256)), device=device, seed=seed)
    if action_low is None or action_high is None or teacher_reward is None:
        raise ValueError(f"{canonical} requires action bounds and teacher_reward")
    planner = _planner_config(raw.get("planner"))
    preference = _preference_config(raw.get("preference"))
    model = dict(raw.get("model", {}))
    sections = {
        "moprl_online_ft": ("pbcwm.baselines.moprl_online_ft", "MoPRLOnlineFT", "moprl"),
        "gpmm": ("pbcwm.baselines.gpmm.online", "GPMMOnline", "gpmm"),
        "hypercrl_adapt": ("pbcwm.baselines.hypercrl.online", "HyperCRLAdaptOnline", "hypercrl"),
        "vblrl_adapt": ("pbcwm.baselines.vblrl.online", "VBLRLAdaptOnline", "vblrl"),
        "curious_replay_adapt": ("pbcwm.baselines.curious_replay.online", "CuriousReplayOnline", "curious_replay"),
    }
    module_name, class_name, section_name = sections[canonical]
    module = __import__(module_name, fromlist=[class_name])
    cls = getattr(module, class_name)
    method_config = dict(raw.get(section_name, {}))
    kwargs = {
        "obs_dim": obs_dim, "action_dim": action_dim, "action_low": action_low, "action_high": action_high,
        "planner_config": planner, "preference_config": preference, "teacher_reward": teacher_reward,
        "device": device, "seed": seed,
    }
    if canonical == "moprl_online_ft":
        # MoP-RL predates the grouped preference config used by the other
        # online adapters.  Expand the shared protocol fields to its explicit
        # constructor names instead of passing an unsupported
        # ``preference_config`` keyword.
        kwargs.pop("preference_config")
        kwargs.update(
            model_hidden_dims=tuple(model.get("hidden_dims", (128, 128))),
            model_learning_rate=float(model.get("learning_rate", 1e-3)),
            preference_ensemble_size=preference["ensemble_size"],
            preference_hidden_dims=tuple(preference["hidden_dims"]),
            preference_learning_rate=preference["learning_rate"],
            preference_batch_size=preference["reward_batch_size"],
            min_preferences_before_planning=preference["min_preferences_before_planning"],
            pair_pool_size=preference["pair_pool_size"],
            teacher_skip_margin=preference["teacher_skip_margin"],
            **method_config,
        )
    else:
        kwargs[section_name + "_config"] = method_config
    return cls(**kwargs)
