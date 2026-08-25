"""Canonical NS-Gym Hopper adapter used by RADIUS component validation."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from ns_gym.schedulers import DiscreteScheduler
from ns_gym.update_functions import StepWiseUpdate
from ns_gym.wrappers import MujocoWrapper

from ..base import BenchmarkSpec
from ..observation_adapter import extract_agent_state
from ..oracle import BenchmarkOracle
from .common import seed_streams
from .metadata import collect_metadata

__all__ = ["MujocoWrapper", "make_mujoco_benchmark", "NSGymHopperBenchmark"]


class NSGymHopperBenchmark(gym.Env):
    """Gym facade with lifetime scheduling and evaluator-only physics metadata."""

    metadata = {"render_modes": []}

    def __init__(self, nsgym_env: MujocoWrapper, spec: BenchmarkSpec, root_seed: int, base_values: dict[str, float]):
        self.nsgym_env = nsgym_env
        self.spec = spec
        self.base_values = dict(base_values)
        self.root_seed = int(root_seed)
        self.seed_streams = seed_streams(self.root_seed)
        self.oracle = BenchmarkOracle()
        self.global_env_step = 0
        self.episode_step = 0
        self._has_initial_seed = False
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=tuple(nsgym_env.env.observation_space.shape),
            dtype=np.float32,
        )
        self.action_space = nsgym_env.action_space
        self.action_space.seed(self.seed_streams["agent"])

    @property
    def unwrapped(self):
        return self.nsgym_env.unwrapped

    def _set_lifetime_clock(self) -> None:
        self.nsgym_env.t = self.global_env_step

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        environment_seed = self.seed_streams["environment"] if seed is None and not self._has_initial_seed else seed
        if seed is not None:
            super().reset(seed=seed)
        raw_obs, _info = self.nsgym_env.reset(seed=environment_seed, options=options)
        self._has_initial_seed = True
        active = self.spec.regime_at(self.global_env_step)
        _apply_hopper_parameters(self.nsgym_env.unwrapped.model, self.nsgym_env.unwrapped.data, self.base_values, dict(active.parameters))
        self._set_lifetime_clock()
        self.episode_step = 0
        self.oracle.reset_episode()
        return extract_agent_state(raw_obs), {"benchmark_global_step": self.global_env_step}

    def step(self, action: np.ndarray):
        self._set_lifetime_clock()
        raw_obs, true_reward, terminated, truncated, _info = self.nsgym_env.step(action)
        self.global_env_step += 1
        self.episode_step += 1
        self.oracle.record_reward(true_reward)
        active = self.spec.regime_at(max(0, self.global_env_step - 1))
        return (
            extract_agent_state(raw_obs),
            float(true_reward),
            terminated,
            truncated,
            {
                "benchmark_global_step": self.global_env_step,
                "episode_step": self.episode_step,
                "evaluation_only": {
                    "true_reward": float(true_reward),
                    "regime_start_step": active.start_step,
                    "parameters": dict(active.parameters),
                },
            },
        )

    def evaluation_metadata(self) -> dict[str, Any]:
        active = self.spec.regime_at(self.global_env_step)
        return {
            **collect_metadata(self.spec, self.root_seed),
            "global_env_step": self.global_env_step,
            "episode_step": self.episode_step,
            "active_regime_start_step": active.start_step,
            "active_regime_parameters": dict(active.parameters),
            "oracle_trajectory_return": self.oracle.trajectory_return,
        }

    def close(self) -> None:
        self.nsgym_env.close()


_HOPPER_PARAMS = ("torso_mass", "floor_friction", "thigh_joint_damping")


def _validate_hopper_spec(spec: BenchmarkSpec) -> None:
    if spec.base_env != "Hopper-v5":
        raise ValueError("the MuJoCo adapter supports only Hopper-v5")
    if set(spec.regimes[0].parameters) != set(_HOPPER_PARAMS):
        raise ValueError(f"Hopper regimes must contain exactly {_HOPPER_PARAMS}")
    if any(set(regime.parameters) != set(_HOPPER_PARAMS) for regime in spec.regimes):
        raise ValueError(f"Hopper regimes must contain exactly {_HOPPER_PARAMS}")
    if any(any(value <= 0.0 for value in regime.parameters.values()) for regime in spec.regimes):
        raise ValueError("Hopper parameter scales must be positive")


def _apply_hopper_parameters(model, data, base_values: dict[str, float], parameters: dict[str, float]) -> None:
    """Apply the initial physical regime directly to the MuJoCo model."""

    torso_id = model.body("torso").id
    floor_id = model.geom("floor").id
    thigh_id = model.joint("thigh_joint").id
    model.body_mass[torso_id] = base_values["torso_mass"] * float(parameters["torso_mass"])
    model.geom_friction[floor_id, 0] = base_values["floor_friction"] * float(parameters["floor_friction"])
    model.dof_damping[thigh_id] = base_values["thigh_joint_damping"] * float(parameters["thigh_joint_damping"])
    mujoco.mj_forward(model, data)


def make_mujoco_benchmark(spec: BenchmarkSpec, root_seed: int = 0) -> NSGymHopperBenchmark:
    """Build Hopper with regime values interpreted as positive parameter scales."""

    _validate_hopper_spec(spec)
    base_env = gym.make(spec.base_env)
    probe = MujocoWrapper(base_env, {}, change_notification=False, delta_change_notification=False, persistent_params=True)
    base_values = {name: float(probe.unwrapped.model.body_mass[probe.unwrapped.model.body("torso").id]) if name == "torso_mass" else float(probe.unwrapped.model.geom_friction[probe.unwrapped.model.geom("floor").id, 0]) if name == "floor_friction" else float(probe.unwrapped.model.dof_damping[probe.unwrapped.model.joint("thigh_joint").id]) for name in _HOPPER_PARAMS}
    probe.close()

    # NS-Gym's DiscreteScheduler rejects an empty event set.  Keep a harmless
    # event strictly after the requested lifetime for a stationary smoke.
    change_steps = {regime.start_step for regime in spec.regimes[1:]} or {spec.total_steps + 1}
    scheduler = DiscreteScheduler(change_steps)
    tunable = {
        name: StepWiseUpdate(scheduler, [base_values[name] * regime.parameters[name] for regime in spec.regimes[1:]])
        for name in _HOPPER_PARAMS
    }
    wrapped = MujocoWrapper(
        gym.make(spec.base_env),
        tunable,
        change_notification=False,
        delta_change_notification=False,
        persistent_params=True,
    )
    _apply_hopper_parameters(wrapped.unwrapped.model, wrapped.unwrapped.data, base_values, dict(spec.regimes[0].parameters))
    return NSGymHopperBenchmark(wrapped, spec, root_seed, base_values)
