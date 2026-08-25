"""Canonical NS-Gym Pendulum adapter with a lifetime clock."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from ns_gym.schedulers import DiscreteScheduler
from ns_gym.update_functions import StepWiseUpdate
from ns_gym.wrappers import NSClassicControlWrapper

from ..base import BenchmarkSpec
from ..observation_adapter import extract_agent_state
from ..oracle import BenchmarkOracle
from .metadata import collect_metadata
from .common import seed_streams


class NSGymPendulumBenchmark(gym.Env):
    """Gym facade exposing raw state while NS-Gym owns parameter updates."""

    metadata = {"render_modes": []}

    def __init__(self, nsgym_env: NSClassicControlWrapper, spec: BenchmarkSpec, root_seed: int):
        self.nsgym_env = nsgym_env
        self.spec = spec
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
        # NS-Gym's wrapper clock is episode-local.  Its scheduler/update-function
        # API remains the source of truth; this assignment supplies lifetime time.
        self.nsgym_env.t = self.global_env_step

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        environment_seed = self.seed_streams["environment"] if seed is None and not self._has_initial_seed else seed
        if seed is not None:
            super().reset(seed=seed)
        raw_obs, info = self.nsgym_env.reset(seed=environment_seed, options=options)
        self._has_initial_seed = True
        self._set_lifetime_clock()
        self.episode_step = 0
        self.oracle.reset_episode()
        state = extract_agent_state(raw_obs)
        return state, {"benchmark_global_step": self.global_env_step}

    def step(self, action: np.ndarray):
        self._set_lifetime_clock()
        raw_obs, true_reward, terminated, truncated, _info = self.nsgym_env.step(action)
        self.global_env_step += 1
        self.episode_step += 1
        self.oracle.record_reward(true_reward)
        state = extract_agent_state(raw_obs)
        active = self.spec.regime_at(max(0, self.global_env_step - 1))
        info = {
            "benchmark_global_step": self.global_env_step,
            "episode_step": self.episode_step,
            "evaluation_only": {
                "true_reward": float(true_reward),
                "regime_start_step": active.start_step,
                "parameters": dict(active.parameters),
            },
        }
        return state, float(true_reward), terminated, truncated, info

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


def _validate_pendulum_spec(spec: BenchmarkSpec) -> None:
    if spec.base_env != "Pendulum-v1" or spec.parameter != "m":
        raise ValueError("the Phase 1 adapter supports only Pendulum-v1 mass schedules")
    if any(set(regime.parameters) != {"m"} for regime in spec.regimes):
        raise ValueError("Pendulum mass regimes must contain only m; l/g/dt stay fixed")
    if spec.fixed_parameters and set(spec.fixed_parameters) != {"l", "g", "dt"}:
        raise ValueError("Pendulum fixed parameters must be exactly l, g, and dt")


def make_pendulum_benchmark(spec: BenchmarkSpec, root_seed: int = 0) -> NSGymPendulumBenchmark:
    _validate_pendulum_spec(spec)
    base_env = gym.make(spec.base_env)
    unwrapped = base_env.unwrapped
    initial = spec.regimes[0].parameters["m"]
    setattr(unwrapped, "m", initial)
    for name, value in (spec.fixed_parameters or {}).items():
        setattr(unwrapped, name, value)

    transitions = spec.regimes[1:]
    scheduler = DiscreteScheduler({regime.start_step for regime in transitions})
    update = StepWiseUpdate(scheduler, [regime.parameters["m"] for regime in transitions])
    wrapped = NSClassicControlWrapper(
        base_env,
        {"m": update},
        change_notification=False,
        delta_change_notification=False,
        persistent_params=True,
    )
    return NSGymPendulumBenchmark(wrapped, spec, root_seed)
