"""Pendulum wrapper with hidden, scheduled physical parameter changes."""

from collections.abc import Mapping, Sequence
from copy import deepcopy

import gymnasium as gym


class NonstationaryPendulum(gym.Wrapper):
    """Change ``m``, ``l`` and/or ``g`` at global step thresholds."""

    def __init__(
        self,
        dynamics_schedule: Sequence[Mapping[str, float]],
        env: gym.Env | None = None,
    ) -> None:
        if not dynamics_schedule:
            raise ValueError("dynamics_schedule must contain at least one entry")
        schedule = [dict(entry) for entry in dynamics_schedule]
        if any("step" not in entry for entry in schedule):
            raise ValueError("each schedule entry needs a step")
        if any(int(entry["step"]) < 0 for entry in schedule):
            raise ValueError("schedule steps must be non-negative")
        if any(schedule[i]["step"] > schedule[i + 1]["step"] for i in range(len(schedule) - 1)):
            raise ValueError("dynamics_schedule must be sorted by step")

        super().__init__(env or gym.make("Pendulum-v1"))
        self.dynamics_schedule = schedule
        self.global_step = 0
        self._next_schedule_index = 0
        self._current_stage = -1
        self._current_params: dict[str, float] = {}
        self._apply_due_parameters()

    @property
    def current_stage(self) -> int:
        return self._current_stage

    @property
    def current_params(self) -> dict[str, float]:
        return dict(self._current_params)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self._apply_due_parameters()
        observation, info = super().reset(seed=seed, options=options)
        return observation, self._with_diagnostics(info)

    def step(self, action):
        self._apply_due_parameters()
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.global_step += 1
        return observation, reward, terminated, truncated, self._with_diagnostics(info)

    def _apply_due_parameters(self) -> None:
        while (
            self._next_schedule_index < len(self.dynamics_schedule)
            and int(self.dynamics_schedule[self._next_schedule_index]["step"]) <= self.global_step
        ):
            entry = self.dynamics_schedule[self._next_schedule_index]
            parameters = {
                key: float(value)
                for key, value in entry.items()
                if key in {"mass", "length", "gravity"}
            }
            attribute_names = {"mass": "m", "length": "l", "gravity": "g"}
            for key, value in parameters.items():
                setattr(self.unwrapped, attribute_names[key], value)
            self._current_stage = self._next_schedule_index
            self._current_params = parameters
            self._next_schedule_index += 1

    def _with_diagnostics(self, info: dict) -> dict:
        diagnostic = dict(info)
        diagnostic["true_dynamics_stage"] = self._current_stage
        diagnostic["true_dynamics_params"] = deepcopy(self._current_params)
        return diagnostic
