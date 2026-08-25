"""Small interfaces shared by benchmark adapters and evaluators.

The benchmark returns a normal Gymnasium reward for the evaluator.  Learner
transitions must be constructed with :func:`build_agent_transition`, which
intentionally sets that reward to zero until a separately learned preference
model is available.
"""

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from pbcwm.core.types import Transition


@dataclass(frozen=True)
class Regime:
    """One lifetime regime, identified by its first global environment step."""

    start_step: int
    parameters: Mapping[str, float]


@dataclass(frozen=True)
class BenchmarkSpec:
    """Validated, config-driven benchmark definition."""

    name: str
    provider: str
    base_env: str
    parameter: str
    regimes: tuple[Regime, ...]
    total_steps: int
    change_notification: bool = False
    delta_change_notification: bool = False
    fixed_parameters: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        if not self.regimes:
            raise ValueError("a benchmark needs at least one regime")
        starts = [regime.start_step for regime in self.regimes]
        if starts[0] != 0 or starts != sorted(set(starts)):
            raise ValueError("regime start steps must be unique, sorted, and start at 0")
        if any(step < 0 for step in starts) or self.total_steps <= 0:
            raise ValueError("steps must be non-negative and total_steps must be positive")
        if self.change_notification or self.delta_change_notification:
            raise ValueError("PB-CWM benchmark agents must not receive NS-Gym notifications")

    def regime_at(self, global_step: int) -> Regime:
        """Return the evaluator-only regime active at ``global_step``."""

        active = self.regimes[0]
        for regime in self.regimes[1:]:
            if regime.start_step > global_step:
                break
            active = regime
        return active


def build_agent_transition(
    obs: np.ndarray,
    action: np.ndarray,
    next_obs: np.ndarray,
    terminated: bool,
    truncated: bool,
) -> Transition:
    """Build the learner view without exposing the environment reward."""

    return Transition(
        obs=np.asarray(obs, dtype=np.float32).copy(),
        action=np.asarray(action, dtype=np.float32).copy(),
        next_obs=np.asarray(next_obs, dtype=np.float32).copy(),
        reward=0.0,
        terminated=bool(terminated),
        truncated=bool(truncated),
    )


class BenchmarkEnvProtocol:
    """Documentation-only protocol-like surface used by smoke/evaluator code."""

    global_env_step: int

    def evaluation_metadata(self) -> dict[str, Any]:  # pragma: no cover - interface only
        raise NotImplementedError
