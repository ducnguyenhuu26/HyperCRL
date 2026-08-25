"""Posterior-predictive acquisition and reacquisition router."""

from collections import deque
from dataclasses import dataclass
from collections.abc import Callable

from pbcwm.core.types import Transition


@dataclass(frozen=True)
class PosteriorRouterDecision:
    current_nll: float
    best_stored_nll: float
    best_stored_posterior_id: int | None
    shift_triggered: bool
    reacquisition_triggered: bool
    acquisition_triggered: bool
    selected_posterior_id: int | None


class PosteriorPredictiveRouter:
    """Use persistent posterior-predictive NLL mismatch, without task IDs."""

    def __init__(
        self,
        window_size: int = 32,
        posterior_samples: int = 5,
        shift_threshold: float = 4.0,
        reuse_threshold: float = 3.0,
        consecutive_trigger_windows: int = 2,
        cooldown_steps: int = 32,
    ) -> None:
        if window_size <= 0 or posterior_samples <= 0:
            raise ValueError("window_size and posterior_samples must be positive")
        if shift_threshold < 0 or reuse_threshold < 0 or consecutive_trigger_windows <= 0 or cooldown_steps < 0:
            raise ValueError("invalid posterior router configuration")
        self.window_size = int(window_size)
        self.posterior_samples = int(posterior_samples)
        self.shift_threshold = float(shift_threshold)
        self.reuse_threshold = float(reuse_threshold)
        self.consecutive_trigger_windows = int(consecutive_trigger_windows)
        self.cooldown_steps = int(cooldown_steps)
        self.window: deque[Transition] = deque(maxlen=self.window_size)
        self.high_nll_windows = 0
        self.cooldown_remaining = 0
        self.switch_count = 0
        self.last_decision = PosteriorRouterDecision(0.0, float("inf"), None, False, False, False, None)

    @property
    def ready(self) -> bool:
        return len(self.window) >= self.window_size

    def add_transition(self, transition: Transition) -> None:
        self.window.append(transition)
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1

    def evaluate(
        self,
        current_id: int,
        stored_ids: list[int],
        nll_fn: Callable[[int, list[Transition], int], float],
    ) -> PosteriorRouterDecision:
        if not self.ready or self.cooldown_remaining > 0:
            return self._remember(PosteriorRouterDecision(0.0, float("inf"), None, False, False, False, None))
        transitions = list(self.window)
        current_nll = float(nll_fn(current_id, transitions, self.posterior_samples))
        self.high_nll_windows = self.high_nll_windows + 1 if current_nll > self.shift_threshold else 0
        if self.high_nll_windows < self.consecutive_trigger_windows:
            return self._remember(PosteriorRouterDecision(current_nll, float("inf"), None, False, False, False, None))
        scores = {posterior_id: float(nll_fn(posterior_id, transitions, self.posterior_samples)) for posterior_id in stored_ids}
        best_id, best_nll = min(scores.items(), key=lambda item: item[1])
        reacquire = best_id != current_id and best_nll < self.reuse_threshold
        acquire = not reacquire
        decision = PosteriorRouterDecision(
            current_nll,
            best_nll,
            best_id,
            True,
            reacquire,
            acquire,
            best_id if reacquire else None,
        )
        return self._remember(decision)

    def commit_switch(self) -> None:
        self.switch_count += 1
        self.high_nll_windows = 0
        self.cooldown_remaining = self.cooldown_steps
        self.window.clear()

    def state_dict(self) -> dict:
        return {
            "window": list(self.window),
            "high_nll_windows": self.high_nll_windows,
            "cooldown_remaining": self.cooldown_remaining,
            "switch_count": self.switch_count,
            "last_decision": self.last_decision,
        }

    def load_state_dict(self, state: dict) -> None:
        self.window.clear()
        self.window.extend(state["window"])
        self.high_nll_windows = int(state["high_nll_windows"])
        self.cooldown_remaining = int(state["cooldown_remaining"])
        self.switch_count = int(state["switch_count"])
        self.last_decision = state["last_decision"]

    def _remember(self, decision: PosteriorRouterDecision) -> PosteriorRouterDecision:
        self.last_decision = decision
        return decision
