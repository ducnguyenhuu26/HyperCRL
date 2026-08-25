"""Transparent residual-based boundary-free regime router."""

from collections import deque
from dataclasses import dataclass
from typing import Callable

from pbcwm.core.types import Transition


@dataclass(frozen=True)
class RouterDecision:
    current_error: float
    best_stored_error: float
    best_stored_embedding_id: int | None
    shift_triggered: bool
    reuse_triggered: bool
    new_embedding_triggered: bool
    selected_embedding_id: int | None


class ResidualRegimeRouter:
    """Use persistent recent residuals to reuse or create an embedding."""

    def __init__(
        self,
        window_size: int = 32,
        shift_threshold: float = 0.05,
        reuse_threshold: float = 0.03,
        consecutive_trigger_windows: int = 2,
        cooldown_steps: int = 32,
    ) -> None:
        if window_size <= 0 or shift_threshold < 0 or reuse_threshold < 0:
            raise ValueError("invalid router window or thresholds")
        if consecutive_trigger_windows <= 0 or cooldown_steps < 0:
            raise ValueError("invalid router persistence or cooldown")
        self.window_size = int(window_size)
        self.shift_threshold = float(shift_threshold)
        self.reuse_threshold = float(reuse_threshold)
        self.consecutive_trigger_windows = int(consecutive_trigger_windows)
        self.cooldown_steps = int(cooldown_steps)
        self.window: deque[Transition] = deque(maxlen=self.window_size)
        self.total_seen = 0
        self.high_residual_windows = 0
        self.cooldown_remaining = 0
        self.switch_count = 0
        self.last_decision = RouterDecision(0.0, float("inf"), None, False, False, False, None)

    @property
    def ready(self) -> bool:
        return len(self.window) >= self.window_size

    def add_transition(self, transition: Transition) -> None:
        self.window.append(transition)
        self.total_seen += 1
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1

    def evaluate(
        self,
        current_embedding_id: int,
        stored_embedding_ids: list[int],
        error_fn: Callable[[int, list[Transition]], float],
    ) -> RouterDecision:
        if not self.ready or self.cooldown_remaining > 0:
            return self._remember(RouterDecision(0.0, float("inf"), None, False, False, False, None))

        transitions = list(self.window)
        current_error = float(error_fn(current_embedding_id, transitions))
        if current_error > self.shift_threshold:
            self.high_residual_windows += 1
        else:
            self.high_residual_windows = 0
        if self.high_residual_windows < self.consecutive_trigger_windows:
            return self._remember(RouterDecision(current_error, float("inf"), None, False, False, False, None))

        errors = {embedding_id: float(error_fn(embedding_id, transitions)) for embedding_id in stored_embedding_ids}
        best_id, best_error = min(errors.items(), key=lambda item: item[1])
        reuse = best_id != current_embedding_id and best_error < self.reuse_threshold
        new_embedding = not reuse
        decision = RouterDecision(
            current_error=current_error,
            best_stored_error=best_error,
            best_stored_embedding_id=best_id,
            shift_triggered=True,
            reuse_triggered=reuse,
            new_embedding_triggered=new_embedding,
            selected_embedding_id=best_id if reuse else None,
        )
        return self._remember(decision)

    def commit_switch(self) -> None:
        self.switch_count += 1
        self.high_residual_windows = 0
        self.cooldown_remaining = self.cooldown_steps
        self.window.clear()

    def state_dict(self) -> dict:
        return {
            "window": list(self.window),
            "total_seen": self.total_seen,
            "high_residual_windows": self.high_residual_windows,
            "cooldown_remaining": self.cooldown_remaining,
            "switch_count": self.switch_count,
            "last_decision": self.last_decision,
        }

    def load_state_dict(self, state: dict) -> None:
        self.window.clear()
        self.window.extend(state["window"])
        self.total_seen = int(state["total_seen"])
        self.high_residual_windows = int(state["high_residual_windows"])
        self.cooldown_remaining = int(state["cooldown_remaining"])
        self.switch_count = int(state["switch_count"])
        self.last_decision = state["last_decision"]

    def _remember(self, decision: RouterDecision) -> RouterDecision:
        self.last_decision = decision
        return decision
