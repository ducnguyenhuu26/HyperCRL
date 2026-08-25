"""Evaluation helpers that restore the exact training state after probing."""

from __future__ import annotations

import copy
from typing import Any, Callable


def isolated_evaluation(learner: Any, evaluator: Callable[[Any], Any]) -> Any:
    """Run an evaluator and restore learner state even if it mutates it."""

    target = learner
    if not hasattr(target, "state_dict") and hasattr(learner, "dynamics"):
        target = learner.dynamics
    if not hasattr(target, "state_dict") or not hasattr(target, "load_state_dict"):
        raise TypeError("isolated evaluation requires checkpointable learner")
    snapshot = copy.deepcopy(target.state_dict())
    try:
        return evaluator(learner)
    finally:
        target.load_state_dict(snapshot)
