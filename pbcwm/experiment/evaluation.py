"""Evaluation helpers that restore the exact training state after probing."""

from __future__ import annotations

import copy
from collections.abc import Iterable
from typing import Any, Callable


def _checkpointable_targets(learner: Any, components: Iterable[Any]) -> list[Any]:
    targets: list[Any] = []
    seen: set[int] = set()
    candidates = [learner]
    if not hasattr(learner, "state_dict") and hasattr(learner, "dynamics"):
        candidates.append(learner.dynamics)
    candidates.extend(components)
    for candidate in candidates:
        if candidate is None or id(candidate) in seen:
            continue
        if hasattr(candidate, "state_dict") and hasattr(candidate, "load_state_dict"):
            targets.append(candidate)
            seen.add(id(candidate))
    return targets

def isolated_evaluation(
    learner: Any,
    evaluator: Callable[[Any], Any],
    *,
    components: Iterable[Any] = (),
) -> Any:
    """Run an evaluator and restore every supplied checkpointable component."""

    targets = _checkpointable_targets(learner, components)
    if not targets:
        raise TypeError("isolated evaluation requires checkpointable learner")
    snapshots = [(target, copy.deepcopy(target.state_dict())) for target in targets]
    try:
        return evaluator(learner)
    finally:
        for target, snapshot in snapshots:
            target.load_state_dict(snapshot)
