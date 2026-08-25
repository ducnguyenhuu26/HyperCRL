"""Observation boundary between NS-Gym and PB-CWM learners."""

from collections.abc import Mapping

import numpy as np

_FORBIDDEN_NS_KEYS = frozenset({"env_change", "delta_change", "relative_time"})


def extract_agent_state(observation: object) -> np.ndarray:
    """Return only NS-Gym's raw ``state`` field as a float32 copy.

    A plain array is accepted for regression-friendly adapters, but a mapping
    must contain ``state`` and is rejected if it has no raw-state field.
    Notification and relative-time fields are deliberately never propagated.
    """

    if isinstance(observation, Mapping):
        if "state" not in observation:
            raise ValueError("NS-Gym observation mapping has no state field")
        state = observation["state"]
    else:
        state = observation
    result = np.asarray(state, dtype=np.float32).copy()
    if result.ndim == 0 or not np.all(np.isfinite(result)):
        raise ValueError("agent state must be a finite non-scalar array")
    return result


def contains_notification_fields(observation: object) -> bool:
    """Return whether a raw observation still contains NS-Gym side channels."""

    return isinstance(observation, Mapping) and bool(_FORBIDDEN_NS_KEYS.intersection(observation))
