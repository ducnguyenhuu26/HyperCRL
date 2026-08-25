"""State-snapshot guard for evaluator-only execution."""

from collections.abc import Callable, Mapping
from contextlib import contextmanager
from copy import deepcopy
from typing import Any, Iterator


@contextmanager
def isolated_evaluation(snapshots: Mapping[str, Callable[[], Any]]) -> Iterator[None]:
    """Assert that evaluation callbacks did not mutate registered training state."""

    before = {name: deepcopy(getter()) for name, getter in snapshots.items()}
    yield
    after = {name: deepcopy(getter()) for name, getter in snapshots.items()}
    changed = [name for name in before if before[name] != after[name]]
    if changed:
        raise RuntimeError(f"evaluation mutated training state: {changed}")
