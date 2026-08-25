"""Common NS-Gym dependency and seed helpers."""

from importlib.metadata import PackageNotFoundError, version
import sys
from typing import Any

import numpy as np


def installed_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "not-installed"


def seed_streams(root_seed: int) -> dict[str, int]:
    """Derive independent deterministic streams from one user-visible seed."""

    sequence = np.random.SeedSequence(int(root_seed))
    children = sequence.spawn(3)
    return {
        name: int(child.generate_state(1, dtype=np.uint32)[0])
        for name, child in zip(("environment", "agent", "evaluator"), children)
    }


def dependency_metadata() -> dict[str, Any]:
    import gymnasium
    import mujoco
    import ns_gym
    import torch

    return {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "gymnasium": gymnasium.__version__,
        "mujoco": mujoco.__version__,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "ns_gym": installed_version("ns-gym"),
        "ns_gym_upstream_commit": None,
        "ns_gym_install_source": "PyPI wheel; upstream commit not applicable",
        "ns_gym_module": ns_gym.__file__,
    }
