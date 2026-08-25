"""Named benchmark registry; no baseline runner is modified here."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from .base import BenchmarkSpec
from .nsgym.classic_control import make_pendulum_benchmark
from .nsgym.mujoco import make_mujoco_benchmark

_PENDULUM_NAME = "nsgym/pendulum-mass-abrupt-return-v0"
_HOPPER_NAME = "nsgym/hopper-physics-abrupt-return-v0"


def available_benchmarks() -> tuple[str, ...]:
    return (_PENDULUM_NAME, _HOPPER_NAME)


def load_benchmark_spec(path: str | Path) -> BenchmarkSpec:
    with Path(path).open("r", encoding="utf-8") as handle:
        data: Mapping[str, Any] = yaml.safe_load(handle)
    from .schedules import benchmark_spec_from_mapping

    return benchmark_spec_from_mapping(data)


def make_benchmark(name: str, spec: BenchmarkSpec, root_seed: int = 0):
    if name == _PENDULUM_NAME and spec.name == _PENDULUM_NAME:
        return make_pendulum_benchmark(spec, root_seed=root_seed)
    if name == _HOPPER_NAME and spec.name == _HOPPER_NAME:
        return make_mujoco_benchmark(spec, root_seed=root_seed)
    raise KeyError(f"unknown benchmark: {name}")
