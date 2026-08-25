"""Named benchmark registry; no baseline runner is modified here."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from .base import BenchmarkSpec
from .nsgym.classic_control import make_pendulum_benchmark

_PENDULUM_NAME = "nsgym/pendulum-mass-abrupt-return-v0"


def available_benchmarks() -> tuple[str, ...]:
    return (_PENDULUM_NAME,)


def load_benchmark_spec(path: str | Path) -> BenchmarkSpec:
    with Path(path).open("r", encoding="utf-8") as handle:
        data: Mapping[str, Any] = yaml.safe_load(handle)
    from .schedules import benchmark_spec_from_mapping

    return benchmark_spec_from_mapping(data)


def make_benchmark(name: str, spec: BenchmarkSpec, root_seed: int = 0):
    if name != _PENDULUM_NAME or spec.name != _PENDULUM_NAME:
        raise KeyError(f"unknown benchmark: {name}")
    return make_pendulum_benchmark(spec, root_seed=root_seed)
