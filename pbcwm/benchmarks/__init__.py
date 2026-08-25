"""External benchmark integrations kept separate from PB-CWM baselines."""

from .base import BenchmarkSpec, build_agent_transition
from .registry import available_benchmarks, make_benchmark

__all__ = [
    "BenchmarkSpec",
    "available_benchmarks",
    "build_agent_transition",
    "make_benchmark",
]
