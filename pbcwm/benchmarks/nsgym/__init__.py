"""NS-Gym-backed benchmark implementations."""

from .classic_control import NSGymPendulumBenchmark, make_pendulum_benchmark

__all__ = ["NSGymPendulumBenchmark", "make_pendulum_benchmark"]
