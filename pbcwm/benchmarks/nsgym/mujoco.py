"""MuJoCo API boundary reserved for a later benchmark phase."""

from ns_gym.wrappers import MujocoWrapper

__all__ = ["MujocoWrapper", "make_mujoco_benchmark"]


def make_mujoco_benchmark(*args, **kwargs):
    raise NotImplementedError(
        "MuJoCo NS-Gym benchmarks are intentionally deferred until the canonical Pendulum adapter is frozen"
    )
