import numpy as np

from pbcwm.benchmarks.base import BenchmarkSpec, Regime, build_agent_transition
from pbcwm.benchmarks.registry import available_benchmarks, make_benchmark


def hopper_spec() -> BenchmarkSpec:
    return BenchmarkSpec(
        name="nsgym/hopper-physics-abrupt-return-v0",
        provider="nsgym",
        base_env="Hopper-v5",
        parameter="hopper_physics",
        regimes=(
            Regime(0, {"torso_mass": 1.0, "floor_friction": 1.0, "thigh_joint_damping": 1.0}),
            Regime(2, {"torso_mass": 1.25, "floor_friction": 1.0, "thigh_joint_damping": 1.0}),
            Regime(4, {"torso_mass": 1.0, "floor_friction": 1.0, "thigh_joint_damping": 1.0}),
        ),
        total_steps=6,
    )


def test_hopper_adapter_registry_notifications_and_mid_episode_change():
    assert hopper_spec().name in available_benchmarks()
    env = make_benchmark(hopper_spec().name, hopper_spec(), root_seed=4)
    try:
        obs, _ = env.reset(seed=4)
        assert env.nsgym_env.change_notification is False
        assert env.nsgym_env.delta_change_notification is False
        initial_mass = float(env.nsgym_env._get_param_value("torso_mass"))
        for _ in range(3):
            next_obs, _reward, terminated, truncated, info = env.step(np.zeros(3, dtype=np.float32))
            assert not terminated and not truncated
            transition = build_agent_transition(obs, np.zeros(3, dtype=np.float32), next_obs, terminated, truncated)
            assert transition.reward == 0.0
            assert not hasattr(transition, "parameters")
            obs = next_obs
        assert env.global_env_step == 3
        assert env.episode_step == 3
        assert float(env.nsgym_env._get_param_value("torso_mass")) > initial_mass
        assert info["evaluation_only"]["parameters"]["torso_mass"] == 1.25
    finally:
        env.close()


def test_hopper_natural_reset_is_not_reseeded_to_same_initial_state():
    env = make_benchmark(hopper_spec().name, hopper_spec(), root_seed=8)
    try:
        first, _ = env.reset()
        second, _ = env.reset()
        assert not np.array_equal(first, second)
    finally:
        env.close()
