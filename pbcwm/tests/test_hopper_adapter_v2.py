import numpy as np

from pbcwm.benchmarks.base import BenchmarkSpec, Regime, build_agent_transition
from pbcwm.benchmarks.registry import available_benchmarks, make_benchmark
from pbcwm.experiments.radius_validation.probes import generate_probe_bank


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


def _stationary_hopper_spec(parameters: dict[str, float]) -> BenchmarkSpec:
    return BenchmarkSpec(
        name="nsgym/hopper-physics-abrupt-return-v0",
        provider="nsgym",
        base_env="Hopper-v5",
        parameter="hopper_physics",
        regimes=(Regime(0, parameters),),
        total_steps=8,
    )


def _model_values(env) -> tuple[float, float, float]:
    model = env.unwrapped.model
    return (
        float(model.body_mass[model.body("torso").id]),
        float(model.geom_friction[model.geom("floor").id, 0]),
        float(model.dof_damping[model.joint("thigh_joint").id]),
    )


def test_hopper_single_regime_applies_initial_physics_to_simulator():
    scales = {"torso_mass": 1.25, "floor_friction": 0.75, "thigh_joint_damping": 1.25}
    baseline_env = make_benchmark(hopper_spec().name, _stationary_hopper_spec({key: 1.0 for key in scales}), root_seed=0)
    scaled_env = make_benchmark(hopper_spec().name, _stationary_hopper_spec(scales), root_seed=0)
    try:
        baseline = _model_values(baseline_env)
        scaled = _model_values(scaled_env)
        assert np.allclose(scaled, np.asarray(baseline) * np.array([1.25, 0.75, 1.25]), rtol=1e-6, atol=1e-8)
        assert np.allclose(_model_values(baseline_env), baseline, rtol=1e-7, atol=1e-9)
    finally:
        baseline_env.close()
        scaled_env.close()


def test_hopper_probe_banks_with_same_seed_differ_by_physics():
    parameters = {
        "P0": {"torso_mass": 1.0, "floor_friction": 1.0, "thigh_joint_damping": 1.0},
        "A": {"torso_mass": 1.25, "floor_friction": 1.0, "thigh_joint_damping": 1.0},
        "B": {"torso_mass": 1.0, "floor_friction": 0.75, "thigh_joint_damping": 1.0},
        "C": {"torso_mass": 1.0, "floor_friction": 1.0, "thigh_joint_damping": 1.25},
    }
    banks = {
        role: generate_probe_bank(
            role,
            lambda role=role: make_benchmark(hopper_spec().name, _stationary_hopper_spec(parameters[role]), root_seed=0),
            seed=123,
            n_probes=2,
            horizon=4,
        )
        for role in parameters
    }
    for role in ("A", "C"):
        assert any(not np.allclose(banks["P0"].probes[i].true_obs, banks[role].probes[i].true_obs, rtol=1e-7, atol=1e-7) for i in range(2))
