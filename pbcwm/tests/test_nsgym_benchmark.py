"""Acceptance tests for the Phase 1 external NS-Gym import."""

import numpy as np

from pbcwm.benchmarks.base import Regime, BenchmarkSpec, build_agent_transition
from pbcwm.benchmarks.observation_adapter import extract_agent_state
from pbcwm.benchmarks.registry import make_benchmark


def small_spec() -> BenchmarkSpec:
    return BenchmarkSpec(
        name="nsgym/pendulum-mass-abrupt-return-v0",
        provider="nsgym",
        base_env="Pendulum-v1",
        parameter="m",
        regimes=(
            Regime(0, {"m": 1.0}),
            Regime(3, {"m": 1.5}),
            Regime(6, {"m": 1.0}),
        ),
        total_steps=10,
        fixed_parameters={"l": 1.0, "g": 10.0, "dt": 0.05},
    )


def test_nsgym_import_and_notifications_are_disabled():
    import ns_gym
    from ns_gym.wrappers import NSClassicControlWrapper

    assert ns_gym is not None
    assert NSClassicControlWrapper is not None
    env = make_benchmark("nsgym/pendulum-mass-abrupt-return-v0", small_spec())
    try:
        assert env.nsgym_env.change_notification is False
        assert env.nsgym_env.delta_change_notification is False
        obs, _ = env.reset(seed=7)
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (3,)
        assert not isinstance(obs, dict)
    finally:
        env.close()


def test_notification_fields_are_removed_at_observation_boundary():
    raw = {"state": np.array([1.0, 2.0]), "env_change": {"m": 1}, "delta_change": {"m": 0.5}, "relative_time": 4}
    state = extract_agent_state(raw)
    assert state.shape == (2,)
    assert np.array_equal(state, np.array([1.0, 2.0], dtype=np.float32))


def test_p1_p2_p1_physical_schedule_and_lifetime_reset():
    env = make_benchmark("nsgym/pendulum-mass-abrupt-return-v0", small_spec())
    try:
        env.reset(seed=11)
        assert env.unwrapped.m == 1.0
        observations = []
        for _ in range(3):
            before = (env.global_env_step, float(env.unwrapped.m))
            env.step(np.array([0.0], dtype=np.float32))
            observations.append((before, (env.global_env_step, float(env.unwrapped.m))))
        assert observations[-1] == ((2, 1.0), (3, 1.0))
        env.step(np.array([0.0], dtype=np.float32))
        assert env.global_env_step == 4
        assert env.unwrapped.m == 1.5
        env.reset()
        assert env.global_env_step == 4
        assert env.unwrapped.m == 1.5
        env.step(np.array([0.0], dtype=np.float32))
        env.step(np.array([0.0], dtype=np.float32))
        assert env.global_env_step == 6
        assert env.unwrapped.m == 1.5
        env.step(np.array([0.0], dtype=np.float32))
        assert env.global_env_step == 7
        assert env.unwrapped.m == 1.0
    finally:
        env.close()


def test_true_reward_is_oracle_only_in_learner_transition():
    env = make_benchmark("nsgym/pendulum-mass-abrupt-return-v0", small_spec())
    try:
        obs, _ = env.reset(seed=5)
        action = np.array([0.25], dtype=np.float32)
        next_obs, true_reward, terminated, truncated, _ = env.step(action)
        transition = build_agent_transition(obs, action, next_obs, terminated, truncated)
        assert transition.reward == 0.0
        assert env.oracle.rewards == [true_reward]
    finally:
        env.close()


def test_root_seed_reproduces_state_and_schedule():
    first = make_benchmark("nsgym/pendulum-mass-abrupt-return-v0", small_spec(), root_seed=23)
    second = make_benchmark("nsgym/pendulum-mass-abrupt-return-v0", small_spec(), root_seed=23)
    try:
        obs_a, _ = first.reset()
        obs_b, _ = second.reset()
        assert np.array_equal(obs_a, obs_b)
        for _ in range(8):
            action = np.array([0.1], dtype=np.float32)
            next_a, _, _, _, _ = first.step(action)
            next_b, _, _, _, _ = second.step(action)
            assert np.array_equal(next_a, next_b)
            assert first.unwrapped.m == second.unwrapped.m
    finally:
        first.close()
        second.close()


def test_static_dynamics_smoke_has_finite_prediction():
    import torch
    from pbcwm.baselines.static import StaticDynamicsLearner
    from pbcwm.planning.cem import CEMPlanner
    from pbcwm.preferences.reward_model import PreferenceRewardEnsemble

    env = make_benchmark("nsgym/pendulum-mass-abrupt-return-v0", small_spec(), root_seed=31)
    learner = StaticDynamicsLearner(3, 1, hidden_dims=(16, 16), batch_size=4, replay_capacity=32, seed=31)
    try:
        obs, _ = env.reset()
        for _ in range(8):
            action = np.array([0.0], dtype=np.float32)
            next_obs, _, terminated, truncated, _ = env.step(action)
            learner.observe(build_agent_transition(obs, action, next_obs, terminated, truncated))
            learner.update()
            obs = next_obs
        prediction = learner.predict(torch.zeros((2, 3)), torch.zeros((2, 1)))
        assert prediction.shape == (2, 3)
        assert torch.isfinite(prediction).all()
        planner = CEMPlanner(
            horizon=2,
            population_size=6,
            elite_size=2,
            num_iterations=1,
            action_low=env.action_space.low,
            action_high=env.action_space.high,
        )
        plan = planner.plan(
            obs,
            learner,
            lambda imagined_obs, imagined_action, imagined_next_obs: imagined_obs.new_zeros(
                imagined_obs.shape[0]
            ),
            return_candidates=False,
        )
        assert plan.action.shape == (1,)
        assert PreferenceRewardEnsemble(3, 1, ensemble_size=1, hidden_dims=(8,), batch_size=1).ensemble_size == 1
    finally:
        env.close()
