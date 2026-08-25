import numpy as np
import torch

from pbcwm.baselines.moprl_online_ft import MoPRLOnlineFT
from pbcwm.core.types import Transition
from pbcwm.envs.nonstationary_pendulum import NonstationaryPendulum
from pbcwm.rewards.pendulum import PendulumReward


def test_moprl_online_ft_collects_preferences_across_hidden_shift() -> None:
    env = NonstationaryPendulum(
        [
            {"step": 0, "mass": 1.0, "length": 1.0, "gravity": 10.0},
            {"step": 24, "mass": 1.5, "length": 1.0, "gravity": 10.0},
        ]
    )
    env.reset(seed=0)
    env.action_space.seed(0)
    baseline = MoPRLOnlineFT(
        obs_dim=3,
        action_dim=1,
        action_low=env.action_space.low,
        action_high=env.action_space.high,
        planner_config={
            "horizon": 4,
            "population_size": 24,
            "elite_size": 4,
            "num_iterations": 2,
            "initial_std": 1.0,
            "candidate_keep_per_iteration": 3,
            "candidate_keep_final_elites": 3,
        },
        model_hidden_dims=(16,),
        model_learning_rate=3e-3,
        dynamics_window_size=32,
        dynamics_batch_size=8,
        preference_ensemble_size=3,
        preference_hidden_dims=(16,),
        preference_learning_rate=3e-3,
        preference_batch_size=4,
        min_preferences_before_planning=4,
        pair_pool_size=32,
        teacher_reward=PendulumReward(),
        seed=0,
    )

    obs, _ = env.reset(seed=0)
    bootstrap_done = False
    query_metrics = None
    for step in range(48):
        if baseline.planning_ready:
            plan = baseline.plan(obs, collect_candidates=step % 8 == 0)
            action = plan.action
        else:
            action = env.action_space.sample()
            plan = None
        next_obs, env_reward, terminated, truncated, info = env.step(action)
        transition = Transition(obs, action, next_obs, float(env_reward), terminated, truncated)
        baseline.observe(transition)
        if baseline.dynamics_ready:
            baseline.update_dynamics(1)
        if baseline.dynamics_ready and not bootstrap_done:
            baseline.bootstrap(8, 4, env.action_space.low, env.action_space.high)
            bootstrap_done = True
        if plan is not None and plan.candidate_trajectories:
            query_metrics = baseline.query_and_update(plan.candidate_trajectories, 2, 2)
        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset()

    assert baseline.dynamics_ready
    assert len(baseline.preference_buffer) >= 4
    assert query_metrics is not None
    assert query_metrics["num_preferences"] >= 4
    assert info["true_dynamics_stage"] == 1
    assert not hasattr(next(iter(baseline.preference_buffer)).traj_a, "true_dynamics_stage")
    env.close()
