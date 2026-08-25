import numpy as np

from pbcwm.baselines.gpmm.online import GPMMOnline
from pbcwm.core.types import Transition
from pbcwm.envs.nonstationary_pendulum import NonstationaryPendulum
from pbcwm.rewards.pendulum import PendulumReward


def test_gpmm_online_reuses_shared_preference_and_cem_components() -> None:
    env = NonstationaryPendulum(
        [
            {"step": 0, "mass": 1.0, "length": 1.0, "gravity": 10.0},
            {"step": 12, "mass": 1.5, "length": 1.0, "gravity": 10.0},
        ]
    )
    env.reset(seed=0)
    env.action_space.seed(0)
    online = GPMMOnline(
        obs_dim=3,
        action_dim=1,
        action_low=env.action_space.low,
        action_high=env.action_space.high,
        planner_config={
            "horizon": 3,
            "population_size": 12,
            "elite_size": 3,
            "num_iterations": 1,
            "initial_std": 1.0,
        },
        gpmm_config={
            "alpha": 1.0,
            "sticky_bonus": 0.0,
            "transition_base_count": 1.0,
            "expert_min_points_before_competition": 3,
            "gp_fit_steps": 1,
            "gp_learning_rate": 0.08,
            "max_points_per_expert": 32,
            "merge_enabled": False,
            "merge_burnin_points": 20,
            "merge_threshold": 0.5,
            "prune_probationary": True,
            "min_predictive_variance": 1e-5,
            "max_predictive_variance": 1e3,
            "prior_variance": 1.0,
            "observation_noise": 0.05,
        },
        preference_config={
            "ensemble_size": 3,
            "hidden_dims": [16],
            "learning_rate": 3e-3,
            "reward_batch_size": 2,
            "min_preferences_before_planning": 2,
            "pair_pool_size": 16,
            "candidate_keep_per_iteration": 2,
            "candidate_keep_final_elites": 2,
            "teacher_skip_margin": 0.0,
        },
        teacher_reward=PendulumReward(),
        seed=0,
    )

    obs, _ = env.reset(seed=0)
    for _ in range(8):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)
        online.observe(Transition(obs, action, next_obs, float(reward), terminated, truncated))
        online.update_dynamics(1)
        obs = next_obs

    assert online.dynamics_ready
    online.bootstrap(4, 3, env.action_space.low, env.action_space.high)
    assert len(online.preference_buffer) >= 2
    plan = online.plan(obs, collect_candidates=True)
    assert plan.action.shape == env.action_space.shape
    assert plan.candidate_trajectories
    assert all(not hasattr(example.traj_a, "true_dynamics_stage") for example in online.preference_buffer)
    env.close()
