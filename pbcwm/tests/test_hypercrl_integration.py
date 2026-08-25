import numpy as np

from pbcwm.baselines.hypercrl.online import HyperCRLAdaptOnline
from pbcwm.core.types import Transition
from pbcwm.envs.nonstationary_pendulum import NonstationaryPendulum
from pbcwm.rewards.pendulum import PendulumReward


def test_hypercrl_online_uses_shared_preference_and_cem_components() -> None:
    env = NonstationaryPendulum(
        [
            {"step": 0, "mass": 1.0, "length": 1.0, "gravity": 10.0},
            {"step": 8, "mass": 1.5, "length": 1.0, "gravity": 10.0},
        ]
    )
    env.reset(seed=0)
    env.action_space.seed(0)
    online = HyperCRLAdaptOnline(
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
        hypercrl_config={
            "embedding_dim": 4,
            "embedding_init_std": 0.1,
            "hyper_hidden_dims": [16],
            "target_hidden_dims": [16],
            "hyper_lr": 0.01,
            "embedding_lr": 0.01,
            "regularization_beta": 0.1,
            "current_regime_buffer_size": 32,
            "dynamics_batch_size": 4,
            "router_window_size": 4,
            "shift_threshold": 1.0,
            "reuse_threshold": 0.5,
            "consecutive_trigger_windows": 2,
            "router_cooldown_steps": 8,
            "gradient_clip_norm": 10.0,
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
        next_obs, reward, terminated, truncated, _ = env.step(action)
        online.observe(Transition(obs, action, next_obs, float(reward), terminated, truncated))
        online.update_dynamics(1)
        obs = next_obs

    assert online.dynamics_ready
    online.bootstrap(4, 3, env.action_space.low, env.action_space.high)
    assert len(online.preference_buffer) >= 2
    plan = online.plan(obs, collect_candidates=True)
    assert plan.action.shape == env.action_space.shape
    assert plan.candidate_trajectories
    env.close()
