import numpy as np
import torch

from pbcwm.baselines.vblrl.learner import VBLRLAdaptDynamicsLearner
from pbcwm.core.types import Transition
from pbcwm.planning.cem import CEMPlanner


class TargetReward:
    def __call__(self, obs, action, next_obs):
        del obs, action
        return -(next_obs[:, 0] - 1.0).square()


def test_vblrl_samples_are_consumed_by_stochastic_cem() -> None:
    learner = VBLRLAdaptDynamicsLearner(
        1,
        1,
        hidden_dims=(8,),
        dynamics_batch_size=4,
        current_buffer_size=16,
        world_buffer_size=16,
        world_updates_per_interval=0,
        world_update_interval_steps=100,
        planning_model_samples=3,
        seed=0,
    )
    for index in range(6):
        obs = np.array([0.0], dtype=np.float32)
        learner.observe(
            Transition(obs, np.array([0.0], dtype=np.float32), np.array([0.2], dtype=np.float32), 0.0, False, False)
        )
        learner.update(1)
    planner = CEMPlanner(
        horizon=2,
        population_size=16,
        elite_size=4,
        num_iterations=2,
        action_low=np.array([-1.0]),
        action_high=np.array([1.0]),
        dynamics_samples=3,
    )
    result = planner.plan(
        np.array([0.0], dtype=np.float32), learner, TargetReward(), return_candidates=True
    )
    assert result.action.shape == (1,)
    assert result.candidate_trajectories
