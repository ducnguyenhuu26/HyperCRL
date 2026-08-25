import numpy as np
import torch

from pbcwm.planning.cem import CEMPlanner


class Integrator:
    def predict(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return obs + action


class TargetReward:
    def __call__(self, obs, action, next_obs):
        del obs, action
        return -(next_obs[:, 0] - 1.0).square()


def test_plan_exposes_candidates_and_act_remains_compatible() -> None:
    torch.manual_seed(0)
    planner = CEMPlanner(
        horizon=3,
        population_size=32,
        elite_size=4,
        num_iterations=2,
        action_low=np.array([-2.0]),
        action_high=np.array([2.0]),
        candidate_keep_per_iteration=3,
        candidate_keep_final_elites=2,
    )
    dynamics = Integrator()
    reward = TargetReward()
    result = planner.plan(np.zeros(1, dtype=np.float32), dynamics, reward)
    action = planner.act(np.zeros(1, dtype=np.float32), dynamics, reward)

    assert result.action.shape == (1,)
    assert result.best_action_sequence.shape == (3, 1)
    assert len(result.candidate_trajectories) == 5
    assert result.candidate_trajectories[0].obs.shape == (3, 1)
    assert result.candidate_trajectories[0].actions.shape == (3, 1)
    assert action.shape == (1,)
