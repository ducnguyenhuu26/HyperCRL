import numpy as np
import torch

from pbcwm.planning.cem import CEMPlanner


class IntegratorDynamics:
    def predict(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return obs + action


class TargetReward:
    def __call__(self, obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
        del obs, action
        return -(next_obs[:, 0] - 1.0).square()


def test_cem_prefers_action_with_better_objective() -> None:
    torch.manual_seed(0)
    planner = CEMPlanner(
        horizon=1,
        population_size=256,
        elite_size=32,
        num_iterations=5,
        initial_std=1.0,
        action_low=np.array([-2.0]),
        action_high=np.array([2.0]),
    )
    action = planner.act(np.array([0.0], dtype=np.float32), IntegratorDynamics(), TargetReward())
    assert action.shape == (1,)
    assert 0.5 < action[0] <= 2.0
