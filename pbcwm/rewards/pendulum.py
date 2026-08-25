"""Analytical reward for Gymnasium's Pendulum observation encoding."""

import torch


class PendulumReward:
    """Return ``-(theta^2 + 0.1 theta_dot^2 + 0.001 action^2)``."""

    def __call__(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        next_obs: torch.Tensor,
    ) -> torch.Tensor:
        del next_obs
        if obs.ndim != 2 or obs.shape[-1] != 3:
            raise ValueError("Pendulum observations must have shape [batch, 3]")
        theta = torch.atan2(obs[:, 1], obs[:, 0])
        theta_dot = obs[:, 2]
        action_cost = action.square().sum(dim=-1)
        return -(theta.square() + 0.1 * theta_dot.square() + 0.001 * action_cost)
