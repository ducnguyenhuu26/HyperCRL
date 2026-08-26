"""Reward-free preference teacher proxy for Hopper imagined rollouts."""

from __future__ import annotations

import torch


class HopperPreferenceReward:
    """State/action proxy used only to label synthetic preference pairs.

    Hopper observations omit the absolute x position, so the exact forward
    velocity reward cannot be reconstructed from imagined state vectors.  The
    proxy retains the observable healthy-posture and control terms.  It is
    never passed to a learner and is not used for benchmark scoring.
    """

    def __call__(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        next_obs: torch.Tensor,
    ) -> torch.Tensor:
        del next_obs
        if obs.ndim != 2 or obs.shape[-1] < 2:
            raise ValueError("Hopper observations must have shape [batch, >=2]")
        height = obs[:, 0]
        angle = obs[:, 1]
        healthy = ((height > 0.7) & (height < 1.3) & (angle > -0.2) & (angle < 0.2)).to(obs.dtype)
        posture_cost = (height - 1.0).square() + angle.square()
        control_cost = 1.0e-3 * action.square().sum(dim=-1)
        return healthy - posture_cost - control_cost
