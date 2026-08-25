"""Adapter from the shared preference ensemble to the Phase-0 reward API."""

import torch

from pbcwm.preferences.reward_model import PreferenceRewardEnsemble


class LearnedPreferenceReward:
    """Use ensemble-mean learned reward while preserving ``RewardFunction``."""

    def __init__(self, ensemble: PreferenceRewardEnsemble) -> None:
        self.ensemble = ensemble

    def __call__(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        next_obs: torch.Tensor,
    ) -> torch.Tensor:
        del next_obs
        return self.ensemble.reward(obs, action)
