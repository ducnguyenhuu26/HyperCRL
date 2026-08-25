"""Synthetic preference labels; this oracle is never used by the planner."""

from pbcwm.rewards.base import RewardFunction

from .types import TrajectorySegment


class SyntheticPreferenceTeacher:
    """Label imagined segments using ground-truth reward only as an oracle."""

    def __init__(self, reward_fn: RewardFunction, skip_margin: float = 0.0) -> None:
        if skip_margin < 0:
            raise ValueError("skip_margin must be non-negative")
        self.reward_fn = reward_fn
        self.skip_margin = float(skip_margin)

    def score(self, trajectory: TrajectorySegment) -> float:
        # This method is for label generation only; callers must not pass this
        # score to CEM or to either learner.
        import torch

        with torch.no_grad():
            rewards = self.reward_fn(trajectory.obs, trajectory.actions, trajectory.next_obs)
        return float(rewards.sum().cpu())

    def label(self, traj_a: TrajectorySegment, traj_b: TrajectorySegment) -> int | None:
        score_a = self.score(traj_a)
        score_b = self.score(traj_b)
        if self.skip_margin > 0 and abs(score_a - score_b) < self.skip_margin:
            return None
        return 0 if score_a > score_b else 1
