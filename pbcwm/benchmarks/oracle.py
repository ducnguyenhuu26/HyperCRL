"""Evaluator-only reward and regime bookkeeping."""

from dataclasses import dataclass, field


@dataclass
class BenchmarkOracle:
    """Stores true rewards for evaluation, never for learner transitions."""

    rewards: list[float] = field(default_factory=list)

    def record_reward(self, reward: float) -> None:
        self.rewards.append(float(reward))

    @property
    def trajectory_return(self) -> float:
        return float(sum(self.rewards))

    def reset_episode(self) -> None:
        self.rewards.clear()
