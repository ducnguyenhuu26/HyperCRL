"""Fail-closed environment, warm-up, and preference budget accounting."""

from dataclasses import dataclass


@dataclass
class BudgetLedger:
    environment_budget: int
    preference_budget: int
    warmup_budget: int
    environment_interactions: int = 0
    preference_labels: int = 0
    warmup_interactions: int = 0

    def consume_environment(self, *, warmup: bool = False) -> None:
        if self.environment_interactions >= self.environment_budget:
            raise RuntimeError("environment interaction budget exceeded")
        self.environment_interactions += 1
        if warmup:
            if self.warmup_interactions >= self.warmup_budget:
                raise RuntimeError("warm-up budget exceeded")
            self.warmup_interactions += 1

    def consume_preferences(self, count: int) -> None:
        if count < 0 or self.preference_labels + count > self.preference_budget:
            raise RuntimeError("preference budget exceeded")
        self.preference_labels += int(count)

    def assert_complete(self) -> None:
        if self.environment_interactions != self.environment_budget:
            raise RuntimeError("lifetime ended before consuming the environment budget")
        if self.preference_labels != self.preference_budget:
            raise RuntimeError("lifetime ended without consuming the preference budget")
        if self.warmup_interactions != self.warmup_budget:
            raise RuntimeError("lifetime ended without consuming the warm-up budget")

    def to_dict(self) -> dict[str, int]:
        return {
            "environment_budget": self.environment_budget,
            "preference_budget": self.preference_budget,
            "warmup_budget": self.warmup_budget,
            "environment_interactions": self.environment_interactions,
            "preference_labels": self.preference_labels,
            "warmup_interactions": self.warmup_interactions,
        }
