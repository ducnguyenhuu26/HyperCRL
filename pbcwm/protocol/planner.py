"""Shared planner configuration and evaluator-only oracle cache keys."""

from dataclasses import dataclass

from .config import ProtocolConfig


@dataclass(frozen=True)
class SharedPlannerConfig:
    planner_type: str
    replan_interval: int
    iterations: int
    elite_fraction: float
    horizon: int
    population: int

    @property
    def elite_count(self) -> int:
        return max(1, round(self.population * self.elite_fraction))

    def to_dict(self) -> dict[str, int | float | str]:
        return {
            "type": self.planner_type,
            "replan_interval": self.replan_interval,
            "iterations": self.iterations,
            "elite_fraction": self.elite_fraction,
            "horizon": self.horizon,
            "population": self.population,
            "elite_count": self.elite_count,
        }


def shared_planner_config(config: ProtocolConfig, environment: str) -> SharedPlannerConfig:
    values = config.planner_for(environment)
    return SharedPlannerConfig(
        planner_type=str(values["type"]),
        replan_interval=int(values["replan_interval"]),
        iterations=int(values["iterations"]),
        elite_fraction=float(values["elite_fraction"]),
        horizon=int(values["horizon"]),
        population=int(values["population"]),
    )


@dataclass(frozen=True)
class OracleCacheKey:
    environment: str
    dynamics_id: str
    planner_config: tuple[tuple[str, int | float | str], ...]
    evaluation_seed: int


class OracleEvaluationCache:
    """Small in-memory hook; actual environment oracle evaluation is scenario-owned."""

    def __init__(self) -> None:
        self._values: dict[OracleCacheKey, float] = {}

    def get(self, key: OracleCacheKey) -> float | None:
        return self._values.get(key)

    def put(self, key: OracleCacheKey, value: float) -> None:
        self._values[key] = float(value)
