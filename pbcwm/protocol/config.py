"""Validated, scenario-agnostic experiment protocol configuration."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(frozen=True)
class EnvironmentProtocolConfig:
    stage_length: int
    warmup_steps: int
    planner_horizon: int
    planner_population: int

    @property
    def nominal_budget(self) -> int:
        return self.stage_length * 6


@dataclass(frozen=True)
class PreferenceProtocolConfig:
    total_budget: int
    bootstrap_queries: int
    bootstrap_rounds: int
    online_queries: int
    online_rounds: int
    queries_per_round: int
    query_jitter_fraction: float = 0.20
    minimum_spacing_fraction: float = 0.50

    def __post_init__(self) -> None:
        if self.bootstrap_queries + self.online_queries != self.total_budget:
            raise ValueError("bootstrap_queries + online_queries must equal total_budget")
        if self.bootstrap_queries != self.bootstrap_rounds * self.queries_per_round:
            raise ValueError("bootstrap budget must equal bootstrap_rounds * queries_per_round")
        if self.online_queries != self.online_rounds * self.queries_per_round:
            raise ValueError("online budget must equal online_rounds * queries_per_round")


@dataclass(frozen=True)
class RewardModelProtocolConfig:
    ensemble_size: int
    batch_size: int
    updates_per_query_round: int


@dataclass(frozen=True)
class WorldModelProtocolConfig:
    min_buffer_before_update: int
    update_opportunities_per_env_step: int
    default_neural_batch_size: int
    update_interval_steps: int = 1


@dataclass(frozen=True)
class EvaluationProtocolConfig:
    stage_fractions: tuple[float, ...]
    recurrence_interactions: tuple[int, ...]
    planning_episodes_regular: int
    planning_episodes_stage_end: int
    heldout_preference_pairs: int = 32
    heldout_preference_horizon: int = 10
    planning_episode_horizon: int = 250
    reward_ablation_episodes_stage_end: int = 0


@dataclass(frozen=True)
class TuningProtocolConfig:
    max_configs_per_method_env: int
    development_seeds: tuple[int, ...]


@dataclass(frozen=True)
class ProtocolConfig:
    version: str
    schedule_template: tuple[str, ...]
    randomize_abc_per_seed: bool
    stage_jitter_fraction: float
    bootstrap_stage_jitter: bool
    lifetime_clock: str
    change_mid_episode: bool
    environments: dict[str, EnvironmentProtocolConfig]
    preference: PreferenceProtocolConfig
    reward_model: RewardModelProtocolConfig
    world_model: WorldModelProtocolConfig
    planner_type: str
    planner_replan_interval: int
    planner_iterations: int
    planner_elite_fraction: float
    evaluation: EvaluationProtocolConfig
    development_seeds: tuple[int, ...]
    final_seeds: tuple[int, ...]
    tuning: TuningProtocolConfig

    def environment(self, name: str) -> EnvironmentProtocolConfig:
        try:
            return self.environments[name]
        except KeyError as exc:
            raise KeyError(f"environment is not configured in protocol: {name}") from exc

    def planner_for(self, environment: str) -> dict[str, int | float | str]:
        env = self.environment(environment)
        return {
            "type": self.planner_type,
            "replan_interval": self.planner_replan_interval,
            "iterations": self.planner_iterations,
            "elite_fraction": self.planner_elite_fraction,
            "horizon": env.planner_horizon,
            "population": env.planner_population,
            "elite_count": max(1, round(env.planner_population * self.planner_elite_fraction)),
        }


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _tuple_ints(value: object, name: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return tuple(int(item) for item in value)


def protocol_from_mapping(data: Mapping[str, Any]) -> ProtocolConfig:
    root = _mapping(data.get("protocol", data), "protocol")
    schedule = _mapping(root.get("schedule"), "protocol.schedule")
    template = tuple(str(item) for item in schedule.get("template", []))
    if template != ("P0", "A", "B", "C", "B", "A"):
        raise ValueError("the canonical protocol template must be [P0, A, B, C, B, A]")

    raw_lengths = _mapping(root.get("stage_length"), "protocol.stage_length")
    raw_warmup = _mapping(root.get("warmup_steps"), "protocol.warmup_steps")
    raw_planner = _mapping(root.get("planner"), "protocol.planner")
    raw_by_env = _mapping(raw_planner.get("by_env"), "protocol.planner.by_env")
    environments: dict[str, EnvironmentProtocolConfig] = {}
    for name, raw_length in raw_lengths.items():
        planner_env = _mapping(raw_by_env.get(name), f"planner.by_env.{name}")
        length = int(raw_length)
        warmup = int(raw_warmup.get(name))
        if length <= 0 or warmup < 0 or warmup >= length:
            raise ValueError(f"invalid stage length/warm-up for {name}")
        environments[str(name)] = EnvironmentProtocolConfig(
            stage_length=length,
            warmup_steps=warmup,
            planner_horizon=int(planner_env["horizon"]),
            planner_population=int(planner_env["population"]),
        )

    raw_preference = _mapping(root.get("preference"), "protocol.preference")
    preference = PreferenceProtocolConfig(
        total_budget=int(raw_preference["total_budget"]),
        bootstrap_queries=int(raw_preference["bootstrap_queries"]),
        bootstrap_rounds=int(raw_preference["bootstrap_rounds"]),
        online_queries=int(raw_preference["online_queries"]),
        online_rounds=int(raw_preference["online_rounds"]),
        queries_per_round=int(raw_preference["queries_per_round"]),
        query_jitter_fraction=float(raw_preference.get("query_jitter_fraction", 0.20)),
        minimum_spacing_fraction=float(raw_preference.get("minimum_spacing_fraction", 0.50)),
    )
    raw_reward = _mapping(root.get("reward_model"), "protocol.reward_model")
    reward_model = RewardModelProtocolConfig(int(raw_reward["ensemble_size"]), int(raw_reward["batch_size"]), int(raw_reward["updates_per_query_round"]))
    raw_world = _mapping(root.get("world_model"), "protocol.world_model")
    world_model = WorldModelProtocolConfig(
        int(raw_world["min_buffer_before_update"]),
        int(raw_world["update_opportunities_per_env_step"]),
        int(raw_world["default_neural_batch_size"]),
        int(raw_world.get("update_interval_steps", 1)),
    )
    if world_model.update_interval_steps <= 0:
        raise ValueError("world_model.update_interval_steps must be positive")
    raw_evaluation = _mapping(root.get("evaluation"), "protocol.evaluation")
    evaluation = EvaluationProtocolConfig(
        stage_fractions=tuple(float(item) for item in raw_evaluation["stage_fractions"]),
        recurrence_interactions=tuple(int(item) for item in raw_evaluation["recurrence_interactions"]),
        planning_episodes_regular=int(raw_evaluation["planning_episodes_regular"]),
        planning_episodes_stage_end=int(raw_evaluation["planning_episodes_stage_end"]),
        heldout_preference_pairs=int(raw_evaluation.get("heldout_preference_pairs", 32)),
        heldout_preference_horizon=int(raw_evaluation.get("heldout_preference_horizon", 10)),
        planning_episode_horizon=int(raw_evaluation.get("planning_episode_horizon", 250)),
        reward_ablation_episodes_stage_end=int(raw_evaluation.get("reward_ablation_episodes_stage_end", 0)),
    )
    if not evaluation.stage_fractions or any(not 0 <= fraction <= 1 for fraction in evaluation.stage_fractions):
        raise ValueError("evaluation.stage_fractions must lie in [0, 1]")
    if min(
        evaluation.planning_episodes_regular,
        evaluation.planning_episodes_stage_end,
        evaluation.reward_ablation_episodes_stage_end,
    ) < 0:
        raise ValueError("evaluation planning episode counts must be non-negative")
    if evaluation.heldout_preference_pairs <= 0 or evaluation.heldout_preference_horizon <= 0 or evaluation.planning_episode_horizon <= 0:
        raise ValueError("evaluation pair, preference-horizon, and planning-horizon values must be positive")
    raw_seeds = _mapping(root.get("seeds"), "protocol.seeds")
    development = _tuple_ints(raw_seeds["development"], "seeds.development")
    final = _tuple_ints(raw_seeds["final"], "seeds.final")
    raw_tuning = _mapping(root.get("tuning"), "protocol.tuning")
    tuning_seeds = _tuple_ints(raw_tuning["development_seeds"], "tuning.development_seeds")
    tuning = TuningProtocolConfig(int(raw_tuning["max_configs_per_method_env"]), tuning_seeds)
    if tuning.max_configs_per_method_env <= 0:
        raise ValueError("tuning budget must be positive")
    return ProtocolConfig(
        version=str(root["version"]),
        schedule_template=template,
        randomize_abc_per_seed=bool(schedule["randomize_abc_per_seed"]),
        stage_jitter_fraction=float(schedule["stage_jitter_fraction"]),
        bootstrap_stage_jitter=bool(schedule["bootstrap_stage_jitter"]),
        lifetime_clock=str(schedule["lifetime_clock"]),
        change_mid_episode=bool(schedule["change_mid_episode"]),
        environments=environments,
        preference=preference,
        reward_model=reward_model,
        world_model=world_model,
        planner_type=str(raw_planner["type"]),
        planner_replan_interval=int(raw_planner["replan_interval"]),
        planner_iterations=int(raw_planner["iterations"]),
        planner_elite_fraction=float(raw_planner["elite_fraction"]),
        evaluation=evaluation,
        development_seeds=development,
        final_seeds=final,
        tuning=tuning,
    )


def load_protocol_config(path: str | Path) -> ProtocolConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        return protocol_from_mapping(yaml.safe_load(handle))
