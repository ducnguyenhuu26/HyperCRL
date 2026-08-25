"""Scenario-agnostic placeholder lifetime runner for protocol validation."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .budget import BudgetLedger
from .checkpoints import EvaluationCheckpoint, build_evaluation_checkpoints
from .config import ProtocolConfig
from .logger import ProtocolLogger
from .queries import QueryRound, build_query_schedule, validate_query_schedule
from .schedule import LifetimeSchedule, StageInstance, build_lifetime_schedule
from .seeds import SeedStreams, spawn_seed_streams


@dataclass(frozen=True)
class PlaceholderRunSummary:
    environment: str
    root_seed: int
    schedule: LifetimeSchedule
    query_rounds: tuple[QueryRound, ...]
    checkpoints: tuple[EvaluationCheckpoint, ...]
    ledger: dict[str, int]
    stage_switch_steps: tuple[int, ...]
    episode_reset_steps: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "environment": self.environment,
            "root_seed": self.root_seed,
            "abc_permutation": list(self.schedule.abc_permutation),
            "schedule": self.schedule.records(),
            "query_rounds": [round_item.__dict__ for round_item in self.query_rounds],
            "checkpoints": [checkpoint.__dict__ for checkpoint in self.checkpoints],
            "ledger": self.ledger,
            "stage_switch_steps": list(self.stage_switch_steps),
            "episode_reset_steps": list(self.episode_reset_steps),
        }


class PlaceholderLifetimeRunner:
    """Run all protocol machinery without inventing physical scenario values.

    ``on_stage_switch`` is the future scenario adapter hook.  The placeholder
    deliberately never calls reset at a stage boundary, so mid-episode change
    semantics can be tested before a physical environment is selected.
    """

    def __init__(self, config: ProtocolConfig, environment: str, root_seed: int, *, episode_length: int = 137):
        self.config = config
        self.environment = environment
        self.root_seed = int(root_seed)
        self.streams: SeedStreams = spawn_seed_streams(root_seed)
        self.schedule = build_lifetime_schedule(config, environment, self.streams)
        self.query_rounds = build_query_schedule(config, self.schedule, self.streams)
        validate_query_schedule(self.query_rounds, self.schedule, config.preference.total_budget)
        self.checkpoints = build_evaluation_checkpoints(config, self.schedule)
        self.episode_length = int(episode_length)
        if self.episode_length <= 0:
            raise ValueError("episode_length must be positive")

    def run(
        self,
        *,
        max_steps: int | None = None,
        on_stage_switch: Callable[[StageInstance], None] | None = None,
        log_path: str | Path | None = None,
    ) -> PlaceholderRunSummary:
        steps = self.schedule.total_steps if max_steps is None else min(int(max_steps), self.schedule.total_steps)
        if steps < 0:
            raise ValueError("max_steps must be non-negative")
        if steps != self.schedule.total_steps:
            raise ValueError("placeholder acceptance run must consume the complete lifetime budget")
        env_config = self.config.environment(self.environment)
        ledger = BudgetLedger(self.schedule.total_steps, self.config.preference.total_budget, env_config.warmup_steps)
        query_by_step = {round_item.global_step: round_item for round_item in self.query_rounds}
        checkpoint_by_step = {checkpoint.global_step: [] for checkpoint in self.checkpoints}
        for checkpoint in self.checkpoints:
            checkpoint_by_step[checkpoint.global_step].append(checkpoint)
        stage_switch_steps: list[int] = []
        episode_reset_steps: list[int] = []
        logger = ProtocolLogger(log_path) if log_path is not None else None
        try:
            if logger is not None:
                logger.write({"kind": "protocol_start", "protocol_version": self.config.version, "root_seed": self.root_seed, "seed_streams": self.streams.to_dict(), "schedule": self.schedule.records()})
            for global_step in range(steps):
                stage = self.schedule.stage_at(global_step)
                if global_step in self.schedule.boundary_steps:
                    stage_switch_steps.append(global_step)
                    if on_stage_switch is not None:
                        on_stage_switch(stage)
                    if logger is not None:
                        logger.write({"kind": "stage_switch", "global_step": global_step, **self.schedule.metadata_at(global_step)})
                ledger.consume_environment(warmup=global_step < env_config.warmup_steps)
                if global_step in query_by_step:
                    query = query_by_step[global_step]
                    ledger.consume_preferences(query.pair_count)
                    if logger is not None:
                        logger.write({"kind": "preference_query_round", "global_step": global_step, "round_id": query.round_id, "pair_count": query.pair_count, "bootstrap": query.bootstrap})
                if global_step in checkpoint_by_step and logger is not None:
                    for checkpoint in checkpoint_by_step[global_step]:
                        logger.write({"kind": "evaluation_checkpoint", **checkpoint.__dict__})
                if (global_step + 1) % self.episode_length == 0 and global_step + 1 < steps:
                    episode_reset_steps.append(global_step + 1)
                    if logger is not None:
                        logger.write({"kind": "episode_reset", "global_step": global_step + 1})
            ledger.assert_complete()
            summary = PlaceholderRunSummary(self.environment, self.root_seed, self.schedule, self.query_rounds, self.checkpoints, ledger.to_dict(), tuple(stage_switch_steps), tuple(episode_reset_steps))
            if logger is not None:
                logger.write({"kind": "protocol_summary", **summary.to_dict()})
            return summary
        finally:
            if logger is not None:
                logger.close()
