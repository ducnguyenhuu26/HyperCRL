"""Canonical real-lifetime runner with protocol-owned timing and budgets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from pbcwm.benchmarks.base import build_agent_transition
from pbcwm.protocol.budget import BudgetLedger
from pbcwm.protocol.checkpoints import build_evaluation_checkpoints
from pbcwm.protocol.config import ProtocolConfig
from pbcwm.protocol.logger import ProtocolLogger
from pbcwm.protocol.queries import build_query_schedule, validate_query_schedule
from pbcwm.protocol.schedule import LifetimeSchedule, StageInstance, build_lifetime_schedule
from pbcwm.protocol.seeds import SeedStreams, spawn_seed_streams

from .evaluation import isolated_evaluation


@dataclass(frozen=True)
class LifetimeRunSummary:
    environment: str
    root_seed: int
    steps: int
    queries: int
    episode_resets: tuple[int, ...]
    stage_switches: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"environment": self.environment, "root_seed": self.root_seed, "steps": self.steps, "queries": self.queries, "episode_resets": list(self.episode_resets), "stage_switches": list(self.stage_switches)}


class CanonicalLifetimeRunner:
    """One loop for baselines and RADIUS; hidden stage metadata stays outside transitions."""

    def __init__(self, config: ProtocolConfig, environment: str, root_seed: int, *, episode_length: int = 137):
        self.config = config
        self.environment = environment
        self.root_seed = int(root_seed)
        self.streams: SeedStreams = spawn_seed_streams(root_seed)
        self.schedule: LifetimeSchedule = build_lifetime_schedule(config, environment, self.streams)
        self.queries = build_query_schedule(config, self.schedule, self.streams)
        validate_query_schedule(self.queries, self.schedule, config.preference.total_budget)
        self.checkpoints = build_evaluation_checkpoints(config, self.schedule)
        self.episode_length = int(episode_length)
        if self.episode_length <= 0:
            raise ValueError("episode_length must be positive")

    def run(
        self,
        env: Any,
        learner: Any,
        planner: Any,
        *,
        reward_fn: Any | None = None,
        query_handler: Callable[[Any, Any, Any, Any], int] | None = None,
        evaluation_handler: Callable[[Any, Any, Any], Any] | None = None,
        stage_handler: Callable[[StageInstance], None] | None = None,
        log_path: str | Path | None = None,
    ) -> LifetimeRunSummary:
        env_config = self.config.environment(self.environment)
        ledger = BudgetLedger(self.schedule.total_steps, self.config.preference.total_budget, env_config.warmup_steps)
        query_by_step = {item.global_step: item for item in self.queries}
        checkpoints_by_step: dict[int, list[Any]] = {}
        for checkpoint in self.checkpoints:
            checkpoints_by_step.setdefault(checkpoint.global_step, []).append(checkpoint)
        logger = ProtocolLogger(log_path) if log_path is not None else None
        resets: list[int] = []
        switches: list[int] = []
        try:
            obs, _ = env.reset(seed=self.streams["environment_seed"])
            for checkpoint in checkpoints_by_step.get(0, []):
                if evaluation_handler is not None:
                    isolated_evaluation(learner, lambda current, item=checkpoint: evaluation_handler(item, current, env))
            for global_step in range(self.schedule.total_steps):
                stage = self.schedule.stage_at(global_step)
                if global_step in self.schedule.boundary_steps:
                    switches.append(global_step)
                    if stage_handler is not None:
                        stage_handler(stage)
                warmup = global_step < env_config.warmup_steps
                action = env.action_space.sample() if warmup else planner.act(obs, learner, reward_fn)
                next_obs, _reward, terminated, truncated, _info = env.step(action)
                ledger.consume_environment(warmup=warmup)
                learner.observe(build_agent_transition(obs, action, next_obs, bool(terminated), bool(truncated)))
                updater = getattr(learner, "update", None)
                if not callable(updater):
                    updater = getattr(learner, "update_dynamics", None)
                if callable(updater):
                    updater(self.config.world_model.update_opportunities_per_env_step)
                if global_step in query_by_step:
                    query = query_by_step[global_step]
                    if query_handler is None:
                        raise RuntimeError("protocol query requires query_handler")
                    produced = int(query_handler(query, learner, planner, reward_fn))
                    if produced != query.pair_count:
                        raise RuntimeError("query handler did not consume the protocol pair count")
                    ledger.consume_preferences(produced)
                for checkpoint in checkpoints_by_step.get(global_step + 1, []):
                    if evaluation_handler is not None:
                        isolated_evaluation(learner, lambda current: evaluation_handler(checkpoint, current, env))
                if terminated or truncated:
                    resets.append(global_step + 1)
                    obs, _ = env.reset()
                else:
                    obs = next_obs
            ledger.assert_complete()
            summary = LifetimeRunSummary(self.environment, self.root_seed, self.schedule.total_steps, ledger.preference_labels, tuple(resets), tuple(switches))
            if logger is not None:
                logger.write({"kind": "protocol_summary", **summary.to_dict()})
            return summary
        finally:
            if logger is not None:
                logger.close()
            close = getattr(env, "close", None)
            if callable(close):
                close()
