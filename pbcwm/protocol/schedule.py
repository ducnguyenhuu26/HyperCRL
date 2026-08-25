"""Canonical six-stage lifetime schedule and evaluator-only metadata."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import ProtocolConfig
from .seeds import SeedStreams, spawn_seed_streams


@dataclass(frozen=True)
class StageInstance:
    segment_id: int
    role: str
    dynamics_id: str
    visit_id: int
    start_step: int
    end_step: int
    realized_length: int
    parameter_vector: dict[str, float] | None = None

    def contains(self, global_step: int) -> bool:
        return self.start_step <= global_step < self.end_step


@dataclass(frozen=True)
class LifetimeSchedule:
    environment: str
    stages: tuple[StageInstance, ...]
    schedule_seed: int
    abc_permutation: tuple[str, str, str]

    @property
    def total_steps(self) -> int:
        return self.stages[-1].end_step

    @property
    def boundary_steps(self) -> tuple[int, ...]:
        return tuple(stage.start_step for stage in self.stages[1:])

    def stage_at(self, global_step: int) -> StageInstance:
        if global_step < 0 or global_step >= self.total_steps:
            raise ValueError(f"global_step must be in [0, {self.total_steps})")
        for stage in self.stages:
            if stage.contains(global_step):
                return stage
        raise RuntimeError("schedule has a gap")

    def metadata_at(self, global_step: int) -> dict[str, Any]:
        stage = self.stage_at(global_step)
        return {
            "global_step": int(global_step),
            "segment_id": stage.segment_id,
            "dynamics_id": stage.dynamics_id,
            "visit_id": stage.visit_id,
            "change_event": global_step in self.boundary_steps,
            "parameter_vector": stage.parameter_vector,
        }

    def records(self) -> list[dict[str, Any]]:
        return [
            {
                "segment_id": stage.segment_id,
                "role": stage.role,
                "dynamics_id": stage.dynamics_id,
                "visit_id": stage.visit_id,
                "start_step": stage.start_step,
                "end_step": stage.end_step,
                "realized_stage_length": stage.realized_length,
                "parameter_vector": stage.parameter_vector,
            }
            for stage in self.stages
        ]


def build_lifetime_schedule(config: ProtocolConfig, environment: str, root_seed: int | SeedStreams) -> LifetimeSchedule:
    streams = root_seed if isinstance(root_seed, SeedStreams) else spawn_seed_streams(root_seed)
    env_config = config.environment(environment)
    rng = np.random.default_rng(streams["schedule_seed"])
    identities = np.array(["P1", "P2", "P3"], dtype=object)
    if config.randomize_abc_per_seed:
        identities = rng.permutation(identities)
    permutation = tuple(str(item) for item in identities)
    lengths = [env_config.stage_length]
    low = round(env_config.stage_length * (1.0 - config.stage_jitter_fraction))
    high = round(env_config.stage_length * (1.0 + config.stage_jitter_fraction))
    if config.bootstrap_stage_jitter:
        lengths[0] = int(rng.integers(low, high + 1))
    lengths.extend(int(rng.integers(low, high + 1)) for _ in range(5))
    role_to_identity = dict(zip(("A", "B", "C"), permutation))
    stage_definitions = (("P0", "P0", 0), ("A", role_to_identity["A"], 0), ("B", role_to_identity["B"], 0), ("C", role_to_identity["C"], 0), ("B", role_to_identity["B"], 1), ("A", role_to_identity["A"], 1))
    stages: list[StageInstance] = []
    cursor = 0
    for segment_id, ((role, dynamics_id, visit_id), length) in enumerate(zip(stage_definitions, lengths)):
        stages.append(StageInstance(segment_id, role, dynamics_id, visit_id, cursor, cursor + length, length))
        cursor += length
    return LifetimeSchedule(environment, tuple(stages), streams["schedule_seed"], permutation)
