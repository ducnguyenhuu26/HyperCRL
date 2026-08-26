"""Irregular evaluation checkpoint generation."""

from dataclasses import dataclass

from .config import ProtocolConfig
from .schedule import LifetimeSchedule


@dataclass(frozen=True)
class EvaluationCheckpoint:
    global_step: int
    segment_id: int
    dynamics_id: str
    visit_id: int
    normalized_fraction: float
    few_shot_interactions: int | None = None


def build_evaluation_checkpoints(config: ProtocolConfig, schedule: LifetimeSchedule) -> tuple[EvaluationCheckpoint, ...]:
    # Keep both sides of a boundary: the preceding stage's final quality and
    # the new stage's entry quality answer different continual-learning
    # questions even though they share a global timestamp.
    points: dict[tuple[int, int, int | None], EvaluationCheckpoint] = {}
    for stage in schedule.stages:
        for fraction in config.evaluation.stage_fractions:
            global_step = min(schedule.total_steps, round(stage.start_step + fraction * stage.realized_length))
            points[(global_step, stage.segment_id, None)] = EvaluationCheckpoint(global_step, stage.segment_id, stage.dynamics_id, stage.visit_id, fraction)
        if stage.visit_id > 0:
            for interactions in config.evaluation.recurrence_interactions:
                global_step = stage.start_step + interactions
                if global_step <= stage.end_step:
                    points[(global_step, stage.segment_id, interactions)] = EvaluationCheckpoint(global_step, stage.segment_id, stage.dynamics_id, stage.visit_id, interactions / stage.realized_length, interactions)
    return tuple(sorted(points.values(), key=lambda point: (point.global_step, point.few_shot_interactions is not None, point.segment_id)))
