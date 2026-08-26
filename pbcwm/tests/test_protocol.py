from pathlib import Path

import pytest

from pbcwm.protocol.checkpoint import REQUIRED_CHECKPOINT_FIELDS, save_checkpoint
from pbcwm.protocol.checkpoints import build_evaluation_checkpoints
from pbcwm.protocol.config import load_protocol_config
from pbcwm.protocol.isolation import isolated_evaluation
from pbcwm.protocol.planner import shared_planner_config
from pbcwm.protocol.queries import build_query_schedule, validate_query_schedule
from pbcwm.protocol.runner import PlaceholderLifetimeRunner
from pbcwm.protocol.schedule import build_lifetime_schedule
from pbcwm.protocol.seeds import SeedStreams, spawn_seed_streams
from pbcwm.protocol.tuning import TuningBudget


CONFIG = "pbcwm/configs/protocol.yaml"


def test_schedule_query_and_child_seed_determinism():
    config = load_protocol_config(CONFIG)
    first = build_lifetime_schedule(config, "Pendulum-v1", 17)
    second = build_lifetime_schedule(config, "Pendulum-v1", 17)
    assert first.records() == second.records()
    assert build_query_schedule(config, first, 17) == build_query_schedule(config, second, 17)
    assert first.stages[0].realized_length == 10000
    assert all(9000 <= stage.realized_length <= 11000 for stage in first.stages[1:])
    assert first.abc_permutation == tuple(stage.dynamics_id for stage in first.stages[1:4])


def test_schedule_seed_is_independent_of_learner_seed():
    config = load_protocol_config(CONFIG)
    streams = spawn_seed_streams(19)
    changed = SeedStreams(streams.root_seed, {**streams.values, "learner_seed": streams["learner_seed"] + 1})
    assert build_lifetime_schedule(config, "Pendulum-v1", streams).records() == build_lifetime_schedule(config, "Pendulum-v1", changed).records()


def test_pairing_and_preference_budget_are_exact():
    config = load_protocol_config(CONFIG)
    schedule = build_lifetime_schedule(config, "Pendulum-v1", 0)
    queries = build_query_schedule(config, schedule, 0)
    validate_query_schedule(queries, schedule, 200)
    assert len(queries) == 20
    assert sum(query.pair_count for query in queries if query.bootstrap) == 40
    assert sum(query.pair_count for query in queries if not query.bootstrap) == 160
    assert not any(query.global_step in schedule.boundary_steps for query in queries)


def test_placeholder_changes_mid_episode_and_preserves_lifetime_clock(tmp_path: Path):
    config = load_protocol_config(CONFIG)
    runner = PlaceholderLifetimeRunner(config, "Pendulum-v1", 0, episode_length=137)
    summary = runner.run(log_path=tmp_path / "protocol.jsonl")
    assert summary.stage_switch_steps == summary.schedule.boundary_steps
    assert all(step not in summary.episode_reset_steps for step in summary.stage_switch_steps)
    assert summary.ledger["environment_interactions"] == 60527
    assert summary.ledger["preference_labels"] == 200
    assert summary.ledger["warmup_interactions"] == 2000
    assert summary.schedule.stages[4].dynamics_id == summary.schedule.stages[2].dynamics_id
    assert summary.schedule.stages[5].dynamics_id == summary.schedule.stages[1].dynamics_id


def test_checkpoints_are_sorted_deduplicated_and_include_recurrence_few_shot():
    config = load_protocol_config(CONFIG)
    schedule = build_lifetime_schedule(config, "Pendulum-v1", 0)
    checkpoints = build_evaluation_checkpoints(config, schedule)
    keys = [(point.global_step, point.segment_id, point.few_shot_interactions) for point in checkpoints]
    assert keys == sorted(keys, key=lambda item: (item[0], item[2] is not None, item[1]))
    assert len(keys) == len(set(keys))
    assert any(point.visit_id == 1 and point.few_shot_interactions == 16 for point in checkpoints)
    boundary = schedule.boundary_steps[0]
    assert {(point.segment_id, point.normalized_fraction) for point in checkpoints if point.global_step == boundary} >= {(0, 1.0), (1, 0.0)}


def test_shared_planner_and_tuning_budget():
    config = load_protocol_config(CONFIG)
    planner = shared_planner_config(config, "Pendulum-v1")
    assert planner.horizon == 25
    assert planner.elite_count == 26
    budget = TuningBudget(2, (0, 1, 2))
    budget.register("method-a", "Pendulum-v1", "config-0")
    budget.register("method-a", "Pendulum-v1", "config-1")
    with pytest.raises(RuntimeError):
        budget.register("method-a", "Pendulum-v1", "config-2")


def test_evaluation_isolation_and_checkpoint_envelope(tmp_path: Path):
    state = {"replay": 4}
    with isolated_evaluation({"replay": lambda: state["replay"]}):
        _ = state["replay"]
    with pytest.raises(RuntimeError):
        with isolated_evaluation({"replay": lambda: state["replay"]}):
            state["replay"] = 5
    checkpoint_state = {field: {} for field in REQUIRED_CHECKPOINT_FIELDS}
    checkpoint_state["global_lifetime_step"] = 10
    save_checkpoint(tmp_path / "checkpoint.pt", checkpoint_state, {"protocol_version": "pbcwm-protocol-v1"})
    assert (tmp_path / "checkpoint.pt").exists()
