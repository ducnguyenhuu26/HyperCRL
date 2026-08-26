from pbcwm.benchmarks.registry import load_benchmark_spec
from pbcwm.protocol.config import load_protocol_config
from pbcwm.protocol.queries import build_query_schedule, validate_query_schedule
from pbcwm.protocol.schedule import build_lifetime_schedule


def test_hopper_screen_has_matched_schedule_budget_and_bounded_compute():
    protocol = load_protocol_config("pbcwm/configs/protocol_hopper_screen_v1.yaml")
    benchmark = load_benchmark_spec("pbcwm/configs/benchmarks/hopper/screen_v1_single_schedule.yaml")
    schedule = build_lifetime_schedule(protocol, "Hopper-v5", 200)
    queries = build_query_schedule(protocol, schedule, 200)
    assert schedule.total_steps == benchmark.total_steps == 60_000
    assert [stage.realized_length for stage in schedule.stages] == [10_000] * 6
    assert schedule.boundary_steps == tuple(regime.start_step for regime in benchmark.regimes[1:])
    assert protocol.planner_replan_interval == 2
    assert protocol.world_model.update_interval_steps == 4
    assert protocol.environment("Hopper-v5").planner_population == 64
    assert protocol.evaluation.heldout_preference_pairs == 32
    assert protocol.evaluation.planning_episodes_stage_end == 3
    assert protocol.evaluation.reward_ablation_episodes_stage_end == 3
    validate_query_schedule(queries, schedule, 120)
    assert sum(query.pair_count for query in queries) == 120
