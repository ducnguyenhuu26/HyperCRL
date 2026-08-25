"""Benchmark-controlled preference query timing and exact budget accounting."""

from dataclasses import dataclass

import numpy as np

from .config import ProtocolConfig
from .schedule import LifetimeSchedule
from .seeds import SeedStreams, spawn_seed_streams


@dataclass(frozen=True)
class QueryRound:
    round_id: int
    global_step: int
    pair_count: int
    bootstrap: bool


def _nearest_valid(candidate: int, *, lower: int, upper: int, boundaries: set[int], used: set[int], minimum_spacing: int) -> int:
    for radius in range(0, max(upper - lower, 1) + 1):
        choices = (candidate,) if radius == 0 else (candidate - radius, candidate + radius)
        for value in choices:
            if value <= lower or value >= upper or value in boundaries or value in used:
                continue
            if all(abs(value - other) >= minimum_spacing for other in used):
                return value
    raise ValueError("unable to place a non-boundary preference query with the requested spacing")


def build_query_schedule(config: ProtocolConfig, schedule: LifetimeSchedule, root_seed: int | SeedStreams) -> tuple[QueryRound, ...]:
    streams = root_seed if isinstance(root_seed, SeedStreams) else spawn_seed_streams(root_seed)
    preference = config.preference
    p0 = schedule.stages[0]
    warmup = config.environment(schedule.environment).warmup_steps
    bootstrap_fractions = np.linspace(0.20, 0.80, preference.bootstrap_rounds)
    rounds: list[QueryRound] = []
    used: set[int] = set()
    boundaries = set(schedule.boundary_steps)
    for index, fraction in enumerate(bootstrap_fractions):
        candidate = round(p0.start_step + warmup + fraction * (p0.realized_length - warmup))
        value = _nearest_valid(candidate, lower=p0.start_step, upper=p0.end_step, boundaries=boundaries, used=used, minimum_spacing=1)
        used.add(value)
        rounds.append(QueryRound(index, value, preference.queries_per_round, True))

    online_count = preference.online_rounds
    post_start = p0.end_step
    post_length = schedule.total_steps - post_start
    spacing = post_length / (online_count + 1)
    minimum_spacing = max(1, round(spacing * preference.minimum_spacing_fraction))
    rng = np.random.default_rng(streams["preference_query_seed"])
    for index in range(online_count):
        base = post_start + (index + 1) * spacing
        jitter = rng.uniform(-preference.query_jitter_fraction, preference.query_jitter_fraction) * spacing
        candidate = round(base + jitter)
        value = _nearest_valid(candidate, lower=post_start, upper=schedule.total_steps, boundaries=boundaries, used=used, minimum_spacing=minimum_spacing)
        used.add(value)
        rounds.append(QueryRound(preference.bootstrap_rounds + index, value, preference.queries_per_round, False))
    rounds.sort(key=lambda item: item.global_step)
    return tuple(rounds)


def validate_query_schedule(rounds: tuple[QueryRound, ...], schedule: LifetimeSchedule, total_budget: int) -> None:
    if any(current.global_step >= following.global_step for current, following in zip(rounds, rounds[1:])):
        raise ValueError("preference query times must be strictly increasing")
    if any(round_item.global_step in schedule.boundary_steps for round_item in rounds):
        raise ValueError("preference query cannot coincide with a dynamics boundary")
    if sum(round_item.pair_count for round_item in rounds) != total_budget:
        raise ValueError("query schedule does not consume the configured preference budget")
