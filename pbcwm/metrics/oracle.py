"""Evaluator-only 2x2 learned/oracle planner diagnostics."""

from dataclasses import dataclass

from .common import MetricResult


@dataclass(frozen=True)
class OraclePlannerScores:
    j_ll: float
    j_ol: float
    j_lo: float
    j_oo: float


def oracle_diagnostics(scores: OraclePlannerScores) -> dict[str, MetricResult]:
    values = {
        "oracle/j_ll": (scores.j_ll, True),
        "oracle/j_ol": (scores.j_ol, True),
        "oracle/j_lo": (scores.j_lo, True),
        "oracle/j_oo": (scores.j_oo, True),
        "oracle/world_side_gap": (scores.j_oo - scores.j_lo, False),
        "oracle/reward_side_gap": (scores.j_oo - scores.j_ol, False),
        "oracle/full_system_gap": (scores.j_oo - scores.j_ll, False),
        "oracle/world_reward_interaction": (scores.j_ll - scores.j_ol - scores.j_lo + scores.j_oo, False),
    }
    return {name: MetricResult(name, float(value), higher) for name, (value, higher) in values.items()}
