"""Reusable, protocol-agnostic PB-CWM evaluation metrics."""

from .aggregation import aggregate_cross_environment, aggregate_seed_values, aggregate_visit_metrics, normalized_auc
from .common import (
    BenchmarkEvalMetadata,
    CandidateTrajectoryBank,
    EvaluationSummary,
    Metric,
    MetricResult,
    PreferenceEvalBatch,
    RolloutProbeBatch,
)
from .registry import METRIC_REGISTRY, get_metric
from .serialization import to_json_records, write_csv, write_json

__all__ = [
    "BenchmarkEvalMetadata",
    "CandidateTrajectoryBank",
    "EvaluationSummary",
    "Metric",
    "MetricResult",
    "PreferenceEvalBatch",
    "RolloutProbeBatch",
    "METRIC_REGISTRY",
    "aggregate_seed_values",
    "aggregate_cross_environment",
    "aggregate_visit_metrics",
    "get_metric",
    "normalized_auc",
    "to_json_records",
    "write_csv",
    "write_json",
]
