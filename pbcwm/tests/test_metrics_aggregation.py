import pytest

from pbcwm.metrics.aggregation import aggregate_cross_environment, aggregate_seed_values


def test_seed_aggregation_reports_uncertainty():
    result = aggregate_seed_values([1.0, 2.0, 3.0], name="wm/r2_h", higher_is_better=True)
    assert result["count"] == 3
    assert result["std"] is not None
    assert result["ci_low"] < result["mean"] < result["ci_high"]


def test_cross_environment_aggregation_rejects_raw_return():
    with pytest.raises(ValueError, match="raw"):
        aggregate_cross_environment([{"metric_name": "planning/return_mean", "value": 1.0, "valid": True}])
