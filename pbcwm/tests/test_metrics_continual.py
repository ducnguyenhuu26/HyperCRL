from pbcwm.metrics.aggregation import normalized_auc
from pbcwm.metrics.continual import (
    VisitMetricSeries,
    acquisition_auc,
    match_first_and_return_visits,
    reacquisition_auc,
    recurrent_aggregation,
    reuse_advantage,
)


def test_irregular_auc_and_reuse_advantage():
    first = VisitMetricSeries(2, 0, [0, 1, 3, 5], [0.1, 0.3, 0.6, 0.8])
    returned = VisitMetricSeries(2, 1, [0, 2, 4, 5], [0.5, 0.7, 0.85, 0.9])
    assert normalized_auc([0, 2, 5], [0, 2, 5]) == 2.5
    assert reacquisition_auc(returned).value > acquisition_auc(first).value
    assert reuse_advantage(acquisition_auc(first), reacquisition_auc(returned)).value > 0


def test_visit_matching_uses_dynamics_id_and_supports_multiple_returns():
    records = [
        VisitMetricSeries(1, 0, [0, 1], [0.1, 0.2]),
        VisitMetricSeries(2, 0, [0, 1], [0.1, 0.2]),
        VisitMetricSeries(3, 0, [0, 1], [0.1, 0.2]),
        VisitMetricSeries(2, 1, [0, 1], [0.3, 0.4]),
        VisitMetricSeries(2, 2, [0, 1], [0.5, 0.6]),
    ]
    pairs = match_first_and_return_visits(records)
    recurrent_pairs = [(first, returns) for first, returns in pairs if returns]
    assert len(pairs) == 3  # dynamics without recurrence remain representable
    assert recurrent_pairs[0][0].dynamics_id == 2
    assert [item.visit_id for item in recurrent_pairs[0][1]] == [1, 2]
    assert recurrent_aggregation(records)["recurrence_count"] == 2
