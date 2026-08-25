from pathlib import Path

from pbcwm.metrics.planning import adaptation_cost, dynamic_regret, return_metrics
from pbcwm.metrics.serialization import to_json_records, write_csv, write_json


def test_planning_metrics_support_irregular_checkpoints():
    result = dynamic_regret([0, 2, 5], [10, 8, 6], [10, 10, 10])
    assert result["planning/dynamic_regret"].value == 11.0
    assert result["planning/mean_dynamic_regret"].value == 2.2
    assert adaptation_cost([0, 2, 5], [10, 8, 6], [10, 10, 10]).value == 2.2
    assert return_metrics([1, 3])["planning/return_mean"].value == 2.0


def test_metric_records_serialize_without_opaque_objects(tmp_path: Path):
    record = return_metrics([1, 3])["planning/return_mean"].to_record(seed=0, metadata={"horizon": 5})
    assert to_json_records([record])[0]["value"] == 2.0
    write_json([record], tmp_path / "metrics.json")
    write_csv([record], tmp_path / "metrics.csv")
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "metrics.csv").exists()
