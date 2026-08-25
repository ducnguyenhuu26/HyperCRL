"""JSON/CSV serialization for flat metric records."""

import csv
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"opaque metric value is not serializable: {type(value).__name__}")


def to_json_records(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [_json_safe(dict(record)) for record in records]


def write_json(records: Iterable[Mapping[str, Any]], path: str | Path) -> None:
    Path(path).write_text(json.dumps(to_json_records(records), indent=2, sort_keys=True), encoding="utf-8")


def write_csv(records: Iterable[Mapping[str, Any]], path: str | Path) -> None:
    materialized = to_json_records(records)
    fields = sorted({key for record in materialized for key in record})
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in materialized:
            writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value for key, value in record.items()})
