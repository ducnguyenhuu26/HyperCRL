"""Append-only JSONL protocol event logger."""

import json
from pathlib import Path
from typing import Any


class ProtocolLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w", encoding="utf-8")

    def write(self, record: dict[str, Any]) -> None:
        self._handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> "ProtocolLogger":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
