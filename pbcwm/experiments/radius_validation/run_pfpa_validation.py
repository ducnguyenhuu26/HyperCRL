"""PFPA isolation entry point; labels and timestamps share one fixed budget."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .generate_fixed_stream import load_fixed_stream


def run_pfpa_validation(stream_path: str | Path, *, labels: tuple[int, ...] = (100, 200)) -> dict:
    stream = load_fixed_stream(stream_path)
    if any(label <= 0 for label in labels):
        raise ValueError("PFPA label budgets must be positive")
    return {
        "protocol": "pfpa_isolation_setup",
        "queries": {"Q0_random": "random", "Q1_disagreement": "disagreement", "Q2_pfpa": "pfpa"},
        "labels": list(labels),
        "same_stream_sha256": stream.learner_payload_sha256,
        "planner_reward_label": "DEV_ORACLE_REWARD_PLANNING",
        "status": "ready_for_online_planner_wiring",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream", required=True)
    parser.add_argument("--output", default="outputs/radius_validation/pfpa_setup.json")
    args = parser.parse_args()
    result = run_pfpa_validation(args.stream)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"status": result["status"], "labels": result["labels"]}))


if __name__ == "__main__":
    main()
