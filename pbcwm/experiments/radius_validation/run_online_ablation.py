"""Online-ablation entry point with explicit protocol/leakage checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from .generate_fixed_stream import load_fixed_stream
from .probes import load_probe_bank
from .run_fixed_stream import run_variant
from .variants import VARIANT_NAMES


def run_online_ablation(stream_path: str | Path, *, probe_dir: str | Path, variants: tuple[str, ...] = ("W1", "W2", "W3", "W4"), seed: int = 0, device: str = "cpu", max_steps: int | None = None) -> dict:
    stream = load_fixed_stream(stream_path)
    if any(name not in VARIANT_NAMES[1:] for name in variants):
        raise ValueError("online ablation accepts W1-W4 only")
    probe_dir = Path(probe_dir)
    probe_banks = {dynamics_id: load_probe_bank(probe_dir / f"{dynamics_id}.npz") for dynamics_id in dict.fromkeys(stream.dynamics_id.tolist())}
    results = [run_variant(stream, name, probe_banks=probe_banks, seed=seed, device=device, max_steps=max_steps) for name in variants]
    return {"protocol": "online_ablation_setup", "planner_reward_label": "DEV_ORACLE_REWARD_PLANNING", "learner_payload_sha256": stream.learner_payload_sha256, "results": results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream", required=True)
    parser.add_argument("--probe-dir", required=True)
    parser.add_argument("--output", default="outputs/radius_validation/online_ablation_setup.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()
    result = run_online_ablation(args.stream, probe_dir=args.probe_dir, device=args.device, max_steps=args.max_steps)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"protocol": result["protocol"], "variants": [item["variant"] for item in result["results"]]}))


if __name__ == "__main__":
    main()
