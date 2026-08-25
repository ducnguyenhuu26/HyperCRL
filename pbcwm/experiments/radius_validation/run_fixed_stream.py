"""Run W0-W4 on one frozen stream with evaluator-only probes."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from .generate_fixed_stream import FixedStream, generate_fixed_stream, load_fixed_stream
from .probe_metrics import evaluate_probe_bank
from .probes import DynamicsProbeBank, generate_synthetic_probe_banks, load_probe_bank, probe_bank_sha256
from .variants import VARIANT_NAMES, build_variant


def build_stage_checkpoints(
    *,
    stage_length: int,
    schedule: tuple[str, ...],
    fractions: tuple[float, ...],
    recurrence_offsets: tuple[int, ...],
    total_steps: int,
) -> dict[int, list[dict[str, Any]]]:
    """Build stage-relative checkpoints, retaining both sides of boundaries."""

    visits: dict[str, int] = {}
    checkpoints: dict[int, list[dict[str, Any]]] = {}
    for stage_index, dynamics_id in enumerate(schedule):
        visit_id = visits.get(dynamics_id, 0)
        visits[dynamics_id] = visit_id + 1
        stage_start = stage_index * stage_length
        for fraction in fractions:
            global_step = stage_start + round(stage_length * fraction)
            if global_step <= total_steps:
                checkpoints.setdefault(global_step, []).append({
                    "global_step": global_step,
                    "stage_index": stage_index,
                    "dynamics_id": dynamics_id,
                    "visit_id": visit_id,
                    "stage_offset": int(round(stage_length * fraction)),
                    "stage_fraction": float(fraction),
                })
        if visit_id > 0:
            for offset in recurrence_offsets:
                global_step = stage_start + int(offset)
                if global_step <= min(total_steps, stage_start + stage_length):
                    checkpoints.setdefault(global_step, []).append({
                        "global_step": global_step,
                        "stage_index": stage_index,
                        "dynamics_id": dynamics_id,
                        "visit_id": visit_id,
                        "stage_offset": int(offset),
                        "stage_fraction": float(offset / stage_length),
                    })
    return checkpoints


def run_variant(
    stream: FixedStream,
    variant: str,
    *,
    probe_banks: dict[str, DynamicsProbeBank],
    seed: int = 0,
    device: str = "cpu",
    max_steps: int | None = None,
) -> dict[str, Any]:
    if max_steps is not None and max_steps < 0:
        raise ValueError("max_steps must be non-negative")
    steps = min(stream.steps, stream.steps if max_steps is None else int(max_steps))
    low = -np.ones(stream.action.shape[1], dtype=np.float32)
    high = np.ones(stream.action.shape[1], dtype=np.float32)
    learner = build_variant(variant, stream.obs.shape[1], stream.action.shape[1], action_low=low, action_high=high, device=device, seed=seed)
    stage_length = 10_000
    schedule = ("P0", "A", "B", "C", "B", "A")
    checkpoints = build_stage_checkpoints(
        stage_length=stage_length,
        schedule=schedule,
        fractions=(0.0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0),
        recurrence_offsets=(16, 32, 64, 128),
        total_steps=steps,
    )
    records: list[dict[str, Any]] = []
    for step in range(steps + 1):
        if step in checkpoints:
            diagnostics = learner.diagnostics() if callable(getattr(learner, "diagnostics", None)) else {}
            for checkpoint in checkpoints[step]:
                bank = probe_banks.get(checkpoint["dynamics_id"])
                if bank is None:
                    raise KeyError(f"missing evaluator probe bank for {checkpoint['dynamics_id']}")
                metric = evaluate_probe_bank(learner, bank)
                records.append({**checkpoint, **metric, "diagnostics": {key: value for key, value in diagnostics.items() if isinstance(value, (float, int, str, bool))}})
        if step >= steps:
            break
        learner.observe(stream.transition(step))
        learner.update(1)
    numeric_values = [value for record in records for value in record["diagnostics"].values() if isinstance(value, (float, int))]
    failure_flags = {
        "nan_or_inf": any(not np.isfinite(value) for value in numeric_values),
        "rank_explosion": any(record["diagnostics"].get("radius/atlas_rank", 0) > 8 for record in records),
        "prototype_explosion": any(record["diagnostics"].get("radius/memory_num_prototypes", 0) > 32 for record in records),
        "pec_step_capped": any(bool(record["diagnostics"].get("radius/pec_step_capped", False)) for record in records),
    }
    try:
        git_commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = None
    config_path = Path(__file__).parent / "configs" / "hopper_component.yaml"
    config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    return {
        "protocol": "radius_hopper_fixed_stream_component_validation",
        "variant": variant,
        "seed": seed,
        "steps": steps,
        "schedule": ["P0", "A", "B", "C", "B", "A"],
        "stage_length": stage_length,
        "horizon": next(iter(probe_banks.values())).horizon,
        "learner_payload_sha256": stream.learner_payload_sha256,
        "fixed_stream_hash": stream.learner_payload_sha256,
        "probe_bank_sha256": {key: probe_bank_sha256(value) for key, value in sorted(probe_banks.items())},
        "config_sha256": config_hash,
        "git_commit": git_commit,
        "failure_flags": failure_flags,
        "checkpoints": records,
    }


def run_fixed_stream(stream_path: str | Path, *, variant: str, probe_dir: str | Path, seed: int = 0, device: str = "cpu", max_steps: int | None = None, output: str | Path | None = None) -> dict[str, Any]:
    stream = load_fixed_stream(stream_path)
    probe_dir = Path(probe_dir)
    probe_banks = {dynamics_id: load_probe_bank(probe_dir / f"{dynamics_id}.npz") for dynamics_id in dict.fromkeys(stream.dynamics_id.tolist())}
    result = run_variant(stream, variant, probe_banks=probe_banks, seed=seed, device=device, max_steps=max_steps)
    if output is not None:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream", default="outputs/radius_validation/hopper_fixed_stream_seed0.npz")
    parser.add_argument("--variant", choices=VARIANT_NAMES, default="W0")
    parser.add_argument("--probe-dir", default="outputs/radius_validation/probe_banks_seed0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--synthetic", action="store_true", help="create a short CPU fixture if the stream is absent; probe banks must still be generated separately")
    parser.add_argument("--output", default="outputs/radius_validation/fixed_stream_result.json")
    args = parser.parse_args()
    stream_path = Path(args.stream)
    if args.synthetic and not Path(args.probe_dir).exists():
        generate_synthetic_probe_banks(Path(__file__).parent / "configs" / "hopper_component.yaml", args.probe_dir, seed=args.seed, n_probes=32, horizon=20)
    if not stream_path.exists():
        if not args.synthetic:
            raise FileNotFoundError(f"missing frozen stream: {stream_path}; generate it before running W0-W4")
        generate_fixed_stream(stream_path, seed=args.seed, steps=args.max_steps or 256, synthetic=True)
    result = run_fixed_stream(stream_path, variant=args.variant, probe_dir=args.probe_dir, seed=args.seed, device=args.device, max_steps=args.max_steps, output=args.output)
    print(json.dumps({"variant": result["variant"], "steps": result["steps"], "learner_payload_sha256": result["learner_payload_sha256"]}))


if __name__ == "__main__":
    main()
