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
from .variants import VARIANT_NAMES, build_variant


def _r2(true: np.ndarray, predicted: np.ndarray) -> float | None:
    if true.size == 0:
        return None
    variance = np.var(true, axis=0)
    valid = variance > 1e-12
    if not np.any(valid):
        return None
    value = 1.0 - np.sum((predicted[:, valid] - true[:, valid]) ** 2, axis=0) / (np.sum((true[:, valid] - np.mean(true[:, valid], axis=0)) ** 2, axis=0) + 1e-12)
    return float(np.mean(value))


def _probe(learner: Any, stream: FixedStream, start: int, horizon: int) -> tuple[float | None, float | None]:
    if start < 0 or start + horizon > stream.steps:
        return None, None
    obs = torch.as_tensor(stream.obs[start:start + 1], dtype=torch.float32)
    true = stream.next_obs[start:start + horizon]
    predictions = []
    for index in range(horizon):
        action = torch.as_tensor(stream.action[start + index:start + index + 1], dtype=torch.float32)
        obs = learner.predict(obs, action).detach().cpu()
        predictions.append(obs.numpy()[0])
    predicted = np.asarray(predictions, dtype=np.float32)
    r2_h1 = _r2(true[:1], predicted[:1])
    r2_h = _r2(true, predicted)
    return r2_h1, r2_h


def run_variant(stream: FixedStream, variant: str, *, seed: int = 0, device: str = "cpu", max_steps: int | None = None) -> dict[str, Any]:
    steps = min(stream.steps, int(max_steps or stream.steps))
    low = -np.ones(stream.action.shape[1], dtype=np.float32)
    high = np.ones(stream.action.shape[1], dtype=np.float32)
    learner = build_variant(variant, stream.obs.shape[1], stream.action.shape[1], action_low=low, action_high=high, device=device, seed=seed)
    fractions = (0.0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0)
    checkpoints = sorted({min(steps, int(round(steps * fraction))) for fraction in fractions})
    records: list[dict[str, Any]] = []
    for step in range(steps + 1):
        if step in checkpoints:
            probe_start = min(step, max(0, steps - 20))
            r2_h1, r2_h = _probe(learner, stream, probe_start, min(20, steps - probe_start))
            diagnostics = learner.diagnostics() if callable(getattr(learner, "diagnostics", None)) else {}
            records.append({"checkpoint": step, "probe_start": probe_start, "r2_at_1": r2_h1, "r2_at_H": r2_h, "diagnostics": {key: value for key, value in diagnostics.items() if isinstance(value, (float, int, str, bool))}})
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
        "stage_length": 10000,
        "horizon": 20,
        "learner_payload_sha256": stream.learner_payload_sha256,
        "config_sha256": config_hash,
        "git_commit": git_commit,
        "failure_flags": failure_flags,
        "checkpoints": records,
    }


def run_fixed_stream(stream_path: str | Path, *, variant: str, seed: int = 0, device: str = "cpu", max_steps: int | None = None, output: str | Path | None = None) -> dict[str, Any]:
    stream = load_fixed_stream(stream_path)
    result = run_variant(stream, variant, seed=seed, device=device, max_steps=max_steps)
    if output is not None:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream", default="outputs/radius_validation/hopper_fixed_stream_seed0.npz")
    parser.add_argument("--variant", choices=VARIANT_NAMES, default="W0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--synthetic", action="store_true", help="create a short CPU fixture if the stream is absent")
    parser.add_argument("--output", default="outputs/radius_validation/fixed_stream_result.json")
    args = parser.parse_args()
    stream_path = Path(args.stream)
    if not stream_path.exists():
        if not args.synthetic:
            raise FileNotFoundError(f"missing frozen stream: {stream_path}; generate it before running W0-W4")
        generate_fixed_stream(stream_path, seed=args.seed, steps=args.max_steps or 256, synthetic=True)
    result = run_fixed_stream(stream_path, variant=args.variant, seed=args.seed, device=args.device, max_steps=args.max_steps, output=args.output)
    print(json.dumps({"variant": result["variant"], "steps": result["steps"], "learner_payload_sha256": result["learner_payload_sha256"]}))


if __name__ == "__main__":
    main()
