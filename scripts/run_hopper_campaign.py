"""Bounded three-process launcher for the Hopper single-schedule campaign."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
import numpy as np
import yaml


def _load(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("campaign config root must be a mapping")
    return value


def _write_aggregate(output_dir: Path, jobs: list[tuple[str, int]]) -> None:
    """Aggregate only completed per-seed summaries; missing values stay missing."""

    summaries: dict[str, list[dict[str, Any]]] = {}
    for method, seed in jobs:
        path = output_dir / method / f"seed_{seed}" / "summary.json"
        if not path.exists():
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("status") == "COMPLETE":
            summaries.setdefault(method, []).append(record)
    aggregate: dict[str, Any] = {"status": "COMPLETE", "methods": {}}
    for method, records in summaries.items():
        metric_names = sorted({name for record in records for name in record.get("metrics", {})})
        method_metrics: dict[str, Any] = {}
        for name in metric_names:
            values = [float(record["metrics"][name]) for record in records if isinstance(record.get("metrics", {}).get(name), (int, float))]
            method_metrics[name] = {
                "count": len(values),
                "mean": float(sum(values) / len(values)) if values else None,
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else (0.0 if values else None),
            }
        aggregate["methods"][method] = {"seeds": sorted(int(record["seed"]) for record in records), "metrics": method_metrics}
    (output_dir / "aggregate.json").write_text(json.dumps(aggregate, indent=2, allow_nan=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="pbcwm/configs/hopper_campaign.yaml")
    parser.add_argument("--device", default=None, help="defaults to config device; explicit cuda is strict")
    parser.add_argument("--max-parallel", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--include-static", action="store_true", help="add the non-preference static MLP sanity baseline")
    args = parser.parse_args()
    config_path = Path(args.config)
    cfg = _load(config_path)
    campaign = cfg["campaign"]
    device = str(args.device or campaign.get("device", "cuda"))
    max_parallel = int(args.max_parallel or campaign.get("max_parallel", 3))
    if max_parallel <= 0:
        raise ValueError("max_parallel must be positive")
    methods = list(campaign["methods"])
    if args.include_static and "static" not in methods:
        methods.insert(0, "static")
    seeds = [int(seed) for seed in campaign["seeds"]]
    output_dir = Path(campaign["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    devices = [str(value) for value in campaign.get("cuda_devices", [0])]
    jobs = [(method, seed) for method in methods for seed in seeds]
    manifest = {"campaign": campaign["name"], "device": device, "max_parallel": max_parallel, "methods": methods, "seeds": seeds, "jobs": [{"method": method, "seed": seed} for method, seed in jobs]}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if args.dry_run:
        for index, (method, seed) in enumerate(jobs):
            gpu = devices[index % len(devices)]
            print(json.dumps({"slot": index % max_parallel, "cuda_visible_devices": gpu, "command": [sys.executable, "scripts/run_hopper_job.py", "--config", str(config_path), "--method", method, "--seed", str(seed), "--device", device, "--output-dir", str(output_dir)]}))
        return
    if device.lower().startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested for the Hopper campaign, but this host has no CUDA-enabled PyTorch. Use a GPU host or pass --device cpu explicitly.")
    pending = list(jobs)
    running: dict[subprocess.Popen[str], tuple[str, int, Path]] = {}
    completed = 0
    while pending or running:
        while pending and len(running) < max_parallel:
            method, seed = pending.pop(0)
            job_dir = output_dir / method / f"seed_{seed}"
            job_dir.mkdir(parents=True, exist_ok=True)
            gpu = devices[completed % len(devices)]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            env["OMP_NUM_THREADS"] = "1"
            env["MKL_NUM_THREADS"] = "1"
            env["PYTHONUNBUFFERED"] = "1"
            stdout = (job_dir / "stdout.log").open("w", encoding="utf-8")
            stderr = (job_dir / "stderr.log").open("w", encoding="utf-8")
            command = [sys.executable, "scripts/run_hopper_job.py", "--config", str(config_path), "--method", method, "--seed", str(seed), "--device", device, "--output-dir", str(output_dir)]
            process = subprocess.Popen(command, env=env, stdout=stdout, stderr=stderr, text=True)
            stdout.close()
            stderr.close()
            running[process] = (method, seed, job_dir)
            print(json.dumps({"event": "START", "method": method, "seed": seed, "cuda_visible_devices": gpu}), flush=True)
        time.sleep(1.0)
        for process, (method, seed, job_dir) in list(running.items()):
            code = process.poll()
            if code is None:
                continue
            running.pop(process)
            completed += 1
            print(json.dumps({"event": "DONE" if code == 0 else "FAILED", "method": method, "seed": seed, "returncode": code, "job_dir": str(job_dir)}), flush=True)
    def is_complete(method: str, seed: int) -> bool:
        status_path = output_dir / method / f"seed_{seed}" / "status.json"
        if not status_path.exists():
            return False
        return json.loads(status_path.read_text(encoding="utf-8")).get("status") == "COMPLETE"

    if not all(is_complete(method, seed) for method, seed in jobs):
        raise SystemExit("one or more Hopper jobs failed; inspect each job's stderr.log and status.json")
    _write_aggregate(output_dir, jobs)
    print(json.dumps({"event": "AGGREGATE", "path": str(output_dir / "aggregate.json")}), flush=True)


if __name__ == "__main__":
    main()
