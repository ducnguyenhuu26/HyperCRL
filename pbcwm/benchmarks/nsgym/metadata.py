"""Self-describing metadata emitted by the benchmark smoke/evaluator."""

import subprocess
from typing import Any

from ..base import BenchmarkSpec
from .common import dependency_metadata, seed_streams


def git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            check=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def collect_metadata(spec: BenchmarkSpec, root_seed: int) -> dict[str, Any]:
    return {
        "benchmark": spec.name,
        "provider": spec.provider,
        "base_env": spec.base_env,
        "tunable_parameter": spec.parameter,
        "fixed_parameters": dict(spec.fixed_parameters or {}),
        "schedule": [
            {"start_step": regime.start_step, "parameters": dict(regime.parameters)}
            for regime in spec.regimes
        ],
        "root_seed": int(root_seed),
        "seed_streams": seed_streams(root_seed),
        "pbcwm_git_commit": git_commit(),
        **dependency_metadata(),
    }
