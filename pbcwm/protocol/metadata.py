"""Run metadata and checkpoint-safe serialization helpers."""

from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
import platform
import subprocess
from typing import Any


def _version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "not-installed"


def repository_commit() -> str:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, check=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


@dataclass(frozen=True)
class RunMetadata:
    experiment_id: str
    protocol_version: str
    scenario_id: str
    method: str
    environment: str
    root_seed: int
    global_step: int
    segment_id: int | None
    dynamics_id: str | None
    visit_id: int | None
    stage_start_step: int | None
    stage_end_step: int | None
    realized_stage_length: int | None
    environment_interactions_used: int
    preference_labels_used: int
    planner_config: dict[str, Any]
    method_config: dict[str, Any] = field(default_factory=dict)
    reward_model_config: dict[str, Any] = field(default_factory=dict)
    git_commit: str = "unknown"
    ns_gym_version_or_commit: str = "not-recorded"
    python_version: str = field(default_factory=platform.python_version)
    torch_version: str = field(default_factory=lambda: _version("torch"))
    device_info: str = field(default_factory=platform.platform)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "protocol_version": self.protocol_version,
            "scenario_id": self.scenario_id,
            "method": self.method,
            "environment": self.environment,
            "root_seed": self.root_seed,
            "global_step": self.global_step,
            "segment_id": self.segment_id,
            "dynamics_id": self.dynamics_id,
            "visit_id": self.visit_id,
            "stage_start_step": self.stage_start_step,
            "stage_end_step": self.stage_end_step,
            "realized_stage_length": self.realized_stage_length,
            "environment_interactions_used": self.environment_interactions_used,
            "preference_labels_used": self.preference_labels_used,
            "planner_config": self.planner_config,
            "method_config": self.method_config,
            "reward_model_config": self.reward_model_config,
            "git_commit": self.git_commit,
            "ns_gym_version_or_commit": self.ns_gym_version_or_commit,
            "python_version": self.python_version,
            "torch_version": self.torch_version,
            "device_info": self.device_info,
        }
