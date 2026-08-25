"""Generate one frozen, reward-separated Hopper stream for W0-W4."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import yaml

from pbcwm.benchmarks.base import BenchmarkSpec, Regime, build_agent_transition
from pbcwm.benchmarks.nsgym.mujoco import make_mujoco_benchmark


@dataclass(frozen=True)
class FixedStream:
    obs: np.ndarray
    action: np.ndarray
    next_obs: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    true_reward: np.ndarray
    stage_id: np.ndarray
    dynamics_id: np.ndarray
    visit_id: np.ndarray
    change_event: np.ndarray
    parameter_vector: tuple[dict[str, float], ...]
    learner_payload_sha256: str

    @property
    def steps(self) -> int:
        return int(self.obs.shape[0])

    def transition(self, index: int):
        return build_agent_transition(self.obs[index], self.action[index], self.next_obs[index], bool(self.terminated[index]), bool(self.truncated[index]))


def _sha256_payload(arrays: list[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(np.ascontiguousarray(array).tobytes(order="C"))
    return digest.hexdigest()


def _component_config() -> dict[str, Any]:
    path = Path(__file__).parent / "configs" / "hopper_component.yaml"
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _make_spec(config: dict[str, Any], steps: int) -> BenchmarkSpec:
    experiment = config["experiment"]
    stage_length = int(experiment["stage_length"])
    regimes = []
    for index, role in enumerate(experiment["schedule"]):
        regimes.append(Regime(index * stage_length, dict(experiment["perturbations"][role])))
    return BenchmarkSpec(
        name=str(experiment["benchmark"]),
        provider="nsgym",
        base_env=str(experiment["environment"]),
        parameter="hopper_physics",
        regimes=tuple(regimes),
        total_steps=steps,
        change_notification=False,
        delta_change_notification=False,
    )


def _synthetic_stream(config: dict[str, Any], seed: int, steps: int) -> FixedStream:
    """Deterministic CPU fixture for tests; it never substitutes for Hopper results."""

    experiment = config["experiment"]
    rng = np.random.default_rng(seed + 17)
    policy_rng = np.random.default_rng(seed + 23)
    obs_dim, action_dim = 11, 3
    obs = np.zeros(obs_dim, dtype=np.float32)
    previous_action = np.zeros(action_dim, dtype=np.float32)
    records: list[tuple[np.ndarray, np.ndarray, np.ndarray, bool, bool, float, int, str, int, bool, dict[str, float]]] = []
    roles = list(experiment["schedule"])
    stage_length = int(experiment["stage_length"])
    for step in range(steps):
        stage_index = min(step // stage_length, len(roles) - 1)
        role = roles[stage_index]
        parameters = dict(experiment["perturbations"][role])
        uniform = policy_rng.uniform(-1.0, 1.0, action_dim).astype(np.float32)
        if step < int(experiment["warmup_steps"]):
            action = uniform
        else:
            action = np.clip(0.8 * previous_action + 0.2 * uniform, -1.0, 1.0).astype(np.float32)
        scale = np.float32(parameters["torso_mass"] / parameters["floor_friction"] * parameters["thigh_joint_damping"])
        delta = np.zeros(obs_dim, dtype=np.float32)
        delta[:action_dim] = 0.05 * action
        delta[action_dim:] = 0.01 * scale
        delta += rng.normal(0.0, 0.002, obs_dim).astype(np.float32)
        next_obs = (obs + delta).astype(np.float32)
        records.append((obs.copy(), action.copy(), next_obs.copy(), False, False, float(np.linalg.norm(next_obs)), stage_index, role, stage_index // 3, step in {i * stage_length for i in range(1, 6)}, parameters))
        obs, previous_action = next_obs, action
    return _records_to_stream(records)


def _records_to_stream(records: list[tuple[Any, ...]]) -> FixedStream:
    obs, action, next_obs, terminated, truncated, reward, stage, dynamics, visit, change, params = zip(*records)
    arrays = [np.asarray(obs, dtype=np.float32), np.asarray(action, dtype=np.float32), np.asarray(next_obs, dtype=np.float32), np.asarray(terminated, dtype=np.bool_), np.asarray(truncated, dtype=np.bool_)]
    return FixedStream(*arrays, np.asarray(reward, dtype=np.float32), np.asarray(stage, dtype=np.int64), np.asarray(dynamics, dtype="U8"), np.asarray(visit, dtype=np.int64), np.asarray(change, dtype=np.bool_), tuple(dict(item) for item in params), _sha256_payload(arrays))


def generate_fixed_stream(output: str | Path, *, seed: int = 0, steps: int | None = None, synthetic: bool = False) -> FixedStream:
    config = _component_config()
    experiment = config["experiment"]
    total_steps = int(steps or (int(experiment["stage_length"]) * len(experiment["schedule"])))
    if synthetic:
        stream = _synthetic_stream(config, seed, total_steps)
    else:
        spec = _make_spec(config, total_steps)
        env = make_mujoco_benchmark(spec, root_seed=seed)
        policy_rng = np.random.default_rng(seed + 23)
        obs, _ = env.reset(seed=seed)
        previous_action = np.zeros(env.action_space.shape, dtype=np.float32)
        records = []
        try:
            for step in range(total_steps):
                uniform = policy_rng.uniform(env.action_space.low, env.action_space.high).astype(np.float32)
                if step < int(experiment["warmup_steps"]):
                    action = uniform
                else:
                    action = np.clip(0.8 * previous_action + 0.2 * uniform, env.action_space.low, env.action_space.high).astype(np.float32)
                next_obs, reward, terminated, truncated, info = env.step(action)
                meta = spec.regime_at(step)
                records.append((np.asarray(obs, dtype=np.float32), action.copy(), np.asarray(next_obs, dtype=np.float32), bool(terminated), bool(truncated), float(reward), step // int(experiment["stage_length"]), experiment["schedule"][min(step // int(experiment["stage_length"]), len(experiment["schedule"]) - 1)], min(step // int(experiment["stage_length"]), len(experiment["schedule"]) - 1) // 3, step in {regime.start_step for regime in spec.regimes[1:]}, dict(meta.parameters)))
                obs = next_obs
                previous_action = action
                if terminated or truncated:
                    obs, _ = env.reset()
        finally:
            env.close()
        stream = _records_to_stream(records)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, obs=stream.obs, action=stream.action, next_obs=stream.next_obs, terminated=stream.terminated, truncated=stream.truncated, true_reward=stream.true_reward, stage_id=stream.stage_id, dynamics_id=stream.dynamics_id, visit_id=stream.visit_id, change_event=stream.change_event)
    sidecar = path.with_suffix(".json")
    sidecar.write_text(json.dumps({"steps": stream.steps, "seed": seed, "learner_payload_sha256": stream.learner_payload_sha256, "parameter_vector": list(stream.parameter_vector)}, indent=2), encoding="utf-8")
    return stream


def load_fixed_stream(path: str | Path) -> FixedStream:
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        arrays = [data[name] for name in ("obs", "action", "next_obs", "terminated", "truncated")]
        sidecar = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        return FixedStream(*arrays, data["true_reward"], data["stage_id"], data["dynamics_id"], data["visit_id"], data["change_event"], tuple(sidecar["parameter_vector"]), sidecar["learner_payload_sha256"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/radius_validation/hopper_fixed_stream_seed0.npz")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--synthetic", action="store_true", help="CPU fixture for setup tests; not Hopper evidence")
    args = parser.parse_args()
    stream = generate_fixed_stream(args.output, seed=args.seed, steps=args.steps, synthetic=args.synthetic)
    print(json.dumps({"steps": stream.steps, "learner_payload_sha256": stream.learner_payload_sha256}))


if __name__ == "__main__":
    main()
