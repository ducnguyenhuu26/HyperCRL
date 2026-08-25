"""Evaluator-only frozen open-loop probe banks for component validation."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from pbcwm.benchmarks.base import BenchmarkSpec, Regime
from pbcwm.benchmarks.nsgym.mujoco import make_mujoco_benchmark


@dataclass(frozen=True)
class DynamicsProbe:
    initial_obs: np.ndarray
    actions: np.ndarray
    true_obs: np.ndarray


@dataclass(frozen=True)
class DynamicsProbeBank:
    dynamics_id: str
    probes: tuple[DynamicsProbe, ...]

    @property
    def horizon(self) -> int:
        return int(self.probes[0].actions.shape[0]) if self.probes else 0

    @property
    def obs_dim(self) -> int:
        return int(self.probes[0].initial_obs.shape[0]) if self.probes else 0


def probe_bank_sha256(bank: DynamicsProbeBank) -> str:
    digest = hashlib.sha256()
    digest.update(bank.dynamics_id.encode("utf-8"))
    for probe in bank.probes:
        for array in (probe.initial_obs, probe.actions, probe.true_obs):
            digest.update(np.ascontiguousarray(array, dtype=np.float32).tobytes())
    return digest.hexdigest()


def generate_probe_bank(
    dynamics_id: str,
    env_factory: Callable[[], object],
    *,
    seed: int,
    n_probes: int = 128,
    horizon: int = 20,
    max_attempts_per_probe: int = 32,
) -> DynamicsProbeBank:
    """Generate probes and reject any rollout containing a reset boundary."""

    if n_probes < 2 or horizon <= 0:
        raise ValueError("probe banks need at least two probes and a positive horizon")
    probes: list[DynamicsProbe] = []
    for probe_index in range(n_probes):
        accepted = False
        for attempt in range(max_attempts_per_probe):
            env = env_factory()
            probe_seed = int(seed + probe_index * 1_000_003 + attempt * 97)
            action_rng = np.random.default_rng(probe_seed + 17)
            try:
                initial_obs, _ = env.reset(seed=probe_seed)
                actions: list[np.ndarray] = []
                true_obs: list[np.ndarray] = []
                low = np.asarray(env.action_space.low, dtype=np.float32)
                high = np.asarray(env.action_space.high, dtype=np.float32)
                valid = True
                for _ in range(horizon):
                    action = action_rng.uniform(low, high).astype(np.float32)
                    next_obs, _reward, terminated, truncated, _info = env.step(action)
                    if terminated or truncated:
                        valid = False
                        break
                    actions.append(action.copy())
                    true_obs.append(np.asarray(next_obs, dtype=np.float32).copy())
                if valid and len(actions) == horizon:
                    probes.append(DynamicsProbe(np.asarray(initial_obs, dtype=np.float32).copy(), np.stack(actions), np.stack(true_obs)))
                    accepted = True
                    break
            finally:
                close = getattr(env, "close", None)
                if callable(close):
                    close()
        if not accepted:
            raise RuntimeError(f"could not generate a non-terminating probe for {dynamics_id} index {probe_index}")
    return DynamicsProbeBank(str(dynamics_id), tuple(probes))


def save_probe_bank(path: str | Path, bank: DynamicsProbeBank) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        dynamics_id=np.asarray(bank.dynamics_id),
        initial_obs=np.stack([probe.initial_obs for probe in bank.probes]),
        actions=np.stack([probe.actions for probe in bank.probes]),
        true_obs=np.stack([probe.true_obs for probe in bank.probes]),
    )
    path.with_suffix(".json").write_text(
        json.dumps({"dynamics_id": bank.dynamics_id, "probes": len(bank.probes), "horizon": bank.horizon, "sha256": probe_bank_sha256(bank)}, indent=2),
        encoding="utf-8",
    )


def load_probe_bank(path: str | Path) -> DynamicsProbeBank:
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        dynamics_id = str(data["dynamics_id"].item())
        initial_obs = np.asarray(data["initial_obs"], dtype=np.float32)
        actions = np.asarray(data["actions"], dtype=np.float32)
        true_obs = np.asarray(data["true_obs"], dtype=np.float32)
    if (
        initial_obs.ndim != 2
        or actions.ndim != 3
        or true_obs.ndim != 3
        or actions.shape[:2] != true_obs.shape[:2]
        or initial_obs.shape[0] != actions.shape[0]
        or initial_obs.shape[0] < 2
        or actions.shape[1] <= 0
        or actions.shape[2] <= 0
        or not all(np.isfinite(array).all() for array in (initial_obs, actions, true_obs))
    ):
        raise ValueError("invalid probe-bank array shapes")
    return DynamicsProbeBank(
        dynamics_id,
        tuple(DynamicsProbe(initial_obs[i], actions[i], true_obs[i]) for i in range(initial_obs.shape[0])),
    )


def _hopper_spec(dynamics_id: str, parameters: dict[str, float], horizon: int) -> BenchmarkSpec:
    return BenchmarkSpec(
        name=f"radius-probe-{dynamics_id}",
        provider="nsgym",
        base_env="Hopper-v5",
        parameter="hopper_physics",
        regimes=(Regime(0, parameters),),
        total_steps=horizon,
        change_notification=False,
        delta_change_notification=False,
    )


def generate_hopper_probe_banks(config_path: str | Path, output_dir: str | Path, *, seed: int = 0, n_probes: int = 128, horizon: int = 20) -> dict[str, str]:
    import yaml

    with Path(config_path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    experiment = config["experiment"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for role in dict.fromkeys(experiment["schedule"]):
        parameters = {str(key): float(value) for key, value in experiment["perturbations"][role].items()}
        spec = _hopper_spec(role, parameters, horizon)
        bank = generate_probe_bank(
            role,
            lambda spec=spec: make_mujoco_benchmark(spec, root_seed=seed),
            seed=seed + len(paths) * 10_000,
            n_probes=n_probes,
            horizon=horizon,
        )
        path = output_dir / f"{role}.npz"
        save_probe_bank(path, bank)
        paths[role] = str(path)
    return paths


def generate_synthetic_probe_banks(config_path: str | Path, output_dir: str | Path, *, seed: int = 0, n_probes: int = 32, horizon: int = 20) -> dict[str, str]:
    """Create deterministic plumbing probes; these are never Hopper evidence."""

    import yaml

    with Path(config_path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    experiment = config["experiment"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for role in dict.fromkeys(experiment["schedule"]):
        rng = np.random.default_rng(seed + len(paths) * 1009)
        scale = experiment["perturbations"][role]["torso_mass"] / experiment["perturbations"][role]["floor_friction"] * experiment["perturbations"][role]["thigh_joint_damping"]
        probes: list[DynamicsProbe] = []
        for _ in range(n_probes):
            initial = rng.normal(0.0, 0.2, 11).astype(np.float32)
            current = initial.copy()
            actions = rng.uniform(-1.0, 1.0, (horizon, 3)).astype(np.float32)
            true_obs = []
            for action in actions:
                delta = np.full(11, 0.01 * scale, dtype=np.float32)
                delta[:3] += 0.05 * action
                current = current + delta
                true_obs.append(current.copy())
            probes.append(DynamicsProbe(initial, actions, np.stack(true_obs)))
        bank = DynamicsProbeBank(role, tuple(probes))
        path = output_dir / f"{role}.npz"
        save_probe_bank(path, bank)
        paths[role] = str(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="pbcwm/experiments/radius_validation/configs/hopper_component.yaml")
    parser.add_argument("--output-dir", default="outputs/radius_validation/probe_banks_seed0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--probes", type=int, default=128)
    parser.add_argument("--horizon", type=int, default=20)
    args = parser.parse_args()
    paths = generate_hopper_probe_banks(args.config, args.output_dir, seed=args.seed, n_probes=args.probes, horizon=args.horizon)
    print(json.dumps(paths, indent=2))


if __name__ == "__main__":
    main()
