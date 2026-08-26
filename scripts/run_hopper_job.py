"""Run one protocol-faithful Hopper method/seed job."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import subprocess
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from pbcwm.benchmarks.base import BenchmarkSpec, Regime
from pbcwm.benchmarks.registry import load_benchmark_spec, make_benchmark
from pbcwm.core.device import configure_torch
from pbcwm.experiment.factory import build_method
from pbcwm.metrics.common import RolloutProbeBatch
from pbcwm.metrics.aggregation import normalized_auc
from pbcwm.metrics.planning import return_metrics
from pbcwm.metrics.world_model import nrmse_at_h, r2_at_H
from pbcwm.evaluation.preference_metrics import preference_accuracy
from pbcwm.planning.cem import CEMPlanner
from pbcwm.preferences.buffer import PreferenceBuffer
from pbcwm.preferences.query import DisagreementQuerySelector
from pbcwm.preferences.reward_model import PreferenceRewardEnsemble
from pbcwm.preferences.teacher import SyntheticPreferenceTeacher
from pbcwm.rewards.hopper import HopperPreferenceReward
from pbcwm.rewards.preference import LearnedPreferenceReward
from pbcwm.protocol.config import load_protocol_config
from pbcwm.protocol.seeds import spawn_seed_streams
from pbcwm.methods.radius import RadiusPbCWM


def _read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return value


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _merge(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    result = dict(left)
    for key, value in right.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class HopperPreferenceAdapter:
    """Attach the shared preference/CEM path to a dynamics-only learner."""

    def __init__(self, dynamics: Any, config: dict[str, Any], *, obs_dim: int, action_dim: int, action_low: np.ndarray, action_high: np.ndarray, device: torch.device, seed: int):
        self.dynamics = dynamics
        self.device = device
        preference = config["preference"]
        planner = dict(config["planner"])
        planner.update({"action_low": action_low, "action_high": action_high, "device": device})
        self.planner = CEMPlanner(**planner)
        self.reward_ensemble = PreferenceRewardEnsemble(
            obs_dim=int(obs_dim),
            action_dim=int(action_dim),
            ensemble_size=int(preference["ensemble_size"]),
            hidden_dims=tuple(preference["hidden_dims"]),
            learning_rate=float(preference["learning_rate"]),
            batch_size=int(preference["reward_batch_size"]),
            device=device,
            seed=seed + 1,
        )
        self.preference_buffer = PreferenceBuffer(seed=seed)
        self.query_selector = DisagreementQuerySelector(int(preference["pair_pool_size"]), seed=seed)
        self.teacher = SyntheticPreferenceTeacher(HopperPreferenceReward(), float(preference["teacher_skip_margin"]))
        self.learned_reward = LearnedPreferenceReward(self.reward_ensemble)
        self.min_preferences_before_planning = int(preference["min_preferences_before_planning"])
        self._rng = np.random.default_rng(seed)
        self._action_low = np.asarray(action_low, dtype=np.float32)
        self._action_high = np.asarray(action_high, dtype=np.float32)

    @property
    def dynamics_ready(self) -> bool:
        if hasattr(self.dynamics, "dynamics_ready"):
            return bool(self.dynamics.dynamics_ready)
        replay = getattr(self.dynamics, "replay_buffer", getattr(self.dynamics, "replay", ()))
        if hasattr(self.dynamics, "batch_size"):
            batch_size = int(self.dynamics.batch_size)
        else:
            batch_size = int(self.dynamics.config.training.batch_size)
        return len(replay) >= batch_size

    def observe(self, transition: Any) -> None:
        self.dynamics.observe(transition)

    def update_dynamics(self, num_steps: int = 1) -> dict[str, float]:
        return self.dynamics.update(num_steps)

    def plan(self, obs: Any, collect_candidates: bool = False):
        return self.planner.plan(obs, self.dynamics, self.learned_reward, return_candidates=collect_candidates)

    def bootstrap(self, num_queries: int, horizon: int, action_low: np.ndarray, action_high: np.ndarray) -> dict[str, float]:
        starts = self._seed_observations()
        if not starts or num_queries <= 0:
            return self.preference_metrics()
        target = len(self.preference_buffer) + int(num_queries)
        attempts = 0
        while len(self.preference_buffer) < target and attempts < max(4 * num_queries, 1):
            attempts += 1
            start = starts[int(self._rng.integers(len(starts)))]
            example = self._label_example(self._random_rollout(start, horizon, action_low, action_high), self._random_rollout(start, horizon, action_low, action_high))
            if example is not None:
                self.preference_buffer.add(example)
        if len(self.preference_buffer) >= self.reward_ensemble.batch_size:
            return {**self.preference_metrics(), **self.reward_ensemble.update(self.preference_buffer, max(1, min(10, num_queries)))}
        return self.preference_metrics()

    def query_and_update(self, candidates: list[Any], num_queries: int, reward_updates: int) -> dict[str, float]:
        scored = self.query_selector.score_pairs_with_ensemble(candidates, self.reward_ensemble)
        selected = scored[: max(0, num_queries)]
        for (index_a, index_b), _score in selected:
            example = self._label_example(candidates[index_a], candidates[index_b])
            if example is not None:
                self.preference_buffer.add(example)
        return {**self.preference_metrics(), **self.reward_ensemble.update(self.preference_buffer, reward_updates)}

    def query_and_update_radius(self, obs: Any, candidates: list[Any], num_queries: int, reward_updates: int) -> dict[str, float]:
        # Reuse the method's PFPA selector while keeping the candidate bank
        # produced by the shared CEM call (no second rollout or hidden budget).
        self.planner.candidate_trajectories = candidates
        selection = self.dynamics.select_preference_queries(obs, self.planner, self.reward_ensemble, num_queries)
        for index_a, index_b in selection.pairs:
            example = self._label_example(candidates[index_a], candidates[index_b])
            if example is not None:
                self.preference_buffer.add(example)
        return {**self.preference_metrics(), **self.reward_ensemble.update(self.preference_buffer, reward_updates)}

    def preference_metrics(self) -> dict[str, float]:
        return {"num_preferences": float(len(self.preference_buffer))}

    def generate_preference_examples(self, num_examples: int, horizon: int, action_low: np.ndarray, action_high: np.ndarray, seed: int):
        starts = self._seed_observations()
        if not starts:
            return []
        rng = np.random.default_rng(seed)
        examples = []
        attempts = 0
        while len(examples) < num_examples and attempts < max(4 * num_examples, 1):
            attempts += 1
            start = starts[int(rng.integers(len(starts)))]
            example = self._label_example(
                self._random_rollout(start, horizon, action_low, action_high, rng),
                self._random_rollout(start, horizon, action_low, action_high, rng),
            )
            if example is not None:
                examples.append(example)
        return examples

    def _seed_observations(self) -> list[np.ndarray]:
        replay = getattr(self.dynamics, "replay_buffer", None)
        if replay is not None:
            return [np.asarray(item.obs, dtype=np.float32).copy() for item in replay]
        replay = getattr(self.dynamics, "replay", None)
        if replay is not None and hasattr(replay, "storage"):
            return [item.obs.detach().cpu().numpy().astype(np.float32) for item in replay.storage]
        seed_fn = getattr(self.dynamics, "seed_observations", None)
        return [] if not callable(seed_fn) else list(seed_fn())

    def _random_rollout(self, start_obs: np.ndarray, horizon: int, action_low: np.ndarray, action_high: np.ndarray, rng: np.random.Generator | None = None):
        from pbcwm.preferences.types import TrajectorySegment
        rng = self._rng if rng is None else rng
        obs = torch.as_tensor(start_obs, dtype=torch.float32, device=self.device)
        observations, actions, next_observations = [], [], []
        with torch.no_grad():
            for _ in range(horizon):
                action_np = rng.uniform(action_low, action_high).astype(np.float32)
                action = torch.as_tensor(action_np, dtype=torch.float32, device=self.device)
                next_obs = self.dynamics.predict(obs.unsqueeze(0), action.unsqueeze(0)).squeeze(0)
                observations.append(obs.clone())
                actions.append(action.clone())
                next_observations.append(next_obs.clone())
                obs = next_obs
        return TrajectorySegment(torch.stack(observations), torch.stack(actions), torch.stack(next_observations))

    def _label_example(self, traj_a: Any, traj_b: Any):
        from pbcwm.preferences.types import PreferenceExample
        label = self.teacher.label(traj_a, traj_b)
        return None if label is None else PreferenceExample(traj_a, traj_b, label)


class _ZeroReward:
    """Evaluator-only reward ablation with the same model and CEM settings."""

    def __call__(self, obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
        del action, next_obs
        return torch.zeros(obs.shape[0], dtype=obs.dtype, device=obs.device)


def _stationary_spec(source: BenchmarkSpec, parameters: dict[str, float]) -> BenchmarkSpec:
    return BenchmarkSpec(source.name, source.provider, source.base_env, source.parameter, (Regime(0, parameters),), 1000, False, False, source.fixed_parameters)


def _safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (int, str, bool)) or value is None:
        return value
    return str(value)


def _probe(env: Any, *, seed: int, count: int = 32, horizon: int = 5) -> RolloutProbeBatch:
    rng = np.random.default_rng(seed)
    initials, action_rows, state_rows = [], [], []
    for index in range(count):
        accepted = False
        for attempt in range(20):
            obs, _ = env.reset(seed=seed + index * 100 + attempt)
            actions, states = [], [np.asarray(obs, dtype=np.float32).copy()]
            done = False
            for _ in range(horizon):
                action = rng.uniform(-0.25, 0.25, size=env.action_space.shape).astype(np.float32)
                next_obs, _reward, terminated, truncated, _info = env.step(action)
                actions.append(action)
                states.append(np.asarray(next_obs, dtype=np.float32).copy())
                done = bool(terminated or truncated)
                if done:
                    break
            if len(actions) == horizon and not done:
                initials.append(states[0])
                action_rows.append(actions)
                state_rows.append(states)
                accepted = True
                break
        if not accepted:
            raise RuntimeError(f"failed to create non-terminal Hopper probe {index}")
    return RolloutProbeBatch(torch.as_tensor(np.stack(initials)), torch.as_tensor(np.stack(action_rows)), torch.as_tensor(np.stack(state_rows)))


def _evaluate_cem_returns(
    env: Any,
    learner: Any,
    *,
    seed: int,
    episodes: int,
    horizon: int,
    replan_interval: int,
    reward_fn: Any | None = None,
) -> dict[str, float]:
    """Evaluate true Hopper return while keeping reward and physics evaluator-only."""

    returns: list[float] = []
    for episode_index in range(episodes):
        obs, _ = env.reset(seed=seed + episode_index)
        total = 0.0
        cached_plan = None
        cached_plan_step = -1
        for step in range(horizon):
            age = step - cached_plan_step
            sequence = None if cached_plan is None else getattr(cached_plan, "best_action_sequence", None)
            if cached_plan is None or age >= replan_interval or sequence is None or age >= len(sequence):
                if reward_fn is None:
                    cached_plan = learner.plan(obs, collect_candidates=False)
                else:
                    cached_plan = learner.planner.plan(obs, learner.dynamics, reward_fn, return_candidates=False)
                cached_plan_step = step
                action = cached_plan.action
            else:
                action = sequence[age].detach().cpu().numpy().astype(np.float32)
            next_obs, true_reward, terminated, truncated, _info = env.step(action)
            total += float(true_reward)
            if terminated or truncated:
                break
            obs = next_obs
        returns.append(total)
    metrics = return_metrics(returns)
    return {
        "planning_return_mean": float(metrics["planning/return_mean"].value),
        "planning_return_std": float(metrics["planning/return_std"].value),
        "planning_episode_count": float(episodes),
    }


def _build_agent(method: str, cfg: dict[str, Any], *, obs_dim: int, action_dim: int, low: np.ndarray, high: np.ndarray, device: torch.device, seed: int):
    method_cfg = _merge(cfg["method_defaults"], cfg.get("methods", {}).get(method, {}))
    # CEM owns its own torch generator; make the per-job seed explicit for
    # every wrapper instead of relying on the generator's nondeterministic
    # default seeding.
    method_cfg["planner"] = dict(method_cfg["planner"])
    method_cfg["planner"]["seed"] = int(seed)
    if method == "radius_pb_cwm":
        radius_path = _resolve(Path.cwd(), str(method_cfg["radius_config"]))
        radius_data = _read_yaml(radius_path)
        dynamics = build_method("radius_pb_cwm", obs_dim=obs_dim, action_dim=action_dim, action_low=low, action_high=high, config={"radius": radius_data}, device=device, seed=seed)
        return HopperPreferenceAdapter(dynamics, method_cfg, obs_dim=obs_dim, action_dim=action_dim, action_low=low, action_high=high, device=device, seed=seed)
    if method == "static":
        dynamics = build_method("static", obs_dim=obs_dim, action_dim=action_dim, action_low=low, action_high=high, config=method_cfg, device=device, seed=seed)
        return HopperPreferenceAdapter(dynamics, method_cfg, obs_dim=obs_dim, action_dim=action_dim, action_low=low, action_high=high, device=device, seed=seed)
    return build_method(method, obs_dim=obs_dim, action_dim=action_dim, action_low=low, action_high=high, config=method_cfg, device=device, seed=seed, teacher_reward=HopperPreferenceReward())


def run_job(config_path: str | Path, method: str, seed: int, device: str, output_dir: str | Path) -> dict[str, Any]:
    root = Path.cwd()
    cfg = _read_yaml(config_path)
    campaign = cfg["campaign"]
    if method not in campaign["methods"] and method != "static":
        raise ValueError(f"method is not in campaign config: {method}")
    if int(seed) not in [int(value) for value in campaign["seeds"]]:
        raise ValueError(f"seed is not in campaign config: {seed}")
    resolved_device = configure_torch(device)
    protocol_path = _resolve(root, str(campaign["protocol_config"]))
    benchmark_path = _resolve(root, str(campaign["benchmark_config"]))
    protocol = load_protocol_config(protocol_path)
    benchmark = load_benchmark_spec(benchmark_path)
    env = make_benchmark(benchmark.name, benchmark, root_seed=seed)
    low = np.asarray(env.action_space.low, dtype=np.float32)
    high = np.asarray(env.action_space.high, dtype=np.float32)
    _seed_everything(seed)
    agent = _build_agent(method, cfg, obs_dim=int(env.observation_space.shape[0]), action_dim=int(env.action_space.shape[0]), low=low, high=high, device=resolved_device, seed=seed)
    runner = __import__("pbcwm.experiment", fromlist=["CanonicalLifetimeRunner"]).CanonicalLifetimeRunner(protocol, campaign["environment"], seed, episode_length=1000)
    evaluations: list[dict[str, Any]] = []

    def query_handler(query: Any, learner: Any, planner: Any, reward_fn: Any, obs: Any) -> int:
        del planner, reward_fn
        preference_count_before = len(learner.preference_buffer)
        if query.bootstrap:
            learner.bootstrap(query.pair_count, 20, low, high)
        else:
            result = learner.plan(obs, collect_candidates=True)
            candidates = result.candidate_trajectories
            if len(candidates) < 2:
                raise RuntimeError(f"{method} produced fewer than two query candidates at step {query.global_step}")
            if isinstance(learner.dynamics, RadiusPbCWM):
                learner.query_and_update_radius(obs, candidates, query.pair_count, int(protocol.reward_model.updates_per_query_round))
            else:
                learner.query_and_update(candidates, query.pair_count, int(protocol.reward_model.updates_per_query_round))
        produced = len(learner.preference_buffer) - preference_count_before
        if produced != query.pair_count:
            raise RuntimeError(
                f"{method} produced {produced}/{query.pair_count} preference labels at step {query.global_step}; "
                "the protocol budget cannot silently consume skipped labels"
            )
        return query.pair_count

    def eval_env_factory(checkpoint: Any, eval_seed: int) -> Any:
        parameters = dict(benchmark.regimes[min(int(checkpoint.segment_id), len(benchmark.regimes) - 1)].parameters)
        return make_benchmark(benchmark.name, _stationary_spec(benchmark, parameters), root_seed=eval_seed + int(checkpoint.segment_id))

    def evaluation_handler(checkpoint: Any, learner: Any, eval_context: Any) -> dict[str, Any]:
        eval_env = eval_context["env"]
        probe = _probe(eval_env, seed=int(eval_context["evaluation_seed"]) + checkpoint.global_step, count=24, horizon=5)
        dynamics = learner.dynamics
        eval_error = None
        try:
            h1 = nrmse_at_h(dynamics, probe, 1)
            h5 = nrmse_at_h(dynamics, probe, 5)
            r2 = r2_at_H(dynamics, probe, 5)
        except (RuntimeError, ValueError, FloatingPointError) as exc:
            # A model that has not reached its minimum fit buffer at the
            # initial checkpoint is an expected undefined metric, not a job
            # failure.  Keep it explicit and fail closed in aggregation.
            eval_error = repr(exc)
            h1 = h5 = r2 = None
        preference_error = None
        preference_examples = []
        preference_accuracy_value = None
        try:
            generate_preferences = getattr(learner, "generate_preference_examples", None)
            if callable(generate_preferences):
                preference_examples = generate_preferences(
                    protocol.evaluation.heldout_preference_pairs,
                    protocol.evaluation.heldout_preference_horizon,
                    low,
                    high,
                    int(eval_context["evaluation_seed"]) + checkpoint.global_step,
                )
                if preference_examples:
                    preference_accuracy_value = preference_accuracy(learner.reward_ensemble, preference_examples)
        except (RuntimeError, ValueError, FloatingPointError) as exc:
            preference_error = repr(exc)
        planning_error = None
        reward_ablation_error = None
        planning_metrics: dict[str, float | None] = {}
        if checkpoint.few_shot_interactions is None and checkpoint.normalized_fraction >= 1.0:
            try:
                initial_planner_state = copy.deepcopy(learner.planner.state_dict())
                planning_metrics = _evaluate_cem_returns(
                    eval_env,
                    learner,
                    seed=int(eval_context["evaluation_seed"]) + checkpoint.global_step,
                    episodes=protocol.evaluation.planning_episodes_stage_end,
                    horizon=protocol.evaluation.planning_episode_horizon,
                    replan_interval=protocol.planner_replan_interval,
                )
                if protocol.evaluation.reward_ablation_episodes_stage_end:
                    learner.planner.load_state_dict(initial_planner_state)
                    zero_metrics = _evaluate_cem_returns(
                        eval_env,
                        learner,
                        seed=int(eval_context["evaluation_seed"]) + checkpoint.global_step,
                        episodes=protocol.evaluation.reward_ablation_episodes_stage_end,
                        horizon=protocol.evaluation.planning_episode_horizon,
                        replan_interval=protocol.planner_replan_interval,
                        reward_fn=_ZeroReward(),
                    )
                    planning_metrics["planning_return_zero_reward_mean"] = zero_metrics["planning_return_mean"]
                    planning_metrics["planning_return_reward_advantage"] = (
                        planning_metrics["planning_return_mean"] - zero_metrics["planning_return_mean"]
                    )
            except (RuntimeError, ValueError, FloatingPointError) as exc:
                if planning_metrics:
                    reward_ablation_error = repr(exc)
                else:
                    planning_error = repr(exc)
        diagnostics = dynamics.diagnostics() if callable(getattr(dynamics, "diagnostics", None)) else {}
        result = {
            "global_step": int(checkpoint.global_step),
            "segment_id": int(checkpoint.segment_id),
            "dynamics_id": str(checkpoint.dynamics_id),
            "visit_id": int(checkpoint.visit_id),
            "normalized_fraction": float(checkpoint.normalized_fraction),
            "few_shot_interactions": checkpoint.few_shot_interactions,
            "wm_nrmse_h1": None if h1 is None or not h1.valid else h1.value,
            "wm_nrmse_h5": None if h5 is None or not h5.valid else h5.value,
            "wm_r2_h5": None if r2 is None or not r2.valid else r2.value,
            "evaluation_error": eval_error,
            "reward_proxy_preference_accuracy": preference_accuracy_value,
            "reward_proxy_preference_pair_count": len(preference_examples),
            "reward_evaluation_error": preference_error,
            **planning_metrics,
            "planning_evaluation_error": planning_error,
            "reward_ablation_evaluation_error": reward_ablation_error,
            "diagnostics": _safe(diagnostics),
        }
        evaluations.append(result)
        return result

    job_dir = Path(output_dir) / method / f"seed_{seed}"
    job_dir.mkdir(parents=True, exist_ok=True)
    config_hash = hashlib.sha256(Path(config_path).read_bytes()).hexdigest()
    protocol_hash = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    benchmark_hash = hashlib.sha256(benchmark_path.read_bytes()).hexdigest()
    status_path = job_dir / "status.json"
    status_path.write_text(json.dumps({"status": "RUNNING", "method": method, "seed": seed}, indent=2), encoding="utf-8")
    try:
        summary = runner.run(env, agent, agent.planner, reward_fn=agent.learned_reward, query_handler=query_handler, evaluation_handler=evaluation_handler, eval_env_factory=eval_env_factory, log_path=job_dir / "protocol.jsonl")
        grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
        for record in evaluations:
            if record["few_shot_interactions"] is None and record["wm_nrmse_h5"] is not None:
                grouped[(record["segment_id"], record["visit_id"])].append(record)
        acq, reacq = [], []
        for (segment_id, visit_id), records in grouped.items():
            records.sort(key=lambda item: item["normalized_fraction"])
            if len(records) < 2:
                continue
            auc = normalized_auc([item["normalized_fraction"] for item in records], [item["wm_nrmse_h5"] for item in records])
            (reacq if visit_id > 0 else acq).append(auc)
        reward_stage_end = [
            float(record["reward_proxy_preference_accuracy"])
            for record in evaluations
            if record["few_shot_interactions"] is None
            and record["normalized_fraction"] >= 1.0
            and isinstance(record.get("reward_proxy_preference_accuracy"), (int, float))
        ]
        planning_stage_end = [
            float(record["planning_return_mean"])
            for record in evaluations
            if isinstance(record.get("planning_return_mean"), (int, float))
        ]
        reward_advantages = [
            float(record["planning_return_reward_advantage"])
            for record in evaluations
            if isinstance(record.get("planning_return_reward_advantage"), (int, float))
        ]
        acq_mean = float(np.mean(acq)) if acq else None
        reacq_mean = float(np.mean(reacq)) if reacq else None
        result = {
            "status": "COMPLETE",
            "method": method,
            "seed": int(seed),
            "device": str(resolved_device),
            "benchmark": benchmark.name,
            "protocol_version": protocol.version,
            "schedule": ["P0", "A", "B", "C", "B", "A"],
            "steps": int(summary.steps),
            "queries": int(summary.queries),
            "metrics": {
                "wm_acq_nrmse_auc_mean": acq_mean,
                "wm_reacq_nrmse_auc_mean": reacq_mean,
                "wm_reacq_advantage_nrmse": None if acq_mean is None or reacq_mean is None else acq_mean - reacq_mean,
                "wm_acq_nrmse_auc_count": len(acq),
                "wm_reacq_nrmse_auc_count": len(reacq),
                "reward_proxy_preference_accuracy_stage_end_mean": float(np.mean(reward_stage_end)) if reward_stage_end else None,
                "reward_proxy_preference_accuracy_stage_end_count": len(reward_stage_end),
                "planning_return_stage_end_mean": float(np.mean(planning_stage_end)) if planning_stage_end else None,
                "planning_return_final": planning_stage_end[-1] if planning_stage_end else None,
                "planning_return_stage_end_count": len(planning_stage_end),
                "planning_reward_advantage_stage_end_mean": float(np.mean(reward_advantages)) if reward_advantages else None,
                "planning_reward_advantage_stage_end_count": len(reward_advantages),
            },
            "evaluations": evaluations,
            "runner_summary": summary.to_dict(),
            "config_sha256": config_hash,
            "protocol_sha256": protocol_hash,
            "benchmark_sha256": benchmark_hash,
            "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip(),
        }
        (job_dir / "summary.json").write_text(json.dumps(_safe(result), indent=2, allow_nan=False), encoding="utf-8")
        status_path.write_text(json.dumps({"status": "COMPLETE", "summary": "summary.json"}, indent=2), encoding="utf-8")
        return result
    except Exception as exc:
        failure = {"status": "FAILED", "method": method, "seed": int(seed), "error": repr(exc), "traceback": traceback.format_exc()}
        status_path.write_text(json.dumps(failure, indent=2), encoding="utf-8")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="pbcwm/configs/hopper_campaign.yaml")
    parser.add_argument("--method", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="outputs/hopper_screen_v1")
    args = parser.parse_args()
    result = run_job(args.config, args.method, args.seed, args.device, args.output_dir)
    print(json.dumps({"status": result["status"], "method": result["method"], "seed": result["seed"], "metrics": result["metrics"]}, allow_nan=False))


if __name__ == "__main__":
    main()
