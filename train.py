"""Run the baseline-agnostic PB-CWM Pendulum smoke test."""

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from pbcwm.baselines.static import StaticDynamicsLearner
from pbcwm.baselines.moprl_online_ft import MoPRLOnlineFT
from pbcwm.baselines.gpmm.online import GPMMOnline
from pbcwm.baselines.hypercrl.online import HyperCRLAdaptOnline
from pbcwm.baselines.vblrl.online import VBLRLAdaptOnline
from pbcwm.baselines.curious_replay.online import CuriousReplayOnline
from pbcwm.core.types import Transition
from pbcwm.envs.nonstationary_pendulum import NonstationaryPendulum
from pbcwm.evaluation.metrics import evaluate_dynamics
from pbcwm.evaluation.gpmm_metrics import (
    assignment_contingency,
    assignment_purity,
    expert_reuse_rate,
)
from pbcwm.evaluation.hypercrl_metrics import (
    embedding_contingency,
    embedding_purity,
    embedding_reuse_rate,
)
from pbcwm.evaluation.vblrl_metrics import (
    posterior_contingency,
    posterior_purity,
    posterior_reuse_rate,
)
from pbcwm.evaluation.curious_replay_metrics import (
    first_recovery_step,
    first_return_recovery_step,
    replay_stage_share,
    sampled_stage_counts,
)
from pbcwm.evaluation.preference_metrics import preference_accuracy
from pbcwm.planning.cem import CEMPlanner
from pbcwm.rewards.pendulum import PendulumReward


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_components(config: dict[str, Any], seed: int):
    env = NonstationaryPendulum(config["env"]["dynamics_schedule"])
    env.reset(seed=seed)
    env.action_space.seed(seed)
    dynamics = StaticDynamicsLearner(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        hidden_dims=config["model"]["hidden_dims"],
        learning_rate=config["model"]["learning_rate"],
        replay_capacity=config["training"]["replay_capacity"],
        batch_size=config["training"]["batch_size"],
        device=config["device"],
        seed=seed,
    )
    planner = CEMPlanner(
        **config["planner"],
        action_low=env.action_space.low,
        action_high=env.action_space.high,
        device=config["device"],
    )
    return env, dynamics, planner, PendulumReward()


def run(config: dict[str, Any], total_steps: int | None = None) -> Path:
    if config.get("baseline", {}).get("name") == "moprl_online_ft":
        return run_moprl_online_ft(config, total_steps=total_steps)
    if config.get("baseline", {}).get("name") == "gpmm":
        return run_gpmm(config, total_steps=total_steps)
    if config.get("baseline", {}).get("name") == "hypercrl_adapt":
        return run_hypercrl_adapt(config, total_steps=total_steps)
    if config.get("baseline", {}).get("name") == "vblrl_adapt":
        return run_vblrl_adapt(config, total_steps=total_steps)
    if config.get("baseline", {}).get("name") == "curious_replay_adapt":
        return run_curious_replay_adapt(config, total_steps=total_steps)

    seed = int(config["seed"])
    set_global_seed(seed)
    env, dynamics, planner, reward_fn = build_components(config, seed)
    training = config["training"]
    total_steps = int(total_steps if total_steps is not None else training["total_steps"])
    log_path = Path(training["log_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    evaluation_capacity = int(training["evaluation_capacity"])
    evaluation_fraction = float(training["evaluation_fraction"])
    evaluation_buffers: dict[int, list[Transition]] = {}
    episode_rewards: list[float] = []
    episode_return_value = 0.0
    obs, info = env.reset(seed=seed)

    def evaluation_snapshot(global_step: int, stage: int, loss: float) -> dict[str, Any]:
        record: dict[str, Any] = {
            "kind": "step",
            "global_step": global_step,
            "true_dynamics_stage": stage,
            "train_loss": loss,
        }
        for stage_id, transitions in sorted(evaluation_buffers.items()):
            record[f"prediction_mse_stage_{stage_id}"] = evaluate_dynamics(
                dynamics, transitions
            )["prediction_mse"]
        return record

    with log_path.open("w", encoding="utf-8") as log_file:
        for step in range(total_steps):
            if step < int(training["random_steps"]):
                action = env.action_space.sample()
            else:
                action = planner.act(obs, dynamics, reward_fn)
            next_obs, env_reward, terminated, truncated, info = env.step(action)
            transition = Transition(
                obs=np.asarray(obs, dtype=np.float32),
                action=np.asarray(action, dtype=np.float32),
                next_obs=np.asarray(next_obs, dtype=np.float32),
                reward=float(env_reward),
                terminated=bool(terminated),
                truncated=bool(truncated),
            )
            if np.random.random() < evaluation_fraction:
                stage = int(info["true_dynamics_stage"])
                evaluation_buffers.setdefault(stage, [])
                if len(evaluation_buffers[stage]) < evaluation_capacity:
                    evaluation_buffers[stage].append(transition)
            else:
                dynamics.observe(transition)
            diagnostics = {"loss": 0.0}
            if step >= int(training["learning_starts"]):
                diagnostics = dynamics.update(int(training["updates_per_step"]))
            episode_return_value += float(env_reward)

            if (step + 1) % int(training["log_interval"]) == 0 or step == total_steps - 1:
                record = evaluation_snapshot(
                    step + 1, int(info["true_dynamics_stage"]), diagnostics["loss"]
                )
                log_file.write(json.dumps(record, allow_nan=True) + "\n")
                log_file.flush()
                print(json.dumps(record, allow_nan=True))

            if terminated or truncated:
                episode_rewards.append(episode_return_value)
                log_file.write(
                    json.dumps(
                        {
                            "kind": "episode",
                            "global_step": step + 1,
                            "episode_return": episode_return_value,
                        }
                    )
                    + "\n"
                )
                episode_return_value = 0.0
                obs, info = env.reset()
            else:
                obs = next_obs

        summary = {
            "kind": "summary",
            "global_step": total_steps,
            "episodes": len(episode_rewards),
            "mean_episode_return": float(np.mean(episode_rewards)) if episode_rewards else None,
        }
        log_file.write(json.dumps(summary) + "\n")
        print(json.dumps(summary))
    env.close()
    return log_path


def run_moprl_online_ft(config: dict[str, Any], total_steps: int | None = None) -> Path:
    """Run Phase-1 PbRL with learned reward planning and naive online FT."""

    seed = int(config["seed"])
    set_global_seed(seed)
    env = NonstationaryPendulum(config["env"]["dynamics_schedule"])
    env.reset(seed=seed)
    env.action_space.seed(seed)
    training = config["training"]
    preference = config["preference"]
    planner_config = dict(config["planner"])
    planner_config["candidate_keep_per_iteration"] = preference["candidate_keep_per_iteration"]
    planner_config["candidate_keep_final_elites"] = preference["candidate_keep_final_elites"]

    baseline = MoPRLOnlineFT(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        action_low=env.action_space.low,
        action_high=env.action_space.high,
        planner_config=planner_config,
        model_hidden_dims=config["model"]["hidden_dims"],
        model_learning_rate=config["model"]["learning_rate"],
        dynamics_window_size=config["baseline"]["dynamics_window_size"],
        dynamics_batch_size=training["dynamics_batch_size"],
        preference_ensemble_size=preference["ensemble_size"],
        preference_hidden_dims=preference["hidden_dims"],
        preference_learning_rate=preference["learning_rate"],
        preference_batch_size=preference["reward_batch_size"],
        min_preferences_before_planning=preference["min_preferences_before_planning"],
        pair_pool_size=preference["pair_pool_size"],
        teacher_skip_margin=preference["teacher_skip_margin"],
        teacher_reward=PendulumReward(),
        device=config["device"],
        seed=seed,
    )

    total_steps = int(total_steps if total_steps is not None else training["total_steps"])
    log_path = Path(training["log_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    evaluation_buffers: dict[int, list[Transition]] = {}
    heldout_examples = []
    bootstrap_done = False
    query_round = 0
    last_preference_metrics: dict[str, float] = baseline.preference_metrics()
    episode_rewards: list[float] = []
    episode_return_value = 0.0
    obs, info = env.reset(seed=seed)

    def snapshot(global_step: int, stage: int, dynamics_loss: float) -> dict[str, Any]:
        record: dict[str, Any] = {
            "kind": "step",
            "global_step": global_step,
            "true_dynamics_stage": stage,
            "dynamics_loss": dynamics_loss,
            "dynamics_ready": baseline.dynamics_ready,
            "planning_ready": baseline.planning_ready,
            "query_round": query_round,
            "preference_accuracy": preference_accuracy(
                baseline.reward_ensemble, heldout_examples
            ),
            **last_preference_metrics,
        }
        for stage_id, transitions in sorted(evaluation_buffers.items()):
            record[f"prediction_mse_stage_{stage_id}"] = evaluate_dynamics(
                baseline.dynamics, transitions
            )["prediction_mse"]
        return record

    with log_path.open("w", encoding="utf-8") as log_file:
        for step in range(total_steps):
            collect_queries = (
                baseline.planning_ready
                and step > 0
                and step % int(preference["query_interval_steps"]) == 0
            )
            plan_result = None
            if step < int(training["dynamics_random_steps"]) or not baseline.planning_ready:
                action = env.action_space.sample()
            else:
                plan_result = baseline.plan(obs, collect_candidates=collect_queries)
                action = plan_result.action

            next_obs, env_reward, terminated, truncated, info = env.step(action)
            transition = Transition(
                obs=np.asarray(obs, dtype=np.float32),
                action=np.asarray(action, dtype=np.float32),
                next_obs=np.asarray(next_obs, dtype=np.float32),
                reward=float(env_reward),
                terminated=bool(terminated),
                truncated=bool(truncated),
            )
            baseline.observe(transition)
            # The real environment reward is evaluation-only. It is never used
            # by dynamics training, preference training, CEM scoring, or action selection.
            stage = int(info["true_dynamics_stage"])
            evaluation_buffers.setdefault(stage, [])
            if len(evaluation_buffers[stage]) < 128 and step % 5 == 0:
                evaluation_buffers[stage].append(transition)

            dynamics_metrics = {"loss": 0.0, "updates": 0.0}
            if step >= int(training["dynamics_learning_starts"]):
                dynamics_metrics = baseline.update_dynamics(
                    int(training["dynamics_updates_per_step"])
                )

            if baseline.dynamics_ready and not bootstrap_done:
                last_preference_metrics = baseline.bootstrap(
                    num_queries=int(preference["bootstrap_queries"]),
                    horizon=int(preference["bootstrap_horizon"]),
                    action_low=env.action_space.low,
                    action_high=env.action_space.high,
                )
                heldout_examples = baseline.generate_preference_examples(
                    num_examples=int(training["heldout_preference_examples"]),
                    horizon=int(preference["bootstrap_horizon"]),
                    action_low=env.action_space.low,
                    action_high=env.action_space.high,
                    seed=seed + 10_000,
                )
                bootstrap_done = True

            if collect_queries and plan_result is not None:
                last_preference_metrics = baseline.query_and_update(
                    candidates=plan_result.candidate_trajectories,
                    num_queries=int(preference["queries_per_round"]),
                    reward_updates=int(preference["reward_updates_per_round"]),
                )
                query_round += 1

            episode_return_value += float(env_reward)
            if (step + 1) % int(training["log_interval"]) == 0 or step == total_steps - 1:
                record = snapshot(step + 1, stage, dynamics_metrics["loss"])
                log_file.write(json.dumps(record, allow_nan=True) + "\n")
                log_file.flush()
                print(json.dumps(record, allow_nan=True))

            if terminated or truncated:
                episode_rewards.append(episode_return_value)
                log_file.write(
                    json.dumps(
                        {
                            "kind": "episode",
                            "global_step": step + 1,
                            "episode_return": episode_return_value,
                        }
                    )
                    + "\n"
                )
                episode_return_value = 0.0
                obs, info = env.reset()
            else:
                obs = next_obs

        summary = {
            "kind": "summary",
            "global_step": total_steps,
            "episodes": len(episode_rewards),
            "mean_episode_return": float(np.mean(episode_rewards)) if episode_rewards else None,
            "num_preferences": len(baseline.preference_buffer),
            "query_round": query_round,
        }
        log_file.write(json.dumps(summary) + "\n")
        print(json.dumps(summary))
    env.close()
    return log_path


def run_gpmm(config: dict[str, Any], total_steps: int | None = None) -> Path:
    """Run Phase-2 GPMM with the unchanged shared preference stack."""

    seed = int(config["seed"])
    set_global_seed(seed)
    env = NonstationaryPendulum(config["env"]["dynamics_schedule"])
    env.reset(seed=seed)
    env.action_space.seed(seed)
    training = config["training"]
    preference = config["preference"]
    baseline = GPMMOnline(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        action_low=env.action_space.low,
        action_high=env.action_space.high,
        planner_config=config["planner"],
        gpmm_config=config["gpmm"],
        preference_config=preference,
        teacher_reward=PendulumReward(),
        device=config["device"],
        seed=seed,
    )

    total_steps = int(total_steps if total_steps is not None else training["total_steps"])
    log_path = Path(training["log_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    evaluation_buffers: dict[int, list[Transition]] = {}
    observed_true_stages: list[int] = []
    heldout_examples = []
    bootstrap_done = False
    query_round = 0
    last_preference_metrics: dict[str, float] = baseline.preference_metrics()
    episode_rewards: list[float] = []
    episode_return_value = 0.0
    obs, info = env.reset(seed=seed)

    def snapshot(global_step: int, stage: int, gp_loss: float) -> dict[str, Any]:
        record: dict[str, Any] = {
            "kind": "step",
            "global_step": global_step,
            "true_dynamics_stage": stage,
            "gp_loss": gp_loss,
            "dynamics_ready": baseline.dynamics_ready,
            "planning_ready": baseline.planning_ready,
            "query_round": query_round,
            "preference_accuracy": preference_accuracy(
                baseline.reward_ensemble, heldout_examples
            ),
            **last_preference_metrics,
            **baseline.dynamics.diagnostics(),
        }
        for stage_id, transitions in sorted(evaluation_buffers.items()):
            record[f"prediction_mse_stage_{stage_id}"] = evaluate_dynamics(
                baseline.dynamics, transitions
            )["prediction_mse"]
        return record

    with log_path.open("w", encoding="utf-8") as log_file:
        for step in range(total_steps):
            collect_queries = (
                baseline.planning_ready
                and step > 0
                and step % int(preference["query_interval_steps"]) == 0
            )
            plan_result = None
            if step < int(training["dynamics_random_steps"]) or not baseline.planning_ready:
                action = env.action_space.sample()
            else:
                plan_result = baseline.plan(obs, collect_candidates=collect_queries)
                action = plan_result.action

            next_obs, env_reward, terminated, truncated, info = env.step(action)
            transition = Transition(
                obs=np.asarray(obs, dtype=np.float32),
                action=np.asarray(action, dtype=np.float32),
                next_obs=np.asarray(next_obs, dtype=np.float32),
                reward=float(env_reward),
                terminated=bool(terminated),
                truncated=bool(truncated),
            )
            baseline.observe(transition)
            # Stage and environment reward are retained for evaluator-only logs;
            # neither enters GP assignment, GP fitting, preference fitting, or CEM.
            stage = int(info["true_dynamics_stage"])
            observed_true_stages.append(stage)
            evaluation_buffers.setdefault(stage, [])
            if len(evaluation_buffers[stage]) < 128 and step % 5 == 0:
                evaluation_buffers[stage].append(transition)

            dynamics_metrics = {"gp_loss": 0.0, "gp_updates": 0.0}
            if step >= int(training["dynamics_learning_starts"]):
                dynamics_metrics = baseline.update_dynamics(
                    int(training["dynamics_updates_per_step"])
                )

            if baseline.dynamics_ready and not bootstrap_done:
                last_preference_metrics = baseline.bootstrap(
                    num_queries=int(preference["bootstrap_queries"]),
                    horizon=int(preference["bootstrap_horizon"]),
                    action_low=env.action_space.low,
                    action_high=env.action_space.high,
                )
                heldout_examples = baseline.generate_preference_examples(
                    num_examples=int(training["heldout_preference_examples"]),
                    horizon=int(preference["bootstrap_horizon"]),
                    action_low=env.action_space.low,
                    action_high=env.action_space.high,
                    seed=seed + 10_000,
                )
                bootstrap_done = True

            if collect_queries and plan_result is not None:
                last_preference_metrics = baseline.query_and_update(
                    candidates=plan_result.candidate_trajectories,
                    num_queries=int(preference["queries_per_round"]),
                    reward_updates=int(preference["reward_updates_per_round"]),
                )
                query_round += 1

            episode_return_value += float(env_reward)
            if (step + 1) % int(training["log_interval"]) == 0 or step == total_steps - 1:
                record = snapshot(step + 1, stage, dynamics_metrics["gp_loss"])
                log_file.write(json.dumps(record, allow_nan=True) + "\n")
                log_file.flush()
                print(json.dumps(record, allow_nan=True))

            if terminated or truncated:
                episode_rewards.append(episode_return_value)
                log_file.write(
                    json.dumps(
                        {
                            "kind": "episode",
                            "global_step": step + 1,
                            "episode_return": episode_return_value,
                        }
                    )
                    + "\n"
                )
                episode_return_value = 0.0
                obs, info = env.reset()
            else:
                obs = next_obs

        summary = {
            "kind": "summary",
            "global_step": total_steps,
            "episodes": len(episode_rewards),
            "mean_episode_return": float(np.mean(episode_rewards)) if episode_rewards else None,
            "num_preferences": len(baseline.preference_buffer),
            "query_round": query_round,
            **baseline.dynamics.diagnostics(),
        }
        summary["stage_expert_contingency"] = assignment_contingency(
            observed_true_stages, baseline.dynamics.assignment_history
        )
        summary["assignment_purity"] = assignment_purity(
            observed_true_stages, baseline.dynamics.assignment_history
        )
        schedule = config["env"]["dynamics_schedule"]
        first_dynamics = {key: value for key, value in schedule[0].items() if key != "step"}
        last_dynamics = {key: value for key, value in schedule[-1].items() if key != "step"}
        return_start = (
            int(schedule[-1]["step"])
            if len(schedule) >= 3 and first_dynamics == last_dynamics
            else total_steps
        )
        summary["expert_reuse_rate"] = expert_reuse_rate(
            baseline.dynamics.assignment_history, return_start
        )
        log_file.write(json.dumps(summary) + "\n")
        print(json.dumps(summary))
    env.close()
    return log_path


def run_hypercrl_adapt(config: dict[str, Any], total_steps: int | None = None) -> Path:
    """Run Phase-3 HyperCRL-Adapt with boundary-free residual routing."""

    seed = int(config["seed"])
    set_global_seed(seed)
    env = NonstationaryPendulum(config["env"]["dynamics_schedule"])
    env.reset(seed=seed)
    env.action_space.seed(seed)
    training = config["training"]
    preference = config["preference"]
    baseline = HyperCRLAdaptOnline(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        action_low=env.action_space.low,
        action_high=env.action_space.high,
        planner_config=config["planner"],
        hypercrl_config=config["hypercrl"],
        preference_config=preference,
        teacher_reward=PendulumReward(),
        device=config["device"],
        seed=seed,
    )

    total_steps = int(total_steps if total_steps is not None else training["total_steps"])
    log_path = Path(training["log_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    evaluation_buffers: dict[int, list[Transition]] = {}
    observed_true_stages: list[int] = []
    heldout_examples = []
    bootstrap_done = False
    query_round = 0
    last_preference_metrics: dict[str, float] = baseline.preference_metrics()
    episode_rewards: list[float] = []
    episode_return_value = 0.0
    obs, info = env.reset(seed=seed)

    def snapshot(global_step: int, stage: int, dynamics_loss: float) -> dict[str, Any]:
        record: dict[str, Any] = {
            "kind": "step",
            "global_step": global_step,
            "true_dynamics_stage": stage,
            "dynamics_loss": dynamics_loss,
            "dynamics_ready": baseline.dynamics_ready,
            "planning_ready": baseline.planning_ready,
            "query_round": query_round,
            "preference_accuracy": preference_accuracy(
                baseline.reward_ensemble, heldout_examples
            ),
            **last_preference_metrics,
            **baseline.dynamics.diagnostics(),
        }
        for stage_id, transitions in sorted(evaluation_buffers.items()):
            record[f"prediction_mse_stage_{stage_id}"] = evaluate_dynamics(
                baseline.dynamics, transitions
            )["prediction_mse"]
        return record

    with log_path.open("w", encoding="utf-8") as log_file:
        for step in range(total_steps):
            collect_queries = (
                baseline.planning_ready
                and step > 0
                and step % int(preference["query_interval_steps"]) == 0
            )
            plan_result = None
            if step < int(training["dynamics_random_steps"]) or not baseline.planning_ready:
                action = env.action_space.sample()
            else:
                plan_result = baseline.plan(obs, collect_candidates=collect_queries)
                action = plan_result.action

            next_obs, env_reward, terminated, truncated, info = env.step(action)
            transition = Transition(
                obs=np.asarray(obs, dtype=np.float32),
                action=np.asarray(action, dtype=np.float32),
                next_obs=np.asarray(next_obs, dtype=np.float32),
                reward=float(env_reward),
                terminated=bool(terminated),
                truncated=bool(truncated),
            )
            baseline.observe(transition)
            # Stage/reward are evaluator-only and are never passed to baseline.
            stage = int(info["true_dynamics_stage"])
            observed_true_stages.append(stage)
            evaluation_buffers.setdefault(stage, [])
            if len(evaluation_buffers[stage]) < 128 and step % 5 == 0:
                evaluation_buffers[stage].append(transition)

            dynamics_metrics = {"L_dyn": 0.0, "L_reg": 0.0, "L_total": 0.0}
            if step >= int(training["dynamics_learning_starts"]):
                dynamics_metrics = baseline.update_dynamics(
                    int(training["dynamics_updates_per_step"])
                )

            if baseline.dynamics_ready and not bootstrap_done:
                last_preference_metrics = baseline.bootstrap(
                    num_queries=int(preference["bootstrap_queries"]),
                    horizon=int(preference["bootstrap_horizon"]),
                    action_low=env.action_space.low,
                    action_high=env.action_space.high,
                )
                heldout_examples = baseline.generate_preference_examples(
                    num_examples=int(training["heldout_preference_examples"]),
                    horizon=int(preference["bootstrap_horizon"]),
                    action_low=env.action_space.low,
                    action_high=env.action_space.high,
                    seed=seed + 10_000,
                )
                bootstrap_done = True

            if collect_queries and plan_result is not None:
                last_preference_metrics = baseline.query_and_update(
                    candidates=plan_result.candidate_trajectories,
                    num_queries=int(preference["queries_per_round"]),
                    reward_updates=int(preference["reward_updates_per_round"]),
                )
                query_round += 1

            episode_return_value += float(env_reward)
            if (step + 1) % int(training["log_interval"]) == 0 or step == total_steps - 1:
                record = snapshot(step + 1, stage, dynamics_metrics["L_dyn"])
                log_file.write(json.dumps(record, allow_nan=True) + "\n")
                log_file.flush()
                print(json.dumps(record, allow_nan=True))

            if terminated or truncated:
                episode_rewards.append(episode_return_value)
                log_file.write(
                    json.dumps(
                        {
                            "kind": "episode",
                            "global_step": step + 1,
                            "episode_return": episode_return_value,
                        }
                    )
                    + "\n"
                )
                episode_return_value = 0.0
                obs, info = env.reset()
            else:
                obs = next_obs

        schedule = config["env"]["dynamics_schedule"]
        first_dynamics = {key: value for key, value in schedule[0].items() if key != "step"}
        last_dynamics = {key: value for key, value in schedule[-1].items() if key != "step"}
        return_start = (
            int(schedule[-1]["step"])
            if len(schedule) >= 3 and first_dynamics == last_dynamics
            else total_steps
        )
        summary = {
            "kind": "summary",
            "global_step": total_steps,
            "episodes": len(episode_rewards),
            "mean_episode_return": float(np.mean(episode_rewards)) if episode_rewards else None,
            "num_preferences": len(baseline.preference_buffer),
            "query_round": query_round,
            **baseline.dynamics.diagnostics(),
            "stage_embedding_contingency": embedding_contingency(
                observed_true_stages, baseline.dynamics.assignment_history
            ),
            "embedding_purity": embedding_purity(
                observed_true_stages, baseline.dynamics.assignment_history
            ),
            "embedding_reuse_rate": embedding_reuse_rate(
                baseline.dynamics.assignment_history, return_start
            ),
        }
        log_file.write(json.dumps(summary) + "\n")
        print(json.dumps(summary))
    env.close()
    return log_path


def run_vblrl_adapt(config: dict[str, Any], total_steps: int | None = None) -> Path:
    """Run Phase-4 VBLRL-Adapt with reward-free Bayesian dynamics."""

    seed = int(config["seed"])
    set_global_seed(seed)
    env = NonstationaryPendulum(config["env"]["dynamics_schedule"])
    env.reset(seed=seed)
    env.action_space.seed(seed)
    training = config["training"]
    preference = config["preference"]
    baseline = VBLRLAdaptOnline(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        action_low=env.action_space.low,
        action_high=env.action_space.high,
        planner_config=config["planner"],
        vblrl_config=config["vblrl"],
        preference_config=preference,
        teacher_reward=PendulumReward(),
        device=config["device"],
        seed=seed,
    )

    total_steps = int(total_steps if total_steps is not None else training["total_steps"])
    log_path = Path(training["log_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    evaluation_buffers: dict[int, list[Transition]] = {}
    observed_true_stages: list[int] = []
    heldout_examples = []
    bootstrap_done = False
    query_round = 0
    last_preference_metrics: dict[str, float] = baseline.preference_metrics()
    episode_rewards: list[float] = []
    episode_return_value = 0.0
    obs, info = env.reset(seed=seed)

    def snapshot(global_step: int, stage: int) -> dict[str, Any]:
        record: dict[str, Any] = {
            "kind": "step",
            "global_step": global_step,
            "true_dynamics_stage": stage,
            "dynamics_ready": baseline.dynamics_ready,
            "planning_ready": baseline.planning_ready,
            "query_round": query_round,
            "preference_accuracy": preference_accuracy(
                baseline.reward_ensemble, heldout_examples
            ),
            **last_preference_metrics,
            **baseline.dynamics.diagnostics(),
        }
        for stage_id, transitions in sorted(evaluation_buffers.items()):
            record[f"prediction_mse_stage_{stage_id}"] = evaluate_dynamics(
                baseline.dynamics, transitions
            )["prediction_mse"]
        return record

    with log_path.open("w", encoding="utf-8") as log_file:
        for step in range(total_steps):
            collect_queries = (
                baseline.planning_ready
                and step > 0
                and step % int(preference["query_interval_steps"]) == 0
            )
            plan_result = None
            if step < int(training["dynamics_random_steps"]) or not baseline.planning_ready:
                action = env.action_space.sample()
            else:
                plan_result = baseline.plan(obs, collect_candidates=collect_queries)
                action = plan_result.action

            next_obs, env_reward, terminated, truncated, info = env.step(action)
            transition = Transition(
                obs=np.asarray(obs, dtype=np.float32),
                action=np.asarray(action, dtype=np.float32),
                next_obs=np.asarray(next_obs, dtype=np.float32),
                reward=float(env_reward),
                terminated=bool(terminated),
                truncated=bool(truncated),
            )
            baseline.observe(transition)
            # Stage and environment reward are evaluator-only; neither enters VBLRL.
            stage = int(info["true_dynamics_stage"])
            observed_true_stages.append(stage)
            evaluation_buffers.setdefault(stage, [])
            if len(evaluation_buffers[stage]) < 128 and step % 5 == 0:
                evaluation_buffers[stage].append(transition)

            if step >= int(training["dynamics_learning_starts"]):
                baseline.update_dynamics(int(training["dynamics_updates_per_step"]))

            if baseline.dynamics_ready and not bootstrap_done:
                last_preference_metrics = baseline.bootstrap(
                    num_queries=int(preference["bootstrap_queries"]),
                    horizon=int(preference["bootstrap_horizon"]),
                    action_low=env.action_space.low,
                    action_high=env.action_space.high,
                )
                heldout_examples = baseline.generate_preference_examples(
                    num_examples=int(training["heldout_preference_examples"]),
                    horizon=int(preference["bootstrap_horizon"]),
                    action_low=env.action_space.low,
                    action_high=env.action_space.high,
                    seed=seed + 10_000,
                )
                bootstrap_done = True

            if collect_queries and plan_result is not None:
                last_preference_metrics = baseline.query_and_update(
                    candidates=plan_result.candidate_trajectories,
                    num_queries=int(preference["queries_per_round"]),
                    reward_updates=int(preference["reward_updates_per_round"]),
                )
                query_round += 1

            episode_return_value += float(env_reward)
            if (step + 1) % int(training["log_interval"]) == 0 or step == total_steps - 1:
                record = snapshot(step + 1, stage)
                log_file.write(json.dumps(record, allow_nan=True) + "\n")
                log_file.flush()
                print(json.dumps(record, allow_nan=True))

            if terminated or truncated:
                episode_rewards.append(episode_return_value)
                log_file.write(
                    json.dumps(
                        {
                            "kind": "episode",
                            "global_step": step + 1,
                            "episode_return": episode_return_value,
                        }
                    )
                    + "\n"
                )
                episode_return_value = 0.0
                obs, info = env.reset()
            else:
                obs = next_obs

        schedule = config["env"]["dynamics_schedule"]
        first_dynamics = {key: value for key, value in schedule[0].items() if key != "step"}
        last_dynamics = {key: value for key, value in schedule[-1].items() if key != "step"}
        return_start = (
            int(schedule[-1]["step"])
            if len(schedule) >= 3 and first_dynamics == last_dynamics
            else total_steps
        )
        summary = {
            "kind": "summary",
            "global_step": total_steps,
            "episodes": len(episode_rewards),
            "mean_episode_return": float(np.mean(episode_rewards)) if episode_rewards else None,
            "num_preferences": len(baseline.preference_buffer),
            "query_round": query_round,
            **baseline.dynamics.diagnostics(),
            "stage_posterior_contingency": posterior_contingency(
                observed_true_stages, baseline.dynamics.assignment_history
            ),
            "posterior_purity": posterior_purity(
                observed_true_stages, baseline.dynamics.assignment_history
            ),
            "posterior_reuse_rate": posterior_reuse_rate(
                baseline.dynamics.assignment_history, return_start
            ),
        }
        log_file.write(json.dumps(summary) + "\n")
        print(json.dumps(summary))
    env.close()
    return log_path


def run_curious_replay_adapt(config: dict[str, Any], total_steps: int | None = None) -> Path:
    """Run Phase-5 Curious Replay-Adapt with one reward-free world model."""

    seed = int(config["seed"])
    set_global_seed(seed)
    env = NonstationaryPendulum(config["env"]["dynamics_schedule"])
    env.reset(seed=seed)
    env.action_space.seed(seed)
    training = config["training"]
    preference = config["preference"]
    baseline = CuriousReplayOnline(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        action_low=env.action_space.low,
        action_high=env.action_space.high,
        planner_config=config["planner"],
        curious_replay_config=config["curious_replay"],
        preference_config=preference,
        teacher_reward=PendulumReward(),
        device=config["device"],
        seed=seed,
    )

    total_steps = int(total_steps if total_steps is not None else training["total_steps"])
    log_path = Path(training["log_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    evaluation_buffers: dict[int, list[Transition]] = {}
    evaluator_stage_by_slot: dict[int, int] = {}
    replay_stage_counts: dict[int, int] = {}
    observed_true_stages: list[int] = []
    heldout_examples = []
    step_records: list[dict[str, Any]] = []
    episode_records: list[dict[str, Any]] = []
    bootstrap_done = False
    query_round = 0
    last_preference_metrics: dict[str, float] = baseline.preference_metrics()
    episode_rewards: list[float] = []
    episode_return_value = 0.0
    obs, info = env.reset(seed=seed)

    def snapshot(global_step: int, stage: int) -> dict[str, Any]:
        record: dict[str, Any] = {
            "kind": "step",
            "global_step": global_step,
            "true_dynamics_stage": stage,
            "dynamics_ready": baseline.dynamics_ready,
            "planning_ready": baseline.planning_ready,
            "query_round": query_round,
            "preference_accuracy": preference_accuracy(
                baseline.reward_ensemble, heldout_examples
            ),
            "replay_stage_share": replay_stage_share(replay_stage_counts),
            **last_preference_metrics,
            **baseline.dynamics.diagnostics(),
        }
        for stage_id, transitions in sorted(evaluation_buffers.items()):
            record[f"prediction_mse_stage_{stage_id}"] = evaluate_dynamics(
                baseline.dynamics, transitions
            )["prediction_mse"]
        return record

    with log_path.open("w", encoding="utf-8") as log_file:
        for step in range(total_steps):
            collect_queries = (
                baseline.planning_ready
                and step > 0
                and step % int(preference["query_interval_steps"]) == 0
            )
            plan_result = None
            if step < int(training["dynamics_random_steps"]) or not baseline.planning_ready:
                action = env.action_space.sample()
            else:
                plan_result = baseline.plan(obs, collect_candidates=collect_queries)
                action = plan_result.action

            next_obs, env_reward, terminated, truncated, info = env.step(action)
            transition = Transition(
                obs=np.asarray(obs, dtype=np.float32),
                action=np.asarray(action, dtype=np.float32),
                next_obs=np.asarray(next_obs, dtype=np.float32),
                reward=float(env_reward),
                terminated=bool(terminated),
                truncated=bool(truncated),
            )
            baseline.observe(transition)
            stage = int(info["true_dynamics_stage"])
            observed_true_stages.append(stage)
            # Evaluator-only slot metadata. The learner receives no stage value.
            if baseline.dynamics.last_observed_index is not None:
                evaluator_stage_by_slot[baseline.dynamics.last_observed_index] = stage
            evaluation_buffers.setdefault(stage, [])
            if len(evaluation_buffers[stage]) < 128 and step % 5 == 0:
                evaluation_buffers[stage].append(transition)

            if step >= int(training["dynamics_learning_starts"]):
                baseline.update_dynamics(int(training["dynamics_updates_per_step"]))
                sampled_counts = sampled_stage_counts(
                    evaluator_stage_by_slot, baseline.dynamics.last_sample_indices
                )
                for sampled_stage, count in sampled_counts.items():
                    replay_stage_counts[sampled_stage] = (
                        replay_stage_counts.get(sampled_stage, 0) + count
                    )

            if baseline.dynamics_ready and not bootstrap_done:
                last_preference_metrics = baseline.bootstrap(
                    num_queries=int(preference["bootstrap_queries"]),
                    horizon=int(preference["bootstrap_horizon"]),
                    action_low=env.action_space.low,
                    action_high=env.action_space.high,
                )
                heldout_examples = baseline.generate_preference_examples(
                    num_examples=int(training["heldout_preference_examples"]),
                    horizon=int(preference["bootstrap_horizon"]),
                    action_low=env.action_space.low,
                    action_high=env.action_space.high,
                    seed=seed + 10_000,
                )
                bootstrap_done = True

            if collect_queries and plan_result is not None:
                last_preference_metrics = baseline.query_and_update(
                    candidates=plan_result.candidate_trajectories,
                    num_queries=int(preference["queries_per_round"]),
                    reward_updates=int(preference["reward_updates_per_round"]),
                )
                query_round += 1

            episode_return_value += float(env_reward)
            if (step + 1) % int(training["log_interval"]) == 0 or step == total_steps - 1:
                record = snapshot(step + 1, stage)
                step_records.append(record)
                log_file.write(json.dumps(record, allow_nan=True) + "\n")
                log_file.flush()
                print(json.dumps(record, allow_nan=True))

            if terminated or truncated:
                episode_record = {
                    "kind": "episode",
                    "global_step": step + 1,
                    "episode_return": episode_return_value,
                }
                episode_records.append(episode_record)
                episode_rewards.append(episode_return_value)
                log_file.write(json.dumps(episode_record) + "\n")
                episode_return_value = 0.0
                obs, info = env.reset()
            else:
                obs = next_obs

        schedule = config["env"]["dynamics_schedule"]
        first_dynamics = {key: value for key, value in schedule[0].items() if key != "step"}
        last_dynamics = {key: value for key, value in schedule[-1].items() if key != "step"}
        return_start = (
            int(schedule[-1]["step"])
            if len(schedule) >= 3 and first_dynamics == last_dynamics
            else total_steps
        )
        has_return_regime = len(schedule) >= 3 and first_dynamics == last_dynamics
        pre_return_records = [
            record for record in step_records if int(record["global_step"]) < return_start
        ]
        pre_return_mse = float("nan")
        for record in reversed(pre_return_records):
            value = record.get("prediction_mse_stage_0")
            if isinstance(value, (int, float)) and np.isfinite(float(value)):
                pre_return_mse = float(value)
                break
        pre_return_episode = float("nan")
        for record in episode_records:
            if int(record["global_step"]) < return_start:
                pre_return_episode = float(record["episode_return"])
        summary = {
            "kind": "summary",
            "global_step": total_steps,
            "episodes": len(episode_rewards),
            "mean_episode_return": float(np.mean(episode_rewards)) if episode_rewards else None,
            "num_preferences": len(baseline.preference_buffer),
            "query_round": query_round,
            "sampled_replay_stage_counts": replay_stage_counts,
            "sampled_replay_stage_share": replay_stage_share(replay_stage_counts),
            "T_reacq_model": first_recovery_step(
                step_records,
                return_start,
                "prediction_mse_stage_0",
                pre_return_mse,
            ) if has_return_regime else float("nan"),
            "T_reacq_return": first_return_recovery_step(
                episode_records,
                return_start,
                pre_return_episode,
            ) if has_return_regime else float("nan"),
            **baseline.dynamics.diagnostics(),
        }
        log_file.write(json.dumps(summary) + "\n")
        print(json.dumps(summary))
    env.close()
    return log_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="pbcwm/configs/pendulum.yaml")
    parser.add_argument("--total-steps", type=int, default=None)
    args = parser.parse_args()
    run(load_config(args.config), total_steps=args.total_steps)


if __name__ == "__main__":
    main()
