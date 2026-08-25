"""Random-policy and static-dynamics smoke commands for the imported benchmark."""

import argparse
import json
from pathlib import Path

import numpy as np

from pbcwm.benchmarks.base import build_agent_transition
from pbcwm.benchmarks.registry import load_benchmark_spec, make_benchmark


def run(args: argparse.Namespace) -> None:
    spec = load_benchmark_spec(args.config)
    env = make_benchmark(args.benchmark, spec, root_seed=args.seed)
    learner = None
    if args.learner == "static":
        from pbcwm.baselines.static import StaticDynamicsLearner
        from pbcwm.planning.cem import CEMPlanner

        learner = StaticDynamicsLearner(
            obs_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            hidden_dims=(32, 32),
            batch_size=4,
            replay_capacity=128,
            seed=args.seed,
        )
    obs, _ = env.reset(seed=args.seed)
    for _ in range(args.steps):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, _info = env.step(action)
        if learner is not None:
            learner.observe(build_agent_transition(obs, action, next_obs, terminated, truncated))
            learner.update()
        print(json.dumps({
            "benchmark": spec.name,
            "learner": args.learner,
            "global_env_step": env.global_env_step,
            "observation_shape": list(next_obs.shape),
            "true_reward_evaluation_only": float(reward),
        }))
        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset()
    if learner is not None:
        planner = CEMPlanner(
            horizon=3,
            population_size=8,
            elite_size=2,
            num_iterations=1,
            action_low=env.action_space.low,
            action_high=env.action_space.high,
        )
        planner.plan(
            obs,
            learner,
            lambda imagined_obs, imagined_action, imagined_next_obs: imagined_obs.new_zeros(
                imagined_obs.shape[0]
            ),
            return_candidates=False,
        )
    print(json.dumps({"metadata": env.evaluation_metadata()}, default=str))
    env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="nsgym/pendulum-mass-abrupt-return-v0")
    parser.add_argument("--config", default="pbcwm/configs/benchmarks/pendulum/p1_p2_p1.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--learner", choices=("random", "static"), default="random")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
