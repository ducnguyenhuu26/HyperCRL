"""Short real-Pendulum RADIUS smoke; no campaign or benchmark result."""

import argparse
import json
from dataclasses import replace

import gymnasium as gym
import numpy as np

from pbcwm.core.types import Transition

from . import RadiusPbCWM, load_radius_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="pbcwm/configs/methods/radius.yaml")
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    loaded = load_radius_config(args.config)
    config = replace(
        loaded,
        atlas=replace(loaded.atlas, hidden_size=32),
        ref=replace(loaded.ref, context_window=32, min_context_samples=8),
        rne=replace(loaded.rne, initialization_updates=2),
        memory=replace(loaded.memory, min_stable_steps=8),
        training=replace(loaded.training, batch_size=8, replay_capacity=128),
    )
    env = gym.make("Pendulum-v1")
    method = RadiusPbCWM(3, 1, config, seed=args.seed)
    obs, _ = env.reset(seed=args.seed)
    env.action_space.seed(args.seed)
    for _ in range(args.steps):
        action = env.action_space.sample().astype(np.float32)
        next_obs, _, terminated, truncated, _ = env.step(action)
        method.observe(Transition(np.asarray(obs, dtype=np.float32), action, np.asarray(next_obs, dtype=np.float32), 0.0, terminated, truncated))
        method.update()
        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset()
    print(json.dumps({"steps": args.steps, "replay_size": len(method.replay), "rank": method.rank, **method.diagnostics()}, sort_keys=True))
    env.close()


if __name__ == "__main__":
    main()
