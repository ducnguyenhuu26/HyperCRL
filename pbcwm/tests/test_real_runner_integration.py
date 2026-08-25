from dataclasses import replace

import numpy as np
import yaml

from pbcwm.protocol.config import EnvironmentProtocolConfig, load_protocol_config
from pbcwm.experiment.runner import CanonicalLifetimeRunner


class _Space:
    def sample(self):
        return np.zeros(1, dtype=np.float32)


class _Env:
    def __init__(self):
        self.action_space = _Space()
        self.step_count = 0
        self.resets = 0

    def reset(self, seed=None):
        self.resets += 1
        return np.zeros(2, dtype=np.float32), {}

    def step(self, action):
        self.step_count += 1
        obs = np.full(2, self.step_count, dtype=np.float32)
        return obs, 123.0, False, False, {"hidden": "metadata"}

    def close(self):
        pass


class _Learner:
    def __init__(self):
        self.transitions = []
        self.updates = 0

    def observe(self, transition):
        self.transitions.append(transition)

    def update(self, steps=1):
        self.updates += steps
        return {"loss": 0.0}

    def predict(self, obs, action):
        return obs

    def state_dict(self):
        return {"updates": self.updates, "count": len(self.transitions)}

    def load_state_dict(self, state):
        self.updates = int(state["updates"])


class _Planner:
    def act(self, obs, learner, reward_fn):
        return np.zeros(1, dtype=np.float32)


def _config():
    config = load_protocol_config("pbcwm/configs/protocol.yaml")
    env = EnvironmentProtocolConfig(stage_length=20, warmup_steps=1, planner_horizon=2, planner_population=4)
    return replace(config, environments={"Pendulum-v1": env})


def test_real_runner_stage_switch_does_not_reset_episode_or_leak_metadata():
    runner = CanonicalLifetimeRunner(_config(), "Pendulum-v1", 0, episode_length=100)
    env = _Env()
    learner = _Learner()
    summary = runner.run(env, learner, _Planner(), query_handler=lambda query, *_: query.pair_count)
    assert summary.stage_switches
    assert summary.episode_resets == ()
    assert all(item.reward == 0.0 for item in learner.transitions)
    assert all(not hasattr(item, "dynamics_id") for item in learner.transitions)
