from dataclasses import replace

import numpy as np

from pbcwm.experiment.runner import CanonicalLifetimeRunner
from pbcwm.protocol.config import EnvironmentProtocolConfig, load_protocol_config


class _Space:
    def sample(self): return np.zeros(1, dtype=np.float32)


class _Env:
    action_space = _Space()
    def reset(self, seed=None): return np.zeros(1, dtype=np.float32), {}
    def step(self, action): return np.zeros(1, dtype=np.float32), 0.0, False, False, {}
    def close(self): pass


class _Learner:
    def observe(self, transition): pass
    def update(self, steps=1): return {"loss": 0.0}
    def state_dict(self): return {}
    def load_state_dict(self, state): pass


class _Planner:
    def act(self, obs, learner, reward_fn): return np.zeros(1, dtype=np.float32)


def test_query_handler_must_match_protocol_budget():
    config = load_protocol_config("pbcwm/configs/protocol.yaml")
    config = replace(config, environments={"Pendulum-v1": EnvironmentProtocolConfig(20, 1, 2, 4)})
    runner = CanonicalLifetimeRunner(config, "Pendulum-v1", 0)
    try:
        runner.run(_Env(), _Learner(), _Planner(), query_handler=lambda query, *_: 0)
    except RuntimeError as exc:
        assert "pair count" in str(exc)
    else:
        raise AssertionError("query budget mismatch must fail closed")
