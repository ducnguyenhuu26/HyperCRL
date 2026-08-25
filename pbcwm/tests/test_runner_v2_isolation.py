import numpy as np

from pbcwm.experiment.runner import CanonicalLifetimeRunner
from pbcwm.protocol.config import EnvironmentProtocolConfig, load_protocol_config
from pbcwm.tests.test_real_runner_integration import _Env, _Learner, _config


class _Agent(_Learner):
    def __init__(self):
        super().__init__()
        self.plan_calls = 0

    def plan(self, obs, collect_candidates=False):
        del obs, collect_candidates
        self.plan_calls += 1
        return type("Plan", (), {"action": np.zeros(1, dtype=np.float32)})()

    def update_dynamics(self, steps=1):
        self.updates += steps
        return {"loss": 0.0}


class _StatefulPlanner:
    def __init__(self):
        self.calls = 0

    def act(self, obs, learner, reward_fn):
        del obs, learner, reward_fn
        self.calls += 1
        return np.zeros(1, dtype=np.float32)

    def state_dict(self):
        return {"calls": self.calls}

    def load_state_dict(self, state):
        self.calls = int(state["calls"])


def test_evaluation_uses_separate_environment_and_agent_plan_interface():
    runner = CanonicalLifetimeRunner(_config(), "Pendulum-v1", 0, episode_length=100)
    train_env = _Env()
    agent = _Agent()
    evaluation_envs = []

    def make_eval(_checkpoint, _seed):
        env = _Env()
        evaluation_envs.append(env)
        return env

    def evaluate(_checkpoint, _learner, context):
        context["env"].reset(seed=123)
        context["env"].step(np.zeros(1, dtype=np.float32))

    summary = runner.run(
        train_env,
        agent,
        None,
        query_handler=lambda query, *_: query.pair_count,
        evaluation_handler=evaluate,
        eval_env_factory=make_eval,
    )
    assert summary.steps == train_env.step_count
    assert train_env.resets == 1
    assert evaluation_envs and all(env.step_count == 1 for env in evaluation_envs)
    assert agent.plan_calls > 0


def test_evaluation_restores_external_planner_state():
    runner = CanonicalLifetimeRunner(_config(), "Pendulum-v1", 0, episode_length=100)
    train_env = _Env()
    learner = _Learner()
    planner = _StatefulPlanner()
    observed = []

    def evaluate(_checkpoint, _learner, _context):
        before = planner.calls
        planner.calls += 100
        observed.append((before, planner.calls))

    runner.run(
        train_env,
        learner,
        planner,
        query_handler=lambda query, *_: query.pair_count,
        evaluation_handler=evaluate,
    )
    assert observed
    assert all(after - before == 100 for before, after in observed)
    assert planner.calls == runner.schedule.total_steps - runner.config.environment("Pendulum-v1").warmup_steps
