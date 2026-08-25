import numpy as np
import torch

from pbcwm.baselines.static import StaticDynamicsLearner
from pbcwm.core.types import Transition


def test_dynamics_learns_synthetic_delta() -> None:
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    learner = StaticDynamicsLearner(
        obs_dim=2,
        action_dim=1,
        hidden_dims=(32, 32),
        batch_size=32,
        replay_capacity=256,
        learning_rate=3e-3,
        seed=0,
    )
    for _ in range(128):
        obs = rng.normal(size=2).astype(np.float32)
        action = rng.normal(size=1).astype(np.float32)
        delta = np.array([0.5 * obs[0] + action[0], -0.25 * obs[1]], dtype=np.float32)
        learner.observe(Transition(obs, action, obs + delta, 0.0, False, False))

    initial_loss = learner.update(1)["loss"]
    for _ in range(60):
        learner.update(1)
    final_loss = learner.update(1)["loss"]

    assert final_loss < initial_loss
    obs = torch.zeros(4, 2)
    action = torch.zeros(4, 1)
    assert learner.predict(obs, action).shape == (4, 2)
