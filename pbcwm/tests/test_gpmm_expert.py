import numpy as np
import torch

from pbcwm.baselines.gpmm.expert import GPExpert


def test_gp_expert_learns_delta_and_returns_bounded_variance() -> None:
    expert = GPExpert(1, 1, max_points=64, learning_rate=0.08, seed=0)
    for x in np.linspace(-1.0, 1.0, 24):
        action = np.sin(x)
        expert.add_transition([x], [action], [x + 0.4 * action + 0.1 * x])

    obs = torch.tensor([[x] for x in np.linspace(-0.9, 0.9, 9)], dtype=torch.float64)
    actions = torch.sin(obs)
    target = obs + 0.4 * actions + 0.1 * obs
    prior_mse = float(torch.mean((target - obs).square()))
    expert.fit(10)
    mean, variance = expert.predict_distribution(obs, actions)

    assert expert.num_points == 24
    assert float(torch.mean((mean - (target - obs)).square())) < prior_mse
    assert torch.all(variance >= expert.min_predictive_variance)
    assert torch.all(variance <= expert.max_predictive_variance)
    assert torch.isfinite(expert.log_likelihood(obs, actions, target)).all()


def test_gp_expert_memory_keeps_recent_and_historical_points() -> None:
    expert = GPExpert(1, 1, max_points=6, seed=0)
    for index in range(20):
        expert.add_transition([index], [0.0], [index + 1])

    assert expert.num_points == 6
    assert torch.equal(expert.training_inputs[-3:, 0], torch.tensor([17.0, 18.0, 19.0]))
    assert torch.any(expert.training_inputs[:3, 0] < 17.0)
