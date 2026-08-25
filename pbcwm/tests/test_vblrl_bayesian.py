import numpy as np
import torch

from pbcwm.baselines.vblrl.bnn import BayesianDynamicsNetwork, BayesianLinear
from pbcwm.baselines.vblrl.posterior import BayesianDynamicsPosterior
from pbcwm.core.types import Transition


def _transition(index: int, delta: float = 0.25) -> Transition:
    obs = np.array([index / 10.0], dtype=np.float32)
    return Transition(obs, np.array([0.2], dtype=np.float32), obs + delta, 0.0, False, False)


def test_bayesian_layer_has_mu_rho_and_stochastic_parameters() -> None:
    torch.manual_seed(0)
    layer = BayesianLinear(2, 3)
    first, _ = layer.sampled_parameters()
    second, _ = layer.sampled_parameters()
    assert first.shape == (3, 2)
    assert not torch.equal(first, second)
    assert torch.all(layer.weight_sigma > 0)
    assert torch.all(layer.bias_sigma > 0)


def test_vblrl_network_predicts_delta_mean_and_logvar_only() -> None:
    network = BayesianDynamicsNetwork(2, 1, hidden_dims=(8, 8))
    obs = torch.zeros(4, 2)
    action = torch.zeros(4, 1)
    mean_a, logvar_a = network(obs, action, deterministic=True)
    mean_b, logvar_b = network(obs, action, deterministic=True)
    assert mean_a.shape == (4, 2)
    assert logvar_a.shape == (4, 2)
    assert torch.equal(mean_a, mean_b)
    assert torch.equal(logvar_a, logvar_b)
    assert not hasattr(network, "reward_head")


def test_posterior_samples_delta_dynamics_and_updates() -> None:
    posterior = BayesianDynamicsPosterior(
        1, 1, hidden_dims=(8,), learning_rate=0.02, seed=0
    )
    transitions = [_transition(index) for index in range(8)]
    before = posterior.log_predictive_likelihood(transitions, num_samples=3)
    metrics = posterior.update(transitions, num_steps=3)
    after = posterior.log_predictive_likelihood(transitions, num_samples=3)
    samples = posterior.sample_next(torch.zeros(2, 1), torch.zeros(2, 1), num_samples=4)
    assert samples.shape == (4, 2, 1)
    assert metrics["updates"] == 3
    assert np.isfinite(before) and np.isfinite(after)
