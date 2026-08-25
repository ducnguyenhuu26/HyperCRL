from dataclasses import replace

import torch

from pbcwm.methods.radius.config import RadiusConfig
from pbcwm.methods.radius.inference import RecurrentEvidenceFilter, gaussian_context_posterior
from pbcwm.methods.radius.memory import ContextMemory
from pbcwm.methods.radius.types import ContextPosterior


def test_ref_sequential_precision_counts_each_sample_once():
    config = replace(RadiusConfig().ref, context_process_noise=0.0, numerical_jitter=0.0)
    memory = ContextMemory(4, 2.5, 0.25)
    ref = RecurrentEvidenceFilter(2, 0.2, config, memory, torch.device("cpu"))
    basis = torch.randn(5, 1, 2)
    residual = torch.randn(5, 1)
    active = ContextPosterior(torch.zeros(2), torch.eye(2) * 2.0, 0.0, "initial")
    for index in range(5):
        active = ref.update_active(basis[index:index + 1], residual[index:index + 1], active)
    expected_mean, expected_cov = gaussian_context_posterior(basis, residual, torch.zeros(2), torch.eye(2) * 2.0, 0.2)
    assert torch.allclose(active.mean, expected_mean, atol=1e-5)
    assert torch.allclose(active.covariance, expected_cov, atol=1e-5)
