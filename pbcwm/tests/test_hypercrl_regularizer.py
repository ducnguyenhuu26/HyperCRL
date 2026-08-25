import numpy as np
import torch

from pbcwm.baselines.hypercrl.learner import HyperCRLAdaptDynamicsLearner
from pbcwm.core.types import Transition


def _transition(sign: float, index: int) -> Transition:
    obs = np.array([((index % 5) - 2) / 2], dtype=np.float32)
    action = np.array([0.7], dtype=np.float32)
    return Transition(obs, action, obs + sign * 0.4, 0.0, False, False)


def _train_two_regimes(beta: float) -> float:
    torch.manual_seed(0)
    learner = HyperCRLAdaptDynamicsLearner(
        1,
        1,
        embedding_dim=4,
        hyper_hidden_dims=(16,),
        target_hidden_dims=(16,),
        hyper_lr=0.05,
        embedding_lr=0.05,
        regularization_beta=beta,
        current_regime_buffer_size=32,
        dynamics_batch_size=8,
        router_window_size=4,
        shift_threshold=0.05,
        reuse_threshold=0.5,
        consecutive_trigger_windows=2,
        router_cooldown_steps=16,
        seed=0,
    )
    for index in range(30):
        learner.observe(_transition(1.0, index))
        learner.update(10)
    protected_before = [
        weight.detach().clone()
        for weight in learner.hypernetwork(learner.embeddings[0])
    ]
    for index in range(30):
        learner.observe(_transition(-1.0, index))
        learner.update(10)
    protected_after = [
        weight.detach().clone()
        for weight in learner.hypernetwork(learner.embeddings[0])
    ]
    return float(torch.cat([
        (after - before).reshape(-1)
        for before, after in zip(protected_before, protected_after)
    ]).norm())


def test_output_space_regularization_reduces_old_generated_weight_drift() -> None:
    drift_without_retention = _train_two_regimes(beta=0.0)
    drift_with_retention = _train_two_regimes(beta=1.0)
    assert drift_with_retention < 0.75 * drift_without_retention
