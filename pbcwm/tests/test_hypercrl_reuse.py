import numpy as np
import torch

from pbcwm.baselines.hypercrl.learner import HyperCRLAdaptDynamicsLearner
from pbcwm.core.types import Transition


def _transition(sign: float, index: int) -> Transition:
    obs = np.array([((index % 5) - 2) / 2], dtype=np.float32)
    action = np.array([0.7], dtype=np.float32)
    return Transition(obs, action, obs + sign * 0.4, 0.0, False, False)


def _learner() -> HyperCRLAdaptDynamicsLearner:
    return HyperCRLAdaptDynamicsLearner(
        1,
        1,
        embedding_dim=4,
        hyper_hidden_dims=(16,),
        target_hidden_dims=(16,),
        hyper_lr=0.05,
        embedding_lr=0.05,
        regularization_beta=1.0,
        current_regime_buffer_size=32,
        dynamics_batch_size=8,
        router_window_size=4,
        shift_threshold=0.05,
        reuse_threshold=0.5,
        consecutive_trigger_windows=2,
        router_cooldown_steps=16,
        seed=0,
    )


def test_hypercrl_adapt_reuses_old_embedding_without_oracle_stage() -> None:
    torch.manual_seed(0)
    learner = _learner()
    for sign in (1.0, -1.0, 1.0):
        for index in range(30):
            learner.observe(_transition(sign, index))
            learner.update(10)

    changes = [
        (index, value)
        for index, value in enumerate(learner.assignment_history)
        if index == 0 or learner.assignment_history[index - 1] != value
    ]
    assert changes == [(0, 0), (31, 1), (61, 0)]
    assert learner.num_embeddings == 2
    assert learner.reuse_count == 1
    assert all(not hasattr(item, "true_dynamics_stage") for item in learner.router.window)


def test_same_transitions_have_same_routing_without_stage_metadata() -> None:
    torch.manual_seed(0)
    first = _learner()
    torch.manual_seed(0)
    second = _learner()
    stream = [_transition(1.0, i) for i in range(15)] + [_transition(-1.0, i) for i in range(15)]
    for transition in stream:
        first.observe(transition)
        second.observe(transition)
        first.update(4)
        second.update(4)

    assert first.assignment_history == second.assignment_history
    assert first.current_embedding_id == second.current_embedding_id
    assert first.num_embeddings == second.num_embeddings
