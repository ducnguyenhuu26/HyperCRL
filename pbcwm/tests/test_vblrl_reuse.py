import numpy as np
import torch

from pbcwm.baselines.vblrl.learner import VBLRLAdaptDynamicsLearner
from pbcwm.core.types import Transition


def _transition(sign: float, index: int) -> Transition:
    obs = np.array([((index % 5) - 2) / 2], dtype=np.float32)
    action = np.array([0.7], dtype=np.float32)
    return Transition(obs, action, obs + sign * 0.4, 0.0, False, False)


def _learner() -> VBLRLAdaptDynamicsLearner:
    return VBLRLAdaptDynamicsLearner(
        1,
        1,
        hidden_dims=(16, 16),
        posterior_learning_rate=0.03,
        world_learning_rate=0.03,
        world_buffer_size=32,
        current_buffer_size=32,
        dynamics_batch_size=8,
        dynamics_updates_per_step=1,
        world_updates_per_interval=0,
        world_update_interval_steps=1000,
        router_window_size=4,
        router_posterior_samples=5,
        shift_threshold=1.0,
        reuse_threshold=0.5,
        consecutive_trigger_windows=2,
        router_cooldown_steps=8,
        planning_model_samples=3,
        seed=0,
    )


def test_vblrl_adapt_acquires_and_reuses_old_posterior_without_stage_metadata() -> None:
    torch.manual_seed(0)
    learner = _learner()

    # Isolate the lifecycle/router contract from optimizer convergence.  The
    # router still only sees transition deltas; no stage/task label is passed.
    def synthetic_predictive_nll(posterior_id, transitions, samples):
        del samples
        mean_delta = float(np.mean([item.next_obs[0] - item.obs[0] for item in transitions]))
        expected_id = 0 if mean_delta > 0 else 1
        return 0.1 if posterior_id == expected_id else 5.0

    learner._window_nll = synthetic_predictive_nll
    for sign in (1.0, -1.0, 1.0):
        for index in range(30):
            learner.observe(_transition(sign, index))
            learner.update(4)

    changes = [
        (index, value)
        for index, value in enumerate(learner.assignment_history)
        if index == 0 or learner.assignment_history[index - 1] != value
    ]
    assert changes == [(0, 0), (32, 1), (62, 0)]
    assert learner.num_regime_posteriors == 2
    assert learner.reacquisition_count == 1
    assert all(not hasattr(item, "true_dynamics_stage") for item in learner.router.window)


def test_vblrl_state_roundtrip_preserves_posterior_lifecycle() -> None:
    learner = _learner()
    for index in range(10):
        learner.observe(_transition(1.0, index))
        learner.update(1)
    state = learner.state_dict()

    restored = _learner()
    restored.load_state_dict(state)
    assert restored.assignment_history == learner.assignment_history
    assert restored.current_posterior_id == learner.current_posterior_id
    assert restored.num_regime_posteriors == learner.num_regime_posteriors
    assert restored.world_buffer_size == learner.world_buffer_size
