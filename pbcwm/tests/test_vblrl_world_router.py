import numpy as np

from pbcwm.baselines.vblrl.router import PosteriorPredictiveRouter
from pbcwm.baselines.vblrl.world_model import WorldPosterior
from pbcwm.core.types import Transition


def _transition(index: int) -> Transition:
    obs = np.array([index / 10.0], dtype=np.float32)
    return Transition(obs, np.array([0.1], dtype=np.float32), obs + 0.2, 7.0, False, False)


def test_world_posterior_initializes_new_regime_from_qw_snapshot() -> None:
    world = WorldPosterior(
        1, 1, hidden_dims=(8,), learning_rate=0.02, world_buffer_size=4, seed=0
    )
    for index in range(6):
        world.observe(_transition(index))
    world.update(num_steps=2, batch_size=4)
    new_posterior, prior = world.initialize_regime()
    for left, right in zip(
        world.posterior.network.state_dict().values(),
        new_posterior.network.state_dict().values(),
    ):
        assert np.array_equal(left.detach().numpy(), right.detach().numpy())
    assert len(world.buffer.storage) == 4
    assert all(item.reward == 0.0 for item in world.buffer.storage)
    assert len(prior["layers"]) == 2


def test_posterior_router_requires_persistent_mismatch_and_can_reacquire() -> None:
    router = PosteriorPredictiveRouter(
        window_size=3,
        posterior_samples=2,
        shift_threshold=1.0,
        reuse_threshold=0.5,
        consecutive_trigger_windows=2,
        cooldown_steps=0,
    )
    for index in range(3):
        router.add_transition(_transition(index))
    scores = {0: 2.0, 1: 0.1}

    def nll(posterior_id, transitions, samples):
        assert len(transitions) == 3
        assert samples == 2
        return scores[posterior_id]

    first = router.evaluate(0, [0, 1], nll)
    second = router.evaluate(0, [0, 1], nll)
    assert not first.reacquisition_triggered
    assert second.reacquisition_triggered
    assert second.selected_posterior_id == 1
