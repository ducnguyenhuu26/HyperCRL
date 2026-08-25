import numpy as np

from pbcwm.baselines.hypercrl.router import ResidualRegimeRouter
from pbcwm.core.types import Transition


def _transition(value: float) -> Transition:
    point = np.array([value], dtype=np.float32)
    return Transition(point, point, point, 0.0, False, False)


def test_router_requires_persistent_residual_before_switching() -> None:
    router = ResidualRegimeRouter(
        window_size=2,
        shift_threshold=0.5,
        reuse_threshold=0.1,
        consecutive_trigger_windows=2,
        cooldown_steps=2,
    )
    errors = {0: 1.0, 1: 0.01}
    for _ in range(2):
        router.add_transition(_transition(0.0))
    first = router.evaluate(0, [0, 1], lambda embedding_id, _: errors[embedding_id])
    second = router.evaluate(0, [0, 1], lambda embedding_id, _: errors[embedding_id])

    assert not first.shift_triggered
    assert second.shift_triggered
    assert second.reuse_triggered
    assert second.selected_embedding_id == 1


def test_router_does_not_switch_when_residual_is_stable() -> None:
    router = ResidualRegimeRouter(window_size=2, shift_threshold=0.5, consecutive_trigger_windows=2)
    for _ in range(8):
        router.add_transition(_transition(0.0))
        decision = router.evaluate(0, [0], lambda _, __: 0.01)
        assert not decision.shift_triggered
        assert not decision.new_embedding_triggered
