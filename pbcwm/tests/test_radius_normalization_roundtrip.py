import numpy as np
import torch

from pbcwm.core.types import Transition
from pbcwm.methods.radius import RadiusPbCWM
from pbcwm.tests.test_radius_core import small_config


def test_radius_normalizers_update_only_from_observed_transitions_and_checkpoint():
    method = RadiusPbCWM(2, 1, small_config(), seed=0)
    for value in range(5):
        obs = np.array([value, 10 * value], dtype=np.float32)
        action = np.array([0.5], dtype=np.float32)
        method.observe(Transition(obs, action, obs + 1.0, 999.0, False, False))
    assert method.state_normalizer.count == 10
    assert method.delta_normalizer.count == 5
    assert torch.isfinite(method.predict(torch.zeros(2, 2), torch.zeros(2, 1))).all()
    restored = RadiusPbCWM(2, 1, small_config(), seed=1)
    restored.load_state_dict(method.state_dict())
    assert restored.state_normalizer.count == method.state_normalizer.count
    assert restored.delta_normalizer.count == method.delta_normalizer.count
