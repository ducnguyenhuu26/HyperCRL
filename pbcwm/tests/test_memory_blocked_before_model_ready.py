import numpy as np

from pbcwm.core.types import Transition
from pbcwm.methods.radius import RadiusPbCWM
from pbcwm.tests.test_radius_core import small_config


def test_memory_has_no_prototype_before_first_model_update():
    method = RadiusPbCWM(2, 1, small_config(), seed=0)
    for _ in range(8):
        obs = np.zeros(2, dtype=np.float32)
        method.observe(Transition(obs, np.zeros(1, dtype=np.float32), np.ones(2, dtype=np.float32), 0.0, False, False))
    assert method.model_updates_total == 0
    assert not method.memory.prototypes
