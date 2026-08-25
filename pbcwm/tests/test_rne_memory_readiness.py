from dataclasses import replace

import numpy as np

from pbcwm.core.types import Transition
from pbcwm.methods.radius import RadiusPbCWM
from pbcwm.tests.test_radius_core import small_config


def test_rne_expansion_is_blocked_before_model_readiness():
    config = replace(small_config(), rne=replace(small_config().rne, min_model_updates_before_expansion=10**6))
    method = RadiusPbCWM(2, 1, config, seed=0)
    for _ in range(8):
        obs = np.zeros(2, dtype=np.float32)
        method.observe(Transition(obs, np.zeros(1, dtype=np.float32), np.ones(2, dtype=np.float32), 0.0, False, False))
    assert method.rank == config.atlas.initial_rank
    assert method.rne_blocked_not_ready >= 0
