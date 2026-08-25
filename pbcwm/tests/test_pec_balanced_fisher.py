import numpy as np

from pbcwm.methods.radius import RadiusPbCWM
from pbcwm.tests.test_radius_core import small_config


def test_pec_fisher_refresh_uses_all_available_prototype_pools():
    method = RadiusPbCWM(2, 1, small_config(), seed=4)
    method.state_normalizer.update(np.zeros(2, dtype=np.float32))
    method.delta_normalizer.update(np.zeros(2, dtype=np.float32))
    method.anchors.add(1, np.zeros(2), np.zeros(1), np.zeros(2))
    method.anchors.add(2, np.ones(2), np.zeros(1), np.ones(2))
    method.refresh_pec_fisher()
    assert 0 < method.pec.rank <= method.config.pec.fisher_sketch_rank
