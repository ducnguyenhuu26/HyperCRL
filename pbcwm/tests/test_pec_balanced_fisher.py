import numpy as np
import torch

from pbcwm.methods.radius import RadiusPbCWM
from pbcwm.methods.radius.types import ContextPosterior
from pbcwm.tests.test_radius_core import small_config


def test_pec_fisher_refresh_uses_all_available_prototype_pools():
    method = RadiusPbCWM(2, 1, small_config(), seed=4)
    method.state_normalizer.update(np.zeros(2, dtype=np.float32))
    method.delta_normalizer.update(np.zeros(2, dtype=np.float32))
    method.anchors.add(1, np.zeros(2), np.zeros(1), np.zeros(2))
    method.anchors.add(2, np.ones(2), np.zeros(1), np.ones(2))
    method.memory.consolidate(ContextPosterior(torch.zeros(2), torch.eye(2), 0.0, "active"), 1)
    method.memory.prototypes[0].prototype_id = 1
    method.memory.consolidate(ContextPosterior(torch.ones(2) * 10.0, torch.eye(2), 0.0, "active"), 2)
    method.memory.prototypes[1].prototype_id = 2
    method.refresh_pec_fisher()
    assert 0 < method.pec.rank <= method.config.pec.fisher_sketch_rank
