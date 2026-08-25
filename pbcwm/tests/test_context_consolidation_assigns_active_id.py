import torch

from pbcwm.methods.radius.memory import ContextMemory
from pbcwm.methods.radius.types import ContextPosterior


def test_consolidation_returns_active_prototype_id():
    result = ContextMemory(2, 2.5, 0.25).consolidate(ContextPosterior(torch.zeros(2), torch.eye(2), 0.0, "active"), 1)
    assert result.prototype_id == 0
    assert result.evicted_prototype_id is None
