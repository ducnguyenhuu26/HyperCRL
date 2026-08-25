import numpy as np
import torch

from pbcwm.methods.radius.memory import ContextMemory
from pbcwm.methods.radius.types import ContextPosterior


def test_context_memory_reports_eviction_for_anchor_cleanup():
    memory = ContextMemory(1, 0.0, 0.25)
    memory.consolidate(ContextPosterior(torch.zeros(1), torch.eye(1), 0.0, "active"), 1)
    result = memory.consolidate(ContextPosterior(torch.ones(1) * 10, torch.eye(1), 0.0, "active"), 2)
    assert result.evicted_prototype_id == 0
