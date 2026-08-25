import torch

from pbcwm.methods.radius.config import RadiusConfig
from pbcwm.methods.radius.inference import RecurrentEvidenceFilter
from pbcwm.methods.radius.memory import ContextMemory
from pbcwm.methods.radius.types import ContextPosterior


def test_ref_does_not_duplicate_active_prototype_in_window_routing():
    config = RadiusConfig().ref
    memory = ContextMemory(4, 2.5, 0.25)
    prototype = ContextPosterior(torch.zeros(2), torch.eye(2), 0.0, "active", prototype_id=7)
    memory.consolidate(prototype, 1)
    memory.prototypes[0].prototype_id = 7
    ref = RecurrentEvidenceFilter(2, 1.0, config, memory, torch.device("cpu"))
    result = ref.evaluate_hypotheses(torch.zeros(4, 1, 2), torch.zeros(4, 1), prototype)
    assert [candidate.source for candidate in result.candidates] == ["new"]
