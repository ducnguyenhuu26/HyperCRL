import torch

from pbcwm.methods.radius.preferences import PFPASelector


def test_pfpa_accepts_shared_planner_elite_fraction():
    scores = torch.tensor([[4.0, 3.9, 0.0, -1.0], [3.8, 4.0, 0.1, -1.0]])
    selection = PFPASelector().select_from_scores(scores, torch.randn(4, 2), 2, elite_fraction=0.5)
    assert selection.elite_fraction == 0.5
