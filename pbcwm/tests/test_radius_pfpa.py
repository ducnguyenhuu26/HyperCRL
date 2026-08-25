import torch

from pbcwm.methods.radius.preferences.frontier import PFPASelector, pair_entropy


def test_pfpa_entropy_and_frontier_score():
    assert pair_entropy(0.5) > pair_entropy(0.99)
    scores = torch.tensor([[10.0, 9.9, 0.0, -1.0], [9.8, 10.0, 0.1, -1.0], [10.1, 9.7, 0.2, -1.0]])
    _, _, frontier = PFPASelector.score_from_samples(scores)
    # Pair (0,1) is both uncertain and planner-frontier relevant.
    assert float(frontier[0]) > float(frontier[-1])


def test_pfpa_budget_split_is_8_frontier_2_coverage():
    torch.manual_seed(0)
    scores = torch.randn(5, 10)
    actions = torch.randn(10, 4)
    selection = PFPASelector(frontier_fraction=0.8, seed=0).select_from_scores(scores, actions, 10)
    assert len(selection.pairs) == 10
    assert selection.frontier_pairs == 8
    assert selection.coverage_pairs == 2


def test_pfpa_coverage_prefers_actionally_distinct_pairs():
    scores = torch.zeros(2, 3)
    actions = torch.tensor([[0.0], [1.0], [10.0]])
    selection = PFPASelector(frontier_fraction=0.0, seed=0).select_from_scores(scores, actions, 1)
    assert selection.pairs == [(0, 2)]
