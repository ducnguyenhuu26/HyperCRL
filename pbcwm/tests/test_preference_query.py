import torch

from pbcwm.preferences.query import DisagreementQuerySelector
from pbcwm.preferences.types import TrajectorySegment


class DeliberatelyDisagreeingEnsemble:
    def preference_probabilities(self, traj_a, traj_b):
        left = int(traj_a.obs[0, 0].item())
        right = int(traj_b.obs[0, 0].item())
        if {left, right} == {0, 1}:
            return torch.tensor([0.01, 0.99, 0.50])
        return torch.tensor([0.50, 0.50, 0.50])


def candidate(index: int) -> TrajectorySegment:
    value = torch.tensor(float(index))
    return TrajectorySegment(value.reshape(1, 1), torch.zeros(1, 1), value.reshape(1, 1))


def test_selector_prioritizes_disagreement_without_duplicate_pairs() -> None:
    candidates = [candidate(index) for index in range(4)]
    selector = DisagreementQuerySelector(pair_pool_size=32, seed=0)
    pairs = selector.select(candidates, DeliberatelyDisagreeingEnsemble(), num_queries=3)

    assert pairs[0] == (0, 1)
    assert all(left != right for left, right in pairs)
    assert len({tuple(sorted(pair)) for pair in pairs}) == len(pairs)
