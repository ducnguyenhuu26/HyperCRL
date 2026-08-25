import torch

from pbcwm.metrics.common import CandidateTrajectoryBank, PreferenceEvalBatch
from pbcwm.metrics.coupling import (
    CandidateScoreSet,
    normalized_selection_regret,
    reward_kendall_imagined,
    reward_kendall_true,
    selection_regret,
    world_reward_kendall,
)
from pbcwm.metrics.oracle import OraclePlannerScores, oracle_diagnostics
from pbcwm.metrics.reward import bradley_terry_nll, pairwise_preference_accuracy
from pbcwm.preferences.types import TrajectorySegment


class FixedPreferenceModel:
    def preference_probabilities(self, traj_a, traj_b):
        score_a = float(traj_a.actions.sum())
        score_b = float(traj_b.actions.sum())
        return torch.tensor([torch.sigmoid(torch.tensor(score_a - score_b))])


def segment(value: float) -> TrajectorySegment:
    return TrajectorySegment(torch.zeros(2, 1), torch.full((2, 1), value), torch.zeros(2, 1))


def test_preference_metrics_and_ties_are_explicit():
    batch = PreferenceEvalBatch([segment(1), segment(0)], [segment(0), segment(1)], torch.tensor([0, 1]))
    assert pairwise_preference_accuracy(FixedPreferenceModel(), batch).value == 1.0
    assert bradley_terry_nll(FixedPreferenceModel(), batch).value < 1.0
    skipped = PreferenceEvalBatch([segment(1)], [segment(1)], torch.tensor([-1]))
    assert not pairwise_preference_accuracy(FixedPreferenceModel(), skipped).valid


def test_reward_coupling_and_selection_regret():
    scores = CandidateScoreSet(torch.tensor([3.0, 2.0, 1.0]), torch.tensor([3.0, 2.0, 1.0]), torch.tensor([3.0, 2.0, 1.0]), torch.tensor([3.0, 2.0, 1.0]))
    assert reward_kendall_true(scores).value == 1.0
    assert reward_kendall_imagined(scores).value == 1.0
    assert world_reward_kendall(scores).value == 1.0
    inverted = selection_regret(torch.tensor([3.0, 2.0, 1.0]), torch.tensor([1.0, 2.0, 3.0]))
    assert inverted.value == 2.0
    assert normalized_selection_regret(torch.tensor([1.0, 1.0]), torch.tensor([0.0, 1.0])).valid is False


def test_oracle_decomposition():
    metrics = oracle_diagnostics(OraclePlannerScores(70, 90, 85, 100))
    assert metrics["oracle/world_side_gap"].value == 15
    assert metrics["oracle/reward_side_gap"].value == 10
    assert metrics["oracle/full_system_gap"].value == 30
    assert metrics["oracle/world_reward_interaction"].value == -5
