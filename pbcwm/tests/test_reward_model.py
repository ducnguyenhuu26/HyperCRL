import torch

from pbcwm.preferences.buffer import PreferenceBuffer
from pbcwm.preferences.reward_model import PreferenceRewardEnsemble
from pbcwm.preferences.types import PreferenceExample, TrajectorySegment


def make_segment(value: float) -> TrajectorySegment:
    obs = torch.tensor([[value, 0.0], [value, 0.0]])
    return TrajectorySegment(obs, torch.zeros(2, 1), obs.clone())


def test_bradley_terry_training_improves_synthetic_ranking() -> None:
    torch.manual_seed(0)
    buffer = PreferenceBuffer(seed=0)
    for value in range(1, 17):
        buffer.add(PreferenceExample(make_segment(float(value)), make_segment(float(value - 1)), 0))
        buffer.add(PreferenceExample(make_segment(float(value - 1)), make_segment(float(value)), 1))

    ensemble = PreferenceRewardEnsemble(
        obs_dim=2,
        action_dim=1,
        ensemble_size=3,
        hidden_dims=(16,),
        learning_rate=5e-3,
        batch_size=8,
        seed=1,
    )
    initial = ensemble.update(buffer, num_steps=1)
    for _ in range(50):
        ensemble.update(buffer, num_steps=1)
    final = ensemble.update(buffer, num_steps=1)

    assert final["preference_loss"] < initial["preference_loss"]
    assert final["preference_accuracy"] > 0.5
    returns = ensemble.trajectory_returns([make_segment(1.0), make_segment(2.0)])
    assert returns.shape == (2,)
    assert ensemble.reward(torch.zeros(4, 2), torch.zeros(4, 1)).shape == (4,)
