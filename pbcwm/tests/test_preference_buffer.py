import torch

from pbcwm.preferences.buffer import PreferenceBuffer
from pbcwm.preferences.types import PreferenceExample, TrajectorySegment


def segment(value: float) -> TrajectorySegment:
    obs = torch.full((3, 2), value)
    actions = torch.zeros(3, 1)
    return TrajectorySegment(obs, actions, obs + 0.1)


def test_preference_buffer_preserves_labels_and_shapes() -> None:
    buffer = PreferenceBuffer(capacity=2, seed=0)
    buffer.add(PreferenceExample(segment(0.0), segment(1.0), 0))
    buffer.add(PreferenceExample(segment(1.0), segment(2.0), 1))
    buffer.add(PreferenceExample(segment(2.0), segment(3.0), 0))

    assert len(buffer) == 2
    example = buffer.sample(1)[0]
    assert example.label in (0, 1)
    assert example.traj_a.obs.shape == (3, 2)
    assert example.traj_a.actions.shape == (3, 1)
    assert example.traj_a.next_obs.shape == (3, 2)
