import torch

from pbcwm.core.normalization import RunningNormalizer


def test_running_normalizer_matches_mean_variance_and_roundtrip():
    normalizer = RunningNormalizer(2, clip=None)
    values = torch.tensor([[1.0, 4.0], [3.0, 8.0], [5.0, 12.0]])
    normalizer.update(values)
    assert torch.allclose(normalizer.mean, torch.tensor([3.0, 8.0]))
    assert torch.allclose(normalizer.variance, torch.tensor([4.0, 16.0]))
    assert torch.allclose(normalizer.denormalize(normalizer.normalize(values)), values, atol=1e-5)
    restored = RunningNormalizer(2, clip=None)
    restored.load_state_dict(normalizer.state_dict())
    assert restored.count == normalizer.count
    assert torch.allclose(restored.mean, normalizer.mean)
