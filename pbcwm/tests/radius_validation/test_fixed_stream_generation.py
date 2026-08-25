import numpy as np

from pbcwm.experiments.radius_validation.generate_fixed_stream import generate_fixed_stream, load_fixed_stream


def test_synthetic_stream_has_separated_payload_and_metadata(tmp_path):
    path = tmp_path / "stream.npz"
    stream = generate_fixed_stream(path, seed=0, steps=32, synthetic=True)
    restored = load_fixed_stream(path)
    assert stream.steps == 32
    assert restored.learner_payload_sha256 == stream.learner_payload_sha256
    assert restored.true_reward.shape == (32,)
    transition = restored.transition(0)
    assert transition.reward == 0.0
    assert np.isfinite(restored.obs).all()
    assert np.isfinite(restored.next_obs).all()
