import numpy as np

from pbcwm.experiments.radius_validation.generate_fixed_stream import generate_fixed_stream, load_fixed_stream


def test_all_variants_share_identical_learner_payload(tmp_path):
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    generate_fixed_stream(first, seed=0, steps=48, synthetic=True)
    generate_fixed_stream(second, seed=0, steps=48, synthetic=True)
    left, right = load_fixed_stream(first), load_fixed_stream(second)
    assert left.learner_payload_sha256 == right.learner_payload_sha256
    for name in ("obs", "action", "next_obs", "terminated", "truncated"):
        assert np.array_equal(getattr(left, name), getattr(right, name))
