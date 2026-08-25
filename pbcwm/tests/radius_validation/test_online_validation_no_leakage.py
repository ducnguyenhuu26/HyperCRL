from pbcwm.experiments.radius_validation.generate_fixed_stream import generate_fixed_stream, load_fixed_stream


def test_online_transition_cannot_see_reward_or_regime_metadata(tmp_path):
    path = tmp_path / "stream.npz"
    generate_fixed_stream(path, seed=3, steps=24, synthetic=True)
    stream = load_fixed_stream(path)
    transition = stream.transition(5)
    assert transition.reward == 0.0
    assert not hasattr(transition, "stage_id")
    assert not hasattr(transition, "dynamics_id")
    assert not hasattr(transition, "parameter_vector")
    assert stream.true_reward[5] != 0.0
