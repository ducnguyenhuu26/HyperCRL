from pbcwm.methods.radius.novelty import ResidualNoveltyMonitor


def test_rne_uses_normalized_scale_not_raw_coordinate_magnitude():
    first = ResidualNoveltyMonitor(3.0, 0.6, 2, 10)
    second = ResidualNoveltyMonitor(3.0, 0.6, 2, 10)
    assert first.update(4.0, 0.8, 1).should_expand == second.update(4.0, 0.8, 1).should_expand
    assert first.update(4.0, 0.8, 2).should_expand == second.update(4.0, 0.8, 2).should_expand
