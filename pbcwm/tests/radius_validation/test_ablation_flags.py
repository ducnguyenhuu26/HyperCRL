from pbcwm.experiments.radius_validation.variants import variant_config


def test_w0_is_plain_and_w1_to_w4_have_explicit_switches():
    assert variant_config("W0") is None
    w1 = variant_config("W1")
    w2 = variant_config("W2")
    w3 = variant_config("W3")
    w4 = variant_config("W4")
    assert w1.atlas.initial_rank == w2.atlas.initial_rank == 3
    assert w1.atlas.max_rank == w2.atlas.max_rank == 3
    assert w1.ablations.disable_recurrent_memory
    assert not w2.ablations.disable_recurrent_memory
    assert w1.ablations.disable_rne and w2.ablations.disable_rne
    assert not w3.ablations.disable_rne and not w4.ablations.disable_rne
    assert w1.ablations.disable_pec and w2.ablations.disable_pec and w3.ablations.disable_pec
    assert not w4.ablations.disable_pec
    assert all(config.ablations.disable_pfpa for config in (w1, w2, w3, w4))
    assert w3.atlas.initial_rank == w4.atlas.initial_rank == 2
    assert w3.atlas.max_rank == w4.atlas.max_rank == 8
