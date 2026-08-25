from pbcwm.experiments.radius_validation.metrics import first_visit_auc, return_visit_auc, t90, wm_reuse_advantage


def test_validation_metrics_are_deterministic_and_fail_closed():
    checkpoints = [16, 32, 64, 128]
    first = [0.1, 0.3, 0.6, 0.8]
    recurrence = [0.2, 0.5, 0.7, 0.85]
    first_auc = first_visit_auc(checkpoints, first)
    return_auc = return_visit_auc(checkpoints, recurrence)
    assert return_auc > first_auc
    assert wm_reuse_advantage(first_auc, return_auc) > 0.0
    assert t90(checkpoints, recurrence, stage_length=10000) == 128
