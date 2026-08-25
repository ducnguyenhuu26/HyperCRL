from pbcwm.metrics.registry import METRIC_REGISTRY, get_metric


def test_registry_exposes_stable_metric_names_without_composite_score():
    required = {
        "wm/r2_h1",
        "wm/r2_h",
        "wm/nrmse_h",
        "continual/wm_acq_auc",
        "continual/wm_reacq_auc",
        "continual/wm_reuse_advantage",
        "reward/pairwise_accuracy",
        "reward/bt_nll",
        "reward/kendall_true",
        "reward/kendall_imagined",
        "coupling/world_reward_kendall",
        "coupling/selection_regret",
        "coupling/normalized_selection_regret",
        "planning/return_mean",
        "planning/planning_deficit",
        "planning/dynamic_regret",
        "planning/adaptation_cost",
        "planning/reacquisition_cost",
        "planning/reuse_advantage",
        "oracle/j_ll",
        "oracle/j_ol",
        "oracle/j_lo",
        "oracle/j_oo",
        "oracle/world_side_gap",
        "oracle/reward_side_gap",
        "oracle/full_system_gap",
    }
    assert required.issubset(METRIC_REGISTRY)
    assert "composite/score" not in METRIC_REGISTRY
    assert get_metric("wm/r2_h").higher_is_better is True
