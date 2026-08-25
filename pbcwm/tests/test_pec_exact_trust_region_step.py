import torch

from pbcwm.methods.radius.elasticity import PredictiveElasticityController


def test_pec_direct_step_matches_trust_region_budget_without_cap():
    config = type("Config", (), {"enabled": True, "mode": "trust_region", "forgetting_budget": 0.2, "fisher_damping": 0.1, "fisher_sketch_rank": 2, "max_step_norm": 100.0})()
    controller = PredictiveElasticityController(3, config)
    controller.refresh_from_sketch(torch.tensor([[1.0], [0.0], [0.0]]))
    delta = controller.direct_step(torch.tensor([1.0, 2.0, 0.5]))
    assert torch.isfinite(delta).all()
    assert abs(controller.last_predicted_forgetting_cost - 0.2) < 1e-5
    assert not controller.last_step_capped
