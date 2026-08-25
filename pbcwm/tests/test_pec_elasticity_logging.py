import torch

from pbcwm.methods.radius.elasticity import PredictiveElasticityController


def test_pec_diagnostics_expose_latest_step_values():
    config = type("Config", (), {"enabled": True, "mode": "trust_region", "forgetting_budget": 0.1, "fisher_damping": 0.1, "fisher_sketch_rank": 1, "max_step_norm": 100.0})()
    controller = PredictiveElasticityController(2, config)
    controller.direct_step(torch.ones(2))
    diagnostics = controller.diagnostics()
    assert diagnostics["radius/pec_continual_elasticity"] > 0.0
    assert diagnostics["radius/pec_predicted_forgetting_cost"] > 0.0
    assert diagnostics["radius/pec_step_norm"] > 0.0
