import torch

from pbcwm.methods.radius.elasticity import LowRankPredictiveFisher, PredictiveElasticityController, woodbury_solve


def test_woodbury_matches_dense_solve():
    torch.manual_seed(0)
    sketch = torch.randn(6, 3)
    vector = torch.randn(6)
    damping = 0.2
    expected = torch.linalg.solve(sketch @ sketch.T + damping * torch.eye(6), vector)
    assert torch.allclose(woodbury_solve(sketch, damping, vector), expected, atol=1e-5)


def test_woodbury_rejects_non_positive_damping():
    with torch.no_grad():
        try:
            woodbury_solve(torch.empty(2, 0), 0.0, torch.ones(2))
        except ValueError:
            pass
        else:
            raise AssertionError("non-positive damping must fail closed")


def test_pec_protects_old_direction_and_allows_free_plasticity():
    fisher = LowRankPredictiveFisher(2, damping=1e-3, max_rank=2)
    fisher.set_sketch(torch.tensor([[10.0], [0.0]]))
    gradient = torch.tensor([1.0, 1.0])
    transformed = fisher.solve(gradient)
    assert abs(float(transformed[0])) < abs(float(transformed[1]))
    free = torch.tensor([0.0, 1.0])
    assert fisher.protected_energy(free) < 1e-8


def test_trust_region_controller_returns_finite_bounded_gradient():
    controller = PredictiveElasticityController(3, type("Config", (), {"enabled": True, "mode": "trust_region", "forgetting_budget": 1e-3, "fisher_damping": 1e-3, "fisher_sketch_rank": 2})())
    gradient = torch.ones(3)
    result = controller.transform_gradient(gradient)
    assert torch.isfinite(result).all()
    assert controller.diagnostics(gradient)["radius/pec_fisher_rank"] == 0.0
