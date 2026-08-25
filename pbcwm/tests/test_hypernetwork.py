import torch

from pbcwm.baselines.hypercrl.hypernetwork import HyperNetwork
from pbcwm.baselines.hypercrl.target_dynamics import TargetDynamics


def test_hypernetwork_generates_exact_target_shapes_and_batched_vectors() -> None:
    target = TargetDynamics(input_dim=3, output_dim=2, hidden_dims=(5, 4))
    hypernetwork = HyperNetwork(embedding_dim=6, target_shapes=target.parameter_shapes, hidden_dims=(7, 8))
    embedding = torch.randn(6)
    weights = hypernetwork(embedding)

    assert [tuple(weight.shape) for weight in weights] == list(target.parameter_shapes)
    batched = hypernetwork(torch.randn(3, 6))
    assert [tuple(weight.shape) for weight in batched] == [
        (3, *shape) for shape in target.parameter_shapes
    ]

    obs = torch.randn(4, 2)
    action = torch.randn(4, 1)
    prediction = target.predict_next(obs, action, weights)
    assert prediction.shape == (4, 2)
    assert torch.isfinite(prediction).all()
