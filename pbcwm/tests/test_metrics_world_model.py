import torch

from pbcwm.metrics.common import RolloutProbeBatch
from pbcwm.metrics.world_model import nrmse_at_H, r2_at_H, r2_at_h, r2_horizon_curve


class LinearDynamics:
    def __init__(self, bias: float = 0.0):
        self.bias = bias

    def predict(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return obs + torch.cat((action, 2.0 * action), dim=-1) + self.bias


def make_probe(batch=8, horizon=4):
    actions = torch.arange(batch * horizon, dtype=torch.float32).reshape(batch, horizon, 1) / 10.0
    initial = torch.zeros(batch, 2)
    states = [initial]
    current = initial
    for step in range(horizon):
        current = LinearDynamics().predict(current, actions[:, step])
        states.append(current)
    return RolloutProbeBatch(initial, actions, torch.stack(states, dim=1))


def test_perfect_linear_dynamics_has_perfect_multistep_metrics():
    probe = make_probe()
    assert abs(r2_at_h(LinearDynamics(), probe, 1).value - 1.0) < 1e-6
    assert abs(r2_at_H(LinearDynamics(), probe, 4).value - 1.0) < 1e-6
    assert nrmse_at_H(LinearDynamics(), probe, 4).value < 1e-6


def test_negative_r2_is_not_clipped_and_curve_is_recursive():
    probe = make_probe()
    bad = LinearDynamics(bias=10.0)
    assert r2_at_h(bad, probe, 1).value < 0.0
    curve = r2_horizon_curve(LinearDynamics(bias=0.01), probe)
    assert curve.values[0] > curve.values[-1]


def test_degenerate_dimension_is_excluded():
    probe = make_probe()
    result = r2_at_h(LinearDynamics(), probe, 1)
    assert result.metadata["valid_dimension_count"] == 2
    constant = RolloutProbeBatch(
        torch.cat((probe.initial_obs[:, :1], torch.ones_like(probe.initial_obs[:, :1])), dim=-1),
        probe.actions,
        torch.cat((probe.true_states[:, :, :1], torch.ones_like(probe.true_states[:, :, :1])), dim=-1),
    )
    result = r2_at_h(LinearDynamics(), constant, 1)
    assert result.metadata["valid_dimension_count"] == 1
