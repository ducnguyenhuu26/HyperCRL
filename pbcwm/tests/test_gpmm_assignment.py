import numpy as np
import torch

from pbcwm.baselines.gpmm.assignment import assign_transition, transition_log_priors
from pbcwm.baselines.gpmm.expert import GPExpert


def _trained_experts() -> list[GPExpert]:
    experts = []
    for sign in (1.0, -1.0):
        expert = GPExpert(1, 1, max_points=32, learning_rate=0.08, seed=0)
        for x in np.linspace(-1.0, 1.0, 12):
            expert.add_transition([x], [0.7], [x + sign * 0.4])
        expert.fit(8)
        experts.append(expert)
    return experts


def test_assignment_uses_predictive_likelihood() -> None:
    experts = _trained_experts()
    result_plus = assign_transition(
        experts,
        torch.tensor([0.2]),
        torch.tensor([0.7]),
        torch.tensor([0.6]),
        previous_expert=0,
        transition_counts=torch.zeros(2, 2, dtype=torch.float64),
        alpha=1.0,
        sticky_bonus=0.0,
        base_count=1.0,
    )
    result_minus = assign_transition(
        experts,
        torch.tensor([0.2]),
        torch.tensor([0.7]),
        torch.tensor([-0.2]),
        previous_expert=0,
        transition_counts=torch.zeros(2, 2, dtype=torch.float64),
        alpha=1.0,
        sticky_bonus=0.0,
        base_count=1.0,
    )

    assert result_plus.selected_index == 0
    assert result_minus.selected_index == 1
    assert result_plus.posterior[0] > result_plus.posterior[1]
    assert result_minus.posterior[1] > result_minus.posterior[0]


def test_sticky_transition_prior_favors_previous_expert() -> None:
    counts = torch.tensor([[0.0, 4.0], [0.0, 0.0]], dtype=torch.float64)
    priors = transition_log_priors(2, 0, counts, 1.0, 10.0, 1.0)
    assert int(torch.argmax(priors).item()) == 0
    assert priors[0] > priors[1]
