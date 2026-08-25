import numpy as np

from pbcwm.baselines.gpmm.learner import GPMMDynamicsLearner
from pbcwm.core.types import Transition


def _transition(sign: float, index: int) -> Transition:
    obs = np.array([((index % 5) - 2) / 2], dtype=np.float32)
    action = np.array([0.7], dtype=np.float32)
    next_obs = obs + sign * 0.4
    return Transition(obs, action, next_obs, 0.0, False, False)


def _learner(**overrides) -> GPMMDynamicsLearner:
    config = {
        "alpha": 1.0,
        "sticky_bonus": 0.0,
        "transition_base_count": 1.0,
        "expert_min_points_before_competition": 3,
        "gp_fit_steps": 5,
        "gp_learning_rate": 0.08,
        "max_points_per_expert": 32,
        "merge_enabled": False,
        "seed": 0,
    }
    config.update(overrides)
    return GPMMDynamicsLearner(1, 1, **config)


def test_new_expert_is_created_for_a_new_dynamics_regime() -> None:
    learner = _learner()
    for index in range(10):
        learner.observe(_transition(1.0, index))
        learner.update(1)
    for index in range(10):
        learner.observe(_transition(-1.0, index))
        learner.update(1)

    assert learner.num_experts == 2
    assert learner.new_expert_count == 2
    assert learner.switch_count >= 1


def test_old_expert_is_recalled_after_return_to_previous_regime() -> None:
    learner = _learner()
    assignments = []
    for sign in (1.0, -1.0, 1.0):
        for index in range(10):
            learner.observe(_transition(sign, index))
            learner.update(1)
            assignments.append(learner.current_expert)

    assert assignments[0] == 0
    assert assignments[10] == 1
    assert assignments[20] == 0
    assert learner.num_experts == 2
    assert learner.new_expert_count == 2


def test_serialization_restores_assignment_state() -> None:
    learner = _learner()
    for index in range(6):
        learner.observe(_transition(1.0, index))
    restored = _learner()
    restored.load_state_dict(learner.state_dict())

    assert restored.num_experts == learner.num_experts
    assert restored.current_expert == learner.current_expert
    assert restored.previous_expert == learner.previous_expert
    assert restored.assignment_history == learner.assignment_history
    assert restored.transition_counts.equal(learner.transition_counts)
    assert [expert.num_points for expert in restored.experts] == [
        expert.num_points for expert in learner.experts
    ]
