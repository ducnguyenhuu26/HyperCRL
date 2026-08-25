import numpy as np

from pbcwm.baselines.gpmm.learner import GPMMDynamicsLearner
from pbcwm.core.types import Transition


def _add_data(learner: GPMMDynamicsLearner, expert_index: int, count: int = 6) -> None:
    for index in range(count):
        obs = np.array([float(index) / 10], dtype=np.float32)
        learner.experts[expert_index].add_transition(obs, np.array([0.2], dtype=np.float32), obs + 0.1)


def test_merge_rewrites_indices_and_transition_counts() -> None:
    learner = GPMMDynamicsLearner(
        1,
        1,
        gp_fit_steps=1,
        merge_enabled=False,
        max_points_per_expert=32,
        seed=0,
    )
    learner._create_expert()
    learner._create_expert()
    _add_data(learner, 0)
    _add_data(learner, 1)
    learner.current_expert = 0
    learner.previous_expert = 1
    learner.assignment_history = [0, 1, 1]
    learner.transition_counts[0, 1] = 2
    learner.transition_counts[1, 0] = 3

    learner._merge_experts(0, 1)

    assert learner.num_experts == 1
    assert learner.current_expert == 0
    assert learner.previous_expert == 0
    assert learner.assignment_history == [0, 0, 0]
    assert learner.transition_counts.shape == (1, 1)
    assert learner.experts[0].num_points == 12
