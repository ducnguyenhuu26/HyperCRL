import numpy as np
import torch

from pbcwm.baselines.curious_replay.learner import CuriousReplayDynamicsLearner
from pbcwm.baselines.curious_replay.online import CuriousReplayOnline
from pbcwm.baselines.curious_replay.replay import (
    CuriousReplayBuffer,
    combined_priority,
    count_priority,
    loss_priority,
)
from pbcwm.core.types import Transition
from pbcwm.envs.nonstationary_pendulum import NonstationaryPendulum
from pbcwm.rewards.pendulum import PendulumReward


def _transition(sign: float, index: int, reward: float = 0.0) -> Transition:
    obs = np.array([((index % 5) - 2) / 2], dtype=np.float32)
    action = np.array([0.7], dtype=np.float32)
    return Transition(obs, action, obs + sign * 0.4, reward, False, False)


def test_priority_formula_and_count_decay() -> None:
    assert count_priority(0, beta=0.7, count_weight_c=1.0) > count_priority(
        3, beta=0.7, count_weight_c=1.0
    )
    assert loss_priority(4.0, alpha=0.6, epsilon=1e-6) > loss_priority(
        1.0, alpha=0.6, epsilon=1e-6
    )
    assert combined_priority(2, 3.0, 0.7, 0.6, 1e-6, 1.0) == (
        count_priority(2, 0.7, 1.0) + loss_priority(3.0, 0.6, 1e-6)
    )


def test_new_entry_gets_max_priority_and_fifo_eviction() -> None:
    buffer = CuriousReplayBuffer(capacity=3, initial_priority=1.0, seed=0)
    for index in range(3):
        buffer.add(_transition(1.0, index))
    buffer.update_priorities(
        np.array([0, 1, 2]), np.array([0.1, 0.2, 4.0], dtype=np.float64)
    )
    previous_max = buffer.max_priority
    new_slot = buffer.add(_transition(-1.0, 99))
    assert new_slot == 0
    assert len(buffer) == 3
    assert buffer.get(new_slot).replay_count == 0
    assert buffer.get(new_slot).priority == previous_max
    assert not np.array_equal(
        buffer.get(new_slot).transition.next_obs,
        buffer.get(1).transition.next_obs,
    )


def test_per_sample_losses_update_independent_priorities() -> None:
    buffer = CuriousReplayBuffer(capacity=4, seed=0)
    buffer.add(_transition(1.0, 0))
    buffer.add(_transition(1.0, 1))
    buffer.update_priorities(np.array([0, 1]), np.array([0.01, 4.0]))
    first, second = buffer.get(0), buffer.get(1)
    assert first.replay_count == second.replay_count == 1
    assert first.last_model_loss != second.last_model_loss
    assert first.priority < second.priority


def _learner(seed: int = 0) -> CuriousReplayDynamicsLearner:
    return CuriousReplayDynamicsLearner(
        1,
        1,
        hidden_dims=(16, 16),
        capacity=64,
        batch_size=8,
        learning_rate=0.02,
        beta=0.7,
        alpha=0.6,
        gradient_clip_norm=10.0,
        seed=seed,
    )


def test_reward_changes_do_not_change_replay_or_prediction() -> None:
    first, second = _learner(), _learner()
    for index in range(20):
        first.observe(_transition(1.0, index, reward=0.0))
        second.observe(_transition(1.0, index, reward=123.0))
        first.update(1)
        second.update(1)
    first_entries = list(first.replay_buffer)
    second_entries = list(second.replay_buffer)
    assert all(entry.transition.reward == 0.0 for entry in second_entries)
    assert [(entry.replay_count, entry.last_model_loss) for entry in first_entries] == [
        (entry.replay_count, entry.last_model_loss) for entry in second_entries
    ]
    obs = torch.zeros(2, 1)
    action = torch.zeros(2, 1)
    torch.testing.assert_close(first.predict(obs, action), second.predict(obs, action))


def test_a_b_a_replay_focus_uses_new_transition_priority_without_regime_state() -> None:
    learner = _learner()
    evaluator_stage_by_slot: dict[int, int] = {}
    sampled_stage_counts: dict[int, int] = {}
    for stage, sign in ((0, 1.0), (1, -1.0), (2, 1.0)):
        for index in range(16):
            learner.observe(_transition(sign, index, reward=stage * 100.0))
            assert learner.last_observed_index is not None
            evaluator_stage_by_slot[learner.last_observed_index] = stage
            learner.update(1)
            for slot in learner.last_sample_indices:
                if slot in evaluator_stage_by_slot:
                    sampled_stage_counts[evaluator_stage_by_slot[slot]] = (
                        sampled_stage_counts.get(evaluator_stage_by_slot[slot], 0) + 1
                    )
    assert learner.replay_buffer.max_priority > learner.replay_buffer.min_priority
    assert sampled_stage_counts.get(1, 0) > 0
    assert sampled_stage_counts.get(2, 0) > 0
    assert not hasattr(learner, "current_regime")
    assert not hasattr(learner, "regime_id")


def test_curious_replay_state_roundtrip() -> None:
    learner = _learner()
    for index in range(10):
        learner.observe(_transition(1.0, index))
        learner.update(1)
    restored = _learner(seed=123)
    restored.load_state_dict(learner.state_dict())
    assert restored.global_step == learner.global_step
    assert restored.update_count == learner.update_count
    assert restored.replay_buffer.statistics() == learner.replay_buffer.statistics()
    torch.testing.assert_close(
        restored.predict(torch.zeros(2, 1), torch.zeros(2, 1)),
        learner.predict(torch.zeros(2, 1), torch.zeros(2, 1)),
    )


def test_curious_replay_online_uses_shared_preference_and_cem() -> None:
    env = NonstationaryPendulum(
        [
            {"step": 0, "mass": 1.0, "length": 1.0, "gravity": 10.0},
            {"step": 8, "mass": 1.5, "length": 1.0, "gravity": 10.0},
        ]
    )
    env.reset(seed=0)
    env.action_space.seed(0)
    online = CuriousReplayOnline(
        obs_dim=3,
        action_dim=1,
        action_low=env.action_space.low,
        action_high=env.action_space.high,
        planner_config={
            "horizon": 3,
            "population_size": 12,
            "elite_size": 3,
            "num_iterations": 1,
            "initial_std": 1.0,
        },
        curious_replay_config={
            "hidden_dims": [16, 16],
            "capacity": 32,
            "batch_size": 4,
            "learning_rate": 0.01,
            "beta": 0.7,
            "alpha": 0.6,
            "epsilon": 1e-6,
            "count_weight_c": 1.0,
            "initial_priority": 1.0,
            "gradient_clip_norm": 10.0,
        },
        preference_config={
            "ensemble_size": 3,
            "hidden_dims": [16],
            "learning_rate": 3e-3,
            "reward_batch_size": 2,
            "min_preferences_before_planning": 2,
            "pair_pool_size": 16,
            "candidate_keep_per_iteration": 2,
            "candidate_keep_final_elites": 2,
            "teacher_skip_margin": 0.0,
        },
        teacher_reward=PendulumReward(),
        seed=0,
    )
    obs, _ = env.reset(seed=0)
    for _ in range(8):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, _ = env.step(action)
        online.observe(Transition(obs, action, next_obs, float(reward), terminated, truncated))
        online.update_dynamics(1)
        obs = next_obs
    assert online.dynamics_ready
    online.bootstrap(4, 3, env.action_space.low, env.action_space.high)
    assert len(online.preference_buffer) >= 2
    assert online.plan(obs, collect_candidates=True).candidate_trajectories
    env.close()
