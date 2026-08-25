from pathlib import Path

import numpy as np
import torch

from pbcwm.core.types import Transition
from pbcwm.methods.radius import RadiusPbCWM, load_radius_config
from pbcwm.planning.cem import CEMPlanner
from pbcwm.preferences.reward_model import PreferenceRewardEnsemble
from pbcwm.rewards.preference import LearnedPreferenceReward
from pbcwm.tests.test_radius_core import small_config


def transition(obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray) -> Transition:
    return Transition(obs, action, next_obs, reward=999.0, terminated=False, truncated=False)


def test_radius_stationary_smoke_is_reward_free_and_finite():
    method = RadiusPbCWM(3, 1, small_config(), seed=0)
    obs = np.zeros(3, dtype=np.float32)
    for index in range(16):
        action = np.array([np.sin(index / 3)], dtype=np.float32)
        next_obs = obs + np.array([action[0], 2.0 * action[0], -action[0]], dtype=np.float32)
        method.observe(transition(obs, action, next_obs))
        diagnostics = method.update()
        assert all(np.isfinite(value) for value in diagnostics.values() if isinstance(value, (float, int)))
        obs = next_obs
    assert method.dynamics_ready
    assert torch.isfinite(method.predict(torch.zeros(2, 3), torch.zeros(2, 1))).all()
    assert method.get_atlas_rank() == 2


def test_radius_rank_migration_checkpoint_roundtrip():
    config = small_config()
    method = RadiusPbCWM(3, 1, config, seed=1)
    obs = torch.zeros(3)
    action = torch.zeros(1)
    for _ in range(4):
        next_obs = obs + torch.tensor([0.1, -0.2, 0.3])
        method.observe(transition(obs.numpy(), action.numpy(), next_obs.numpy()))
        obs = next_obs
    method._expand_atlas()
    assert method.rank == 3
    checkpoint = method.state_dict()
    restored = RadiusPbCWM(3, 1, config, seed=2)
    restored.load_state_dict(checkpoint)
    assert restored.rank == 3
    assert len(restored.get_context_prototypes()) == len(method.get_context_prototypes())
    test_obs = torch.randn(2, 3)
    test_action = torch.randn(2, 1)
    assert torch.allclose(method.predict(test_obs, test_action), restored.predict(test_obs, test_action))
    original_batch = method.replay.sample(2, method.rank, method.device)
    restored_batch = restored.replay.sample(2, restored.rank, restored.device)
    for original, replayed in zip(original_batch, restored_batch):
        assert torch.allclose(original, replayed)


def test_radius_source_does_not_accept_evaluator_or_reward_channels():
    source = "\n".join(path.read_text(encoding="utf-8") for path in Path("pbcwm/methods/radius").rglob("*.py"))
    forbidden = ("transition.reward", "true_reward", "segment_id", "dynamics_id", "visit_id", "change_event", "physical_parameter")
    assert all(token not in source for token in forbidden)


def test_radius_config_is_serializable_and_shared_protocol_values_are_absent():
    config = load_radius_config("pbcwm/configs/methods/radius.yaml")
    assert config.name == "radius_pb_cwm"
    assert not hasattr(config, "planner_horizon")
    assert not hasattr(config, "preference_budget")


def test_pfpa_uses_shared_cem_candidate_output():
    method = RadiusPbCWM(3, 1, small_config(), seed=3)
    obs = np.zeros(3, dtype=np.float32)
    for index in range(4):
        action = np.array([0.1 * index], dtype=np.float32)
        next_obs = obs + np.array([0.1, -0.1, 0.05], dtype=np.float32)
        method.observe(transition(obs, action, next_obs))
        obs = next_obs
    planner = CEMPlanner(horizon=2, population_size=10, elite_size=2, num_iterations=2, candidate_keep_per_iteration=10, action_low=[-1.0], action_high=[1.0])
    ensemble = PreferenceRewardEnsemble(3, 1, ensemble_size=2, hidden_dims=(8,), batch_size=1, seed=3)
    plan = planner.plan(obs, method, LearnedPreferenceReward(ensemble), return_candidates=True)
    selection = method.select_preference_queries(obs, plan, ensemble, 4)
    assert len(selection.pairs) == 4
    diagnostics = method.diagnostics()
    assert diagnostics["radius/pfpa_mean_entropy"] >= 0.0
    assert diagnostics["radius/pfpa_mean_frontier_score"] >= 0.0
