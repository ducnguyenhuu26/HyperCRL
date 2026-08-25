from dataclasses import replace

import numpy as np
import torch

from pbcwm.core.types import Transition
from pbcwm.methods.radius import RadiusPbCWM
from pbcwm.methods.radius.atlas.losses import atom_orthogonality_loss, atlas_loss
from pbcwm.methods.radius.inference import RecurrentEvidenceFilter
from pbcwm.methods.radius.memory import ContextMemory
from pbcwm.methods.radius.types import ContextPosterior, ContextPrototype
from pbcwm.planning.cem import CEMPlanner
from pbcwm.preferences.reward_model import PreferenceRewardEnsemble
from pbcwm.preferences.types import TrajectorySegment
from pbcwm.tests.test_radius_core import small_config


def test_atom_gram_has_unit_scale_and_penalizes_collinearity_and_explosion():
    orthonormal = (2.0**0.5 * torch.eye(2)).reshape(1, 2, 2)
    collinear = orthonormal.clone()
    collinear[..., 1] = collinear[..., 0]
    exploded = orthonormal * 4.0
    assert atom_orthogonality_loss(orthonormal) < 1e-8
    assert atom_orthogonality_loss(collinear) > atom_orthogonality_loss(orthonormal)
    assert atom_orthogonality_loss(exploded) > atom_orthogonality_loss(orthonormal)


def test_ref_active_prior_is_present_once_and_new_can_win():
    config = replace(small_config().ref, active_prior_bonus=1.0, context_window=4, min_context_samples=2)
    memory = ContextMemory(4, 2.5, 0.25)
    active = ContextPosterior(torch.ones(2) * 10.0, torch.eye(2), 0.0, "active", prototype_id=7)
    memory.prototypes.append(ContextPrototype(7, torch.zeros(2), torch.eye(2), 1, 0, 0, 1))
    ref = RecurrentEvidenceFilter(2, 0.1, config, memory, torch.device("cpu"))
    result = ref.evaluate_hypotheses(torch.zeros(3, 1, 2), torch.zeros(3, 1), active, active_prior=ContextPosterior(torch.zeros(2), torch.eye(2), 0.0, "active", 7))
    assert [candidate.source for candidate in result.candidates].count("active") == 1
    assert [candidate.source for candidate in result.candidates].count("new") == 1
    assert all(candidate.source != "memory" for candidate in result.candidates)
    selected = ref.resolve_context(active, result)
    assert selected.prototype_id is None or selected.prototype_id == 7


def test_ref_window_uses_pre_window_prior_not_current_active_state():
    config = replace(small_config().ref, active_prior_bonus=1.0)
    ref = RecurrentEvidenceFilter(1, 1.0, config, ContextMemory(4, 2.5, 0.25), torch.device("cpu"))
    current = ContextPosterior(torch.tensor([20.0]), torch.eye(1), 0.0, "active", 3)
    before = ContextPosterior(torch.tensor([0.0]), torch.eye(1), 0.0, "active", 3)
    result = ref.evaluate_hypotheses(torch.ones(2, 1, 1), torch.zeros(2, 1), current, active_prior=before)
    active_candidate = next(candidate for candidate in result.candidates if candidate.source == "active")
    assert abs(float(active_candidate.mean.item())) < 1.0


def test_unknown_radius_config_key_fails_closed():
    from pbcwm.methods.radius.config import radius_config_from_mapping

    try:
        radius_config_from_mapping({"method": {"ref": {"min_model_update_before_tracking": 2}}})
    except ValueError as error:
        assert "unknown" in str(error)
    else:
        raise AssertionError("a misspelled RADIUS config key was silently accepted")


def test_memory_touch_does_not_precision_collapse_or_increment_usage_each_step():
    memory = ContextMemory(2, 2.5, 0.25)
    memory.consolidate(ContextPosterior(torch.zeros(2), torch.eye(2), 0.0, "active"), 1)
    prototype = memory.prototypes[0]
    covariance_before = prototype.covariance.clone()
    usage_before = prototype.usage_count
    for step in range(2, 100):
        memory.touch(prototype.prototype_id, step)
    assert torch.allclose(prototype.covariance, covariance_before)
    assert prototype.usage_count == usage_before
    assert prototype.reuse_count == 0


def test_pec_is_not_ready_without_old_prototype_fisher():
    method = RadiusPbCWM(2, 1, small_config(), seed=0)
    assert not method.pec_ready
    assert method.pec.rank == 0


def test_recent_window_is_raw_and_normalizer_counts_obs_once():
    method = RadiusPbCWM(2, 1, small_config(), seed=0)
    for value in range(3):
        obs = np.array([value, value + 1], dtype=np.float32)
        method.observe(Transition(obs, np.zeros(1, dtype=np.float32), obs + 1.0, 0.0, False, False))
    assert method.state_normalizer.count == 3
    assert method.recent[0].obs.equal(torch.tensor([0.0, 1.0]))
    assert method.recent[0].next_obs.equal(torch.tensor([1.0, 2.0]))


def test_ref_does_not_track_before_model_ready():
    method = RadiusPbCWM(2, 1, small_config(), seed=0)
    for _ in range(16):
        obs = np.zeros(2, dtype=np.float32)
        method.observe(Transition(obs, np.zeros(1, dtype=np.float32), np.ones(2, dtype=np.float32), 0.0, False, False))
        method.update()
    assert not method.ref_initialized
    assert method.novelty.consecutive_trigger_count == 0


def test_context_l2_is_not_a_fake_atlas_gradient_term():
    method = RadiusPbCWM(2, 1, small_config(), seed=0)
    obs = torch.randn(4, 2)
    action = torch.randn(4, 1)
    target = torch.randn(4, 2)
    context = torch.randn(4, 2)
    first, _ = atlas_loss(method.atlas, obs, action, target, context, context_l2=0.0)
    second, _ = atlas_loss(method.atlas, obs, action, target, context, context_l2=1000.0)
    assert torch.allclose(first, second)


def test_pfpa_uses_raw_prediction_boundary():
    method = RadiusPbCWM(2, 1, small_config(), seed=1)
    original = method.atlas.predict_next
    method.atlas.predict_next = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("PFPA bypassed raw API"))
    candidates = [
        TrajectorySegment(torch.zeros(2, 2), torch.zeros(2, 1), torch.zeros(2, 2)),
        TrajectorySegment(torch.zeros(2, 2), torch.ones(2, 1), torch.zeros(2, 2)),
    ]
    ensemble = PreferenceRewardEnsemble(2, 1, ensemble_size=1, hidden_dims=(4,), batch_size=1, seed=2)
    try:
        scores = method._pfpa_context_reward_scores(candidates, ensemble)
        assert torch.isfinite(scores).all()
    finally:
        method.atlas.predict_next = original


def test_radius_constructor_and_cem_do_not_change_global_torch_rng():
    torch.manual_seed(1234)
    before = torch.get_rng_state().clone()
    RadiusPbCWM(2, 1, small_config(), seed=9)
    after_radius = torch.get_rng_state().clone()
    assert torch.equal(before, after_radius)
    PreferenceRewardEnsemble(2, 1, ensemble_size=2, hidden_dims=(4,), seed=11)
    after_reward = torch.get_rng_state().clone()
    assert torch.equal(after_radius, after_reward)

    class Dynamics:
        def predict(self, obs, action):
            return obs + action

    def reward(obs, action, next_obs):
        del obs, action
        return -next_obs.square().sum(dim=-1)

    planner = CEMPlanner(1, 8, 2, 1, action_low=[-1.0], action_high=[1.0], seed=10)
    planner.plan(np.zeros(1, dtype=np.float32), Dynamics(), reward, return_candidates=False)
    after_cem = torch.get_rng_state().clone()
    assert torch.equal(after_reward, after_cem)


def test_radius_checkpoint_uses_local_rng_and_context_samples_roundtrip():
    first = RadiusPbCWM(2, 1, small_config(), seed=21)
    state = first.state_dict()
    sample_a = first.sample_context(3)
    second = RadiusPbCWM(2, 1, small_config(), seed=99)
    torch.manual_seed(4321)
    before = torch.get_rng_state().clone()
    second.load_state_dict(state)
    after = torch.get_rng_state().clone()
    assert torch.equal(before, after)
    sample_b = second.sample_context(3)
    assert torch.allclose(sample_a, sample_b)
