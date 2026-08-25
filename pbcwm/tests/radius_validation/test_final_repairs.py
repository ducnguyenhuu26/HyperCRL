from dataclasses import replace

import numpy as np
import pytest
import torch

from pbcwm.core.types import Transition
from pbcwm.experiments.radius_validation.generate_fixed_stream import _visit_ids
from pbcwm.experiments.radius_validation.probe_metrics import evaluate_probe_bank
from pbcwm.experiments.radius_validation.probes import DynamicsProbe, DynamicsProbeBank, generate_probe_bank
from pbcwm.experiments.radius_validation.run_fixed_stream import build_stage_checkpoints
from pbcwm.experiments.radius_validation.variants import build_variant
from pbcwm.methods.radius import RadiusPbCWM
from pbcwm.methods.radius.config import RadiusConfig
from pbcwm.methods.radius.inference import RecurrentEvidenceFilter
from pbcwm.methods.radius.inference.evidence_filter import RoutingResult
from pbcwm.methods.radius.memory import ContextMemory
from pbcwm.methods.radius.memory.replay import RadiusReplayBuffer
from pbcwm.methods.radius.types import ContextPosterior, ContextPrototype, RadiusReplayItem
from pbcwm.tests.test_radius_core import small_config


def test_radius_replay_preserves_prototype_id_and_uses_current_mean():
    buffer = RadiusReplayBuffer(4, seed=0)
    buffer.add(RadiusReplayItem(torch.zeros(2), torch.zeros(1), torch.ones(2), torch.zeros(2), 7))
    assert buffer.storage[0].prototype_id == 7
    *_prefix, contexts = buffer.sample(1, 2, torch.device("cpu"), prototype_means={7: torch.tensor([3.0, 4.0])})
    assert torch.equal(contexts[0], torch.tensor([3.0, 4.0]))


def test_w0_uses_radius_coordinate_convention_without_method_components():
    learner = build_variant("W0", 2, 1, action_low=np.array([-2.0]), action_high=np.array([2.0]), seed=0)
    transition = Transition(np.array([1.0, 2.0], dtype=np.float32), np.array([1.0], dtype=np.float32), np.array([2.0, 4.0], dtype=np.float32), 0.0, False, False)
    learner.observe(transition)
    assert learner.state_normalizer.count == 1
    assert learner.delta_normalizer.count == 1
    assert np.array_equal(learner.replay.storage[0].obs, np.array([1.0, 2.0], dtype=np.float32))
    assert torch.equal(learner.action_scale, torch.tensor([2.0]))
    assert not hasattr(learner, "atlas")
    prediction = learner.predict(torch.ones(1, 2), torch.zeros(1, 1))
    assert prediction.shape == (1, 2) and torch.isfinite(prediction).all()


class _PerfectDynamics:
    def predict(self, obs, action):
        del action
        return obs + 1.0


class _BiasedDynamics:
    def predict(self, obs, action):
        del action
        return obs + 11.0


def _probe_bank(n=4, horizon=3):
    probes = []
    for index in range(n):
        initial = np.array([float(index)], dtype=np.float32)
        actions = np.zeros((horizon, 1), dtype=np.float32)
        true_obs = np.arange(index + 1, index + horizon + 1, dtype=np.float32).reshape(horizon, 1)
        probes.append(DynamicsProbe(initial, actions, true_obs))
    return DynamicsProbeBank("P0", tuple(probes))


def test_probe_bank_metrics_perfect_negative_and_single_probe():
    bank = _probe_bank()
    perfect = evaluate_probe_bank(_PerfectDynamics(), bank)
    assert perfect == {"r2_at_1": 1.0, "r2_at_H": 1.0, "nrmse_at_H": 0.0}
    biased = evaluate_probe_bank(_BiasedDynamics(), bank)
    assert biased["r2_at_1"] < 0.0
    assert biased["r2_at_H"] < 0.0
    assert evaluate_probe_bank(_PerfectDynamics(), DynamicsProbeBank("P0", bank.probes[:1])) == {"r2_at_1": None, "r2_at_H": None, "nrmse_at_H": None}


def test_stage_relative_checkpoints_include_recurrence_offsets_and_boundary_identity():
    checkpoints = build_stage_checkpoints(
        stage_length=10_000,
        schedule=("P0", "A", "B", "C", "B", "A"),
        fractions=(0.0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0),
        recurrence_offsets=(16, 32, 64, 128),
        total_steps=60_000,
    )
    b_return = [item for item in checkpoints[40000] if item["stage_index"] == 4]
    expected = {40000, 40016, 40032, 40064, 40128, 40200, 40500, 41000, 42000, 44000, 46000, 48000, 50000}
    observed = {item["global_step"] for items in checkpoints.values() for item in items if item["stage_index"] == 4}
    assert expected <= observed
    assert b_return[0]["visit_id"] == 1 and b_return[0]["stage_offset"] == 0


class _ActionSpace:
    low = np.array([-1.0], dtype=np.float32)
    high = np.array([1.0], dtype=np.float32)


class _TerminatingProbeEnv:
    action_space = _ActionSpace()

    def __init__(self):
        self.step_count = 0

    def reset(self, seed=None):
        del seed
        self.step_count = 0
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        del action
        self.step_count += 1
        return np.array([self.step_count], dtype=np.float32), 0.0, self.step_count >= 3, False, {}

    def close(self):
        pass


def test_probe_generation_rejects_episode_reset_crossing():
    with pytest.raises(RuntimeError, match="non-terminating probe"):
        generate_probe_bank("P0", _TerminatingProbeEnv, seed=0, n_probes=2, horizon=4, max_attempts_per_probe=2)


def test_visit_ids_follow_dynamics_identity():
    assert _visit_ids(["P0", "A", "B", "C", "B", "A"]) == [0, 0, 0, 0, 1, 1]


def test_rne_refit_does_not_call_sequential_update_for_newest_transition():
    config = replace(small_config(), rne=replace(small_config().rne, initialization_updates=1))
    method = RadiusPbCWM(2, 1, config, seed=0)
    for index in range(5):
        obs = np.array([index, index + 1], dtype=np.float32)
        method.observe(Transition(obs, np.zeros(1, dtype=np.float32), obs + 0.1, 0.0, False, False))
    method.ref_initialized = True
    original = method.ref.update_active

    def fail_if_called(*args, **kwargs):
        raise AssertionError("post-expansion refit sequentially re-ingested the newest row")

    method.ref.update_active = fail_if_called
    try:
        method._expand_atlas()
    finally:
        method.ref.update_active = original
    assert method.rank == 3


def test_active_route_restores_only_a_confident_stored_prototype_id():
    config = replace(small_config().ref, prototype_assignment_probability=0.8, active_prior_bonus=100.0)
    memory = ContextMemory(4, 2.5, 0.25)
    memory.prototypes.append(ContextPrototype(7, torch.zeros(2), torch.eye(2), 1, 0, 0, 1))
    ref = RecurrentEvidenceFilter(2, 1.0, config, memory, torch.device("cpu"))
    active = ContextPosterior(torch.zeros(2), torch.eye(2), 0.0, "active", None)
    prior = ContextPosterior(torch.zeros(2), torch.eye(2), 0.0, "active", 7)
    result = ref.evaluate_hypotheses(torch.zeros(4, 1, 2), torch.zeros(4, 1), active, active_prior=prior)
    assert ref.resolve_context(active, result).prototype_id == 7
    low = RoutingResult(
        [ContextPosterior(torch.zeros(2), torch.eye(2), 0.0, "active", 7), ContextPosterior(torch.zeros(2), torch.eye(2), 0.0, "new", None)],
        torch.tensor([0.5, 0.5]),
        0.5,
    )
    active.prototype_id = None
    assert ref.resolve_context(active, low).prototype_id is None
