from dataclasses import replace

import torch

from pbcwm.methods.radius.atlas import FactorizedDynamicsAtlas
from pbcwm.methods.radius.config import RadiusConfig
from pbcwm.methods.radius.inference.gaussian_context import gaussian_context_posterior, log_marginal_evidence, moment_match
from pbcwm.methods.radius.memory import ContextMemory
from pbcwm.methods.radius.novelty import orthogonal_residual
from pbcwm.methods.radius.novelty.residual_monitor import ResidualNoveltyMonitor
from pbcwm.methods.radius.memory.replay import RadiusReplayBuffer
from pbcwm.methods.radius.types import RadiusReplayItem
from pbcwm.methods.radius.types import ContextPosterior


def small_config() -> RadiusConfig:
    return replace(
        RadiusConfig(),
        atlas=replace(RadiusConfig().atlas, hidden_size=16, initial_rank=2, max_rank=4),
        ref=replace(RadiusConfig().ref, context_window=8, min_context_samples=4),
        memory=replace(RadiusConfig().memory, min_stable_steps=2),
        rne=replace(RadiusConfig().rne, persistence_steps=2, cooldown_steps=4, initialization_updates=2),
        pec=replace(RadiusConfig().pec, fisher_sketch_rank=4),
        training=replace(RadiusConfig().training, batch_size=4, replay_capacity=32),
    )


def test_fda_factorized_delta_matches_definition_and_rank_shapes():
    atlas = FactorizedDynamicsAtlas(3, 2, hidden_size=8, rank=2)
    obs = torch.randn(5, 3)
    action = torch.randn(5, 2)
    context = torch.randn(5, 2)
    base = atlas.backbone(obs, action)
    basis = atlas.basis_outputs(obs, action)
    expected = base + torch.einsum("bdr,br->bd", basis, context)
    assert torch.allclose(atlas.predict_delta(obs, action, context), expected)
    assert basis.shape == (5, 3, 2)
    atlas.append_atom()
    assert atlas.rank == 3


def test_gaussian_ref_matches_closed_form_and_evidence_is_finite():
    torch.manual_seed(0)
    n, state_dim, rank = 20, 3, 2
    basis = torch.randn(n, state_dim, rank)
    true_context = torch.tensor([0.7, -0.4])
    residual = torch.einsum("ndr,r->nd", basis, true_context)
    prior_mean = torch.zeros(rank)
    prior_covariance = torch.eye(rank) * 2.0
    mean, covariance = gaussian_context_posterior(basis, residual, prior_mean, prior_covariance, 0.1)
    precision = torch.linalg.solve(prior_covariance, torch.eye(rank)) + torch.einsum("ndr,ndk->rk", basis, basis) / 0.01
    expected_covariance = torch.linalg.solve(precision, torch.eye(rank))
    expected_mean = expected_covariance @ (torch.einsum("ndr,nd->r", basis, residual) / 0.01)
    assert torch.allclose(covariance, expected_covariance, atol=1e-5)
    assert torch.allclose(mean, expected_mean, atol=1e-5)
    assert torch.isfinite(torch.tensor(log_marginal_evidence(basis, residual, prior_mean, prior_covariance, 0.1)))


def test_gaussian_ref_rejects_invalid_noise_scale_and_shapes():
    basis = torch.zeros(2, 1, 2)
    residual = torch.zeros(2, 1)
    with torch.no_grad():
        try:
            gaussian_context_posterior(basis, residual, torch.zeros(2), torch.eye(2), 0.0)
        except ValueError:
            pass
        else:
            raise AssertionError("non-positive sigma must fail closed")
    try:
        gaussian_context_posterior(basis, residual, torch.zeros(3), torch.eye(3), 1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("inconsistent REF shapes must fail closed")


def test_prior_ordering_and_soft_mixture():
    basis = torch.eye(2).reshape(2, 1, 2).repeat(2, 1, 1)
    residual = torch.zeros(4, 1)
    _, informative = gaussian_context_posterior(basis, residual, torch.zeros(2), torch.eye(2) * 0.1, 1.0)
    _, diffuse = gaussian_context_posterior(basis, residual, torch.zeros(2), torch.eye(2) * 2.0, 1.0)
    assert torch.linalg.eigvalsh(diffuse - informative).min() >= -1e-6
    mean, covariance = moment_match(torch.tensor([[0.0, 0.0], [2.0, 0.0]]), torch.stack([torch.eye(2), torch.eye(2)]), torch.tensor([0.25, 0.75]))
    assert torch.allclose(mean, torch.tensor([1.5, 0.0]))
    assert torch.allclose(covariance, torch.tensor([[1.75, 0.0], [0.0, 1.0]]))


def test_orthogonal_residual_and_context_memory_merge():
    atoms = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    residual = orthogonal_residual(atoms, torch.tensor([1.0, 1.0, 2.0]), ridge=1e-8)
    assert torch.linalg.vector_norm(atoms.T @ residual) < 1e-5
    memory = ContextMemory(4, 2.5, 0.25)
    posterior = ContextPosterior(torch.zeros(2), torch.eye(2) * 0.1, 0.0, "active")
    assert memory.consolidate(posterior, 1)[0] == "CONTEXT_PROTOTYPE_CREATED"
    near = ContextPosterior(torch.ones(2) * 0.01, torch.eye(2) * 0.1, 0.0, "active")
    assert memory.consolidate(near, 2)[0] == "CONTEXT_PROTOTYPE_TOUCHED"
    assert len(memory.prototypes) == 1


def test_novelty_requires_residual_and_new_evidence_persistently():
    monitor = ResidualNoveltyMonitor(3.0, 0.6, persistence_steps=2, cooldown_steps=4)
    assert not monitor.update(5.0, 0.2, 1).should_expand
    assert not monitor.update(5.0, 0.9, 2).should_expand
    assert monitor.update(5.0, 0.9, 3).should_expand
    monitor.mark_expanded(3)
    assert not monitor.update(5.0, 0.9, 4).should_expand


def test_replay_right_pads_historical_context_after_rank_expansion():
    replay = RadiusReplayBuffer(4, seed=0)
    replay.add(RadiusReplayItem(torch.zeros(2), torch.zeros(1), torch.ones(2), torch.tensor([1.0, 2.0])))
    _, _, _, context = replay.sample(1, rank=3, device=torch.device("cpu"))
    assert torch.equal(context, torch.tensor([[1.0, 2.0, 0.0]]))
