"""End-to-end RADIUS-PbCWM learner with reward-free dynamics learning."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from pbcwm.core.dynamics import DynamicsLearner
from pbcwm.core.normalization import RunningNormalizer
from pbcwm.core.types import Transition
from pbcwm.preferences.types import TrajectorySegment
from pbcwm.preferences.query import DisagreementQuerySelector

from .atlas import FactorizedDynamicsAtlas, atlas_loss
from .config import RadiusConfig, radius_config_from_mapping
from .elasticity import AnchorMemory, PredictiveElasticityController
from .inference import RecurrentEvidenceFilter
from .memory import ContextMemory
from .memory.replay import RadiusReplayBuffer
from .novelty import ResidualNoveltyMonitor, explained_residual, orthogonal_residual
from .preferences import PFPASelection, PFPASelector
from .types import ActivePriorSnapshot, ContextPosterior, RadiusEvent, RadiusPrediction, RadiusRecentItem, RadiusReplayItem


class RadiusPbCWM(DynamicsLearner):
    """RADIUS method contribution, separated from shared planner/reward code."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        config: RadiusConfig | dict[str, Any] | None = None,
        *,
        device: str | torch.device = "cpu",
        seed: int | None = None,
        action_scale: np.ndarray | torch.Tensor | None = None,
    ) -> None:
        if config is None:
            config = RadiusConfig()
        elif isinstance(config, dict):
            config = radius_config_from_mapping(config)
        self.config = config
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.device = torch.device(device)
        self.state_normalizer = RunningNormalizer(self.obs_dim)
        self.delta_normalizer = RunningNormalizer(self.obs_dim)
        scale = torch.ones(self.action_dim, dtype=torch.float32) if action_scale is None else torch.as_tensor(action_scale, dtype=torch.float32).flatten()
        if scale.numel() != self.action_dim or (scale <= 0).any():
            raise ValueError("action_scale must be positive and match action_dim")
        self.action_scale = scale
        self.rng = np.random.default_rng(seed)
        local_seed = int(seed if seed is not None else self.rng.integers(0, 2**63 - 1))
        self.torch_generator = torch.Generator(device=self.device).manual_seed(local_seed)
        fork_devices = [] if self.device.type == "cpu" else [self.device.index or torch.cuda.current_device()]
        with torch.random.fork_rng(devices=fork_devices):
            torch.manual_seed(local_seed)
            self.atlas = FactorizedDynamicsAtlas(self.obs_dim, self.action_dim, config.atlas.hidden_size, config.atlas.initial_rank).to(self.device)
        self.optimizer = torch.optim.Adam(self.atlas.parameters(), lr=config.training.learning_rate, weight_decay=config.training.weight_decay)
        self.replay = RadiusReplayBuffer(config.training.replay_capacity, seed=seed)
        self.recent: deque[RadiusRecentItem] = deque(maxlen=config.ref.context_window)
        self.active_prior_history: deque[ActivePriorSnapshot] = deque(maxlen=config.ref.context_window)
        self.memory = ContextMemory(config.memory.max_prototypes, config.memory.prototype_merge_mahalanobis, config.memory.fusion_weight)
        self.context = self._initial_context()
        self.ref = RecurrentEvidenceFilter(self.atlas.rank, config.ref.residual_sigma, config.ref, self.memory, self.device, disable_memory=config.ablations.disable_recurrent_memory, hard_routing=config.ablations.hard_context_routing)
        self.novelty = ResidualNoveltyMonitor(config.rne.residual_threshold, config.rne.new_hypothesis_threshold, config.rne.persistence_steps, config.rne.cooldown_steps)
        self.anchors = AnchorMemory(config.pec.anchors_per_prototype, seed=seed)
        self.fisher_rng = np.random.default_rng(None if seed is None else seed + 104729)
        self.pec = PredictiveElasticityController(self._shared_parameter_count(), config.pec)
        self.pfpa = PFPASelector(config.pfpa.frontier_fraction, config.pfpa.max_pair_action_similarity, seed=seed)
        self.shared_query_selector = DisagreementQuerySelector(pair_pool_size=256, seed=seed)
        self.global_step = 0
        self.stable_steps = 0
        self.model_updates_total = 0
        self.rne_blocked_not_ready = 0
        self.memory_blocked_not_ready = 0
        self.ref_initialized = False
        self.last_pec_ready = False
        self.last_update = {"loss": 0.0, "dynamics_loss": 0.0, "updates": 0.0}
        self.last_pfpa = {
            "mean_entropy": 0.0,
            "mean_frontier_score": 0.0,
            "frontier_pairs": 0.0,
            "coverage_pairs": 0.0,
            "elite_fraction": 0.0,
        }
        self.events: list[RadiusEvent] = []

    def _initial_context(self) -> ContextPosterior:
        rank = self.config.atlas.initial_rank
        return ContextPosterior(
            mean=torch.zeros(rank, device=self.device),
            covariance=torch.eye(rank, device=self.device) * self.config.ref.new_prior_std**2,
            log_evidence=0.0,
            source="initial",
        )

    def _shared_parameters(self) -> list[torch.nn.Parameter]:
        return list(self.atlas.backbone.parameters()) + list(self.atlas.atom_bank.parameters())

    def _shared_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self._shared_parameters())

    @property
    def rank(self) -> int:
        return self.atlas.rank

    @property
    def dynamics_ready(self) -> bool:
        return len(self.replay) >= self.config.training.batch_size and self.model_updates_total > 0

    @property
    def ref_ready(self) -> bool:
        return len(self.recent) >= self.config.ref.min_context_samples and self.model_updates_total >= self.config.ref.min_model_updates_before_tracking

    @property
    def rne_ready(self) -> bool:
        return self.ref_initialized and self.model_updates_total >= self.config.rne.min_model_updates_before_expansion and bool(torch.isfinite(self.context.mean).all())

    @property
    def memory_ready(self) -> bool:
        return self.ref_initialized and self.model_updates_total >= self.config.memory.min_model_updates_before_consolidation

    @property
    def pec_ready(self) -> bool:
        return bool(self.config.pec.enabled and not self.config.ablations.disable_pec and len(self.memory.prototypes) > 0 and self.pec.rank >= self.config.pec.min_fisher_rank)

    def _normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        return action / self.action_scale.to(action.device, action.dtype)

    def _observe_normalized(self, obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.state_normalizer.update(obs)
        delta = next_obs - obs
        self.delta_normalizer.update(delta)
        return self.state_normalizer.normalize(obs), self._normalize_action(action), self.delta_normalizer.normalize(delta)

    def _snapshot(self) -> ActivePriorSnapshot:
        return ActivePriorSnapshot(self.context.mean.detach().clone(), self.context.covariance.detach().clone(), self.context.prototype_id)

    def _snapshot_posterior(self, snapshot: ActivePriorSnapshot) -> ContextPosterior:
        return ContextPosterior(snapshot.mean, snapshot.covariance, 0.0, "active", snapshot.prototype_id)

    def _window_features(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.recent:
            raise RuntimeError("REF window is empty")
        observations = torch.stack([item.obs for item in self.recent]).to(self.device)
        actions = torch.stack([item.action for item in self.recent]).to(self.device)
        next_observations = torch.stack([item.next_obs for item in self.recent]).to(self.device)
        normalized_obs = self.state_normalizer.normalize(observations)
        normalized_actions = self._normalize_action(actions)
        normalized_deltas = self.delta_normalizer.normalize(next_observations - observations)
        base = self.atlas.backbone(normalized_obs, normalized_actions)
        basis = self.atlas.basis_outputs(normalized_obs, normalized_actions)
        if not all(torch.isfinite(value).all() for value in (normalized_obs, normalized_actions, normalized_deltas, base, basis)):
            raise FloatingPointError("non-finite RADIUS window representation")
        return normalized_obs, normalized_actions, normalized_deltas, base, basis

    def _route_window(self, *, initialize: bool = False) -> float:
        _obs, _actions, deltas, base, basis = self._window_features()
        residual = deltas - base
        prior = self._snapshot_posterior(self.active_prior_history[0]) if self.active_prior_history else self._initial_context()
        with torch.no_grad():
            if initialize:
                current = self._initial_context()
                for index in range(basis.shape[0]):
                    current = self.ref.update_active(basis[index:index + 1], residual[index:index + 1], current)
                self.context = current
                self.ref_initialized = True
            else:
                current_basis = basis[-1:]
                current_residual = residual[-1:]
                self.context = self.ref.update_active(current_basis, current_residual, self.context)
            routing = self.ref.evaluate_hypotheses(basis, residual, self.context, active_prior=prior)
            self.context = self.ref.resolve_context(self.context, routing)
            current_context = self.context.mean.unsqueeze(0).expand(len(self.recent), -1)
            residual_energy = float(explained_residual(deltas, base, basis, current_context, self.config.ref.residual_sigma))
        return residual_energy

    def observe(self, transition: Transition) -> None:
        """Consume only obs/action/next_obs; reward and metadata are not read."""

        self.global_step += 1
        obs = torch.as_tensor(np.asarray(transition.obs, dtype=np.float32), device=self.device)
        action = torch.as_tensor(np.asarray(transition.action, dtype=np.float32), device=self.device)
        next_obs = torch.as_tensor(np.asarray(transition.next_obs, dtype=np.float32), device=self.device)
        self._observe_normalized(obs, action, next_obs)
        self.active_prior_history.append(self._snapshot())
        self.recent.append(RadiusRecentItem(obs.detach().clone(), action.detach().clone(), next_obs.detach().clone()))

        if self.ref_ready:
            residual_energy = self._route_window(initialize=not self.ref_initialized)
            novelty = self.novelty.update(residual_energy, self.context.new_hypothesis_probability, self.global_step, allow_trigger=self.rne_ready)
        else:
            novelty = self.novelty.update(0.0, 0.0, self.global_step, allow_trigger=False)

        self.replay.add(RadiusReplayItem(obs, action, next_obs, self.context.mean, self.context.prototype_id))
        if not self.config.ablations.disable_rne and not self.config.ablations.fixed_atlas_rank and novelty.should_expand and self.rank < self.config.atlas.max_rank:
            if self.rne_ready:
                self._expand_atlas()
                self.novelty.mark_expanded(self.global_step)
            else:
                self.rne_blocked_not_ready += 1
        if not self.config.ablations.disable_recurrent_memory and self.memory_ready and self.ref_initialized:
            if novelty.standardized_residual < self.config.memory.stable_residual_threshold:
                self.stable_steps += 1
            else:
                self.stable_steps = 0
            mean_variance = float(self.context.covariance.trace()) / max(1, self.rank)
            if self.stable_steps >= self.config.memory.min_stable_steps and mean_variance <= self.config.memory.max_mean_variance_for_consolidation:
                result = self.memory.consolidate(self.context, self.global_step)
                self.context.prototype_id = result.prototype_id
                if result.evicted_prototype_id is not None:
                    self.anchors.remove(result.evicted_prototype_id)
                    self.events.append(RadiusEvent("ANCHORS_EVICTED", self.global_step, {"prototype_id": result.evicted_prototype_id}))
                self.events.append(RadiusEvent(result.event, self.global_step, {"prototype_id": -1 if result.prototype_id is None else result.prototype_id}))
                self.stable_steps = 0
        elif not self.config.ablations.disable_recurrent_memory and self.ref_initialized:
            self.memory_blocked_not_ready += 1

        if self.context.prototype_id is not None and any(item.prototype_id == self.context.prototype_id for item in self.memory.prototypes):
            self.anchors.add(self.context.prototype_id, obs.detach().cpu().numpy(), action.detach().cpu().numpy(), self.context.mean.detach().cpu().numpy())

    def _expand_atlas(self) -> None:
        if self.rank >= self.config.atlas.max_rank or len(self.recent) < 1:
            return
        observations, actions, target_deltas, base, basis = self._window_features()
        with torch.no_grad():
            current_context = self.context.mean.unsqueeze(0).expand(len(self.recent), -1)
            target = target_deltas - base - torch.einsum("bdr,br->bd", basis, current_context)
            flat_basis = basis.reshape(-1, self.rank)
            residual_target = orthogonal_residual(flat_basis, target.reshape(-1), self.config.rne.orthogonalization_ridge).reshape_as(target)
        new_head = self.atlas.append_atom()
        self.optimizer.add_param_group({"params": new_head.parameters()})
        new_parameters = list(new_head.parameters())
        for parameter in self.atlas.parameters():
            parameter.requires_grad = any(parameter is candidate for candidate in new_parameters)
        initializer = torch.optim.Adam(new_head.parameters(), lr=self.config.rne.initialization_lr)
        features = self.atlas.atom_bank.trunk(torch.cat((observations, actions), dim=-1)).detach()
        for _ in range(self.config.rne.initialization_updates):
            prediction = new_head(features)
            loss = (prediction - residual_target).square().mean()
            initializer.zero_grad(set_to_none=True)
            loss.backward()
            initializer.step()
        for parameter in self.atlas.parameters():
            parameter.requires_grad = True
        old_mean = self.context.mean
        old_covariance = self.context.covariance
        self.context.mean = torch.nn.functional.pad(old_mean, (0, 1))
        new_covariance = torch.zeros(self.rank, self.rank, device=self.device)
        new_covariance[:-1, :-1] = old_covariance
        new_covariance[-1, -1] = self.config.rne.new_context_variance
        self.context.covariance = new_covariance
        for prototype in self.memory.prototypes:
            prototype.mean = torch.nn.functional.pad(prototype.mean, (0, 1))
            covariance = torch.zeros(self.rank, self.rank, device=prototype.covariance.device)
            covariance[:-1, :-1] = prototype.covariance
            covariance[-1, -1] = self.config.rne.new_context_variance
            prototype.covariance = covariance
        self.active_prior_history = deque(
            [ActivePriorSnapshot(torch.nn.functional.pad(item.mean, (0, 1)), self._pad_covariance(item.covariance), item.prototype_id) for item in self.active_prior_history],
            maxlen=self.config.ref.context_window,
        )
        self.ref.rank = self.rank
        old_sketch = self.pec.fisher.sketch
        resized = torch.zeros(
            self._shared_parameter_count(), old_sketch.shape[1], device=self.device
        )
        resized[: old_sketch.shape[0]] = old_sketch.to(self.device)
        self.pec = PredictiveElasticityController(self._shared_parameter_count(), self.config.pec)
        self.pec.refresh_from_sketch(resized)
        self.events.append(RadiusEvent("ATLAS_EXPANDED", self.global_step, {"atlas_rank": self.rank}))
        self.refresh_pec_fisher()
        if self.recent and self.ref_initialized:
            self._refit_context_after_expansion()

    def _refit_context_after_expansion(self) -> float:
        """Refit the expanded context once without re-consuming the newest row."""

        _obs, _actions, deltas, base, basis = self._window_features()
        residual = deltas - base
        prior = self._snapshot_posterior(self.active_prior_history[0]) if self.active_prior_history else self._initial_context()
        with torch.no_grad():
            routing = self.ref.evaluate_hypotheses(basis, residual, self.context, active_prior=prior)
            self.context = self.ref.resolve_context(self.context, routing)
            current_context = self.context.mean.unsqueeze(0).expand(len(self.recent), -1)
            return float(explained_residual(deltas, base, basis, current_context, self.config.ref.residual_sigma))

    def _pad_covariance(self, covariance: torch.Tensor) -> torch.Tensor:
        padded = torch.zeros(self.rank, self.rank, device=covariance.device, dtype=covariance.dtype)
        old_rank = min(covariance.shape[0], self.rank - 1)
        padded[:old_rank, :old_rank] = covariance[:old_rank, :old_rank]
        padded[-1, -1] = self.config.rne.new_context_variance
        return padded

    def refresh_pec_fisher(self) -> None:
        """Refresh a balanced Jacobian sketch from learner-observed anchors only."""

        if not self.config.pec.enabled:
            return
        parameters = self._shared_parameters()
        prototype_ids = sorted(prototype_id for prototype_id, entries in self.anchors.storage.items() if entries and any(item.prototype_id == prototype_id for item in self.memory.prototypes))
        columns: list[torch.Tensor] = []
        if not prototype_ids:
            self.pec.refresh_from_sketch(torch.empty(self._shared_parameter_count(), 0, device=self.device))
            self.events.append(RadiusEvent("FISHER_REFRESHED", self.global_step, {"fisher_rank": self.pec.rank, "prototype_counts": {}}))
            return
        shuffled_ids = list(prototype_ids)
        self.fisher_rng.shuffle(shuffled_ids)
        prototype_counts = {prototype_id: 0 for prototype_id in prototype_ids}
        selected_ids = [shuffled_ids[index % len(shuffled_ids)] for index in range(self.config.pec.fisher_sketch_rank)]
        for prototype_id in selected_ids:
            entries = self.anchors.storage[prototype_id]
            prototype_counts[prototype_id] += 1
            obs_np, action_np, context_np = entries[int(self.fisher_rng.integers(len(entries)))]
            obs_raw = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)
            action_raw = torch.as_tensor(action_np, dtype=torch.float32, device=self.device).unsqueeze(0)
            obs = self.state_normalizer.normalize(obs_raw)
            action = self._normalize_action(action_raw)
            context = torch.as_tensor(context_np, dtype=torch.float32, device=self.device)
            context = torch.nn.functional.pad(context, (0, max(0, self.rank - context.numel())))[: self.rank].unsqueeze(0)
            projection = torch.as_tensor(self.fisher_rng.normal(size=self.obs_dim), dtype=torch.float32, device=self.device)
            projection = projection / projection.norm().clamp_min(1e-12)
            prediction = self.atlas.predict_delta(obs, action, context).squeeze(0)
            scalar = prediction @ projection
            gradients = torch.autograd.grad(scalar, parameters, retain_graph=False, allow_unused=True)
            flat = torch.cat([gradient.reshape(-1) if gradient is not None else parameter.new_zeros(parameter.numel()) for gradient, parameter in zip(gradients, parameters)])
            columns.append(flat.detach() / float(self.config.pec.predictive_sigma))
        if columns:
            self.pec.refresh_from_sketch(torch.stack(columns, dim=1) / max(1.0, len(columns) ** 0.5))
        self.events.append(RadiusEvent("FISHER_REFRESHED", self.global_step, {"fisher_rank": self.pec.rank, "prototype_counts": prototype_counts}))

    def update(self, num_steps: int = 1) -> dict[str, float]:
        if num_steps < 0:
            raise ValueError("num_steps must be non-negative")
        if len(self.replay) < self.config.training.batch_size or num_steps == 0:
            return dict(self.last_update)
        losses: list[float] = []
        dynamics_losses: list[float] = []
        atom_gram_losses: list[float] = []
        for _ in range(num_steps):
            prototype_means = None
            if self.config.training.replay_context_mode == "prototype_if_available":
                prototype_means = {prototype.prototype_id: prototype.mean for prototype in self.memory.prototypes}
            raw_obs, raw_action, raw_next_obs, context = self.replay.sample(self.config.training.batch_size, self.rank, self.device, prototype_means)
            obs = self.state_normalizer.normalize(raw_obs)
            action = self._normalize_action(raw_action)
            target_delta = self.delta_normalizer.normalize(raw_next_obs - raw_obs)
            total, diagnostics = atlas_loss(self.atlas, obs, action, target_delta, context, context_l2=self.config.atlas.context_l2, atom_orthogonality=self.config.atlas.atom_orthogonality)
            self.optimizer.zero_grad(set_to_none=True)
            total.backward()
            shared = self._shared_parameters()
            torch.nn.utils.clip_grad_norm_(self.atlas.parameters(), 10.0)
            flat = torch.cat([parameter.grad.reshape(-1) if parameter.grad is not None else parameter.new_zeros(parameter.numel()) for parameter in shared])
            use_direct_pec = self.pec_ready and self.config.pec.optimizer_integration == "direct_parameter_step"
            if use_direct_pec:
                delta = self.pec.direct_step(flat)
                cursor = 0
                with torch.no_grad():
                    for parameter in shared:
                        size = parameter.numel()
                        parameter.add_(delta[cursor:cursor + size].reshape_as(parameter))
                        cursor += size
            else:
                if self.pec_ready and flat.numel() == self.pec.fisher.parameter_dim and self.config.pec.optimizer_integration == "transformed_gradient":
                    transformed = self.pec.transform_gradient(flat)
                    cursor = 0
                    for parameter in shared:
                        size = parameter.numel()
                        if parameter.grad is not None:
                            parameter.grad.copy_(transformed[cursor:cursor + size].reshape_as(parameter))
                        cursor += size
                self.optimizer.step()
            if not all(torch.isfinite(parameter).all() for parameter in self.atlas.parameters()):
                raise FloatingPointError("non-finite RADIUS parameters after update")
            self.model_updates_total += 1
            losses.append(diagnostics["loss"])
            dynamics_losses.append(diagnostics["dynamics_loss"])
            atom_gram_losses.append(diagnostics.get("atom_gram_loss", 0.0))
        self.last_update = {"loss": float(np.mean(losses)), "dynamics_loss": float(np.mean(dynamics_losses)), "atom_gram_loss": float(np.mean(atom_gram_losses)), "updates": float(num_steps)}
        if self.config.pec.enabled and self.global_step > 0 and self.global_step % self.config.pec.fisher_refresh_interval == 0:
            self.refresh_pec_fisher()
        if self.pec_ready != self.last_pec_ready:
            self.events.append(RadiusEvent("PEC_READY" if self.pec_ready else "PEC_NOT_READY", self.global_step, {"fisher_rank": self.pec.rank}))
            self.last_pec_ready = self.pec_ready
        return {**self.last_update, **self.diagnostics()}

    def predict_with_context(self, obs: torch.Tensor, action: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        obs_device = obs.to(self.device)
        action_device = self._normalize_action(action.to(self.device))
        context_device = context.to(self.device)
        with torch.no_grad():
            normalized_obs = self.state_normalizer.normalize(obs_device)
            normalized_delta = self.atlas.predict_delta(normalized_obs, action_device, context_device)
            raw_delta = self.delta_normalizer.denormalize(normalized_delta)
            result = obs_device + raw_delta
        if not torch.isfinite(result).all():
            raise FloatingPointError("non-finite raw RADIUS prediction")
        return result.to(obs.device, dtype=obs.dtype)

    def predict(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        context = self.context.mean.to(self.device).unsqueeze(0).expand(obs.shape[0], -1)
        return self.predict_with_context(obs, action, context)

    def predict_detailed(self, obs: torch.Tensor, action: torch.Tensor) -> RadiusPrediction:
        context = self.context.mean.to(self.device).unsqueeze(0).expand(obs.shape[0], -1)
        with torch.no_grad():
            normalized_delta = self.atlas.predict_delta(self.state_normalizer.normalize(obs.to(self.device)), self._normalize_action(action.to(self.device)), context)
            delta = self.delta_normalizer.denormalize(normalized_delta)
        return RadiusPrediction(delta + obs.to(self.device), delta, self.context.mean.detach().clone(), self.context.covariance.detach().clone())

    def sample_context(self, num_samples: int) -> torch.Tensor:
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        covariance = 0.5 * (self.context.covariance + self.context.covariance.T) + self.config.ref.numerical_jitter * torch.eye(self.rank, device=self.device)
        chol = torch.linalg.cholesky(covariance)
        samples = torch.randn(num_samples, self.rank, device=self.device, generator=self.torch_generator)
        return self.context.mean + samples @ chol.T

    def select_preference_queries(self, obs, planner, reward_model, num_queries: int):
        candidates = getattr(planner, "candidate_trajectories", None)
        if candidates is None and hasattr(planner, "plan"):
            planner_reward = reward_model if callable(reward_model) else lambda state, action, next_state: reward_model.reward(state, action)
            plan_result = planner.plan(obs, self, planner_reward, return_candidates=True)
            candidates = plan_result.candidate_trajectories
        if candidates is None:
            raise ValueError("shared planner candidate trajectories are required")
        if not self.config.pfpa.enabled or self.config.ablations.disable_pfpa:
            pairs = self.shared_query_selector.select(candidates, reward_model, num_queries)
            self.last_pfpa = {
                "mean_entropy": 0.0,
                "mean_frontier_score": 0.0,
                "frontier_pairs": 0.0,
                "coverage_pairs": float(len(pairs)),
                "elite_fraction": 0.0,
            }
            return PFPASelection(pairs, 0, len(pairs), 0.0, 0.0)
        candidate_scores = self._pfpa_context_reward_scores(candidates, reward_model)
        actions = torch.stack([trajectory.actions.flatten() for trajectory in candidates])
        elite_fraction = float(getattr(planner, "elite_fraction", 0.1))
        if hasattr(planner, "elite_size") and hasattr(planner, "population_size"):
            elite_fraction = float(planner.elite_size / planner.population_size)
        selection = self.pfpa.select_from_scores(candidate_scores, actions, num_queries, elite_fraction=elite_fraction)
        self.last_pfpa = {
            "mean_entropy": selection.mean_entropy,
            "mean_frontier_score": selection.mean_frontier_score,
            "frontier_pairs": float(selection.frontier_pairs),
            "coverage_pairs": float(selection.coverage_pairs),
            "elite_fraction": selection.elite_fraction,
        }
        self.events.append(RadiusEvent("PFPA_QUERY_ROUND", self.global_step, {"num_queries": len(selection.pairs), "frontier_pairs": selection.frontier_pairs, "coverage_pairs": selection.coverage_pairs}))
        return selection

    def _pfpa_context_reward_scores(self, candidates: Sequence[TrajectorySegment], reward_model: Any) -> torch.Tensor:
        """Roll the shared candidate actions under sampled FDA contexts."""

        models = getattr(reward_model, "models", None)
        if models is None:
            raise TypeError("PFPA requires the shared reward ensemble")
        contexts = self.sample_context(self.config.pfpa.context_samples)
        scores: list[torch.Tensor] = []
        with torch.no_grad():
            horizon = candidates[0].actions.shape[0]
            if any(trajectory.actions.shape[0] != horizon for trajectory in candidates):
                raise ValueError("PFPA candidates must have a common horizon")
            actions = torch.stack([trajectory.actions for trajectory in candidates]).to(self.device)
            initial_states = torch.stack([trajectory.obs[0] for trajectory in candidates]).to(self.device)
            for context in contexts:
                state = initial_states
                states: list[torch.Tensor] = []
                context_batch = context.unsqueeze(0).expand(len(candidates), -1)
                for step in range(horizon):
                    states.append(state)
                    state = self.predict_with_context(state, actions[:, step], context_batch)
                candidate_states = torch.stack(states, dim=1)
                for model in models:
                    flat_scores = model(
                        candidate_states.reshape(-1, self.obs_dim),
                        actions.reshape(-1, self.action_dim),
                    ).reshape(len(candidates), horizon)
                    scores.append(flat_scores.sum(dim=1))
        return torch.stack(scores)

    def diagnostics(self) -> dict[str, float | str]:
        novelty = self.novelty.last_state
        atom_min_rms = 0.0
        atom_max_rms = 0.0
        if self.recent:
            with torch.no_grad():
                _obs, _actions, _deltas, _base, basis = self._window_features()
                rms = torch.sqrt(basis.square().mean(dim=(0, 1)).clamp_min(0.0))
                atom_min_rms = float(rms.min())
                atom_max_rms = float(rms.max())
        return {
            "radius/atlas_rank": float(self.rank),
            "radius/context_trace": float(self.context.covariance.trace()),
            "radius/context_mean_variance": float(self.context.covariance.trace()) / max(1, self.rank),
            "radius/context_norm": float(self.context.mean.norm()),
            "radius/context_source": self.context.source,
            "radius/context_max_hypothesis_prob": max(self.context.hypothesis_probabilities.values(), default=0.0),
            "radius/memory_num_prototypes": float(len(self.memory.prototypes)),
            "radius/novelty_residual": novelty.standardized_residual,
            "radius/novelty_p_new": novelty.new_hypothesis_probability,
            "radius/novelty_trigger_count": float(novelty.consecutive_trigger_count),
            "radius/rne_num_expansions": float(self.novelty.expansion_count),
            "radius/rne_blocked_not_ready": float(self.rne_blocked_not_ready),
            "radius/model_updates_total": float(self.model_updates_total),
            "radius/dynamics_ready": float(self.dynamics_ready),
            "radius/normalizer_state_count": float(self.state_normalizer.count),
            "radius/normalizer_delta_count": float(self.delta_normalizer.count),
            "radius/context_new_probability": self.context.new_hypothesis_probability,
            "radius/context_active_probability": self.context.hypothesis_probabilities.get("active", 0.0),
            "radius/context_selected_prototype": float(self.context.prototype_id if self.context.prototype_id is not None else -1),
            "radius/ref_ready": float(self.ref_ready),
            "radius/ref_initialized": float(self.ref_initialized),
            "radius/context_assignment_confident": float(self.context.prototype_id is not None),
            "radius/context_route_entropy": float(-sum(probability * np.log(max(probability, 1e-12)) for probability in self.context.hypothesis_probabilities.values())),
            "radius/memory_reuse_count": float(sum(prototype.reuse_count for prototype in self.memory.prototypes)),
            "radius/memory_orphan_ids": float(sum(1 for prototype_id in self.anchors.storage if not any(item.prototype_id == prototype_id for item in self.memory.prototypes))),
            "radius/memory_consolidation_blocked_not_ready": float(self.memory_blocked_not_ready),
            "radius/rne_ready": float(self.rne_ready),
            "radius/pec_ready": float(self.pec_ready),
            "radius/pec_anchor_prototypes": float(sum(1 for prototype_id, entries in self.anchors.storage.items() if entries and any(item.prototype_id == prototype_id for item in self.memory.prototypes))),
            "radius/atom_gram_loss": float(self.last_update.get("atom_gram_loss", 0.0)),
            "radius/atom_min_rms": atom_min_rms,
            "radius/atom_max_rms": atom_max_rms,
            **self.pec.diagnostics(),
            "radius/pfpa_mean_entropy": self.last_pfpa["mean_entropy"],
            "radius/pfpa_mean_frontier_score": self.last_pfpa["mean_frontier_score"],
            "radius/pfpa_elite_fraction": self.last_pfpa.get("elite_fraction", 0.0),
        }

    def get_context_mean(self) -> torch.Tensor:
        return self.context.mean.detach().clone()

    def get_context_covariance(self) -> torch.Tensor:
        return self.context.covariance.detach().clone()

    def get_context_prototypes(self):
        return list(self.memory.prototypes)

    def get_atlas_rank(self) -> int:
        return self.rank

    def get_atom_outputs(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.atlas.basis_outputs(self.state_normalizer.normalize(obs.to(self.device)), self._normalize_action(action.to(self.device)))

    def get_pec_spectrum_summary(self) -> dict[str, float]:
        sketch = self.pec.fisher.sketch
        if sketch.numel() == 0:
            return {"max": 0.0, "mean": 0.0, "median": 0.0, "effective_rank": 0.0}
        values = torch.linalg.svdvals(sketch)
        return {"max": float(values.max()), "mean": float(values.mean()), "median": float(values.median()), "effective_rank": float((values > 1e-6).sum())}

    def state_dict(self) -> dict:
        return {
            "atlas": self.atlas.state_dict(),
            "atlas_rank": self.rank,
            "optimizer": self.optimizer.state_dict(),
            "replay": self.replay.state_dict(),
            "recent": list(self.recent),
            "context": self.context.__dict__,
            "memory": self.memory.state_dict(),
            "novelty": self.novelty.state_dict(),
            "pec": self.pec.state_dict(),
            "anchors": self.anchors.state_dict(),
            "fisher_rng_state": self.fisher_rng.bit_generator.state,
            "state_normalizer": self.state_normalizer.state_dict(),
            "delta_normalizer": self.delta_normalizer.state_dict(),
            "action_scale": self.action_scale.clone(),
            "pfpa": self.pfpa.state_dict(),
            "shared_query_selector": self.shared_query_selector.state_dict(),
            "global_step": self.global_step,
            "stable_steps": self.stable_steps,
            "model_updates_total": self.model_updates_total,
            "rne_blocked_not_ready": self.rne_blocked_not_ready,
            "last_update": self.last_update,
            "last_pfpa": self.last_pfpa,
            "events": [event.__dict__ for event in self.events],
            "rng_state": self.rng.bit_generator.state,
            "torch_generator_state": self.torch_generator.get_state(),
            "active_prior_history": list(self.active_prior_history),
            "ref_initialized": self.ref_initialized,
            "memory_blocked_not_ready": self.memory_blocked_not_ready,
            "last_pec_ready": self.last_pec_ready,
        }

    def load_state_dict(self, state: dict) -> None:
        target_rank = int(state["atlas_rank"])
        if target_rank < self.config.atlas.initial_rank or target_rank > self.config.atlas.max_rank:
            raise ValueError("checkpoint atlas rank is outside configured bounds")
        while self.rank < target_rank:
            new_head = self.atlas.append_atom()
            self.optimizer.add_param_group({"params": new_head.parameters()})
        if self.rank != target_rank:
            raise ValueError("checkpoint rank is below initial atlas rank")
        self.atlas.load_state_dict(state["atlas"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.replay.load_state_dict(state["replay"])
        self.recent = deque(state["recent"], maxlen=self.config.ref.context_window)
        self.active_prior_history = deque(state.get("active_prior_history", []), maxlen=self.config.ref.context_window)
        self.context = ContextPosterior(**state["context"])
        self.memory.load_state_dict(state["memory"])
        self.novelty.load_state_dict(state["novelty"])
        self.pec.load_state_dict(state["pec"])
        self.anchors.load_state_dict(state["anchors"])
        self.fisher_rng = np.random.default_rng()
        self.fisher_rng.bit_generator.state = state.get("fisher_rng_state", self.fisher_rng.bit_generator.state)
        self.state_normalizer.load_state_dict(state["state_normalizer"])
        self.delta_normalizer.load_state_dict(state["delta_normalizer"])
        loaded_action_scale = torch.as_tensor(state["action_scale"], dtype=torch.float32)
        if loaded_action_scale.shape != self.action_scale.shape:
            raise ValueError("checkpoint action scale mismatch")
        self.action_scale = loaded_action_scale
        self.pfpa.load_state_dict(state["pfpa"])
        if "shared_query_selector" in state:
            self.shared_query_selector.load_state_dict(state["shared_query_selector"])
        self.ref.rank = self.rank
        self.global_step = int(state["global_step"])
        self.stable_steps = int(state["stable_steps"])
        self.model_updates_total = int(state.get("model_updates_total", 0))
        self.rne_blocked_not_ready = int(state.get("rne_blocked_not_ready", 0))
        self.memory_blocked_not_ready = int(state.get("memory_blocked_not_ready", 0))
        self.ref_initialized = bool(state.get("ref_initialized", False))
        self.last_pec_ready = bool(state.get("last_pec_ready", False))
        self.last_update = dict(state["last_update"])
        self.last_pfpa = dict(state.get("last_pfpa", self.last_pfpa))
        self.events = [RadiusEvent(**event) for event in state["events"]]
        self.rng = np.random.default_rng()
        self.rng.bit_generator.state = state["rng_state"]
        self.torch_generator.set_state(state.get("torch_generator_state", self.torch_generator.get_state()))
