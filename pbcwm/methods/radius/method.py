"""End-to-end RADIUS-PbCWM learner with reward-free dynamics learning."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from pbcwm.core.dynamics import DynamicsLearner
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
from .types import ContextPosterior, RadiusEvent, RadiusPrediction, RadiusReplayItem


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
    ) -> None:
        if config is None:
            config = RadiusConfig()
        elif isinstance(config, dict):
            config = radius_config_from_mapping(config)
        self.config = config
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.device = torch.device(device)
        self.rng = np.random.default_rng(seed)
        if seed is not None:
            torch.manual_seed(seed)

        self.atlas = FactorizedDynamicsAtlas(self.obs_dim, self.action_dim, config.atlas.hidden_size, config.atlas.initial_rank).to(self.device)
        self.optimizer = torch.optim.Adam(self.atlas.parameters(), lr=config.training.learning_rate, weight_decay=config.training.weight_decay)
        self.replay = RadiusReplayBuffer(config.training.replay_capacity, seed=seed)
        self.recent: deque[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = deque(maxlen=config.ref.context_window)
        self.memory = ContextMemory(config.memory.max_prototypes, config.memory.prototype_merge_mahalanobis, config.memory.fusion_weight)
        self.context = self._initial_context()
        self.ref = RecurrentEvidenceFilter(self.atlas.rank, config.ref.residual_sigma, config.ref, self.memory, self.device, disable_memory=config.ablations.disable_recurrent_memory, hard_routing=config.ablations.hard_context_routing)
        self.novelty = ResidualNoveltyMonitor(config.rne.residual_threshold, config.rne.new_hypothesis_threshold, config.rne.persistence_steps, config.rne.cooldown_steps)
        self.anchors = AnchorMemory(config.pec.anchors_per_prototype, seed=seed)
        self.pec = PredictiveElasticityController(self._shared_parameter_count(), config.pec)
        self.pfpa = PFPASelector(config.pfpa.frontier_fraction, config.pfpa.max_pair_action_similarity, seed=seed)
        self.shared_query_selector = DisagreementQuerySelector(pair_pool_size=256, seed=seed)
        self.global_step = 0
        self.stable_steps = 0
        self.last_update = {"loss": 0.0, "dynamics_loss": 0.0, "updates": 0.0}
        self.last_pfpa = {
            "mean_entropy": 0.0,
            "mean_frontier_score": 0.0,
            "frontier_pairs": 0.0,
            "coverage_pairs": 0.0,
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
        return len(self.replay) >= self.config.training.batch_size

    def observe(self, transition: Transition) -> None:
        """Consume only obs/action/next_obs; reward and metadata are not read."""

        self.global_step += 1
        obs = torch.as_tensor(np.asarray(transition.obs, dtype=np.float32), device=self.device)
        action = torch.as_tensor(np.asarray(transition.action, dtype=np.float32), device=self.device)
        next_obs = torch.as_tensor(np.asarray(transition.next_obs, dtype=np.float32), device=self.device)
        self.recent.append((obs.detach(), action.detach(), next_obs.detach()))

        if len(self.recent) >= self.config.ref.min_context_samples:
            window_obs = torch.stack([item[0] for item in self.recent])
            window_action = torch.stack([item[1] for item in self.recent])
            window_next = torch.stack([item[2] for item in self.recent])
            with torch.no_grad():
                base = self.atlas.backbone(window_obs, window_action)
                basis = self.atlas.basis_outputs(window_obs, window_action)
                residual = window_next - window_obs - base
                self.context = self.ref.infer(basis, residual, self.context)
                current_context = self.context.mean.unsqueeze(0).expand(len(self.recent), -1)
                residual_energy = float(explained_residual(window_next - window_obs, base, basis, current_context, self.config.ref.residual_sigma))
            novelty = self.novelty.update(residual_energy, self.context.new_hypothesis_probability, self.global_step)
            if self.context.prototype_id is not None:
                self.memory.touch(self.context.prototype_id, self.global_step)
        else:
            novelty = self.novelty.last_state

        self.replay.add(RadiusReplayItem(obs, action, next_obs, self.context.mean))
        if not self.config.ablations.disable_rne and not self.config.ablations.fixed_atlas_rank and novelty.should_expand and self.rank < self.config.atlas.max_rank:
            self._expand_atlas()
            self.novelty.mark_expanded(self.global_step)
        if not self.config.ablations.disable_recurrent_memory and len(self.recent) >= self.config.ref.min_context_samples:
            if novelty.standardized_residual < 1.0:
                self.stable_steps += 1
            else:
                self.stable_steps = 0
            if self.stable_steps >= self.config.memory.min_stable_steps and self.context.covariance.trace() <= self.config.memory.max_trace_for_consolidation:
                event, prototype_id = self.memory.consolidate(self.context, self.global_step)
                self.events.append(RadiusEvent(event, self.global_step, {"prototype_id": -1 if prototype_id is None else prototype_id}))
                self.stable_steps = 0

        if self.context.prototype_id is not None:
            self.anchors.add(self.context.prototype_id, obs.detach().cpu().numpy(), action.detach().cpu().numpy(), self.context.mean.detach().cpu().numpy())

    def _expand_atlas(self) -> None:
        if self.rank >= self.config.atlas.max_rank or len(self.recent) < 1:
            return
        observations = torch.stack([item[0] for item in self.recent])
        actions = torch.stack([item[1] for item in self.recent])
        next_observations = torch.stack([item[2] for item in self.recent])
        with torch.no_grad():
            base = self.atlas.backbone(observations, actions)
            basis = self.atlas.basis_outputs(observations, actions)
            current_context = self.context.mean.unsqueeze(0).expand(len(self.recent), -1)
            target = next_observations - observations - base - torch.einsum("bdr,br->bd", basis, current_context)
            flat_basis = basis.permute(0, 1, 2).reshape(-1, self.rank)
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

    def refresh_pec_fisher(self) -> None:
        """Refresh a Jacobian sketch from learner-observed anchors only."""

        if not self.config.pec.enabled:
            return
        columns: list[torch.Tensor] = []
        parameters = self._shared_parameters()
        for entries in self.anchors.storage.values():
            for obs_np, action_np, context_np in entries:
                obs = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)
                action = torch.as_tensor(action_np, dtype=torch.float32, device=self.device).unsqueeze(0)
                context = torch.as_tensor(context_np, dtype=torch.float32, device=self.device)
                context = torch.nn.functional.pad(context, (0, max(0, self.rank - context.numel())))[: self.rank].unsqueeze(0)
                prediction = self.atlas.predict_delta(obs, action, context).squeeze(0)
                for output_index in range(min(self.obs_dim, self.config.pec.fisher_sketch_rank - len(columns))):
                    gradients = torch.autograd.grad(prediction[output_index], parameters, retain_graph=True, allow_unused=True)
                    flat = torch.cat([gradient.reshape(-1) if gradient is not None else parameter.new_zeros(parameter.numel()) for gradient, parameter in zip(gradients, parameters)])
                    columns.append(flat.detach())
                if len(columns) >= self.config.pec.fisher_sketch_rank:
                    break
            if len(columns) >= self.config.pec.fisher_sketch_rank:
                break
        if columns:
            self.pec.refresh_from_sketch(torch.stack(columns, dim=1) / max(1.0, len(columns) ** 0.5))
        self.events.append(RadiusEvent("FISHER_REFRESHED", self.global_step, {"fisher_rank": self.pec.rank}))

    def update(self, num_steps: int = 1) -> dict[str, float]:
        if num_steps < 0:
            raise ValueError("num_steps must be non-negative")
        if len(self.replay) < self.config.training.batch_size or num_steps == 0:
            return dict(self.last_update)
        losses: list[float] = []
        dynamics_losses: list[float] = []
        for _ in range(num_steps):
            obs, action, next_obs, context = self.replay.sample(self.config.training.batch_size, self.rank, self.device)
            total, diagnostics = atlas_loss(self.atlas, obs, action, next_obs - obs, context, context_l2=self.config.atlas.context_l2, atom_orthogonality=self.config.atlas.atom_orthogonality)
            self.optimizer.zero_grad(set_to_none=True)
            total.backward()
            shared = self._shared_parameters()
            gradients = [parameter.grad.reshape(-1) for parameter in shared if parameter.grad is not None]
            if gradients:
                flat = torch.cat(gradients)
                if self.config.pec.enabled and not self.config.ablations.disable_pec and flat.numel() == self.pec.fisher.parameter_dim:
                    transformed = self.pec.transform_gradient(flat)
                    cursor = 0
                    for parameter in shared:
                        if parameter.grad is not None:
                            size = parameter.numel()
                            parameter.grad.copy_(transformed[cursor : cursor + size].reshape_as(parameter))
                            cursor += size
            torch.nn.utils.clip_grad_norm_(self.atlas.parameters(), 10.0)
            self.optimizer.step()
            losses.append(diagnostics["loss"])
            dynamics_losses.append(diagnostics["dynamics_loss"])
        self.last_update = {"loss": float(np.mean(losses)), "dynamics_loss": float(np.mean(dynamics_losses)), "updates": float(num_steps)}
        if self.config.pec.enabled and self.global_step > 0 and self.global_step % self.config.pec.fisher_refresh_interval == 0:
            self.refresh_pec_fisher()
        return {**self.last_update, **self.diagnostics()}

    def predict_with_context(self, obs: torch.Tensor, action: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        obs_device = obs.to(self.device)
        action_device = action.to(self.device)
        context_device = context.to(self.device)
        with torch.no_grad():
            result = self.atlas.predict_next(obs_device, action_device, context_device)
        return result.to(obs.device, dtype=obs.dtype)

    def predict(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        context = self.context.mean.to(self.device).unsqueeze(0).expand(obs.shape[0], -1)
        return self.predict_with_context(obs, action, context)

    def predict_detailed(self, obs: torch.Tensor, action: torch.Tensor) -> RadiusPrediction:
        context = self.context.mean.to(self.device).unsqueeze(0).expand(obs.shape[0], -1)
        with torch.no_grad():
            delta = self.atlas.predict_delta(obs.to(self.device), action.to(self.device), context)
        return RadiusPrediction(delta + obs.to(self.device), delta, self.context.mean.detach().clone(), self.context.covariance.detach().clone())

    def sample_context(self, num_samples: int) -> torch.Tensor:
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        covariance = 0.5 * (self.context.covariance + self.context.covariance.T) + self.config.ref.numerical_jitter * torch.eye(self.rank, device=self.device)
        chol = torch.linalg.cholesky(covariance)
        return self.context.mean + torch.randn(num_samples, self.rank, device=self.device) @ chol.T

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
            }
            return PFPASelection(pairs, 0, len(pairs), 0.0, 0.0)
        candidate_scores = self._pfpa_context_reward_scores(candidates, reward_model)
        actions = torch.stack([trajectory.actions.flatten() for trajectory in candidates])
        selection = self.pfpa.select_from_scores(candidate_scores, actions, num_queries)
        self.last_pfpa = {
            "mean_entropy": selection.mean_entropy,
            "mean_frontier_score": selection.mean_frontier_score,
            "frontier_pairs": float(selection.frontier_pairs),
            "coverage_pairs": float(selection.coverage_pairs),
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
                    state = self.atlas.predict_next(state, actions[:, step], context_batch)
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
        return {
            "radius/atlas_rank": float(self.rank),
            "radius/context_trace": float(self.context.covariance.trace()),
            "radius/context_norm": float(self.context.mean.norm()),
            "radius/context_source": self.context.source,
            "radius/context_max_hypothesis_prob": max(self.context.hypothesis_probabilities.values(), default=0.0),
            "radius/memory_num_prototypes": float(len(self.memory.prototypes)),
            "radius/novelty_residual": novelty.standardized_residual,
            "radius/novelty_p_new": novelty.new_hypothesis_probability,
            "radius/novelty_trigger_count": float(novelty.consecutive_trigger_count),
            "radius/rne_num_expansions": float(self.novelty.expansion_count),
            **self.pec.diagnostics(),
            "radius/pfpa_mean_entropy": self.last_pfpa["mean_entropy"],
            "radius/pfpa_mean_frontier_score": self.last_pfpa["mean_frontier_score"],
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
        return self.atlas.basis_outputs(obs.to(self.device), action.to(self.device))

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
            "pfpa": self.pfpa.state_dict(),
            "global_step": self.global_step,
            "stable_steps": self.stable_steps,
            "last_update": self.last_update,
            "last_pfpa": self.last_pfpa,
            "events": [event.__dict__ for event in self.events],
            "rng_state": self.rng.bit_generator.state,
            "torch_rng_state": torch.get_rng_state(),
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
        self.context = ContextPosterior(**state["context"])
        self.memory.load_state_dict(state["memory"])
        self.novelty.load_state_dict(state["novelty"])
        self.pec.load_state_dict(state["pec"])
        self.anchors.load_state_dict(state["anchors"])
        self.pfpa.load_state_dict(state["pfpa"])
        self.ref.rank = self.rank
        self.global_step = int(state["global_step"])
        self.stable_steps = int(state["stable_steps"])
        self.last_update = dict(state["last_update"])
        self.last_pfpa = dict(state.get("last_pfpa", self.last_pfpa))
        self.events = [RadiusEvent(**event) for event in state["events"]]
        self.rng = np.random.default_rng()
        self.rng.bit_generator.state = state["rng_state"]
        torch.set_rng_state(state["torch_rng_state"])
