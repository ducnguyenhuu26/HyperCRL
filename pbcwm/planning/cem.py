"""Batch Cross-Entropy Method planner with optional imagined rollouts."""

from dataclasses import dataclass

import numpy as np
import torch

from pbcwm.core.dynamics import DynamicsLearner
from pbcwm.preferences.types import TrajectorySegment
from pbcwm.rewards.base import RewardFunction


@dataclass
class CEMPlanResult:
    """Action and optional imagined candidates from one CEM planning call."""

    action: np.ndarray
    best_action_sequence: torch.Tensor
    candidate_trajectories: list[TrajectorySegment]
    elite_fraction: float = 0.0


class CEMPlanner:
    """Receding-horizon CEM over bounded continuous action sequences."""

    def __init__(
        self,
        horizon: int,
        population_size: int,
        elite_size: int,
        num_iterations: int,
        initial_std: float | np.ndarray = 1.0,
        action_low: np.ndarray | list[float] = (-2.0,),
        action_high: np.ndarray | list[float] = (2.0,),
        device: str | torch.device = "cpu",
        discount: float = 1.0,
        min_std: float = 1e-3,
        candidate_keep_per_iteration: int = 8,
        candidate_keep_final_elites: int = 8,
        dynamics_samples: int = 1,
        seed: int | None = None,
    ) -> None:
        if horizon <= 0 or population_size <= 0 or elite_size <= 0 or num_iterations <= 0:
            raise ValueError("planner sizes and iteration count must be positive")
        if elite_size > population_size:
            raise ValueError("elite_size cannot exceed population_size")
        if not 0 < discount <= 1:
            raise ValueError("discount must be in (0, 1]")
        if candidate_keep_per_iteration < 0 or candidate_keep_final_elites < 0:
            raise ValueError("candidate keep counts must be non-negative")
        if dynamics_samples <= 0:
            raise ValueError("dynamics_samples must be positive")

        self.horizon = int(horizon)
        self.population_size = int(population_size)
        self.elite_size = int(elite_size)
        self.num_iterations = int(num_iterations)
        self.device = torch.device(device)
        self.discount = float(discount)
        self.min_std = float(min_std)
        self.candidate_keep_per_iteration = int(candidate_keep_per_iteration)
        self.candidate_keep_final_elites = int(candidate_keep_final_elites)
        self.dynamics_samples = int(dynamics_samples)
        self.generator = torch.Generator(device=self.device)
        if seed is None:
            self.generator.seed()
        else:
            self.generator.manual_seed(int(seed))

        self.action_low = torch.as_tensor(action_low, dtype=torch.float32, device=self.device).flatten()
        self.action_high = torch.as_tensor(action_high, dtype=torch.float32, device=self.device).flatten()
        if self.action_low.shape != self.action_high.shape:
            raise ValueError("action bounds must have the same shape")
        if torch.any(self.action_low >= self.action_high):
            raise ValueError("each action_low must be smaller than action_high")
        self.action_dim = int(self.action_low.numel())

        std = torch.as_tensor(initial_std, dtype=torch.float32, device=self.device).flatten()
        if std.numel() == 1:
            std = std.repeat(self.action_dim)
        if std.numel() != self.action_dim or torch.any(std <= 0):
            raise ValueError("initial_std must be positive and match action_dim")
        self.initial_std = std

    def act(
        self,
        obs: np.ndarray | torch.Tensor,
        dynamics: DynamicsLearner,
        reward_fn: RewardFunction,
    ) -> np.ndarray:
        return self.plan(obs, dynamics, reward_fn, return_candidates=False).action

    def plan(
        self,
        obs: np.ndarray | torch.Tensor,
        dynamics: DynamicsLearner,
        reward_fn: RewardFunction,
        return_candidates: bool = True,
    ) -> CEMPlanResult:
        """Optimize an action distribution and optionally retain query candidates."""

        current_obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device).flatten()
        mean = torch.zeros(self.horizon, self.action_dim, device=self.device)
        std = self.initial_std.expand(self.horizon, self.action_dim).clone()
        candidates: list[TrajectorySegment] = []
        best_action_sequence = mean.clone()

        with torch.no_grad():
            for iteration in range(self.num_iterations):
                noise = torch.randn(
                    self.population_size,
                    self.horizon,
                    self.action_dim,
                    device=self.device,
                    generator=self.generator,
                )
                sequences = torch.clamp(
                    mean.unsqueeze(0) + std.unsqueeze(0) * noise,
                    self.action_low,
                    self.action_high,
                )
                returns, states, next_states = self._rollout(
                    sequences,
                    current_obs,
                    dynamics,
                    reward_fn,
                    collect_states=return_candidates,
                )
                elite_indices = torch.topk(returns, self.elite_size, largest=True).indices
                best_action_sequence = sequences[returns.argmax()].clone()
                if return_candidates:
                    if iteration == self.num_iterations - 1:
                        keep_indices = elite_indices[: self.candidate_keep_final_elites]
                    else:
                        keep_count = min(self.candidate_keep_per_iteration, self.population_size)
                        keep_indices = torch.randperm(self.population_size, device=self.device, generator=self.generator)[:keep_count]
                    if states is not None and next_states is not None:
                        for index in keep_indices.tolist():
                            candidates.append(
                                TrajectorySegment(
                                    obs=states[index].detach().clone(),
                                    actions=sequences[index].detach().clone(),
                                    next_obs=next_states[index].detach().clone(),
                                )
                            )
                elites = sequences[elite_indices]
                mean = elites.mean(dim=0)
                std = elites.std(dim=0, unbiased=False).clamp_min(self.min_std)

        return CEMPlanResult(
            action=mean[0].detach().cpu().numpy().astype(np.float32),
            best_action_sequence=best_action_sequence.detach().clone(),
            candidate_trajectories=candidates,
            elite_fraction=float(self.elite_size / self.population_size),
        )

    def state_dict(self) -> dict:
        return {
            "generator_state": self.generator.get_state(),
            "horizon": self.horizon,
            "population_size": self.population_size,
            "elite_size": self.elite_size,
        }

    def load_state_dict(self, state: dict) -> None:
        if int(state.get("horizon", self.horizon)) != self.horizon or int(state.get("population_size", self.population_size)) != self.population_size or int(state.get("elite_size", self.elite_size)) != self.elite_size:
            raise ValueError("CEM checkpoint shape/configuration mismatch")
        self.generator.set_state(state["generator_state"])

    def _rollout(
        self,
        sequences: torch.Tensor,
        initial_obs: torch.Tensor,
        dynamics: DynamicsLearner,
        reward_fn: RewardFunction,
        collect_states: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        population = sequences.shape[0]
        state = initial_obs.unsqueeze(0).expand(population, -1).clone()
        particle_states: torch.Tensor | None = None
        returns = torch.zeros(population, dtype=torch.float32, device=self.device)
        discount = 1.0
        state_rollouts: list[torch.Tensor] = []
        next_state_rollouts: list[torch.Tensor] = []
        for time_index in range(self.horizon):
            action = sequences[:, time_index, :]
            next_particles = self._sample_particles(dynamics, state, particle_states, action)
            current_particles = (
                state.unsqueeze(0).expand(self.dynamics_samples, -1, -1)
                if particle_states is None
                else particle_states
            )
            particle_actions = action.unsqueeze(0).expand(self.dynamics_samples, -1, -1)
            reward = reward_fn(
                current_particles.reshape(-1, current_particles.shape[-1]),
                particle_actions.reshape(-1, particle_actions.shape[-1]),
                next_particles.reshape(-1, next_particles.shape[-1]),
            ).reshape(self.dynamics_samples, population)
            returns += discount * reward.mean(dim=0)
            if collect_states:
                state_rollouts.append(current_particles.mean(dim=0).clone())
                next_state_rollouts.append(next_particles.mean(dim=0).clone())
            state = next_particles.mean(dim=0)
            particle_states = next_particles
            discount *= self.discount
        states = torch.stack(state_rollouts, dim=1) if collect_states else None
        next_states = torch.stack(next_state_rollouts, dim=1) if collect_states else None
        return returns, states, next_states

    def _sample_particles(
        self,
        dynamics: DynamicsLearner,
        state: torch.Tensor,
        particle_states: torch.Tensor | None,
        action: torch.Tensor,
    ) -> torch.Tensor:
        sampler = getattr(dynamics, "sample_next", None)
        if particle_states is None:
            if callable(sampler):
                samples = sampler(state, action, self.dynamics_samples)
                if samples.shape != (self.dynamics_samples, state.shape[0], state.shape[1]):
                    raise ValueError("sample_next must return [num_samples, batch, obs_dim]")
                return samples
            next_state = dynamics.predict(state, action)
            return next_state.unsqueeze(0).expand(self.dynamics_samples, -1, -1).clone()

        particle_count, population, obs_dim = particle_states.shape
        flat_state = particle_states.reshape(particle_count * population, obs_dim)
        flat_action = action.unsqueeze(0).expand(particle_count, -1, -1).reshape(
            particle_count * population, action.shape[-1]
        )
        if callable(sampler):
            samples = sampler(flat_state, flat_action, 1)
            if samples.shape != (1, particle_count * population, obs_dim):
                raise ValueError("sample_next must return [num_samples, batch, obs_dim]")
            return samples[0].reshape(particle_count, population, obs_dim)
        next_state = dynamics.predict(flat_state, flat_action)
        return next_state.reshape(particle_count, population, obs_dim)
