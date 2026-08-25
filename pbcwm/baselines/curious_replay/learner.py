"""Single-model reward-free Curious Replay dynamics learner."""

from collections.abc import Sequence

import numpy as np
import torch
from torch import nn

from pbcwm.core.dynamics import DynamicsLearner
from pbcwm.core.types import Transition
from pbcwm.models.mlp_dynamics import MLPDynamicsModel

from .replay import CuriousReplayBuffer


class CuriousReplayDynamicsLearner(DynamicsLearner):
    """Train one delta MLP from a bounded lifetime curiosity-prioritized buffer."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
        capacity: int = 50_000,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        beta: float = 0.7,
        alpha: float = 0.6,
        epsilon: float = 1e-6,
        count_weight_c: float = 1.0,
        initial_priority: float = 1.0,
        gradient_clip_norm: float = 10.0,
        device: str | torch.device = "cpu",
        seed: int | None = None,
    ) -> None:
        if batch_size <= 0 or learning_rate <= 0 or gradient_clip_norm <= 0:
            raise ValueError("batch_size, learning_rate and gradient_clip_norm must be positive")
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.batch_size = int(batch_size)
        self.gradient_clip_norm = float(gradient_clip_norm)
        self.device = torch.device(device)
        if seed is not None:
            torch.manual_seed(seed)
        self.model = MLPDynamicsModel(self.obs_dim, self.action_dim, hidden_dims).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.replay_buffer = CuriousReplayBuffer(
            capacity=capacity,
            beta=beta,
            alpha=alpha,
            epsilon=epsilon,
            count_weight_c=count_weight_c,
            initial_priority=initial_priority,
            seed=seed,
        )
        self.global_step = 0
        self.update_count = 0
        self.last_observed_index: int | None = None
        self.last_sample_indices: list[int] = []
        self._last_sampled_losses: list[float] = []
        self._last_update = {"loss": 0.0, "updates": 0.0, "mean_sampled_model_loss": 0.0}

    @property
    def dynamics_ready(self) -> bool:
        return len(self.replay_buffer) >= self.batch_size

    def seed_observations(self) -> list[np.ndarray]:
        entries = list(self.replay_buffer)
        return [entry.transition.obs.copy() for entry in entries[-128:]]

    def observe(self, transition: Transition) -> None:
        self.global_step += 1
        self.last_observed_index = self.replay_buffer.add(transition)

    def update(self, num_steps: int = 1) -> dict[str, float]:
        if num_steps < 0:
            raise ValueError("num_steps must be non-negative")
        if len(self.replay_buffer) == 0 or num_steps == 0:
            return dict(self._last_update)

        self.model.train()
        losses: list[float] = []
        sampled_losses: list[float] = []
        sampled_indices: list[int] = []
        sample_size = min(self.batch_size, len(self.replay_buffer))
        for _ in range(num_steps):
            indices, entries = self.replay_buffer.sample(sample_size)
            obs = torch.as_tensor(
                np.stack([entry.transition.obs for entry in entries]),
                dtype=torch.float32,
                device=self.device,
            )
            action = torch.as_tensor(
                np.stack([entry.transition.action for entry in entries]),
                dtype=torch.float32,
                device=self.device,
            )
            target_delta = torch.as_tensor(
                np.stack(
                    [entry.transition.next_obs - entry.transition.obs for entry in entries]
                ),
                dtype=torch.float32,
                device=self.device,
            )
            prediction = self.model(obs, action)
            per_sample_loss = (prediction - target_delta).square().mean(dim=-1)
            batch_loss = per_sample_loss.mean()
            if not torch.isfinite(batch_loss):
                raise FloatingPointError("non-finite Curious Replay dynamics loss")
            self.optimizer.zero_grad(set_to_none=True)
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
            self.optimizer.step()

            with torch.no_grad():
                refreshed_loss = (self.model(obs, action) - target_delta).square().mean(dim=-1)
            refreshed_loss_np = refreshed_loss.detach().cpu().numpy().astype(np.float64)
            self.replay_buffer.update_priorities(indices, refreshed_loss_np)
            if not np.all(np.isfinite(refreshed_loss_np)):
                raise FloatingPointError("non-finite Curious Replay per-sample loss")
            losses.append(float(batch_loss.detach().cpu()))
            sampled_losses.extend(float(value) for value in refreshed_loss_np)
            sampled_indices.extend(int(index) for index in indices)
            self.update_count += 1

        self.last_sample_indices = sampled_indices
        self._last_sampled_losses = sampled_losses
        self._last_update = {
            "loss": float(np.mean(losses)),
            "updates": float(len(losses)),
            "mean_sampled_model_loss": float(np.mean(sampled_losses)),
        }
        return dict(self._last_update)

    def predict(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if obs.ndim != 2 or action.ndim != 2 or obs.shape[0] != action.shape[0]:
            raise ValueError("obs and action must have matching batched shapes")
        obs_device = obs.to(self.device)
        action_device = action.to(self.device)
        self.model.eval()
        with torch.no_grad():
            prediction = obs_device + self.model(obs_device, action_device)
        return prediction.to(device=obs.device, dtype=obs.dtype)

    def diagnostics(self) -> dict[str, float]:
        return {
            **self.replay_buffer.statistics(),
            "mean_sampled_model_loss": float(self._last_update["mean_sampled_model_loss"]),
            "dynamics_global_step": float(self.global_step),
            "dynamics_update_count": float(self.update_count),
            **self._last_update,
        }

    def state_dict(self) -> dict:
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "replay_buffer": self.replay_buffer.state_dict(),
            "global_step": self.global_step,
            "update_count": self.update_count,
            "last_update": dict(self._last_update),
        }

    def load_state_dict(self, state: dict) -> None:
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.replay_buffer.load_state_dict(state["replay_buffer"])
        self.global_step = int(state["global_step"])
        self.update_count = int(state["update_count"])
        self._last_update = dict(state["last_update"])
