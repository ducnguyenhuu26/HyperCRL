"""Preprocessing-matched plain dynamics control for the RADIUS ladder."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import nn

from pbcwm.core.dynamics import DynamicsLearner
from pbcwm.core.device import move_batch, resolve_device
from pbcwm.core.normalization import RunningNormalizer
from pbcwm.core.types import Transition
from pbcwm.methods.radius.atlas.backbone import SharedDynamicsBackbone


class _RawReplay:
    def __init__(self, capacity: int, seed: int | None) -> None:
        self.capacity = int(capacity)
        self.storage: list[Transition] = []
        self.next_index = 0
        self.rng = np.random.default_rng(seed)

    def add(self, transition: Transition) -> None:
        copied = Transition(
            np.asarray(transition.obs, dtype=np.float32).copy(),
            np.asarray(transition.action, dtype=np.float32).copy(),
            np.asarray(transition.next_obs, dtype=np.float32).copy(),
            0.0,
            bool(transition.terminated),
            bool(transition.truncated),
        )
        if len(self.storage) < self.capacity:
            self.storage.append(copied)
        else:
            self.storage[self.next_index] = copied
        self.next_index = (self.next_index + 1) % self.capacity

    def sample(self, batch_size: int) -> list[Transition]:
        indices = self.rng.choice(len(self.storage), size=batch_size, replace=False)
        return [self.storage[int(index)] for index in indices]


class NormalizedPlainDynamicsLearner(DynamicsLearner):
    """Plain SiLU backbone using exactly RADIUS's lifetime convention.

    W0 stores raw transitions, updates each normalizer once per transition, and
    exposes raw-coordinate predictions. It intentionally owns no atlas,
    context, memory, novelty, elasticity, or preference machinery.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
        learning_rate: float = 3e-4,
        replay_capacity: int = 50_000,
        batch_size: int = 256,
        device: str | torch.device = "cpu",
        seed: int | None = None,
        action_scale: np.ndarray | torch.Tensor | None = None,
    ) -> None:
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.device = resolve_device(device)
        self.batch_size = int(batch_size)
        scale = torch.ones(self.action_dim, dtype=torch.float32) if action_scale is None else torch.as_tensor(action_scale, dtype=torch.float32).flatten()
        if scale.numel() != self.action_dim or (scale <= 0).any():
            raise ValueError("action_scale must be positive and match action_dim")
        self.action_scale = scale
        self.state_normalizer = RunningNormalizer(self.obs_dim)
        self.delta_normalizer = RunningNormalizer(self.obs_dim)
        local_seed = int(seed if seed is not None else np.random.default_rng().integers(0, 2**63 - 1))
        fork_devices = [] if self.device.type == "cpu" else [self.device.index or torch.cuda.current_device()]
        with torch.random.fork_rng(devices=fork_devices):
            torch.manual_seed(local_seed)
            self.model = SharedDynamicsBackbone(self.obs_dim, self.action_dim, hidden_size=int(hidden_dims[0])).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(learning_rate))
        self.replay = _RawReplay(replay_capacity, seed)
        self.model_updates_total = 0
        self.last_update = {"loss": 0.0, "updates": 0.0}

    def _normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        return action / self.action_scale.to(action.device, action.dtype)

    def observe(self, transition: Transition) -> None:
        obs = torch.as_tensor(transition.obs, dtype=torch.float32)
        next_obs = torch.as_tensor(transition.next_obs, dtype=torch.float32)
        self.state_normalizer.update(obs)
        self.delta_normalizer.update(next_obs - obs)
        self.replay.add(transition)

    def update(self, num_steps: int = 1) -> dict[str, float]:
        if num_steps < 0:
            raise ValueError("num_steps must be non-negative")
        if len(self.replay.storage) < self.batch_size or num_steps == 0:
            self.last_update = {"loss": 0.0, "updates": 0.0}
            return dict(self.last_update)
        losses: list[float] = []
        self.model.train()
        for _ in range(num_steps):
            batch = self.replay.sample(self.batch_size)
            obs = move_batch(torch.as_tensor(np.stack([item.obs for item in batch]), dtype=torch.float32), self.device)
            action = move_batch(torch.as_tensor(np.stack([item.action for item in batch]), dtype=torch.float32), self.device)
            next_obs = move_batch(torch.as_tensor(np.stack([item.next_obs for item in batch]), dtype=torch.float32), self.device)
            normalized_obs = self.state_normalizer.normalize(obs)
            normalized_action = self._normalize_action(action)
            target_delta = self.delta_normalizer.normalize(next_obs - obs)
            loss = nn.functional.mse_loss(self.model(normalized_obs, normalized_action), target_delta)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            losses.append(float(loss.detach().cpu()))
            self.model_updates_total += 1
        self.last_update = {"loss": sum(losses) / len(losses), "updates": float(len(losses))}
        return dict(self.last_update)

    def predict(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if obs.ndim != 2 or action.ndim != 2 or obs.shape[0] != action.shape[0]:
            raise ValueError("obs and action must have matching [batch, feature] shapes")
        self.model.eval()
        raw_obs = obs.to(self.device)
        raw_action = action.to(self.device)
        with torch.no_grad():
            normalized_delta = self.model(self.state_normalizer.normalize(raw_obs), self._normalize_action(raw_action))
            return raw_obs + self.delta_normalizer.denormalize(normalized_delta)

    def diagnostics(self) -> dict[str, float]:
        prediction_finite = all(torch.isfinite(parameter).all() for parameter in self.model.parameters())
        return {
            "w0/loss": float(self.last_update["loss"]),
            "w0/model_updates": float(self.model_updates_total),
            "w0/normalizer_state_count": float(self.state_normalizer.count),
            "w0/normalizer_delta_count": float(self.delta_normalizer.count),
            "w0/finite_parameters": float(prediction_finite),
        }

    def state_dict(self) -> dict:
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "state_normalizer": self.state_normalizer.state_dict(),
            "delta_normalizer": self.delta_normalizer.state_dict(),
            "model_updates_total": self.model_updates_total,
        }

    def load_state_dict(self, state: dict) -> None:
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.state_normalizer.load_state_dict(state["state_normalizer"])
        self.delta_normalizer.load_state_dict(state["delta_normalizer"])
        self.model_updates_total = int(state["model_updates_total"])
