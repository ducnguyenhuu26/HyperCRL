"""Naive online learner used as the first PB-CWM sanity baseline."""

from collections.abc import Sequence

import torch
from torch import nn

from pbcwm.core.buffer import ReplayBuffer
from pbcwm.core.device import move_batch, resolve_device
from pbcwm.core.dynamics import DynamicsLearner
from pbcwm.core.types import Transition
from pbcwm.models.mlp_dynamics import MLPDynamicsModel


class StaticDynamicsLearner(DynamicsLearner):
    """MLP plus uniform replay, with no explicit retention mechanism."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
        learning_rate: float = 1e-3,
        replay_capacity: int = 100_000,
        batch_size: int = 256,
        device: str | torch.device = "cpu",
        seed: int | None = None,
    ) -> None:
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.device = resolve_device(device)
        self.batch_size = int(batch_size)
        self.model = MLPDynamicsModel(obs_dim, action_dim, hidden_dims).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(replay_capacity, seed=seed)

    def observe(self, transition: Transition) -> None:
        self.replay_buffer.add(transition)

    def update(self, num_steps: int = 1) -> dict[str, float]:
        if num_steps < 0:
            raise ValueError("num_steps must be non-negative")
        if len(self.replay_buffer) < self.batch_size or num_steps == 0:
            return {"loss": 0.0, "updates": 0.0}
        self.model.train()
        losses: list[float] = []
        for _ in range(num_steps):
            batch = self.replay_buffer.sample(self.batch_size)
            obs = move_batch(batch.obs, self.device)
            action = move_batch(batch.action, self.device)
            target_delta = move_batch(batch.next_obs, self.device) - obs
            prediction = self.model(obs, action)
            loss = nn.functional.mse_loss(prediction, target_delta)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            losses.append(float(loss.detach().cpu()))
        return {"loss": sum(losses) / len(losses), "updates": float(len(losses))}

    def predict(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if obs.ndim != 2 or action.ndim != 2:
            raise ValueError("obs and action must have shape [batch, feature]")
        if obs.shape[0] != action.shape[0]:
            raise ValueError("obs and action must have the same batch size")
        self.model.eval()
        obs_device = obs.to(self.device)
        action_device = action.to(self.device)
        with torch.no_grad():
            return obs_device + self.model(obs_device, action_device)

    def state_dict(self) -> dict:
        return {"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        self.model.load_state_dict(state["model"])
        if "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])
