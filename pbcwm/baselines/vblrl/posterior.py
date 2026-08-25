"""Bayesian regime posterior abstraction for reward-free dynamics."""

from collections.abc import Sequence
import copy

import numpy as np
import torch

from pbcwm.core.types import Transition

from .bnn import BayesianDynamicsNetwork


class BayesianDynamicsPosterior:
    """Variational posterior over a delta-state Bayesian neural network."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
        learning_rate: float = 1e-3,
        min_logvar: float = -10.0,
        max_logvar: float = 2.0,
        gradient_clip_norm: float = 10.0,
        device: str | torch.device = "cpu",
        seed: int | None = None,
    ) -> None:
        if learning_rate <= 0 or gradient_clip_norm <= 0:
            raise ValueError("learning_rate and gradient_clip_norm must be positive")
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.hidden_dims = tuple(int(size) for size in hidden_dims)
        self.learning_rate = float(learning_rate)
        self.min_logvar = float(min_logvar)
        self.max_logvar = float(max_logvar)
        self.gradient_clip_norm = float(gradient_clip_norm)
        self.device = torch.device(device)
        if seed is not None:
            torch.manual_seed(seed)
        self.network = BayesianDynamicsNetwork(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_dims=self.hidden_dims,
            min_logvar=self.min_logvar,
            max_logvar=self.max_logvar,
        ).to(self.device)
        self._optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate)
        self._updates = 0

    def predict_distribution(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        num_samples: int = 5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        obs_t, action_t = self._inputs(obs, action)
        means = []
        aleatoric = []
        with torch.no_grad():
            for _ in range(num_samples):
                mean, logvar = self.network(obs_t, action_t)
                means.append(mean)
                aleatoric.append(logvar.exp())
        mean_stack = torch.stack(means)
        variance = torch.stack(aleatoric).mean(dim=0) + mean_stack.var(dim=0, unbiased=False)
        return mean_stack.mean(dim=0), variance.clamp_min(1e-8)

    def sample_next(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        num_samples: int,
    ) -> torch.Tensor:
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        obs_t, action_t = self._inputs(obs, action)
        samples = []
        with torch.no_grad():
            for _ in range(num_samples):
                mean, logvar = self.network(obs_t, action_t)
                delta = mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)
                samples.append(obs_t + delta)
        return torch.stack(samples)

    def predict_next(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        obs_t, action_t = self._inputs(obs, action)
        with torch.no_grad():
            mean, _ = self.network(obs_t, action_t, deterministic=True)
        return obs_t + mean

    def log_predictive_likelihood(
        self,
        transitions: Sequence[Transition],
        num_samples: int = 5,
    ) -> float:
        if not transitions:
            return float("inf")
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        obs = torch.as_tensor(np.stack([item.obs for item in transitions]), dtype=torch.float32, device=self.device)
        action = torch.as_tensor(np.stack([item.action for item in transitions]), dtype=torch.float32, device=self.device)
        target = torch.as_tensor(
            np.stack([item.next_obs - item.obs for item in transitions]), dtype=torch.float32, device=self.device
        )
        log_probs = []
        with torch.no_grad():
            for _ in range(num_samples):
                mean, logvar = self.network(obs, action)
                log_prob = -0.5 * (
                    logvar + (target - mean).square() * torch.exp(-logvar) + np.log(2.0 * np.pi)
                ).sum(dim=-1)
                log_probs.append(log_prob)
        return float(torch.logsumexp(torch.stack(log_probs), dim=0).sub(np.log(num_samples)).mean().neg().cpu())

    def update(
        self,
        transitions: Sequence[Transition],
        prior: dict | None = None,
        num_steps: int = 1,
    ) -> dict[str, float]:
        if num_steps < 0:
            raise ValueError("num_steps must be non-negative")
        if not transitions or num_steps == 0:
            return {"nll": 0.0, "kl": 0.0, "vb_loss": 0.0, "updates": 0.0}
        observations = torch.as_tensor(np.stack([item.obs for item in transitions]), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(np.stack([item.action for item in transitions]), dtype=torch.float32, device=self.device)
        targets = torch.as_tensor(
            np.stack([item.next_obs - item.obs for item in transitions]), dtype=torch.float32, device=self.device
        )
        losses = []
        for _ in range(num_steps):
            mean, logvar = self.network(observations, actions)
            nll = 0.5 * (
                logvar + (targets - mean).square() * torch.exp(-logvar) + np.log(2.0 * np.pi)
            ).sum(dim=-1).mean()
            kl = self.network.kl_to_snapshot(prior)
            vb_loss = nll + kl / max(1, len(transitions))
            self._optimizer.zero_grad(set_to_none=True)
            vb_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.gradient_clip_norm)
            if not torch.isfinite(vb_loss):
                raise FloatingPointError("non-finite VBLRL variational loss")
            self._optimizer.step()
            losses.append((float(nll.detach().cpu()), float(kl.detach().cpu()), float(vb_loss.detach().cpu())))
            self._updates += 1
        return {
            "nll": float(np.mean([item[0] for item in losses])),
            "kl": float(np.mean([item[1] for item in losses])),
            "vb_loss": float(np.mean([item[2] for item in losses])),
            "updates": float(num_steps),
        }

    def clone(self) -> "BayesianDynamicsPosterior":
        clone = BayesianDynamicsPosterior(
            self.obs_dim,
            self.action_dim,
            hidden_dims=self.hidden_dims,
            learning_rate=self.learning_rate,
            min_logvar=self.min_logvar,
            max_logvar=self.max_logvar,
            gradient_clip_norm=self.gradient_clip_norm,
            device=self.device,
        )
        clone.network.load_state_dict(copy.deepcopy(self.network.state_dict()))
        clone._updates = self._updates
        return clone

    def snapshot(self) -> dict:
        return {
            "network": copy.deepcopy(self.network.state_dict()),
            "posterior": self.network.posterior_snapshot(),
            "updates": self._updates,
        }

    def state_dict(self) -> dict:
        return {
            "network": copy.deepcopy(self.network.state_dict()),
            "optimizer": copy.deepcopy(self._optimizer.state_dict()),
            "updates": self._updates,
        }

    def load_state_dict(self, state: dict) -> None:
        self.network.load_state_dict(state["network"])
        self._optimizer.load_state_dict(state["optimizer"])
        self._updates = int(state["updates"])

    @property
    def parameter_std_mean(self) -> float:
        sigmas = [
            layer.weight_sigma.mean()
            for layer in [*self.network.hidden_layers, self.network.output_layer]
        ]
        return float(torch.stack(sigmas).mean().detach().cpu())

    def _inputs(self, obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        action_t = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        if obs_t.ndim != 2 or action_t.ndim != 2 or obs_t.shape[0] != action_t.shape[0]:
            raise ValueError("obs and action must be batched tensors with matching rows")
        if obs_t.shape[-1] != self.obs_dim or action_t.shape[-1] != self.action_dim:
            raise ValueError("posterior input dimensions do not match")
        return obs_t, action_t
