"""Lifetime reservoir and world posterior q_W."""

from collections.abc import Sequence

import numpy as np

from pbcwm.core.types import Transition

from .posterior import BayesianDynamicsPosterior


class ReservoirTransitionBuffer:
    """Regime-agnostic bounded reservoir of real transitions."""

    def __init__(self, capacity: int, seed: int | None = None) -> None:
        if capacity <= 0:
            raise ValueError("world buffer capacity must be positive")
        self.capacity = int(capacity)
        self.storage: list[Transition] = []
        self.seen = 0
        self.rng = np.random.default_rng(seed)

    def add(self, transition: Transition) -> None:
        stored = Transition(
            np.asarray(transition.obs, dtype=np.float32).copy(),
            np.asarray(transition.action, dtype=np.float32).copy(),
            np.asarray(transition.next_obs, dtype=np.float32).copy(),
            0.0,
            bool(transition.terminated),
            bool(transition.truncated),
        )
        self.seen += 1
        if len(self.storage) < self.capacity:
            self.storage.append(stored)
            return
        index = int(self.rng.integers(self.seen))
        if index < self.capacity:
            self.storage[index] = stored

    def sample(self, count: int) -> list[Transition]:
        if not self.storage or count <= 0:
            return []
        indices = self.rng.choice(len(self.storage), size=min(count, len(self.storage)), replace=False)
        return [self.storage[int(index)] for index in indices]

    def state_dict(self) -> dict:
        return {"storage": list(self.storage), "seen": self.seen}

    def load_state_dict(self, state: dict) -> None:
        self.storage = list(state["storage"])
        self.seen = int(state["seen"])


class WorldPosterior:
    """The lifetime Bayesian posterior used only to initialize new regimes."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
        learning_rate: float = 1e-3,
        world_buffer_size: int = 10_000,
        min_logvar: float = -10.0,
        max_logvar: float = 2.0,
        gradient_clip_norm: float = 10.0,
        device: str = "cpu",
        seed: int | None = None,
    ) -> None:
        self.posterior = BayesianDynamicsPosterior(
            obs_dim,
            action_dim,
            hidden_dims=hidden_dims,
            learning_rate=learning_rate,
            min_logvar=min_logvar,
            max_logvar=max_logvar,
            gradient_clip_norm=gradient_clip_norm,
            device=device,
            seed=seed,
        )
        self.buffer = ReservoirTransitionBuffer(world_buffer_size, seed=seed)
        self.world_updates = 0
        self._generic_prior = self.posterior.snapshot()["posterior"]

    @property
    def size(self) -> int:
        return len(self.buffer.storage)

    def observe(self, transition: Transition) -> None:
        self.buffer.add(transition)

    def update(self, num_steps: int = 1, batch_size: int = 256) -> dict[str, float]:
        batch = self.buffer.sample(batch_size)
        if not batch or num_steps <= 0:
            return {"world_nll": 0.0, "world_kl": 0.0, "world_vb_loss": 0.0, "world_updates": 0.0}
        metrics = self.posterior.update(batch, prior=self._generic_prior, num_steps=num_steps)
        self.world_updates += int(num_steps)
        return {
            "world_nll": metrics["nll"],
            "world_kl": metrics["kl"],
            "world_vb_loss": metrics["vb_loss"],
            "world_updates": metrics["updates"],
        }

    def initialize_regime(self) -> tuple[BayesianDynamicsPosterior, dict]:
        """Return q_new <- q_W and a frozen q_W snapshot as its prior."""

        return self.posterior.clone(), self.posterior.snapshot()["posterior"]

    def state_dict(self) -> dict:
        return {
            "posterior": self.posterior.state_dict(),
            "buffer": self.buffer.state_dict(),
            "world_updates": self.world_updates,
            "generic_prior": self._generic_prior,
        }

    def load_state_dict(self, state: dict) -> None:
        self.posterior.load_state_dict(state["posterior"])
        self.buffer.load_state_dict(state["buffer"])
        self.world_updates = int(state["world_updates"])
        self._generic_prior = state["generic_prior"]
