"""HyperCRL-Adapt dynamics learner with a minimal residual router."""

from collections import deque
from collections.abc import Sequence

import numpy as np
import torch
from torch import nn

from pbcwm.core.dynamics import DynamicsLearner
from pbcwm.core.types import Transition

from .hypernetwork import HyperNetwork
from .regularizer import (
    normalized_output_drift,
    output_space_regularizer,
    snapshot_output_targets,
)
from .router import ResidualRegimeRouter, RouterDecision
from .target_dynamics import TargetDynamics


class HyperCRLAdaptDynamicsLearner(DynamicsLearner):
    """Shared hypernetwork plus boundary-free embedding management.

    Only the active embedding's transition buffer is retained. Older regime
    transition datasets are deliberately discarded; retention is performed in
    generated-weight space by the HyperCRL output regularizer.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        embedding_dim: int = 16,
        embedding_init_std: float = 0.1,
        hyper_hidden_dims: Sequence[int] = (128, 128),
        target_hidden_dims: Sequence[int] = (256, 256),
        hyper_lr: float = 1e-3,
        embedding_lr: float = 1e-3,
        regularization_beta: float = 0.1,
        current_regime_buffer_size: int = 5000,
        dynamics_batch_size: int = 256,
        router_window_size: int = 32,
        shift_threshold: float = 0.05,
        reuse_threshold: float = 0.03,
        consecutive_trigger_windows: int = 2,
        router_cooldown_steps: int = 32,
        gradient_clip_norm: float = 10.0,
        device: str | torch.device = "cpu",
        seed: int | None = None,
    ) -> None:
        if obs_dim <= 0 or action_dim <= 0:
            raise ValueError("obs_dim and action_dim must be positive")
        if embedding_init_std <= 0 or hyper_lr <= 0 or embedding_lr <= 0:
            raise ValueError("embedding and optimizer scales must be positive")
        if regularization_beta < 0 or current_regime_buffer_size <= 0 or dynamics_batch_size <= 0:
            raise ValueError("invalid retention or dynamics-buffer configuration")
        if gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive")

        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.embedding_dim = int(embedding_dim)
        self.embedding_init_std = float(embedding_init_std)
        self.hyper_lr = float(hyper_lr)
        self.embedding_lr = float(embedding_lr)
        self.regularization_beta = float(regularization_beta)
        self.current_regime_buffer_size = int(current_regime_buffer_size)
        self.dynamics_batch_size = int(dynamics_batch_size)
        self.gradient_clip_norm = float(gradient_clip_norm)
        self.device = torch.device(device)
        self._rng = torch.Generator(device="cpu")
        if seed is not None:
            self._rng.manual_seed(seed)

        self.target_dynamics = TargetDynamics(
            input_dim=self.obs_dim + self.action_dim,
            output_dim=self.obs_dim,
            hidden_dims=target_hidden_dims,
        )
        self.hypernetwork = HyperNetwork(
            embedding_dim=self.embedding_dim,
            target_shapes=self.target_dynamics.parameter_shapes,
            hidden_dims=hyper_hidden_dims,
        ).to(self.device)
        self.embeddings = nn.ParameterList()
        self.embedding_birth_steps: list[int] = []
        self.current_embedding_id: int | None = None
        self.previous_embedding_id: int | None = None
        self._protected_targets: dict[int, list[torch.Tensor]] = {}
        self._current_buffer: deque[Transition] = deque(maxlen=self.current_regime_buffer_size)
        self.router = ResidualRegimeRouter(
            window_size=router_window_size,
            shift_threshold=shift_threshold,
            reuse_threshold=reuse_threshold,
            consecutive_trigger_windows=consecutive_trigger_windows,
            cooldown_steps=router_cooldown_steps,
        )
        self.global_step = 0
        self.assignment_history: list[int] = []
        self.switch_count = 0
        self.new_embedding_count = 0
        self.reuse_count = 0
        self._optimizer: torch.optim.Optimizer | None = None
        self._last_update = {
            "L_dyn": 0.0,
            "L_reg": 0.0,
            "L_total": 0.0,
            "hypernetwork_grad_norm": 0.0,
            "dynamics_updates": 0.0,
        }

    @property
    def num_embeddings(self) -> int:
        return len(self.embeddings)

    @property
    def dynamics_ready(self) -> bool:
        return len(self._current_buffer) >= self.dynamics_batch_size

    @property
    def current_buffer_size(self) -> int:
        return len(self._current_buffer)

    def seed_observations(self) -> list[np.ndarray]:
        """Return active-regime real observations for preference bootstrap."""

        return [np.asarray(transition.obs, dtype=np.float32).copy() for transition in self._current_buffer]

    def observe(self, transition: Transition) -> None:
        self.global_step += 1
        if self.current_embedding_id is None:
            self._activate(self._create_embedding(), reason="new")

        self.router.add_transition(transition)
        if self.current_buffer_size >= self.dynamics_batch_size:
            decision = self.router.evaluate(
                current_embedding_id=self.current_embedding_id,
                stored_embedding_ids=list(range(self.num_embeddings)),
                error_fn=self._window_error,
            )
        else:
            decision = RouterDecision(0.0, float("inf"), None, False, False, False, None)
        if decision.reuse_triggered and decision.selected_embedding_id is not None:
            self._activate(decision.selected_embedding_id, reason="reuse")
            self.router.commit_switch()
            self.switch_count += 1
            self.reuse_count += 1
        elif decision.new_embedding_triggered:
            self._activate(self._create_embedding(), reason="new")
            self.router.commit_switch()
            self.switch_count += 1
            self.new_embedding_count += 1

        self._current_buffer.append(transition)
        self.assignment_history.append(self.current_embedding_id)

    def update(self, num_steps: int = 1) -> dict[str, float]:
        if num_steps < 0:
            raise ValueError("num_steps must be non-negative")
        if (
            num_steps == 0
            or self.current_embedding_id is None
            or not self._current_buffer
            or self._optimizer is None
        ):
            return dict(self._last_update)

        batch_size = min(self.dynamics_batch_size, len(self._current_buffer))
        transitions = list(self._current_buffer)
        losses: list[tuple[float, float, float, float]] = []
        for _ in range(num_steps):
            indices = torch.randperm(len(transitions), generator=self._rng)[:batch_size].tolist()
            batch = [transitions[index] for index in indices]
            obs = torch.as_tensor(np.stack([item.obs for item in batch]), dtype=torch.float32, device=self.device)
            action = torch.as_tensor(np.stack([item.action for item in batch]), dtype=torch.float32, device=self.device)
            target_delta = torch.as_tensor(
                np.stack([item.next_obs - item.obs for item in batch]), dtype=torch.float32, device=self.device
            )
            weights = self.hypernetwork(self.embeddings[self.current_embedding_id])
            prediction = self.target_dynamics.predict_delta(obs, action, weights)
            loss_dyn = torch.mean((prediction - target_delta.to(prediction.device, prediction.dtype)).square())
            loss_reg = output_space_regularizer(
                self.hypernetwork, self.embeddings, self._protected_targets
            )
            loss_total = loss_dyn + self.regularization_beta * loss_reg
            self._optimizer.zero_grad(set_to_none=True)
            loss_total.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(
                list(self.hypernetwork.parameters()) + [self.embeddings[self.current_embedding_id]],
                self.gradient_clip_norm,
            ).detach().cpu())
            self._optimizer.step()
            losses.append((float(loss_dyn.detach().cpu()), float(loss_reg.detach().cpu()), float(loss_total.detach().cpu()), grad_norm))

        self._last_update = {
            "L_dyn": float(np.mean([item[0] for item in losses])),
            "L_reg": float(np.mean([item[1] for item in losses])),
            "L_total": float(np.mean([item[2] for item in losses])),
            "hypernetwork_grad_norm": float(np.mean([item[3] for item in losses])),
            "dynamics_updates": float(num_steps),
        }
        return dict(self._last_update)

    def predict(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if self.current_embedding_id is None:
            raise RuntimeError("HyperCRL-Adapt has no active embedding")
        weights = self.hypernetwork(self.embeddings[self.current_embedding_id])
        prediction = self.target_dynamics.predict_next(obs, action, weights)
        return prediction.to(device=obs.device, dtype=obs.dtype)

    def predict_embedding(self, embedding_id: int, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if not 0 <= embedding_id < self.num_embeddings:
            raise IndexError("embedding_id is out of range")
        weights = self.hypernetwork(self.embeddings[embedding_id])
        prediction = self.target_dynamics.predict_next(obs, action, weights)
        return prediction.to(device=obs.device, dtype=obs.dtype)

    def diagnostics(self) -> dict:
        decision = self.router.last_decision
        current_error = decision.current_error
        if self.router.ready and self.current_embedding_id is not None:
            current_error = self._window_error(self.current_embedding_id, list(self.router.window))
        return {
            "active_embedding_id": -1 if self.current_embedding_id is None else self.current_embedding_id,
            "num_embeddings": self.num_embeddings,
            "current_window_error": current_error,
            "best_stored_window_error": decision.best_stored_error,
            "best_stored_embedding_id": -1 if decision.best_stored_embedding_id is None else decision.best_stored_embedding_id,
            "shift_triggered": decision.shift_triggered,
            "reuse_triggered": decision.reuse_triggered,
            "new_embedding_triggered": decision.new_embedding_triggered,
            "router_switch_count": self.router.switch_count,
            "new_embedding_count": self.new_embedding_count,
            "reuse_count": self.reuse_count,
            "embedding_norms": [float(embedding.detach().norm().cpu()) for embedding in self.embeddings],
            "old_generated_weight_drift": normalized_output_drift(
                self.hypernetwork, self.embeddings, self._protected_targets
            ),
            **self._last_update,
        }

    def state_dict(self) -> dict:
        return {
            "hypernetwork": self.hypernetwork.state_dict(),
            "embeddings": [embedding.detach().clone() for embedding in self.embeddings],
            "embedding_birth_steps": list(self.embedding_birth_steps),
            "current_embedding_id": self.current_embedding_id,
            "previous_embedding_id": self.previous_embedding_id,
            "protected_targets": {
                int(key): [value.detach().clone() for value in values]
                for key, values in self._protected_targets.items()
            },
            "current_buffer": list(self._current_buffer),
            "router": self.router.state_dict(),
            "global_step": self.global_step,
            "assignment_history": list(self.assignment_history),
            "switch_count": self.switch_count,
            "new_embedding_count": self.new_embedding_count,
            "reuse_count": self.reuse_count,
            "last_update": dict(self._last_update),
            "optimizer": None if self._optimizer is None else self._optimizer.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.hypernetwork.load_state_dict(state["hypernetwork"])
        self.embeddings = nn.ParameterList([
            nn.Parameter(value.clone().to(self.device)) for value in state["embeddings"]
        ])
        self.embedding_birth_steps = list(state["embedding_birth_steps"])
        self.current_embedding_id = state["current_embedding_id"]
        self.previous_embedding_id = state["previous_embedding_id"]
        self._protected_targets = {
            int(key): [value.clone().to(self.device) for value in values]
            for key, values in state["protected_targets"].items()
        }
        self._current_buffer.clear()
        self._current_buffer.extend(state["current_buffer"])
        for index, embedding in enumerate(self.embeddings):
            embedding.requires_grad_(index == self.current_embedding_id)
        self._make_optimizer()
        if state.get("optimizer") is not None:
            self._optimizer.load_state_dict(state["optimizer"])
        self.router.load_state_dict(state["router"])
        self.global_step = int(state["global_step"])
        self.assignment_history = list(state["assignment_history"])
        self.switch_count = int(state["switch_count"])
        self.new_embedding_count = int(state["new_embedding_count"])
        self.reuse_count = int(state["reuse_count"])
        self._last_update = dict(state["last_update"])

    def _create_embedding(self) -> int:
        # Keep the private RNG on CPU for reproducible indexing and generate
        # the infrequent embedding sample there before transferring it to the
        # learner device.  A CPU generator cannot be passed to torch.randn
        # with a CUDA output device.
        embedding = torch.randn(
            self.embedding_dim, generator=self._rng, device="cpu", dtype=torch.float32
        ).to(self.device) * self.embedding_init_std
        self.embeddings.append(nn.Parameter(embedding))
        self.embedding_birth_steps.append(self.global_step)
        return self.num_embeddings - 1

    def _activate(self, embedding_id: int, reason: str) -> None:
        del reason
        if not 0 <= embedding_id < self.num_embeddings:
            raise IndexError("embedding_id is out of range")
        if self.current_embedding_id == embedding_id and self._optimizer is not None:
            return
        previous = self.current_embedding_id
        inactive = [index for index in range(self.num_embeddings) if index != embedding_id]
        self._protected_targets = snapshot_output_targets(
            self.hypernetwork, self.embeddings, inactive
        )
        for index, embedding in enumerate(self.embeddings):
            embedding.requires_grad_(index == embedding_id)
        self.previous_embedding_id = previous
        self.current_embedding_id = embedding_id
        self._current_buffer.clear()
        self._make_optimizer()

    def _make_optimizer(self) -> None:
        if self.current_embedding_id is None:
            self._optimizer = None
            return
        self._optimizer = torch.optim.Adam(
            [
                {"params": list(self.hypernetwork.parameters()), "lr": self.hyper_lr},
                {"params": [self.embeddings[self.current_embedding_id]], "lr": self.embedding_lr},
            ]
        )

    def _window_error(self, embedding_id: int, transitions: list[Transition]) -> float:
        if not transitions:
            return float("inf")
        obs = torch.as_tensor(np.stack([item.obs for item in transitions]), dtype=torch.float32, device=self.device)
        action = torch.as_tensor(np.stack([item.action for item in transitions]), dtype=torch.float32, device=self.device)
        target_delta = torch.as_tensor(
            np.stack([item.next_obs - item.obs for item in transitions]), dtype=torch.float32, device=self.device
        )
        with torch.no_grad():
            weights = self.hypernetwork(self.embeddings[embedding_id])
            prediction = self.target_dynamics.predict_delta(obs, action, weights)
            return float(torch.mean((prediction - target_delta.to(prediction.device, prediction.dtype)).square()).cpu())
