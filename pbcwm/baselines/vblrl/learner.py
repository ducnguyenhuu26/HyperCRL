"""VBLRL-Adapt learner: q_W, stored q_k posteriors, and predictive routing."""

from collections import deque
from collections.abc import Sequence

import numpy as np
import torch

from pbcwm.core.dynamics import StochasticDynamicsLearner
from pbcwm.core.types import Transition

from .posterior import BayesianDynamicsPosterior
from .router import PosteriorPredictiveRouter, PosteriorRouterDecision
from .world_model import WorldPosterior


class VBLRLAdaptDynamicsLearner(StochasticDynamicsLearner):
    """Reward-free Bayesian lifelong dynamics with minimal acquisition routing."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
        posterior_learning_rate: float = 1e-3,
        world_learning_rate: float = 1e-3,
        world_buffer_size: int = 10_000,
        current_buffer_size: int = 5_000,
        dynamics_batch_size: int = 256,
        dynamics_updates_per_step: int = 1,
        world_updates_per_interval: int = 20,
        world_update_interval_steps: int = 500,
        router_window_size: int = 32,
        router_posterior_samples: int = 5,
        shift_threshold: float = 4.0,
        reuse_threshold: float = 3.0,
        consecutive_trigger_windows: int = 2,
        router_cooldown_steps: int = 32,
        planning_model_samples: int = 5,
        min_logvar: float = -10.0,
        max_logvar: float = 2.0,
        gradient_clip_norm: float = 10.0,
        device: str | torch.device = "cpu",
        seed: int | None = None,
    ) -> None:
        if dynamics_batch_size <= 0 or current_buffer_size <= 0 or dynamics_updates_per_step < 0:
            raise ValueError("invalid current-regime dynamics configuration")
        if world_update_interval_steps <= 0 or world_updates_per_interval < 0:
            raise ValueError("invalid world posterior update configuration")
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.hidden_dims = tuple(int(size) for size in hidden_dims)
        self.dynamics_batch_size = int(dynamics_batch_size)
        self.current_buffer_size = int(current_buffer_size)
        self.dynamics_updates_per_step = int(dynamics_updates_per_step)
        self.world_updates_per_interval = int(world_updates_per_interval)
        self.world_update_interval_steps = int(world_update_interval_steps)
        self.router_posterior_samples = int(router_posterior_samples)
        self.planning_model_samples = int(planning_model_samples)
        self.device = torch.device(device)
        self.seed = seed
        self.world = WorldPosterior(
            obs_dim,
            action_dim,
            hidden_dims=self.hidden_dims,
            learning_rate=world_learning_rate,
            world_buffer_size=world_buffer_size,
            min_logvar=min_logvar,
            max_logvar=max_logvar,
            gradient_clip_norm=gradient_clip_norm,
            device=device,
            seed=seed,
        )
        self.router = PosteriorPredictiveRouter(
            window_size=router_window_size,
            posterior_samples=router_posterior_samples,
            shift_threshold=shift_threshold,
            reuse_threshold=reuse_threshold,
            consecutive_trigger_windows=consecutive_trigger_windows,
            cooldown_steps=router_cooldown_steps,
        )
        self.posteriors: list[BayesianDynamicsPosterior] = []
        self.regime_priors: list[dict] = []
        self.current_posterior_id: int | None = None
        self.previous_posterior_id: int | None = None
        self._current_buffer: deque[Transition] = deque(maxlen=self.current_buffer_size)
        self.assignment_history: list[int] = []
        self.global_step = 0
        self.acquisition_count = 0
        self.reacquisition_count = 0
        self._last_update = {
            "active_regime_nll": 0.0,
            "active_regime_kl": 0.0,
            "active_regime_vb_loss": 0.0,
            "world_nll": 0.0,
            "world_kl": 0.0,
            "world_vb_loss": 0.0,
            "dynamics_updates": 0.0,
            "world_updates": 0.0,
        }

    @property
    def num_regime_posteriors(self) -> int:
        return len(self.posteriors)

    @property
    def dynamics_ready(self) -> bool:
        return len(self._current_buffer) >= self.dynamics_batch_size

    @property
    def world_buffer_size(self) -> int:
        return self.world.size

    def seed_observations(self) -> list[np.ndarray]:
        return [np.asarray(item.obs, dtype=np.float32).copy() for item in self._current_buffer]

    def observe(self, transition: Transition) -> None:
        self.global_step += 1
        self.world.observe(transition)
        if self.current_posterior_id is None:
            self._activate_new_posterior()

        self.router.add_transition(transition)
        if len(self._current_buffer) >= self.dynamics_batch_size:
            decision = self.router.evaluate(
                current_id=self.current_posterior_id,
                stored_ids=list(range(self.num_regime_posteriors)),
                nll_fn=self._window_nll,
            )
        else:
            decision = PosteriorRouterDecision(0.0, float("inf"), None, False, False, False, None)

        if decision.reacquisition_triggered and decision.selected_posterior_id is not None:
            self._activate_existing(decision.selected_posterior_id)
            self.router.commit_switch()
            self.reacquisition_count += 1
        elif decision.acquisition_triggered:
            self._activate_new_posterior()
            self.router.commit_switch()
            self.acquisition_count += 1

        self._current_buffer.append(transition)
        self.assignment_history.append(self.current_posterior_id)

    def update(self, num_steps: int = 1) -> dict[str, float]:
        if num_steps < 0:
            raise ValueError("num_steps must be non-negative")
        if self.current_posterior_id is None or not self._current_buffer:
            return dict(self._last_update)
        active_metrics = self.posteriors[self.current_posterior_id].update(
            list(self._current_buffer),
            prior=self.regime_priors[self.current_posterior_id],
            num_steps=min(num_steps, self.dynamics_updates_per_step) if num_steps else 0,
        )
        world_metrics = {"world_nll": 0.0, "world_kl": 0.0, "world_vb_loss": 0.0, "world_updates": 0.0}
        if self.global_step % self.world_update_interval_steps == 0:
            world_metrics = self.world.update(
                num_steps=self.world_updates_per_interval,
                batch_size=self.dynamics_batch_size,
            )
        self._last_update = {
            "active_regime_nll": active_metrics["nll"],
            "active_regime_kl": active_metrics["kl"],
            "active_regime_vb_loss": active_metrics["vb_loss"],
            "world_nll": world_metrics["world_nll"],
            "world_kl": world_metrics["world_kl"],
            "world_vb_loss": world_metrics["world_vb_loss"],
            "dynamics_updates": active_metrics["updates"],
            "world_updates": world_metrics["world_updates"],
        }
        return dict(self._last_update)

    def predict(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if self.current_posterior_id is None:
            raise RuntimeError("VBLRL-Adapt has no active posterior")
        prediction = self.posteriors[self.current_posterior_id].predict_next(obs, action)
        return prediction.to(device=obs.device, dtype=obs.dtype)

    def sample_next(self, obs: torch.Tensor, action: torch.Tensor, num_samples: int) -> torch.Tensor:
        if self.current_posterior_id is None:
            raise RuntimeError("VBLRL-Adapt has no active posterior")
        return self.posteriors[self.current_posterior_id].sample_next(obs, action, num_samples).to(
            device=obs.device, dtype=obs.dtype
        )

    def diagnostics(self) -> dict:
        current_nll = 0.0
        if self.current_posterior_id is not None and self.router.window:
            current_nll = self._window_nll(
                self.current_posterior_id, list(self.router.window), self.router_posterior_samples
            )
        decision = self.router.last_decision
        return {
            "num_regime_posteriors": self.num_regime_posteriors,
            "active_posterior_id": -1 if self.current_posterior_id is None else self.current_posterior_id,
            "current_predictive_nll": current_nll,
            "best_stored_predictive_nll": decision.best_stored_nll,
            "best_stored_posterior_id": -1 if decision.best_stored_posterior_id is None else decision.best_stored_posterior_id,
            "acquisition_count": self.acquisition_count,
            "reacquisition_count": self.reacquisition_count,
            "router_switch_count": self.router.switch_count,
            "world_buffer_size": self.world_buffer_size,
            "posterior_parameter_std_mean": self._active_std_mean(),
            "predictive_epistemic_variance": self._active_uncertainty()[0],
            "predictive_aleatoric_variance": self._active_uncertainty()[1],
            "shift_triggered": decision.shift_triggered,
            "reacquisition_triggered": decision.reacquisition_triggered,
            "acquisition_triggered": decision.acquisition_triggered,
            **self._last_update,
        }

    def state_dict(self) -> dict:
        return {
            "world": self.world.state_dict(),
            "posteriors": [posterior.state_dict() for posterior in self.posteriors],
            "regime_priors": self.regime_priors,
            "current_posterior_id": self.current_posterior_id,
            "previous_posterior_id": self.previous_posterior_id,
            "current_buffer": list(self._current_buffer),
            "router": self.router.state_dict(),
            "assignment_history": list(self.assignment_history),
            "global_step": self.global_step,
            "acquisition_count": self.acquisition_count,
            "reacquisition_count": self.reacquisition_count,
            "last_update": dict(self._last_update),
        }

    def load_state_dict(self, state: dict) -> None:
        self.world.load_state_dict(state["world"])
        self.posteriors = []
        for posterior_state in state["posteriors"]:
            posterior = BayesianDynamicsPosterior(
                self.obs_dim,
                self.action_dim,
                hidden_dims=self.hidden_dims,
                learning_rate=self.world.posterior.learning_rate,
                min_logvar=self.world.posterior.min_logvar,
                max_logvar=self.world.posterior.max_logvar,
                gradient_clip_norm=self.world.posterior.gradient_clip_norm,
                device=self.device,
            )
            posterior.load_state_dict(posterior_state)
            self.posteriors.append(posterior)
        self.regime_priors = state["regime_priors"]
        self.current_posterior_id = state["current_posterior_id"]
        self.previous_posterior_id = state["previous_posterior_id"]
        self._current_buffer.clear()
        self._current_buffer.extend(state["current_buffer"])
        self.router.load_state_dict(state["router"])
        self.assignment_history = list(state["assignment_history"])
        self.global_step = int(state["global_step"])
        self.acquisition_count = int(state["acquisition_count"])
        self.reacquisition_count = int(state["reacquisition_count"])
        self._last_update = dict(state["last_update"])

    def _activate_new_posterior(self) -> None:
        if not self.posteriors:
            posterior = self.world.posterior.clone()
            prior = self.world.posterior.snapshot()["posterior"]
        else:
            posterior, prior = self.world.initialize_regime()
        self.posteriors.append(posterior)
        self.regime_priors.append(prior)
        self._activate_existing(len(self.posteriors) - 1)

    def _activate_existing(self, posterior_id: int) -> None:
        if not 0 <= posterior_id < self.num_regime_posteriors:
            raise IndexError("posterior_id is out of range")
        self.previous_posterior_id = self.current_posterior_id
        self.current_posterior_id = posterior_id
        self._current_buffer.clear()

    def _window_nll(self, posterior_id: int, transitions: list[Transition], samples: int) -> float:
        return self.posteriors[posterior_id].log_predictive_likelihood(transitions, samples)

    def _active_std_mean(self) -> float:
        if self.current_posterior_id is None:
            return 0.0
        return self.posteriors[self.current_posterior_id].parameter_std_mean

    def _active_uncertainty(self) -> tuple[float, float]:
        if self.current_posterior_id is None or not self._current_buffer:
            return 0.0, 0.0
        batch = list(self._current_buffer)[-min(8, len(self._current_buffer)):]
        obs = torch.as_tensor(np.stack([item.obs for item in batch]), dtype=torch.float32, device=self.device)
        action = torch.as_tensor(np.stack([item.action for item in batch]), dtype=torch.float32, device=self.device)
        posterior = self.posteriors[self.current_posterior_id]
        means = []
        aleatoric = []
        with torch.no_grad():
            for _ in range(min(self.planning_model_samples, 5)):
                mean, logvar = posterior.network(obs, action)
                means.append(mean)
                aleatoric.append(logvar.exp())
        return float(torch.stack(means).var(0, unbiased=False).mean().cpu()), float(torch.stack(aleatoric).mean().cpu())
