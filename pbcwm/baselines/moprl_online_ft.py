"""Standardized MoP-RL core adapted to naive online dynamics fine-tuning."""

from collections.abc import Sequence

import numpy as np
import torch

from pbcwm.core.types import Transition
from pbcwm.planning.cem import CEMPlanResult, CEMPlanner
from pbcwm.preferences.buffer import PreferenceBuffer
from pbcwm.preferences.query import DisagreementQuerySelector
from pbcwm.preferences.reward_model import PreferenceRewardEnsemble
from pbcwm.preferences.teacher import SyntheticPreferenceTeacher
from pbcwm.preferences.types import PreferenceExample, TrajectorySegment
from pbcwm.rewards.preference import LearnedPreferenceReward

from .static import StaticDynamicsLearner


class MoPRLOnlineFT:
    """Compose shared preference/planning components with rolling-window FT.

    The dynamics learner sees only real ``(obs, action, next_obs)`` data. The
    synthetic teacher is retained solely as a preference-label oracle; its
    ground-truth scores never reach the planner or reward learner.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        planner_config: dict,
        model_hidden_dims: Sequence[int] = (256, 256),
        model_learning_rate: float = 1e-3,
        dynamics_window_size: int = 5_000,
        dynamics_batch_size: int = 256,
        preference_ensemble_size: int = 3,
        preference_hidden_dims: Sequence[int] = (256, 256),
        preference_learning_rate: float = 1e-3,
        preference_batch_size: int = 32,
        preference_buffer_capacity: int | None = None,
        min_preferences_before_planning: int = 40,
        pair_pool_size: int = 256,
        teacher_skip_margin: float = 0.0,
        teacher_reward=None,
        device: str | torch.device = "cpu",
        seed: int | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.min_preferences_before_planning = int(min_preferences_before_planning)
        self.dynamics = StaticDynamicsLearner(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dims=model_hidden_dims,
            learning_rate=model_learning_rate,
            replay_capacity=dynamics_window_size,
            batch_size=dynamics_batch_size,
            device=device,
            seed=seed,
        )
        self.reward_ensemble = PreferenceRewardEnsemble(
            obs_dim=obs_dim,
            action_dim=action_dim,
            ensemble_size=preference_ensemble_size,
            hidden_dims=preference_hidden_dims,
            learning_rate=preference_learning_rate,
            batch_size=preference_batch_size,
            device=device,
            seed=None if seed is None else seed + 1,
        )
        self.preference_buffer = PreferenceBuffer(preference_buffer_capacity, seed=seed)
        self.query_selector = DisagreementQuerySelector(pair_pool_size=pair_pool_size, seed=seed)
        if teacher_reward is None:
            raise ValueError("teacher_reward is required for synthetic preference labels")
        self.teacher = SyntheticPreferenceTeacher(teacher_reward, teacher_skip_margin)
        self.learned_reward = LearnedPreferenceReward(self.reward_ensemble)

        planner_kwargs = dict(planner_config)
        planner_kwargs["action_low"] = action_low
        planner_kwargs["action_high"] = action_high
        planner_kwargs["device"] = device
        self.planner = CEMPlanner(**planner_kwargs)
        self._rng = np.random.default_rng(seed)

    @property
    def dynamics_ready(self) -> bool:
        return len(self.dynamics.replay_buffer) >= self.dynamics.batch_size

    @property
    def planning_ready(self) -> bool:
        return self.dynamics_ready and len(self.preference_buffer) >= self.min_preferences_before_planning

    def observe(self, transition: Transition) -> None:
        self.dynamics.observe(transition)

    def update_dynamics(self, num_steps: int = 1) -> dict[str, float]:
        return self.dynamics.update(num_steps)

    def plan(self, obs: np.ndarray | torch.Tensor, collect_candidates: bool = False) -> CEMPlanResult:
        return self.planner.plan(
            obs=obs,
            dynamics=self.dynamics,
            reward_fn=self.learned_reward,
            return_candidates=collect_candidates,
        )

    def bootstrap(self, num_queries: int, horizon: int, action_low: np.ndarray, action_high: np.ndarray) -> dict[str, float]:
        """Seed preference replay from random imagined rollouts after dynamics warm-up."""

        if not self.dynamics_ready or num_queries <= 0:
            return self.preference_metrics()
        transitions = list(self.dynamics.replay_buffer)
        attempts = 0
        target = len(self.preference_buffer) + int(num_queries)
        while len(self.preference_buffer) < target and attempts < max(4 * num_queries, 1):
            attempts += 1
            start = transitions[int(self._rng.integers(len(transitions)))].obs
            traj_a = self._random_rollout(start, horizon, action_low, action_high)
            traj_b = self._random_rollout(start, horizon, action_low, action_high)
            example = self._label_example(traj_a, traj_b)
            if example is not None:
                self.preference_buffer.add(example)
        if len(self.preference_buffer) >= self.reward_ensemble.batch_size:
            update_metrics = self.reward_ensemble.update(
                self.preference_buffer,
                num_steps=max(1, min(10, num_queries)),
            )
        else:
            update_metrics = {}
        return {**self.preference_metrics(), **update_metrics}

    def query_and_update(
        self,
        candidates: Sequence[TrajectorySegment],
        num_queries: int,
        reward_updates: int,
    ) -> dict[str, float]:
        """Label disagreement-selected CEM candidates and train the ensemble."""

        scored = self.query_selector.score_pairs_with_ensemble(candidates, self.reward_ensemble)
        selected = scored[: max(0, num_queries)]
        added = 0
        for (index_a, index_b), _ in selected:
            example = self._label_example(candidates[index_a], candidates[index_b])
            if example is not None:
                self.preference_buffer.add(example)
                added += 1
        update_metrics = self.reward_ensemble.update(self.preference_buffer, reward_updates)
        disagreement = float(np.mean([score for _, score in selected])) if selected else 0.0
        return {
            **self.preference_metrics(),
            **update_metrics,
            "queries_added": float(added),
            "ensemble_disagreement": disagreement,
        }

    def generate_preference_examples(
        self,
        num_examples: int,
        horizon: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        seed: int,
    ) -> list[PreferenceExample]:
        """Generate held-out labeled examples without inserting them into replay."""

        if not self.dynamics_ready:
            return []
        transitions = list(self.dynamics.replay_buffer)
        rng = np.random.default_rng(seed)
        examples: list[PreferenceExample] = []
        attempts = 0
        while len(examples) < num_examples and attempts < max(4 * num_examples, 1):
            attempts += 1
            start = transitions[int(rng.integers(len(transitions)))].obs
            traj_a = self._random_rollout(start, horizon, action_low, action_high, rng)
            traj_b = self._random_rollout(start, horizon, action_low, action_high, rng)
            example = self._label_example(traj_a, traj_b)
            if example is not None:
                examples.append(example)
        return examples

    def preference_metrics(self) -> dict[str, float]:
        return {"num_preferences": float(len(self.preference_buffer))}

    def _label_example(
        self,
        traj_a: TrajectorySegment,
        traj_b: TrajectorySegment,
    ) -> PreferenceExample | None:
        label = self.teacher.label(traj_a, traj_b)
        return None if label is None else PreferenceExample(traj_a, traj_b, label)

    def _random_rollout(
        self,
        start_obs: np.ndarray,
        horizon: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        rng: np.random.Generator | None = None,
    ) -> TrajectorySegment:
        rng = self._rng if rng is None else rng
        obs = torch.as_tensor(start_obs, dtype=torch.float32, device=self.device)
        observations = []
        actions = []
        next_observations = []
        with torch.no_grad():
            for _ in range(horizon):
                action_np = rng.uniform(action_low, action_high).astype(np.float32)
                action = torch.as_tensor(action_np, dtype=torch.float32, device=self.device)
                next_obs = self.dynamics.predict(obs.unsqueeze(0), action.unsqueeze(0)).squeeze(0)
                observations.append(obs.clone())
                actions.append(action.clone())
                next_observations.append(next_obs.clone())
                obs = next_obs
        return TrajectorySegment(
            obs=torch.stack(observations),
            actions=torch.stack(actions),
            next_obs=torch.stack(next_observations),
        )
