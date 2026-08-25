"""Online PB-CWM composition for the GPMM dynamics baseline."""

from collections.abc import Sequence

import numpy as np
import torch

from pbcwm.planning.cem import CEMPlanResult, CEMPlanner
from pbcwm.preferences.buffer import PreferenceBuffer
from pbcwm.preferences.query import DisagreementQuerySelector
from pbcwm.preferences.reward_model import PreferenceRewardEnsemble
from pbcwm.preferences.teacher import SyntheticPreferenceTeacher
from pbcwm.preferences.types import PreferenceExample, TrajectorySegment
from pbcwm.rewards.preference import LearnedPreferenceReward

from .learner import GPMMDynamicsLearner


class GPMMOnline:
    """Compose shared preference/planning code with GPMM dynamics.

    The wrapper deliberately mirrors the Phase-1 online preference interface.
    The GPMM learner receives only the transition fields; the synthetic reward
    is used only by the preference teacher to create labels.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        planner_config: dict,
        gpmm_config: dict,
        preference_config: dict,
        teacher_reward,
        device: str | torch.device = "cpu",
        seed: int | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.min_dynamics_points = int(
            gpmm_config["expert_min_points_before_competition"]
        )
        self.min_preferences_before_planning = int(
            preference_config["min_preferences_before_planning"]
        )
        self.dynamics = GPMMDynamicsLearner(
            obs_dim=obs_dim,
            action_dim=action_dim,
            device=device,
            seed=seed,
            **gpmm_config,
        )
        self.reward_ensemble = PreferenceRewardEnsemble(
            obs_dim=obs_dim,
            action_dim=action_dim,
            ensemble_size=preference_config["ensemble_size"],
            hidden_dims=preference_config["hidden_dims"],
            learning_rate=preference_config["learning_rate"],
            batch_size=preference_config["reward_batch_size"],
            device=device,
            seed=None if seed is None else seed + 1,
        )
        self.preference_buffer = PreferenceBuffer(seed=seed)
        self.query_selector = DisagreementQuerySelector(
            pair_pool_size=preference_config["pair_pool_size"], seed=seed
        )
        self.teacher = SyntheticPreferenceTeacher(
            teacher_reward, preference_config["teacher_skip_margin"]
        )
        self.learned_reward = LearnedPreferenceReward(self.reward_ensemble)

        planner_kwargs = dict(planner_config)
        planner_kwargs["action_low"] = action_low
        planner_kwargs["action_high"] = action_high
        planner_kwargs["device"] = device
        planner_kwargs["candidate_keep_per_iteration"] = preference_config[
            "candidate_keep_per_iteration"
        ]
        planner_kwargs["candidate_keep_final_elites"] = preference_config[
            "candidate_keep_final_elites"
        ]
        self.planner = CEMPlanner(**planner_kwargs)
        self._rng = np.random.default_rng(seed)

    @property
    def dynamics_ready(self) -> bool:
        return (
            self.dynamics.current_expert is not None
            and self.dynamics.experts[self.dynamics.current_expert].num_points
            >= self.min_dynamics_points
        )

    @property
    def planning_ready(self) -> bool:
        return (
            self.dynamics_ready
            and len(self.preference_buffer) >= self.min_preferences_before_planning
        )

    def observe(self, transition) -> None:
        self.dynamics.observe(transition)

    def update_dynamics(self, num_steps: int = 1) -> dict[str, float]:
        return self.dynamics.update(num_steps)

    def plan(self, obs, collect_candidates: bool = False) -> CEMPlanResult:
        return self.planner.plan(
            obs=obs,
            dynamics=self.dynamics,
            reward_fn=self.learned_reward,
            return_candidates=collect_candidates,
        )

    def bootstrap(
        self,
        num_queries: int,
        horizon: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
    ) -> dict[str, float]:
        if not self.dynamics_ready or num_queries <= 0:
            return self.preference_metrics()
        starts = self._seed_observations()
        attempts = 0
        target = len(self.preference_buffer) + int(num_queries)
        while len(self.preference_buffer) < target and attempts < max(4 * num_queries, 1):
            attempts += 1
            start = starts[int(self._rng.integers(len(starts)))]
            example = self._label_example(
                self._random_rollout(start, horizon, action_low, action_high),
                self._random_rollout(start, horizon, action_low, action_high),
            )
            if example is not None:
                self.preference_buffer.add(example)
        return self._update_preference_model(num_queries)

    def query_and_update(
        self,
        candidates: Sequence[TrajectorySegment],
        num_queries: int,
        reward_updates: int,
    ) -> dict[str, float]:
        scored = self.query_selector.score_pairs_with_ensemble(
            candidates, self.reward_ensemble
        )
        selected = scored[: max(0, num_queries)]
        added = 0
        for (index_a, index_b), _ in selected:
            example = self._label_example(candidates[index_a], candidates[index_b])
            if example is not None:
                self.preference_buffer.add(example)
                added += 1
        metrics = self.reward_ensemble.update(self.preference_buffer, reward_updates)
        disagreement = float(np.mean([score for _, score in selected])) if selected else 0.0
        return {
            **self.preference_metrics(),
            **metrics,
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
        if not self.dynamics_ready:
            return []
        starts = self._seed_observations()
        rng = np.random.default_rng(seed)
        examples: list[PreferenceExample] = []
        attempts = 0
        while len(examples) < num_examples and attempts < max(4 * num_examples, 1):
            attempts += 1
            start = starts[int(rng.integers(len(starts)))]
            example = self._label_example(
                self._random_rollout(start, horizon, action_low, action_high, rng),
                self._random_rollout(start, horizon, action_low, action_high, rng),
            )
            if example is not None:
                examples.append(example)
        return examples

    def preference_metrics(self) -> dict[str, float]:
        return {"num_preferences": float(len(self.preference_buffer))}

    def _update_preference_model(self, num_steps: int) -> dict[str, float]:
        if len(self.preference_buffer) < self.reward_ensemble.batch_size:
            return self.preference_metrics()
        return {
            **self.preference_metrics(),
            **self.reward_ensemble.update(
                self.preference_buffer, max(1, min(10, num_steps))
            ),
        }

    def _seed_observations(self) -> list[np.ndarray]:
        observations = [
            expert.training_inputs[:, : self.dynamics.obs_dim]
            for expert in self.dynamics.experts
        ]
        non_empty = [values for values in observations if values.shape[0] > 0]
        if not non_empty:
            raise RuntimeError("cannot bootstrap preferences before collecting dynamics data")
        return [
            value.detach().cpu().numpy().astype(np.float32)
            for values in non_empty
            for value in values
        ]

    def _label_example(
        self, traj_a: TrajectorySegment, traj_b: TrajectorySegment
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
