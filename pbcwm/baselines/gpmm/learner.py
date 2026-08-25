"""GPMM dynamics learner with latent online expert assignment."""

from collections.abc import Sequence

import torch

from pbcwm.core.dynamics import DynamicsLearner
from pbcwm.core.types import Transition

from .assignment import AssignmentResult, assign_transition
from .expert import GPExpert
from .merge import closest_expert, expert_distance


class GPMMDynamicsLearner(DynamicsLearner):
    """Standardized PB-CWM adaptation of the GPMM dynamics mechanism."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        alpha: float = 1.0,
        sticky_bonus: float = 10.0,
        transition_base_count: float = 1.0,
        expert_min_points_before_competition: int = 5,
        gp_fit_steps: int = 5,
        gp_learning_rate: float = 0.05,
        max_points_per_expert: int = 256,
        merge_enabled: bool = True,
        merge_burnin_points: int = 20,
        merge_threshold: float = 0.5,
        prune_probationary: bool = True,
        min_predictive_variance: float = 1e-5,
        max_predictive_variance: float = 1e3,
        prior_variance: float = 1.0,
        observation_noise: float = 0.05,
        device: str | torch.device = "cpu",
        seed: int | None = None,
    ) -> None:
        del device  # GPyTorch expert state is CPU double in this first baseline.
        if alpha <= 0 or transition_base_count <= 0:
            raise ValueError("alpha and transition_base_count must be positive")
        if sticky_bonus < 0 or expert_min_points_before_competition < 1 or gp_fit_steps < 0:
            raise ValueError("invalid GPMM assignment or fit configuration")
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.alpha = float(alpha)
        self.sticky_bonus = float(sticky_bonus)
        self.transition_base_count = float(transition_base_count)
        self.expert_min_points_before_competition = int(expert_min_points_before_competition)
        self.gp_fit_steps = int(gp_fit_steps)
        self.gp_learning_rate = float(gp_learning_rate)
        self.max_points_per_expert = int(max_points_per_expert)
        self.merge_enabled = bool(merge_enabled)
        self.merge_burnin_points = int(merge_burnin_points)
        self.merge_threshold = float(merge_threshold)
        self.prune_probationary = bool(prune_probationary)
        self.min_predictive_variance = float(min_predictive_variance)
        self.max_predictive_variance = float(max_predictive_variance)
        self.prior_variance = float(prior_variance)
        self.observation_noise = float(observation_noise)
        self.seed = seed
        self._next_seed = 0 if seed is None else int(seed)

        self.experts: list[GPExpert] = []
        self.current_expert: int | None = None
        self.previous_expert: int | None = None
        self.transition_counts = torch.zeros(0, 0, dtype=torch.float64)
        self.assignment_history: list[int] = []
        self.expert_birth_steps: list[int] = []
        self.expert_probationary: list[bool] = []
        self.global_step = 0
        self.switch_count = 0
        self.new_expert_count = 0
        self.merge_count = 0
        self.prune_count = 0
        self._dirty_experts: set[int] = set()
        self.last_assignment: AssignmentResult | None = None
        self.last_fit_metrics: dict[str, float] = {}

    @property
    def num_experts(self) -> int:
        return len(self.experts)

    def observe(self, transition: Transition) -> None:
        self.global_step += 1
        if not self.experts:
            selected = self._create_expert()
            self.last_assignment = None
        elif self.current_expert is not None and (
            self.experts[self.current_expert].num_points < self.expert_min_points_before_competition
        ):
            selected = self.current_expert
            self.last_assignment = None
        else:
            result = assign_transition(
                experts=self.experts,
                obs=torch.as_tensor(transition.obs, dtype=torch.float64),
                action=torch.as_tensor(transition.action, dtype=torch.float64),
                next_obs=torch.as_tensor(transition.next_obs, dtype=torch.float64),
                previous_expert=self.current_expert,
                transition_counts=self.transition_counts,
                alpha=self.alpha,
                sticky_bonus=self.sticky_bonus,
                base_count=self.transition_base_count,
                allow_new=True,
            )
            self.last_assignment = result
            if result.selected_index == len(self.experts):
                selected = self._create_expert()
            else:
                selected = result.selected_index

        previous = self.current_expert
        self.experts[selected].add_transition(transition.obs, transition.action, transition.next_obs)
        self._dirty_experts.add(selected)
        self._update_transition_statistics(previous, selected)
        if previous is not None and previous != selected:
            self.switch_count += 1
        self.previous_expert = previous
        self.current_expert = selected
        self.assignment_history.append(selected)

    def update(self, num_steps: int = 1) -> dict[str, float]:
        if num_steps < 0:
            raise ValueError("num_steps must be non-negative")
        if not self._dirty_experts or num_steps == 0:
            return {"gp_loss": 0.0, "gp_updates": 0.0}
        losses = []
        updates = 0.0
        dirty = sorted(self._dirty_experts)
        self._dirty_experts.clear()
        for index in dirty:
            if index >= len(self.experts):
                continue
            metrics = self.experts[index].fit(min(num_steps, self.gp_fit_steps))
            losses.append(metrics["gp_loss"])
            updates += metrics["gp_updates"]
        self.last_fit_metrics = {
            "gp_loss": sum(losses) / len(losses) if losses else 0.0,
            "gp_updates": updates,
        }
        if self.merge_enabled:
            self._maybe_merge_current()
        return dict(self.last_fit_metrics)

    def predict(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if self.current_expert is None:
            raise RuntimeError("GPMM has no expert; observe a transition before predict")
        prediction = self.experts[self.current_expert].predict_next(obs, action)
        return prediction.to(device=obs.device, dtype=obs.dtype)

    def predict_distribution(self, obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.current_expert is None:
            raise RuntimeError("GPMM has no expert; observe a transition before predict")
        mean, variance = self.experts[self.current_expert].predict_distribution(obs, action)
        return mean, variance

    def diagnostics(self) -> dict[str, float | int | list[int]]:
        selected_log_likelihood = 0.0
        assignment_entropy = 0.0
        assignment_margin = 0.0
        new_posterior = 0.0
        if self.last_assignment is not None:
            assignment_entropy = self.last_assignment.entropy
            assignment_margin = self.last_assignment.margin
            new_posterior = self.last_assignment.new_component_posterior
            selected_log_likelihood = float(self.last_assignment.log_scores[self.last_assignment.selected_index])
        return {
            "num_experts": self.num_experts,
            "current_expert_id": -1 if self.current_expert is None else self.current_expert,
            "expert_switch_count": self.switch_count,
            "new_expert_count": self.new_expert_count,
            "merge_count": self.merge_count,
            "prune_count": self.prune_count,
            "next_seed": self._next_seed,
            "assignment_entropy": assignment_entropy,
            "assignment_margin": assignment_margin,
            "selected_log_likelihood": selected_log_likelihood,
            "new_component_posterior": new_posterior,
            "points_per_expert": [expert.num_points for expert in self.experts],
        }

    def state_dict(self) -> dict:
        return {
            "experts": [expert.state_dict() for expert in self.experts],
            "current_expert": self.current_expert,
            "previous_expert": self.previous_expert,
            "transition_counts": self.transition_counts.clone(),
            "assignment_history": list(self.assignment_history),
            "expert_birth_steps": list(self.expert_birth_steps),
            "expert_probationary": list(self.expert_probationary),
            "global_step": self.global_step,
            "switch_count": self.switch_count,
            "new_expert_count": self.new_expert_count,
            "merge_count": self.merge_count,
            "prune_count": self.prune_count,
        }

    def load_state_dict(self, state: dict) -> None:
        self.experts = []
        for _ in state["experts"]:
            self.experts.append(self._new_expert())
        for expert, expert_state in zip(self.experts, state["experts"]):
            expert.load_state_dict(expert_state)
        self.current_expert = state["current_expert"]
        self.previous_expert = state["previous_expert"]
        self.transition_counts = state["transition_counts"].clone().double()
        self.assignment_history = list(state["assignment_history"])
        self.expert_birth_steps = list(state["expert_birth_steps"])
        self.expert_probationary = list(state["expert_probationary"])
        self.global_step = int(state["global_step"])
        self.switch_count = int(state["switch_count"])
        self.new_expert_count = int(state["new_expert_count"])
        self.merge_count = int(state["merge_count"])
        self.prune_count = int(state["prune_count"])
        self._next_seed = int(state.get("next_seed", len(self.experts)))
        self._dirty_experts.clear()

    def _new_expert(self) -> GPExpert:
        expert = GPExpert(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            max_points=self.max_points_per_expert,
            learning_rate=self.gp_learning_rate,
            min_predictive_variance=self.min_predictive_variance,
            max_predictive_variance=self.max_predictive_variance,
            prior_variance=self.prior_variance,
            observation_noise=self.observation_noise,
            seed=self._next_seed,
        )
        self._next_seed += 1
        return expert

    def _create_expert(self) -> int:
        self.experts.append(self._new_expert())
        self.expert_birth_steps.append(self.global_step)
        self.expert_probationary.append(True)
        self.new_expert_count += 1
        old_size = self.transition_counts.shape[0]
        expanded = torch.zeros(old_size + 1, old_size + 1, dtype=torch.float64)
        if old_size:
            expanded[:old_size, :old_size] = self.transition_counts
        self.transition_counts = expanded
        return old_size

    def _update_transition_statistics(self, previous: int | None, selected: int) -> None:
        if previous is not None:
            self.transition_counts[previous, selected] += 1.0

    def _maybe_merge_current(self) -> None:
        if self.current_expert is None or self.current_expert >= len(self.experts):
            return
        source_index = self.current_expert
        if not self.expert_probationary[source_index] or self.experts[source_index].num_points < self.merge_burnin_points:
            return
        target_index, distance = closest_expert(self.experts[source_index], self.experts, source_index)
        if target_index is None or distance >= self.merge_threshold:
            self.expert_probationary[source_index] = False
            return
        self._merge_experts(source_index, target_index)

    def _merge_experts(self, source_index: int, target_index: int) -> None:
        if source_index == target_index:
            return
        if not (0 <= source_index < len(self.experts) and 0 <= target_index < len(self.experts)):
            raise IndexError("expert merge indices are out of range")
        destination = target_index - int(target_index > source_index)
        self.experts[target_index].merge_from(self.experts[source_index])
        self.experts[target_index].fit(self.gp_fit_steps)
        self.experts.pop(source_index)
        self.expert_birth_steps.pop(source_index)
        self.expert_probationary.pop(source_index)
        self.assignment_history = [
            destination if value == source_index else value - int(value > source_index)
            for value in self.assignment_history
        ]
        if self.current_expert == source_index:
            self.current_expert = destination
        elif self.current_expert is not None and self.current_expert > source_index:
            self.current_expert -= 1
        if self.previous_expert == source_index:
            self.previous_expert = destination
        elif self.previous_expert is not None and self.previous_expert > source_index:
            self.previous_expert -= 1
        counts = self.transition_counts
        keep = [index for index in range(counts.shape[0]) if index != source_index]
        reduced = counts[keep][:, keep].clone()
        source_row = counts[source_index, keep]
        source_col = counts[keep, source_index]
        reduced[destination, :] += source_row
        reduced[:, destination] += source_col
        self.transition_counts = reduced
        self.merge_count += 1
