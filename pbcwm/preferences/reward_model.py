"""Bradley-Terry reward models and an ensemble for query uncertainty."""

from collections.abc import Sequence

import torch
from torch import nn

from .buffer import PreferenceBuffer
from .types import PreferenceExample, TrajectorySegment


class RewardMLP(nn.Module):
    """Predict a scalar reward from one observation-action pair."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        input_dim = obs_dim + action_dim
        for hidden_dim in hidden_dims:
            if hidden_dim <= 0:
                raise ValueError("hidden dimensions must be positive")
            layers.extend((nn.Linear(input_dim, hidden_dim), nn.ReLU()))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat((obs, action), dim=-1)).squeeze(-1)


class PreferenceRewardEnsemble:
    """Independent Bradley-Terry reward models trained on pairwise labels."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        ensemble_size: int = 3,
        hidden_dims: Sequence[int] = (256, 256),
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        device: str | torch.device = "cpu",
        seed: int | None = None,
    ) -> None:
        if ensemble_size <= 0 or batch_size <= 0 or learning_rate <= 0:
            raise ValueError("ensemble_size, batch_size, and learning_rate must be positive")
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        torch_generator = torch.Generator(device="cpu")
        if seed is None:
            torch_generator.seed()
        else:
            torch_generator.manual_seed(seed)
        self.models = nn.ModuleList()
        self.optimizers: list[torch.optim.Optimizer] = []
        for _ in range(ensemble_size):
            with torch.random.fork_rng(devices=[]):
                model = RewardMLP(obs_dim, action_dim, hidden_dims)
                self._initialize_with_generator(model, torch_generator)
            model.to(self.device)
            self.models.append(model)
            self.optimizers.append(torch.optim.Adam(model.parameters(), lr=learning_rate))

    @staticmethod
    def _initialize_with_generator(model: nn.Module, generator: torch.Generator) -> None:
        for module in model.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, generator=generator)
                nn.init.zeros_(module.bias)

    @property
    def ensemble_size(self) -> int:
        return len(self.models)

    def reward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Return the ensemble-mean reward for batched planner inputs."""

        obs_device = obs.to(self.device)
        action_device = action.to(self.device)
        with torch.no_grad():
            rewards = torch.stack([model(obs_device, action_device) for model in self.models])
        return rewards.mean(dim=0)

    def trajectory_returns(self, trajectories: Sequence[TrajectorySegment]) -> torch.Tensor:
        """Return ensemble-mean scores, one scalar per trajectory."""

        if not trajectories:
            return torch.empty(0, device=self.device)
        obs, actions = self._stack_trajectories(trajectories)
        with torch.no_grad():
            per_model = []
            for model in self.models:
                rewards = model(obs, actions).reshape(len(trajectories), -1)
                per_model.append(rewards.sum(dim=1))
            return torch.stack(per_model).mean(dim=0)

    def preference_probabilities(
        self,
        traj_a: TrajectorySegment,
        traj_b: TrajectorySegment,
    ) -> torch.Tensor:
        """Return ``P(A preferred)`` for every ensemble member."""

        obs_a, actions_a = self._stack_trajectories([traj_a])
        obs_b, actions_b = self._stack_trajectories([traj_b])
        with torch.no_grad():
            probabilities = []
            for model in self.models:
                score_a = model(obs_a, actions_a).sum()
                score_b = model(obs_b, actions_b).sum()
                probabilities.append(torch.sigmoid(score_a - score_b))
            return torch.stack(probabilities)

    def state_dict(self) -> dict:
        """Checkpoint model/optimizer state without touching global RNG."""

        return {
            "models": [model.state_dict() for model in self.models],
            "optimizers": [optimizer.state_dict() for optimizer in self.optimizers],
            "batch_size": self.batch_size,
        }

    def load_state_dict(self, state: dict) -> None:
        if len(state["models"]) != len(self.models) or len(state["optimizers"]) != len(self.optimizers):
            raise ValueError("reward ensemble size mismatch")
        for model, model_state, optimizer, optimizer_state in zip(self.models, state["models"], self.optimizers, state["optimizers"]):
            model.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)

    def update(self, preference_buffer: PreferenceBuffer, num_steps: int = 1) -> dict[str, float]:
        if num_steps < 0:
            raise ValueError("num_steps must be non-negative")
        if len(preference_buffer) < self.batch_size or num_steps == 0:
            return {
                "preference_loss": 0.0,
                "preference_accuracy": 0.0,
                "mean_abs_logit": 0.0,
                "reward_model_updates": 0.0,
            }

        losses: list[float] = []
        accuracies: list[float] = []
        mean_abs_logits: list[float] = []
        for _ in range(num_steps):
            examples = preference_buffer.sample(self.batch_size)
            labels = torch.tensor(
                [example.label for example in examples], dtype=torch.float32, device=self.device
            )
            batch_metrics = []
            for model, optimizer in zip(self.models, self.optimizers):
                logits = self._preference_logits(model, examples)
                loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                batch_metrics.append((loss.detach(), logits.detach()))

            losses.append(float(torch.stack([loss for loss, _ in batch_metrics]).mean().cpu()))
            ensemble_logits = torch.stack([logits for _, logits in batch_metrics]).mean(dim=0)
            predictions = (ensemble_logits >= 0).to(torch.float32)
            accuracies.append(float((predictions == labels).float().mean().cpu()))
            mean_abs_logits.append(float(ensemble_logits.abs().mean().cpu()))

        return {
            "preference_loss": sum(losses) / len(losses),
            "preference_accuracy": sum(accuracies) / len(accuracies),
            "mean_abs_logit": sum(mean_abs_logits) / len(mean_abs_logits),
            "reward_model_updates": float(num_steps),
        }

    def predict_preference_accuracy(self, examples: Sequence[PreferenceExample]) -> float:
        if not examples:
            return float("nan")
        correct = 0
        for example in examples:
            probabilities = self.preference_probabilities(example.traj_a, example.traj_b)
            predicted_label = int(probabilities.mean() < 0.5)
            correct += int(predicted_label == example.label)
        return correct / len(examples)

    def _preference_logits(
        self,
        model: RewardMLP,
        examples: Sequence[PreferenceExample],
    ) -> torch.Tensor:
        logits = []
        for example in examples:
            obs_a, actions_a = self._stack_trajectories([example.traj_a])
            obs_b, actions_b = self._stack_trajectories([example.traj_b])
            score_a = model(obs_a, actions_a).sum()
            score_b = model(obs_b, actions_b).sum()
            # Label 1 means B preferred, so the BCE logit is score(B)-score(A).
            logits.append(score_b - score_a)
        return torch.stack(logits)

    def _stack_trajectories(
        self,
        trajectories: Sequence[TrajectorySegment],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        horizons = {trajectory.obs.shape[0] for trajectory in trajectories}
        if len(horizons) != 1:
            raise ValueError("all trajectories in a batch must have the same horizon")
        obs = torch.stack([trajectory.obs for trajectory in trajectories]).to(self.device)
        actions = torch.stack([trajectory.actions for trajectory in trajectories]).to(self.device)
        return obs.reshape(-1, obs.shape[-1]), actions.reshape(-1, actions.shape[-1])
