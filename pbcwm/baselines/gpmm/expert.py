"""Modern, bounded-memory exact GP expert for one dynamics component."""

from collections.abc import Sequence

import gpytorch
import torch
from torch import nn


class _ScalarExactGP(gpytorch.models.ExactGP):
    def __init__(self, input_dim: int, likelihood: gpytorch.likelihoods.GaussianLikelihood):
        super().__init__(None, None, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=input_dim)
        )

    def forward(self, inputs: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        mean = self.mean_module(inputs)
        covariance = self.covar_module(inputs)
        return gpytorch.distributions.MultivariateNormal(mean, covariance)


class GPExpert:
    """Independent scalar ExactGPs predicting each observation delta dimension."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        max_points: int = 256,
        learning_rate: float = 0.05,
        min_predictive_variance: float = 1e-5,
        max_predictive_variance: float = 1e3,
        prior_variance: float = 1.0,
        observation_noise: float = 0.05,
        seed: int | None = None,
    ) -> None:
        if obs_dim <= 0 or action_dim <= 0 or max_points <= 0:
            raise ValueError("dimensions and max_points must be positive")
        if learning_rate <= 0 or prior_variance <= 0 or observation_noise <= 0:
            raise ValueError("learning_rate, prior_variance, and observation_noise must be positive")
        if min_predictive_variance <= 0 or max_predictive_variance < min_predictive_variance:
            raise ValueError("invalid predictive variance bounds")
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.input_dim = self.obs_dim + self.action_dim
        self.max_points = int(max_points)
        self.learning_rate = float(learning_rate)
        self.min_predictive_variance = float(min_predictive_variance)
        self.max_predictive_variance = float(max_predictive_variance)
        self.prior_variance = float(prior_variance)
        self.observation_noise = float(observation_noise)
        self._rng = torch.Generator(device="cpu")
        if seed is not None:
            self._rng.manual_seed(seed)

        self._inputs = torch.empty(0, self.input_dim, dtype=torch.float64)
        self._targets = torch.empty(0, self.obs_dim, dtype=torch.float64)
        self.likelihoods = []
        self.models = nn.ModuleList()
        self.optimizers: list[torch.optim.Optimizer] = []
        self.mlls = []
        for _ in range(self.obs_dim):
            likelihood = gpytorch.likelihoods.GaussianLikelihood(
                noise_constraint=gpytorch.constraints.GreaterThan(self.min_predictive_variance)
            ).double()
            likelihood.noise = self.observation_noise
            model = _ScalarExactGP(self.input_dim, likelihood).double()
            model.covar_module.base_kernel.lengthscale = 1.0
            model.covar_module.outputscale = self.prior_variance
            self.likelihoods.append(likelihood)
            self.models.append(model)
            self.optimizers.append(torch.optim.Adam(model.parameters(), lr=self.learning_rate))
            self.mlls.append(gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model))

    @property
    def num_points(self) -> int:
        return int(self._inputs.shape[0])

    @property
    def training_inputs(self) -> torch.Tensor:
        return self._inputs.clone()

    @property
    def training_targets(self) -> torch.Tensor:
        return self._targets.clone()

    def add_transition(self, obs, action, next_obs) -> None:
        obs_t = torch.as_tensor(obs, dtype=torch.float64).flatten()
        action_t = torch.as_tensor(action, dtype=torch.float64).flatten()
        next_obs_t = torch.as_tensor(next_obs, dtype=torch.float64).flatten()
        if obs_t.numel() != self.obs_dim or action_t.numel() != self.action_dim:
            raise ValueError("transition dimensions do not match the GP expert")
        input_point = torch.cat((obs_t, action_t)).unsqueeze(0)
        target_point = (next_obs_t - obs_t).unsqueeze(0)
        self._replace_data(torch.cat((self._inputs, input_point)), torch.cat((self._targets, target_point)))

    def fit(self, num_steps: int) -> dict[str, float]:
        if num_steps < 0:
            raise ValueError("num_steps must be non-negative")
        if self.num_points == 0 or num_steps == 0:
            return {"gp_loss": 0.0, "gp_updates": 0.0}
        self._set_train_data()
        losses: list[float] = []
        for _ in range(num_steps):
            for output_index, (model, likelihood, optimizer, mll) in enumerate(
                zip(self.models, self.likelihoods, self.optimizers, self.mlls)
            ):
                model.train()
                likelihood.train()
                optimizer.zero_grad(set_to_none=True)
                try:
                    with gpytorch.settings.cholesky_jitter(double_value=1e-6):
                        output = model(self._inputs)
                        loss = -mll(output, self._targets[:, output_index])
                    loss.backward()
                    optimizer.step()
                except RuntimeError as error:
                    optimizer.zero_grad(set_to_none=True)
                    try:
                        with gpytorch.settings.cholesky_jitter(double_value=1e-4):
                            output = model(self._inputs)
                            target = self._targets[:, output_index]
                            loss = -mll(output, target)
                        loss.backward()
                        optimizer.step()
                    except RuntimeError as retry_error:
                        raise RuntimeError("GPExpert fit failed after jitter retry") from retry_error
                losses.append(float(loss.detach().cpu()))
        self._set_eval_mode()
        return {"gp_loss": sum(losses) / len(losses), "gp_updates": float(len(losses))}

    def predict_distribution(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        obs_t, action_t = self._validate_batch(obs, action)
        if self.num_points == 0:
            mean = torch.zeros(obs_t.shape[0], self.obs_dim, dtype=torch.float64)
            variance = torch.full_like(mean, self.prior_variance + self.observation_noise)
            return mean, variance
        self._set_eval_mode()
        inputs = torch.cat((obs_t, action_t), dim=-1)
        means = []
        variances = []
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            for model, likelihood in zip(self.models, self.likelihoods):
                prediction = likelihood(model(inputs))
                means.append(prediction.mean)
                variances.append(
                    prediction.variance.clamp(self.min_predictive_variance, self.max_predictive_variance)
                )
        return torch.stack(means, dim=-1), torch.stack(variances, dim=-1)

    def predict_next(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        mean, _ = self.predict_distribution(obs, action)
        return obs.to(dtype=torch.float64) + mean

    def log_likelihood(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        next_obs: torch.Tensor,
    ) -> torch.Tensor:
        obs_t, action_t = self._validate_batch(obs, action)
        target = torch.as_tensor(next_obs, dtype=torch.float64) - obs_t
        mean, variance = self.predict_distribution(obs_t, action_t)
        distribution = torch.distributions.Normal(mean, variance.sqrt())
        return distribution.log_prob(target).sum(dim=-1)

    def prior_predictive_log_likelihood(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        next_obs: torch.Tensor,
    ) -> torch.Tensor:
        obs_t, _ = self._validate_batch(obs, action)
        target = torch.as_tensor(next_obs, dtype=torch.float64) - obs_t
        variance = self.prior_variance + self.observation_noise
        distribution = torch.distributions.Normal(
            torch.zeros_like(target), torch.full_like(target, variance).sqrt()
        )
        return distribution.log_prob(target).sum(dim=-1)

    def merge_from(self, other: "GPExpert") -> None:
        if (self.obs_dim, self.action_dim) != (other.obs_dim, other.action_dim):
            raise ValueError("cannot merge experts with different dimensions")
        self._replace_data(
            torch.cat((self._inputs, other._inputs)),
            torch.cat((self._targets, other._targets)),
        )

    def state_dict(self) -> dict:
        return {
            "inputs": self._inputs.clone(),
            "targets": self._targets.clone(),
            "models": [model.state_dict() for model in self.models],
            "likelihoods": [likelihood.state_dict() for likelihood in self.likelihoods],
            "optimizers": [optimizer.state_dict() for optimizer in self.optimizers],
        }

    def load_state_dict(self, state: dict) -> None:
        self._inputs = state["inputs"].clone().double()
        self._targets = state["targets"].clone().double()
        if self.num_points:
            self._set_train_data()
        for model, model_state in zip(self.models, state["models"]):
            model.load_state_dict(model_state)
        for likelihood, likelihood_state in zip(self.likelihoods, state["likelihoods"]):
            likelihood.load_state_dict(likelihood_state)
        for optimizer, optimizer_state in zip(self.optimizers, state["optimizers"]):
            optimizer.load_state_dict(optimizer_state)
        self._set_eval_mode()

    def _replace_data(self, inputs: torch.Tensor, targets: torch.Tensor) -> None:
        inputs = inputs.double()
        targets = targets.double()
        if inputs.shape[0] > self.max_points:
            recent_count = max(1, self.max_points // 2)
            recent_start = inputs.shape[0] - recent_count
            recent_inputs = inputs[recent_start:]
            recent_targets = targets[recent_start:]
            historical_count = self.max_points - recent_count
            historical_inputs = inputs[:recent_start]
            historical_targets = targets[:recent_start]
            if historical_count and historical_inputs.shape[0]:
                indices = torch.randperm(historical_inputs.shape[0], generator=self._rng)[:historical_count]
                inputs = torch.cat((historical_inputs[indices], recent_inputs))
                targets = torch.cat((historical_targets[indices], recent_targets))
            else:
                inputs, targets = recent_inputs, recent_targets
        self._inputs = inputs.contiguous()
        self._targets = targets.contiguous()

    def _set_train_data(self) -> None:
        for index, model in enumerate(self.models):
            model.set_train_data(self._inputs, self._targets[:, index], strict=False)

    def _set_eval_mode(self) -> None:
        for model, likelihood in zip(self.models, self.likelihoods):
            model.eval()
            likelihood.eval()

    def _validate_batch(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        obs_t = torch.as_tensor(obs, dtype=torch.float64)
        action_t = torch.as_tensor(action, dtype=torch.float64)
        if obs_t.ndim != 2 or action_t.ndim != 2:
            raise ValueError("obs and action must have shape [batch, feature]")
        if obs_t.shape[0] != action_t.shape[0]:
            raise ValueError("obs and action must have the same batch size")
        return obs_t, action_t
