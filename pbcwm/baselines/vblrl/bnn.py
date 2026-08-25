"""Compact reward-free variational Bayesian delta dynamics network."""

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


class BayesianLinear(nn.Module):
    """Bayesian linear layer with Normal(mu, softplus(rho)^2) parameters."""

    def __init__(self, in_features: int, out_features: int, prior_std: float = 1.0) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0 or prior_std <= 0:
            raise ValueError("Bayesian layer dimensions and prior_std must be positive")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_rho = nn.Parameter(torch.empty(out_features))
        self.reset_parameters(prior_std)

    def reset_parameters(self, prior_std: float) -> None:
        del prior_std  # The KL prior is explicit; initialization stays numerically stable.
        nn.init.xavier_uniform_(self.weight_mu, gain=0.5)
        nn.init.zeros_(self.bias_mu)
        initial_sigma = 0.05
        rho = torch.log(torch.expm1(torch.tensor(initial_sigma)))
        nn.init.constant_(self.weight_rho, float(rho))
        nn.init.constant_(self.bias_rho, float(rho))

    @property
    def weight_sigma(self) -> torch.Tensor:
        return F.softplus(self.weight_rho).clamp_min(1e-6)

    @property
    def bias_sigma(self) -> torch.Tensor:
        return F.softplus(self.bias_rho).clamp_min(1e-6)

    def sampled_parameters(self, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        if deterministic:
            return self.weight_mu, self.bias_mu
        return (
            self.weight_mu + self.weight_sigma * torch.randn_like(self.weight_mu),
            self.bias_mu + self.bias_sigma * torch.randn_like(self.bias_mu),
        )

    def forward(self, inputs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        weight, bias = self.sampled_parameters(deterministic)
        return F.linear(inputs, weight, bias)

    def kl_to_normal(self, prior_mu: float | torch.Tensor = 0.0, prior_sigma: float | torch.Tensor = 1.0) -> torch.Tensor:
        prior_mu_t = torch.as_tensor(prior_mu, dtype=self.weight_mu.dtype, device=self.weight_mu.device)
        prior_sigma_t = torch.as_tensor(prior_sigma, dtype=self.weight_mu.dtype, device=self.weight_mu.device)
        def kl(mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
            return (
                torch.log(prior_sigma_t / sigma)
                + (sigma.square() + (mu - prior_mu_t).square()) / (2.0 * prior_sigma_t.square())
                - 0.5
            ).sum()
        return kl(self.weight_mu, self.weight_sigma) + kl(self.bias_mu, self.bias_sigma)


class BayesianDynamicsNetwork(nn.Module):
    """Bayesian MLP predicting only mean/log-variance of state deltas."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
        min_logvar: float = -10.0,
        max_logvar: float = 2.0,
    ) -> None:
        super().__init__()
        if obs_dim <= 0 or action_dim <= 0:
            raise ValueError("obs_dim and action_dim must be positive")
        if not hidden_dims or any(int(size) <= 0 for size in hidden_dims):
            raise ValueError("hidden_dims must contain positive dimensions")
        if min_logvar >= max_logvar:
            raise ValueError("min_logvar must be smaller than max_logvar")
        dims = (int(obs_dim + action_dim), *[int(size) for size in hidden_dims])
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.min_logvar = float(min_logvar)
        self.max_logvar = float(max_logvar)
        self.hidden_layers = nn.ModuleList([
            BayesianLinear(left, right) for left, right in zip(dims[:-1], dims[1:])
        ])
        last_dim = dims[-1] if hidden_dims else dims[0]
        self.output_layer = BayesianLinear(last_dim, 2 * self.obs_dim)

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = torch.cat((obs, action), dim=-1)
        hidden = inputs
        for layer in self.hidden_layers:
            hidden = F.elu(layer(hidden, deterministic=deterministic))
        output = self.output_layer(hidden, deterministic=deterministic)
        mean, logvar = output.split(self.obs_dim, dim=-1)
        return mean, logvar.clamp(self.min_logvar, self.max_logvar)

    def kl_to_snapshot(self, snapshot: dict | None) -> torch.Tensor:
        if snapshot is None:
            return sum(layer.kl_to_normal() for layer in [*self.hidden_layers, self.output_layer])
        total = self.hidden_layers[0].weight_mu.new_zeros(())
        for layer, saved in zip([*self.hidden_layers, self.output_layer], snapshot["layers"]):
            total = total + self._kl_layer_snapshot(layer, saved)
        return total

    @staticmethod
    def _kl_layer_snapshot(layer: BayesianLinear, saved: dict) -> torch.Tensor:
        def kl(mu, sigma, prior_mu, prior_sigma):
            return (
                torch.log(prior_sigma / sigma)
                + (sigma.square() + (mu - prior_mu).square()) / (2.0 * prior_sigma.square())
                - 0.5
            ).sum()
        return kl(
            layer.weight_mu,
            layer.weight_sigma,
            saved["weight_mu"].to(layer.weight_mu),
            saved["weight_sigma"].to(layer.weight_mu),
        ) + kl(
            layer.bias_mu,
            layer.bias_sigma,
            saved["bias_mu"].to(layer.bias_mu),
            saved["bias_sigma"].to(layer.bias_mu),
        )

    def posterior_snapshot(self) -> dict:
        return {
            "layers": [
                {
                    "weight_mu": layer.weight_mu.detach().clone(),
                    "weight_sigma": layer.weight_sigma.detach().clone(),
                    "bias_mu": layer.bias_mu.detach().clone(),
                    "bias_sigma": layer.bias_sigma.detach().clone(),
                }
                for layer in [*self.hidden_layers, self.output_layer]
            ]
        }
