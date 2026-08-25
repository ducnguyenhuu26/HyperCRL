"""Functional delta-state target network used by HyperCRL-Adapt."""

from collections.abc import Sequence

import torch
from torch.nn import functional as F


class TargetDynamics:
    """An MLP whose weights are supplied by a hypernetwork at call time."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
    ) -> None:
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("input_dim and output_dim must be positive")
        if any(int(size) <= 0 for size in hidden_dims):
            raise ValueError("target hidden dimensions must be positive")
        dims = (int(input_dim), *[int(size) for size in hidden_dims], int(output_dim))
        shapes: list[tuple[int, ...]] = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            shapes.extend(((out_dim, in_dim), (out_dim,)))
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden_dims = tuple(int(size) for size in hidden_dims)
        self.parameter_shapes = tuple(shapes)

    def predict_delta(self, obs: torch.Tensor, action: torch.Tensor, weights: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(weights) != len(self.parameter_shapes):
            raise ValueError("generated weights do not match target dynamics")
        x = torch.cat((obs, action), dim=-1)
        if x.shape[-1] != self.input_dim:
            raise ValueError("observation/action dimensions do not match target dynamics")
        for layer_index in range(0, len(weights), 2):
            weight, bias = weights[layer_index], weights[layer_index + 1]
            if tuple(weight.shape) != self.parameter_shapes[layer_index]:
                raise ValueError("generated weight shape does not match target dynamics")
            x = F.linear(x.to(weight.device, weight.dtype), weight, bias)
            if layer_index + 2 < len(weights):
                x = F.relu(x)
        return x

    def predict_next(self, obs: torch.Tensor, action: torch.Tensor, weights: Sequence[torch.Tensor]) -> torch.Tensor:
        delta = self.predict_delta(obs, action, weights)
        return obs.to(delta.device, delta.dtype) + delta
