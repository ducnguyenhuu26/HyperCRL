"""Small shared hypernetwork that generates functional target MLP weights."""

from collections.abc import Sequence
from math import prod

import torch
from torch import nn


class HyperNetwork(nn.Module):
    """Map one regime embedding to all parameters of a target MLP."""

    def __init__(
        self,
        embedding_dim: int,
        target_shapes: Sequence[Sequence[int]],
        hidden_dims: Sequence[int] = (128, 128),
    ) -> None:
        super().__init__()
        if embedding_dim <= 0 or not target_shapes:
            raise ValueError("embedding_dim and target_shapes must be non-empty")
        if any(not shape or any(int(size) <= 0 for size in shape) for shape in target_shapes):
            raise ValueError("target parameter shapes must contain positive dimensions")
        if any(int(size) <= 0 for size in hidden_dims):
            raise ValueError("hypernetwork hidden dimensions must be positive")

        self.embedding_dim = int(embedding_dim)
        self.target_shapes = tuple(tuple(int(size) for size in shape) for shape in target_shapes)
        self.target_sizes = tuple(prod(shape) for shape in self.target_shapes)
        layers: list[nn.Module] = []
        input_dim = self.embedding_dim
        for hidden_dim in hidden_dims:
            layers.extend((nn.Linear(input_dim, int(hidden_dim)), nn.ReLU()))
            input_dim = int(hidden_dim)
        layers.append(nn.Linear(input_dim, sum(self.target_sizes)))
        self.network = nn.Sequential(*layers)

    @property
    def output_size(self) -> int:
        return sum(self.target_sizes)

    def forward(self, embedding: torch.Tensor) -> list[torch.Tensor]:
        embedding = torch.as_tensor(embedding, dtype=self.network[0].weight.dtype)
        if embedding.ndim not in (1, 2) or embedding.shape[-1] != self.embedding_dim:
            raise ValueError("embedding must have shape [embedding_dim] or [batch, embedding_dim]")
        flat = self.network(embedding)
        if embedding.ndim == 1:
            chunks = torch.split(flat, self.target_sizes, dim=-1)
            return [chunk.reshape(shape) for chunk, shape in zip(chunks, self.target_shapes)]
        return [chunk.reshape(embedding.shape[0], *shape) for chunk, shape in zip(
            torch.split(flat, self.target_sizes, dim=-1), self.target_shapes
        )]

    def parameter_shapes(self) -> tuple[tuple[int, ...], ...]:
        return self.target_shapes
