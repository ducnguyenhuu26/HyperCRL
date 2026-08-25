"""Lifetime running normalization for real online transitions."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class RunningNormalizer:
    """Numerically stable vector normalizer using Welford moments."""

    size: int
    epsilon: float = 1e-6
    clip: float | None = 10.0

    def __post_init__(self) -> None:
        if self.size <= 0 or self.epsilon <= 0.0:
            raise ValueError("normalizer size must be positive and epsilon must be positive")
        self.count = 0
        self.mean = torch.zeros(self.size, dtype=torch.float32)
        self.m2 = torch.zeros(self.size, dtype=torch.float32)

    @property
    def variance(self) -> torch.Tensor:
        if self.count < 2:
            return torch.ones_like(self.mean)
        return (self.m2 / (self.count - 1)).clamp_min(0.0)

    def update(self, x: torch.Tensor) -> None:
        values = torch.as_tensor(x, dtype=torch.float32).detach().cpu()
        if values.shape[-1:] != (self.size,):
            raise ValueError(f"expected final dimension {self.size}, got {tuple(values.shape)}")
        for row in values.reshape(-1, self.size):
            self.count += 1
            delta = row - self.mean
            self.mean += delta / self.count
            self.m2 += delta * (row - self.mean)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        values = torch.as_tensor(x)
        if values.shape[-1:] != (self.size,):
            raise ValueError(f"expected final dimension {self.size}, got {tuple(values.shape)}")
        mean = self.mean.to(values.device, values.dtype)
        scale = torch.sqrt(self.variance.to(values.device, values.dtype) + self.epsilon)
        result = (values - mean) / scale
        return result.clamp(-self.clip, self.clip) if self.clip is not None else result

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        values = torch.as_tensor(x)
        if values.shape[-1:] != (self.size,):
            raise ValueError(f"expected final dimension {self.size}, got {tuple(values.shape)}")
        mean = self.mean.to(values.device, values.dtype)
        scale = torch.sqrt(self.variance.to(values.device, values.dtype) + self.epsilon)
        return values * scale + mean

    def state_dict(self) -> dict:
        return {"size": self.size, "epsilon": self.epsilon, "clip": self.clip, "count": self.count, "mean": self.mean.clone(), "m2": self.m2.clone()}

    def load_state_dict(self, state: dict) -> None:
        if int(state["size"]) != self.size:
            raise ValueError("normalizer size mismatch")
        count = int(state["count"])
        mean = torch.as_tensor(state["mean"], dtype=torch.float32)
        m2 = torch.as_tensor(state["m2"], dtype=torch.float32)
        if count < 0 or mean.shape != (self.size,) or m2.shape != (self.size,) or not torch.isfinite(mean).all() or not torch.isfinite(m2).all() or (m2 < 0).any():
            raise ValueError("invalid normalizer state")
        self.count, self.mean, self.m2 = count, mean.clone(), m2.clone()
