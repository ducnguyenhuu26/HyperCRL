"""Frozen-probe world-model metrics with explicit degeneracy semantics."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .probes import DynamicsProbeBank


def _macro_r2(true: np.ndarray, predicted: np.ndarray, *, variance_epsilon: float = 1e-12) -> float | None:
    if true.ndim != 2 or predicted.shape != true.shape or true.shape[0] < 2:
        return None
    variance = np.var(true, axis=0)
    valid = variance > variance_epsilon
    if not np.any(valid):
        return None
    centered = true[:, valid] - np.mean(true[:, valid], axis=0)
    scores = 1.0 - np.sum((predicted[:, valid] - true[:, valid]) ** 2, axis=0) / np.sum(centered**2, axis=0)
    return float(np.mean(scores))


def _macro_nrmse(true: np.ndarray, predicted: np.ndarray, *, variance_epsilon: float = 1e-12, epsilon: float = 1e-8) -> float | None:
    if true.ndim != 2 or predicted.shape != true.shape or true.shape[0] < 2:
        return None
    std = np.std(true, axis=0)
    valid = std > variance_epsilon
    if not np.any(valid):
        return None
    rmse = np.sqrt(np.mean((predicted[:, valid] - true[:, valid]) ** 2, axis=0))
    return float(np.mean(rmse / (std[valid] + epsilon)))


def evaluate_probe_bank(learner: Any, bank: DynamicsProbeBank) -> dict[str, float | None]:
    """Evaluate recursive open-loop predictions without teacher forcing."""

    if len(bank.probes) < 2:
        return {"r2_at_1": None, "r2_at_H": None, "nrmse_at_H": None}
    device = getattr(learner, "device", torch.device("cpu"))
    if not isinstance(device, torch.device):
        device = torch.device(device)
    true = np.stack([probe.true_obs for probe in bank.probes]).astype(np.float32)
    obs = torch.as_tensor(
        np.stack([probe.initial_obs for probe in bank.probes]),
        dtype=torch.float32,
        device=device,
    )
    actions = torch.as_tensor(
        np.stack([probe.actions for probe in bank.probes]),
        dtype=torch.float32,
        device=device,
    )
    predictions = torch.empty(
        (len(bank.probes), bank.horizon, bank.obs_dim),
        dtype=obs.dtype,
        device=device,
    )
    for horizon_index in range(bank.horizon):
        obs = learner.predict(obs, actions[:, horizon_index])
        if obs.device != device:
            obs = obs.to(device)
        predictions[:, horizon_index] = obs
    predictions_np = predictions.detach().cpu().numpy()
    r2_by_horizon = [_macro_r2(true[:, index], predictions_np[:, index]) for index in range(bank.horizon)]
    nrmse_by_horizon = [_macro_nrmse(true[:, index], predictions_np[:, index]) for index in range(bank.horizon)]
    valid_r2 = [value for value in r2_by_horizon if value is not None]
    valid_nrmse = [value for value in nrmse_by_horizon if value is not None]
    return {
        "r2_at_1": r2_by_horizon[0],
        "r2_at_H": float(np.mean(valid_r2)) if valid_r2 else None,
        "nrmse_at_H": float(np.mean(valid_nrmse)) if valid_nrmse else None,
    }
