"""Raw-state, teacher-forcing-free world-model metrics."""

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch

from .common import MetricResult, RolloutProbeBatch, invalid_result


@dataclass(frozen=True)
class HorizonCurve:
    horizons: list[int]
    values: list[float]


def rollout_predictions(predictor: Any, probe: RolloutProbeBatch) -> torch.Tensor:
    """Recursively predict from the true initial state without teacher forcing."""

    states = [probe.initial_obs]
    current = probe.initial_obs
    with torch.no_grad():
        for horizon in range(probe.horizon):
            action = probe.actions[:, horizon]
            if hasattr(predictor, "predict"):
                current = predictor.predict(current, action)
            else:
                current = predictor(current, action)
            current = current.detach()
            states.append(current)
    return torch.stack(states, dim=1)


def _errors_at_h(predictor: Any, probe: RolloutProbeBatch, horizon: int) -> tuple[torch.Tensor, torch.Tensor]:
    if horizon < 1 or horizon > probe.horizon:
        raise ValueError("horizon must be inside the probe horizon")
    predicted = rollout_predictions(predictor, probe)
    return probe.true_states[:, horizon], predicted[:, horizon]


def r2_at_h(
    predictor: Any, probe: RolloutProbeBatch, horizon: int, *, min_variance: float = 1e-12
) -> MetricResult:
    name = f"wm/r2_h{horizon}"
    try:
        true, predicted = _errors_at_h(predictor, probe, horizon)
    except (ValueError, RuntimeError) as exc:
        return invalid_result(name, True, str(exc))
    true = true.detach().cpu().double().numpy()
    predicted = predicted.detach().cpu().double().numpy()
    variance = np.var(true, axis=0)
    valid_dims = variance >= min_variance
    if not np.any(valid_dims):
        return invalid_result(name, True, "all state dimensions are numerically degenerate", valid_dimension_count=0)
    sse = np.sum((true - predicted) ** 2, axis=0)
    sst = np.sum((true - np.mean(true, axis=0)) ** 2, axis=0)
    values = 1.0 - sse[valid_dims] / sst[valid_dims]
    return MetricResult(name, float(np.mean(values)), True, metadata={"horizon": horizon, "valid_dimension_count": int(valid_dims.sum()), "sample_count": int(true.shape[0])})


def r2_at_horizon(predictor: Any, probe: RolloutProbeBatch, horizon: int, **kwargs: Any) -> MetricResult:
    return r2_at_h(predictor, probe, horizon, **kwargs)


def r2_horizon_curve(predictor: Any, probe: RolloutProbeBatch, *, min_variance: float = 1e-12) -> HorizonCurve:
    values = [r2_at_h(predictor, probe, horizon, min_variance=min_variance).value for horizon in range(1, probe.horizon + 1)]
    return HorizonCurve(list(range(1, probe.horizon + 1)), [float("nan") if value is None else float(value) for value in values])


def r2_at_H(predictor: Any, probe: RolloutProbeBatch, H: int, **kwargs: Any) -> MetricResult:
    """Macro-average R² over horizons 1..H."""

    name = "wm/r2_h"
    if H < 1 or H > probe.horizon:
        return invalid_result(name, True, "H must be inside the probe horizon", horizon=H)
    results = [r2_at_h(predictor, probe, horizon, **kwargs) for horizon in range(1, H + 1)]
    valid = [result.value for result in results if result.valid and result.value is not None]
    if len(valid) != H:
        return invalid_result(name, True, "one or more horizons are undefined", horizon=H)
    return MetricResult(name, float(np.mean(valid)), True, metadata={"horizon": H, "sample_count": int(probe.initial_obs.shape[0])})


def nrmse_at_h(
    predictor: Any, probe: RolloutProbeBatch, horizon: int, *, eps: float = 1e-8, min_variance: float = 1e-12
) -> MetricResult:
    name = f"wm/nrmse_h{horizon}"
    try:
        true, predicted = _errors_at_h(predictor, probe, horizon)
    except (ValueError, RuntimeError) as exc:
        return invalid_result(name, False, str(exc))
    true_np = true.detach().cpu().double().numpy()
    predicted_np = predicted.detach().cpu().double().numpy()
    std = np.std(true_np, axis=0)
    valid_dims = std**2 >= min_variance
    if not np.any(valid_dims):
        return invalid_result(name, False, "all state dimensions are numerically degenerate", valid_dimension_count=0)
    rmse = np.sqrt(np.mean((predicted_np - true_np) ** 2, axis=0))
    value = float(np.mean(rmse[valid_dims] / (std[valid_dims] + eps)))
    return MetricResult(name, value, False, metadata={"horizon": horizon, "valid_dimension_count": int(valid_dims.sum()), "sample_count": int(true_np.shape[0])})


def nrmse_at_H(predictor: Any, probe: RolloutProbeBatch, H: int, **kwargs: Any) -> MetricResult:
    name = "wm/nrmse_h"
    if H < 1 or H > probe.horizon:
        return invalid_result(name, False, "H must be inside the probe horizon", horizon=H)
    results = [nrmse_at_h(predictor, probe, horizon, **kwargs) for horizon in range(1, H + 1)]
    valid = [result.value for result in results if result.valid and result.value is not None]
    if len(valid) != H:
        return invalid_result(name, False, "one or more horizons are undefined", horizon=H)
    return MetricResult(name, float(np.mean(valid)), False, metadata={"horizon": H})


def one_step_raw_errors(predictor: Any, probe: RolloutProbeBatch) -> dict[str, MetricResult]:
    true, predicted = _errors_at_h(predictor, probe, 1)
    mse = float(torch.mean((predicted - true).square()).detach().cpu())
    return {"wm/mse_h1": MetricResult("wm/mse_h1", mse, False), "wm/rmse_h1": MetricResult("wm/rmse_h1", float(np.sqrt(mse)), False)}
