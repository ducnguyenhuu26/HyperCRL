"""Small symmetric-KL merge/prune helpers for GPMM experts."""

import torch

from .expert import GPExpert


def symmetric_kl_diagonal(
    mean_p: torch.Tensor,
    variance_p: torch.Tensor,
    mean_q: torch.Tensor,
    variance_q: torch.Tensor,
) -> torch.Tensor:
    """Return elementwise symmetric KL for diagonal Gaussian predictions."""

    var_p = variance_p.clamp_min(1e-8)
    var_q = variance_q.clamp_min(1e-8)
    kl_pq = 0.5 * ((var_p / var_q) + (mean_q - mean_p).square() / var_q - 1 + (var_q / var_p).log())
    kl_qp = 0.5 * ((var_q / var_p) + (mean_p - mean_q).square() / var_p - 1 + (var_p / var_q).log())
    return 0.5 * (kl_pq + kl_qp)


def expert_distance(source: GPExpert, target: GPExpert) -> float:
    """Average predictive symmetric KL over the source expert's points."""

    if source.num_points == 0 or target.num_points == 0:
        return float("inf")
    inputs = source.training_inputs
    obs = inputs[:, : source.obs_dim]
    action = inputs[:, source.obs_dim :]
    source_mean, source_variance = source.predict_distribution(obs, action)
    target_mean, target_variance = target.predict_distribution(obs, action)
    return float(
        symmetric_kl_diagonal(source_mean, source_variance, target_mean, target_variance).mean().cpu()
    )


def closest_expert(source: GPExpert, experts: list[GPExpert], exclude: int) -> tuple[int | None, float]:
    candidates = [
        (index, expert_distance(source, expert))
        for index, expert in enumerate(experts)
        if index != exclude
    ]
    return min(candidates, key=lambda item: item[1]) if candidates else (None, float("inf"))
