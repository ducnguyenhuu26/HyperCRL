"""Stable linear-Gaussian context inference for REF."""

import math

import torch


def _symmetrize(matrix: torch.Tensor) -> torch.Tensor:
    return 0.5 * (matrix + matrix.T)


def gaussian_context_posterior(
    basis: torch.Tensor,
    residual: torch.Tensor,
    prior_mean: torch.Tensor,
    prior_covariance: torch.Tensor,
    sigma: float,
    jitter: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return posterior mean/covariance without explicit matrix inverses."""

    if sigma <= 0.0 or jitter < 0.0:
        raise ValueError("sigma must be positive and jitter must be non-negative")
    if basis.ndim != 3 or residual.ndim != 2:
        raise ValueError("basis must be [N,state_dim,rank] and residual [N,state_dim]")
    rank = prior_mean.numel()
    if (
        basis.shape[:2] != residual.shape
        or basis.shape[2] != rank
        or prior_mean.ndim != 1
        or prior_covariance.shape != (rank, rank)
    ):
        raise ValueError("basis, residual, prior mean, and covariance shapes are inconsistent")
    eye = torch.eye(rank, dtype=basis.dtype, device=basis.device)
    prior_covariance = _symmetrize(prior_covariance) + jitter * eye
    prior_precision = torch.linalg.solve(prior_covariance, eye)
    precision = prior_precision + torch.einsum("ndr,ndk->rk", basis, basis) / (sigma**2)
    precision = _symmetrize(precision) + jitter * eye
    covariance = torch.linalg.solve(precision, eye)
    rhs = prior_precision @ prior_mean + torch.einsum("ndr,nd->r", basis, residual) / (sigma**2)
    mean = covariance @ rhs
    covariance = _symmetrize(covariance)
    if not torch.isfinite(mean).all() or not torch.isfinite(covariance).all():
        raise FloatingPointError("non-finite REF posterior")
    if torch.linalg.eigvalsh(covariance).min() < -1e-6:
        raise FloatingPointError("REF posterior covariance is not positive semidefinite")
    return mean, covariance


def log_marginal_evidence(
    basis: torch.Tensor,
    residual: torch.Tensor,
    prior_mean: torch.Tensor,
    prior_covariance: torch.Tensor,
    sigma: float,
    jitter: float = 1e-6,
) -> float:
    """Compute linear-Gaussian marginal evidence with stable solves/log-dets."""

    if sigma <= 0.0 or jitter < 0.0:
        raise ValueError("sigma must be positive and jitter must be non-negative")
    posterior_mean, posterior_covariance = gaussian_context_posterior(basis, residual, prior_mean, prior_covariance, sigma, jitter)
    rank = prior_mean.numel()
    eye = torch.eye(rank, dtype=basis.dtype, device=basis.device)
    prior_precision = torch.linalg.solve(_symmetrize(prior_covariance) + jitter * eye, eye)
    precision = torch.linalg.solve(posterior_covariance, eye)
    data_term = residual.square().sum() / (sigma**2)
    prior_term = prior_mean @ prior_precision @ prior_mean
    rhs = prior_precision @ prior_mean + torch.einsum("ndr,nd->r", basis, residual) / (sigma**2)
    posterior_term = rhs @ posterior_covariance @ rhs
    sign_prior, logdet_prior_precision = torch.linalg.slogdet(prior_precision)
    sign_precision, logdet_precision = torch.linalg.slogdet(precision)
    if sign_prior <= 0 or sign_precision <= 0:
        raise FloatingPointError("context precision is not positive definite")
    sample_count = residual.numel()
    value = -0.5 * (sample_count * math.log(2.0 * math.pi * sigma**2) + logdet_precision - logdet_prior_precision + data_term + prior_term - posterior_term)
    return float(value.detach())


def moment_match(means: torch.Tensor, covariances: torch.Tensor, weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if means.ndim != 2 or covariances.ndim != 3 or weights.ndim != 1:
        raise ValueError("mixture inputs must be [K,R], [K,R,R], and [K]")
    if not torch.isfinite(means).all() or not torch.isfinite(covariances).all() or not torch.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise FloatingPointError("invalid REF mixture state")
    weights = weights / weights.sum().clamp_min(1e-12)
    mean = torch.sum(weights[:, None] * means, dim=0)
    centered = means - mean
    covariance = torch.sum(weights[:, None, None] * (covariances + centered[:, :, None] * centered[:, None, :]), dim=0)
    covariance = _symmetrize(covariance)
    if torch.linalg.eigvalsh(covariance).min() < -1e-6:
        raise FloatingPointError("REF mixture covariance is not positive semidefinite")
    return mean, covariance
