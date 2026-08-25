"""HyperCRL output-space retention targets and drift diagnostics."""

from collections.abc import Iterable, Sequence

import torch

from .hypernetwork import HyperNetwork


def snapshot_output_targets(
    hypernetwork: HyperNetwork,
    embeddings: Sequence[torch.Tensor],
    inactive_ids: Iterable[int],
) -> dict[int, list[torch.Tensor]]:
    """Capture detached generated weights before activating another embedding."""

    targets: dict[int, list[torch.Tensor]] = {}
    with torch.no_grad():
        for embedding_id in inactive_ids:
            targets[int(embedding_id)] = [
                weight.detach().clone() for weight in hypernetwork(embeddings[int(embedding_id)])
            ]
    return targets


def output_space_regularizer(
    hypernetwork: HyperNetwork,
    embeddings: Sequence[torch.Tensor],
    targets: dict[int, list[torch.Tensor]],
) -> torch.Tensor:
    """Average squared drift of generated weights for inactive embeddings."""

    if not targets:
        return next(hypernetwork.parameters()).new_zeros(())
    penalties = []
    for embedding_id, target_weights in targets.items():
        current_weights = hypernetwork(embeddings[embedding_id])
        penalties.append(torch.stack([
            (current - target).square().mean()
            for current, target in zip(current_weights, target_weights)
        ]).mean())
    return torch.stack(penalties).mean()


def normalized_output_drift(
    hypernetwork: HyperNetwork,
    embeddings: Sequence[torch.Tensor],
    targets: dict[int, list[torch.Tensor]],
) -> dict[int, float]:
    """Return normalized generated-weight drift for each protected embedding."""

    drift: dict[int, float] = {}
    with torch.no_grad():
        for embedding_id, target_weights in targets.items():
            current_weights = hypernetwork(embeddings[embedding_id])
            numerator = torch.cat([
                (current - target).reshape(-1) for current, target in zip(current_weights, target_weights)
            ]).norm()
            denominator = torch.cat([target.reshape(-1) for target in target_weights]).norm().clamp_min(1e-8)
            drift[int(embedding_id)] = float((numerator / denominator).cpu())
    return drift
