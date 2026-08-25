"""Device selection and CUDA performance policy shared by all runners."""

from __future__ import annotations

import torch


def resolve_device(requested: str | torch.device = "auto") -> torch.device:
    """Resolve ``auto`` to the first CUDA device, with an explicit CPU fallback.

    An explicit ``cuda`` request remains strict: PyTorch will report a missing
    CUDA runtime instead of silently running a supposedly GPU experiment on
    the CPU.  This keeps experiment provenance honest.
    """

    if isinstance(requested, torch.device):
        return requested
    value = str(requested).strip().lower()
    if value in {"auto", "best", "gpu"}:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def configure_torch(device: str | torch.device = "auto") -> torch.device:
    """Apply safe CUDA fast-path settings once and return the resolved device."""

    resolved = resolve_device(device)
    if resolved.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        torch.cuda.set_device(resolved.index if resolved.index is not None else 0)
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    return resolved


def is_cuda(device: str | torch.device) -> bool:
    """Return whether a resolved device is CUDA-backed."""

    return resolve_device(device).type == "cuda"


def move_batch(value: torch.Tensor, device: str | torch.device) -> torch.Tensor:
    """Move a CPU minibatch asynchronously when pinned CUDA transfer is usable."""

    target = resolve_device(device)
    if target.type == "cuda" and value.device.type == "cpu":
        if torch.cuda.is_available() and not value.is_pinned():
            value = value.pin_memory()
        return value.to(target, non_blocking=True)
    return value.to(target)
