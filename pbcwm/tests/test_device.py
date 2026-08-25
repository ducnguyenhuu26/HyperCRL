import torch

from pbcwm.core.device import configure_torch, move_batch, resolve_device


def test_auto_device_is_valid_and_cpu_transfer_stays_cpu() -> None:
    resolved = configure_torch("auto")
    assert resolved.type in {"cpu", "cuda"}
    batch = torch.ones(3, 2)
    moved = move_batch(batch, "cpu")
    assert moved.device.type == "cpu"
    assert torch.equal(moved, batch)


def test_explicit_cpu_resolution_is_stable() -> None:
    assert resolve_device("cpu") == torch.device("cpu")
