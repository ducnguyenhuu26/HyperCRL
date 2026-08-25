"""Optional plotting helpers for development artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def plot_checkpoint_curves(records_by_variant: dict[str, list[dict[str, Any]]], output: str | Path) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7, 4))
    for variant, records in records_by_variant.items():
        x = [record["global_step"] for record in records]
        y = [float("nan") if record["r2_at_H"] is None else record["r2_at_H"] for record in records]
        axis.plot(x, y, marker="o", label=variant)
    axis.set_xlabel("interactions")
    axis.set_ylabel("R²@H (H=20)")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)
