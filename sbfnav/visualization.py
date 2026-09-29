"""Diagnostic visualizations for planning states and belief targets."""

from __future__ import annotations

from pathlib import Path
import textwrap
from typing import Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .planning_state import CHANNEL_NAMES, PlanningState


def save_planning_state(
    state: PlanningState,
    output_path: str | Path,
    *,
    field: np.ndarray | None = None,
    title: str | None = None,
    metadata: Mapping[str, object] | None = None,
) -> None:
    """Save all nine channels plus an optional 28×28 target/prediction."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(3, 4, figsize=(16, 12), constrained_layout=True)
    axes = axes.ravel()
    rgb = np.moveaxis(state.values[:3], 0, -1)
    axes[0].imshow(np.clip(rgb, 0, 1))
    axes[0].set_title("accumulated observed RGB")
    for index in range(3, 9):
        axis = axes[index - 2]
        image = axis.imshow(state.values[index], cmap="viridis", vmin=0, vmax=1)
        axis.set_title(f"ch{index}: {CHANNEL_NAMES[index]}")
        figure.colorbar(image, ax=axis, fraction=0.046)
    axes[7].imshow(state.valid_canvas, cmap="gray", vmin=0, vmax=1)
    axes[7].set_title("valid canvas mask")
    if field is not None:
        image = axes[8].imshow(np.asarray(field), cmap="magma")
        axes[8].set_title("belief field / supervision")
        figure.colorbar(image, ax=axes[8], fraction=0.046)
    else:
        axes[8].axis("off")
    axes[9].imshow(state.values[4] + 2 * state.values[5], cmap="plasma")
    axes[9].set_title("landmarks (referenced highlighted)")
    axes[10].imshow(state.values[7] + state.values[8], cmap="cividis")
    axes[10].set_title("exploration + trajectory")
    axes[11].axis("off")
    if metadata:
        metadata_lines = []
        for key, value in metadata.items():
            metadata_lines.extend(
                textwrap.wrap(
                    f"{key}: {value}",
                    width=52,
                    subsequent_indent="  ",
                    break_long_words=False,
                )
            )
        axes[11].text(
            0,
            1,
            "\n".join(metadata_lines),
            va="top",
            family="monospace",
            fontsize=7,
        )
    for axis in axes[:11]:
        axis.set_xticks([])
        axis.set_yticks([])
    if title:
        figure.suptitle(title)
    figure.savefig(output_path, dpi=140)
    plt.close(figure)
