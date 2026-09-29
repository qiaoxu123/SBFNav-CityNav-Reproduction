#!/usr/bin/env python3
"""Render representative successful and failed closed-loop SBFNav episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sbfnav.dataset import CityNavDataset, CityReferCatalog
from sbfnav.planning_state import CHANNEL_NAMES, PlanningStateBuilder


def select_cases(predictions: list[dict], count: int) -> list[tuple[str, dict]]:
    successes = [item for item in predictions if item["benchmark_2d"]["success"]]
    failures = [item for item in predictions if not item["benchmark_2d"]["success"]]
    successes.sort(key=lambda item: (-item["benchmark_2d"]["spl"], item["benchmark_2d"]["navigation_error"]))
    failures.sort(key=lambda item: -item["benchmark_2d"]["navigation_error"])
    return [("success", item) for item in successes[:count]] + [
        ("failure", item) for item in failures[:count]
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", choices=("val_seen", "val_unseen"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count-per-outcome", type=int, default=5)
    args = parser.parse_args()
    if args.count_per_outcome < 1:
        parser.error("--count-per-outcome must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    predictions = [json.loads(line) for line in args.predictions.read_text().splitlines() if line]
    dataset = CityNavDataset(
        args.data_root,
        args.split,
    )
    episode_index = dataset.episode_index
    catalog = CityReferCatalog(args.data_root / "cityrefer")
    builder = PlanningStateBuilder(args.data_root, catalog)
    manifest = []
    for ordinal, (outcome, prediction) in enumerate(
        select_cases(predictions, args.count_per_outcome)
    ):
        episode_id = tuple(prediction["episode_id"])
        record = dataset[episode_index[episode_id]]
        poses = np.asarray(prediction["trajectory_xyzyaw"], dtype=np.float64)
        state = builder.build_from_poses(record, poses)
        transform = state.transform
        path_rc = transform.world_to_canvas(poses[:, :2])
        target_rc = transform.world_to_canvas(np.asarray(prediction["target_xyz"][:2]))
        diagnostic = prediction.get("final_diagnostic") or {}
        candidate_xy = np.asarray(diagnostic.get("candidate_xy", []), dtype=np.float64)
        candidate_rc = (
            transform.world_to_canvas(candidate_xy) if len(candidate_xy) else np.empty((0, 2))
        )
        selected = diagnostic.get("selected_index")

        figure, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
        observed_rgb = np.moveaxis(state.values[:3], 0, -1)
        axes[0, 0].imshow(observed_rgb)
        axes[0, 0].plot(path_rc[:, 1], path_rc[:, 0], "c.-", linewidth=2, label="trajectory")
        if len(candidate_rc):
            axes[0, 0].scatter(candidate_rc[:, 1], candidate_rc[:, 0], s=20, c="yellow", label="candidates")
        if selected is not None and selected < len(candidate_rc):
            axes[0, 0].scatter(candidate_rc[selected, 1], candidate_rc[selected, 0], s=100, facecolors="none", edgecolors="lime", linewidths=2, label="selected")
        axes[0, 0].scatter(target_rc[1], target_rc[0], s=140, marker="*", c="red", label="GT (visualization only)")
        axes[0, 0].set_title("Observed RGB, candidates, and closed-loop path")
        axes[0, 0].legend(fontsize=7, loc="best")

        axes[0, 1].imshow(state.values[5], cmap="magma", vmin=0, vmax=1)
        axes[0, 1].plot(path_rc[:, 1], path_rc[:, 0], "c.-", linewidth=1)
        axes[0, 1].set_title(CHANNEL_NAMES[5])
        axes[1, 0].imshow(state.values[7], cmap="gray", vmin=0, vmax=1)
        axes[1, 0].plot(path_rc[:, 1], path_rc[:, 0], "r.-", linewidth=1)
        axes[1, 0].set_title(CHANNEL_NAMES[7])
        axes[1, 1].imshow(state.values[8], cmap="viridis", vmin=0, vmax=1)
        axes[1, 1].set_title(CHANNEL_NAMES[8])
        for axis in axes.flat:
            axis.set_xlim(0, transform.canvas_size - 1)
            axis.set_ylim(transform.canvas_size - 1, 0)
            axis.set_xticks([])
            axis.set_yticks([])
        metric = prediction["benchmark_2d"]
        figure.suptitle(
            f"{outcome} | {episode_id} | NE={metric['navigation_error']:.1f} m "
            f"SPL={100 * metric['spl']:.1f} | {prediction['termination']}\n"
            f"{textwrap.fill(prediction['instruction'], width=110)}",
            fontsize=10,
        )
        filename = f"{ordinal:02d}_{outcome}_{record.map_name}_{record.index}.png"
        figure.savefig(args.output_dir / filename, dpi=160)
        plt.close(figure)
        manifest.append(
            {
                "file": filename,
                "outcome": outcome,
                "episode_id": list(episode_id),
                "metrics": metric,
                "termination": prediction["termination"],
            }
        )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps({"output_dir": str(args.output_dir), "rendered": len(manifest)}))


if __name__ == "__main__":
    main()
