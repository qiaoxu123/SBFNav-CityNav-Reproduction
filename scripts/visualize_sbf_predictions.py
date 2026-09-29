#!/usr/bin/env python3
"""Render deterministic real-data planning-state and target diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sbfnav.dataset import CityNavDataset, CityReferCatalog
from sbfnav.field_targets import gaussian_field_target
from sbfnav.planning_state import PlanningStateBuilder
from sbfnav.visualization import save_planning_state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--version", default="processed_citynav")
    parser.add_argument("--samples-per-split", type=int, default=7)
    parser.add_argument("--sigma-m", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.samples_per_split < 1:
        parser.error("--samples-per-split must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    catalog = CityReferCatalog(args.data_root / "cityrefer")
    builder = PlanningStateBuilder(args.data_root, catalog)
    rng = np.random.default_rng(args.seed)
    manifest = []
    for split in ("train_seen", "val_seen", "val_unseen"):
        dataset = CityNavDataset(args.data_root, split, version=args.version)
        indices = rng.choice(len(dataset), size=args.samples_per_split, replace=False)
        for ordinal, index in enumerate(indices):
            record = dataset[int(index)]
            prefix_length = int(rng.integers(1, len(record.trajectory) + 1))
            state = builder.build(record, prefix_length=prefix_length)
            target = gaussian_field_target(
                state.transform,
                record.target_position[:2],
                sigma_m=args.sigma_m,
                valid_mask=state.valid_field,
            )
            round_trip = state.transform.normalized_to_world(
                state.transform.world_to_normalized(record.target_position[:2])
            )
            error_m = float(np.linalg.norm(round_trip - record.target_position[:2]))
            filename = f"{split}_{ordinal:02d}_{record.map_name}_{record.index}.png"
            save_planning_state(
                state,
                args.output_dir / filename,
                field=target,
                title=f"{split} | {record.map_name} | prefix {prefix_length}",
                metadata={
                    "episode_id": record.episode_id,
                    "instruction": record.instruction,
                    "target_xyz": record.target_position.tolist(),
                    "round_trip_error_m": error_m,
                    "referenced_landmarks": [item.name for item in state.referenced_landmarks],
                },
            )
            manifest.append(
                {
                    "file": filename,
                    "split": split,
                    "episode_id": list(record.episode_id),
                    "prefix_length": prefix_length,
                    "target_xyz": record.target_position.tolist(),
                    "round_trip_error_m": error_m,
                    "field_sum": float(target.sum()),
                    "valid_canvas_fraction": float(state.valid_canvas.mean()),
                    "explored_fraction": float(state.values[7].mean()),
                }
            )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps({"output_dir": str(args.output_dir), "samples": len(manifest)}))


if __name__ == "__main__":
    main()

