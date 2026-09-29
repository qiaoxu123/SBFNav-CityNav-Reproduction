#!/usr/bin/env python3
"""Measure static teacher-prefix SBF proposal coverage on development data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import transformers
import platform

from sbfnav.candidate_proposal import oracle_candidate_metrics, propose_candidates
from sbfnav.checkpointing import git_commit, load_checkpoint, sha256
from sbfnav.config import (
    apply_overrides,
    load_config,
    resolved_config_sha256,
    validate_config,
)
from sbfnav.dataset import CityNavDataset, CityReferCatalog
from sbfnav.model_factory import build_backbone, build_model
from sbfnav.planning_state import PlanningStateBuilder
from sbfnav.runtime import TextFeatureCache
from sbfnav.validation import evenly_spaced_indices


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("train_seen", "val_seen", "val_unseen"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--prefix-fraction", type=float, default=0.5)
    args = parser.parse_args()
    if not 0 < args.prefix_fraction <= 1:
        parser.error("--prefix-fraction must be in (0, 1]")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    config = apply_overrides(load_config(args.config), args.override)
    validate_config(config)
    seed = int(config["experiment"]["seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("candidate analysis requires CUDA")

    data = config["data"]
    dataset = CityNavDataset(data["root"], args.split, version=data["version"])
    indices = evenly_spaced_indices(len(dataset), args.max_episodes)
    catalog = CityReferCatalog(Path(data["root"]) / "cityrefer")
    state_config = config["state"]
    builder = PlanningStateBuilder(
        data["root"],
        catalog,
        canvas_size=config["model"]["canvas_size"],
        field_size=config["model"]["field_size"],
        max_height_above_ground_m=state_config["max_height_above_ground_m"],
        trajectory_radius_m=state_config["trajectory_radius_m"],
    )
    backbone = build_backbone(config).to(device)
    model = build_model(config).to(device)
    checkpoint = load_checkpoint(
        args.checkpoint, model=model, backbone=backbone, map_location=device
    )
    model.eval()
    text_cache = TextFeatureCache(backbone, device)
    proposal = config["proposal"]
    ks = config["evaluation"]["candidate_ks"]
    thresholds = config["evaluation"]["candidate_thresholds_m"]
    totals: dict[str, dict[str, float]] = {"field_only": {}, "with_landmarks": {}}
    prediction_path = args.output_dir / "candidate_predictions.jsonl"
    with torch.inference_mode(), prediction_path.open("w") as stream:
        for ordinal, index in enumerate(indices):
            record = dataset[index]
            prefix = max(1, round(args.prefix_fraction * len(record.trajectory)))
            state = builder.build(record, prefix_length=prefix)
            language = text_cache.get([record.instruction])
            output = model.belief_field(
                torch.from_numpy(state.values).unsqueeze(0).to(device),
                language.tokens,
                language.attention_mask,
                torch.from_numpy(state.valid_field).unsqueeze(0).to(device),
                pooled_language=language.pooled,
            )
            variants = {}
            for name, add_landmarks in (("field_only", False), ("with_landmarks", True)):
                candidates = propose_candidates(
                    output.logits[0],
                    state.valid_field,
                    state.transform,
                    state.referenced_landmarks,
                    top_k=proposal["top_k"],
                    nms_kernel=proposal["nms_kernel"],
                    dedup_radius_m=proposal["dedup_radius_m"],
                    add_landmarks=add_landmarks,
                )
                metrics = dict(
                    oracle_candidate_metrics(
                        candidates, record.target_position[:2], ks=ks, thresholds_m=thresholds
                    )
                )
                variants[name] = {
                    "metrics": metrics,
                    "candidate_xy": [candidate.xy.tolist() for candidate in candidates],
                    "candidate_sources": [candidate.sources for candidate in candidates],
                }
                for key, value in metrics.items():
                    totals[name][key] = totals[name].get(key, 0.0) + value
            stream.write(
                json.dumps(
                    {
                        "ordinal": ordinal,
                        "dataset_index": index,
                        "episode_id": list(record.episode_id),
                        "prefix_length": prefix,
                        "target_xy": record.target_position[:2].tolist(),
                        "variants": variants,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    result = {
        "paper_based_reimplementation": True,
        "analysis_state": "fixed teacher-trajectory prefix",
        "split": args.split,
        "data_version": data["version"],
        "seed": seed,
        "prefix_fraction": args.prefix_fraction,
        "episodes": len(indices),
        "indices": indices,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "checkpoint_completed_epoch": checkpoint["completed_epoch"],
        "config": str(args.config.resolve()),
        "config_sha256": sha256(args.config),
        "config_overrides": args.override,
        "effective_config_sha256": resolved_config_sha256(config),
        "resolved_config": config,
        "git_commit": git_commit(Path(__file__).resolve().parents[1]),
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "gpu": torch.cuda.get_device_name(device),
        },
        "metrics": {
            name: {key: value / len(indices) for key, value in values.items()}
            for name, values in totals.items()
        },
        "command": " ".join(__import__("sys").argv),
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
