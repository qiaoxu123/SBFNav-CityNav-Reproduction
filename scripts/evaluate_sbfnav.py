#!/usr/bin/env python3
"""Closed-loop CityNav evaluation for a frozen SBFNav checkpoint."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import torch
import transformers
import yaml

from sbfnav.candidate_proposal import Candidate, oracle_candidate_metrics
from sbfnav.checkpointing import load_checkpoint, sha256
from sbfnav.config import (
    apply_overrides,
    load_config,
    resolved_config_sha256,
    validate_config,
)
from sbfnav.dataset import CityNavDataset, CityReferCatalog, difficulty_membership
from sbfnav.metrics import aggregate_metrics, evaluate_episode
from sbfnav.model_factory import build_backbone, build_model
from sbfnav.navigation import BoundedWaypointController, RecedingHorizonNavigator
from sbfnav.planning_state import PlanningStateBuilder
from sbfnav.runtime import SBFNavRuntime


def stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def git_commit(root: Path) -> str:
    injected = os.environ.get("SBFNAV_GIT_COMMIT")
    if injected:
        return injected
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.strip()


def validate_final_test_authorization(
    manifest_path: Path,
    checkpoint_path: Path,
    config_path: Path,
    config: dict,
    overrides: list[str],
    output_dir: Path,
) -> dict:
    manifest = json.loads(manifest_path.read_text())
    if not manifest.get("test_unseen_authorized"):
        raise PermissionError("freeze manifest does not authorize Test-Unseen")
    actual_checkpoint = sha256(checkpoint_path)
    actual_config = sha256(config_path)
    if manifest.get("checkpoint_sha256") != actual_checkpoint:
        raise ValueError("checkpoint hash differs from frozen Test-Unseen manifest")
    if manifest.get("config_sha256") != actual_config:
        raise ValueError("config hash differs from frozen Test-Unseen manifest")
    if manifest.get("config_overrides", []) != overrides:
        raise ValueError("config overrides differ from frozen Test-Unseen manifest")
    if manifest.get("effective_config_sha256") != resolved_config_sha256(config):
        raise ValueError("effective configuration differs from frozen Test-Unseen manifest")
    sentinel = Path(manifest["single_use_sentinel"]).resolve()
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(sentinel, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w") as stream:
        json.dump(
            {"started": stamp(), "manifest": str(manifest_path.resolve()),
             "checkpoint": str(checkpoint_path.resolve()), "output_dir": str(output_dir.resolve())},
            stream,
            indent=2,
        )
        stream.write("\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("val_seen", "val_unseen", "test_unseen"), required=True)
    parser.add_argument("--difficulty", choices=("easy", "medium", "hard"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--allow-test-unseen", action="store_true")
    parser.add_argument("--freeze-manifest", type=Path)
    parser.add_argument(
        "--record-step-diagnostics",
        action="store_true",
        help="persist per-step candidates/selector decisions for closed-loop failure auditing",
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("output directory already exists")
    if args.split == "test_unseen" and (
        not args.allow_test_unseen or args.freeze_manifest is None
    ):
        parser.error("Test-Unseen requires --allow-test-unseen and --freeze-manifest")
    if args.split == "test_unseen" and args.difficulty is not None:
        parser.error(
            "the single final Test-Unseen call must evaluate the full split; "
            "difficulty rows are aggregated within that run"
        )
    if args.split == "test_unseen" and args.max_episodes is not None:
        parser.error("the single final Test-Unseen call cannot truncate the split")
    args.output_dir.mkdir(parents=True)
    config = apply_overrides(load_config(args.config), args.override)
    validate_config(config)
    final_manifest = None
    if args.split == "test_unseen":
        final_manifest = validate_final_test_authorization(
            args.freeze_manifest,
            args.checkpoint,
            args.config,
            config,
            args.override,
            args.output_dir,
        )
    seed = int(config["experiment"]["seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("closed-loop SBFNav evaluation requires CUDA")

    data = config["data"]
    dataset = CityNavDataset(
        data["root"],
        args.split,
        version=data["version"],
        difficulty=args.difficulty,
        allow_test_unseen=args.split == "test_unseen" and args.allow_test_unseen,
        max_records=args.max_episodes,
    )
    difficulty_lookup = None
    if args.difficulty is None and args.max_episodes is None:
        difficulty_lookup = difficulty_membership(
            dataset,
            tuple(config["evaluation"]["difficulties"]),
            allow_test_unseen=args.split == "test_unseen" and args.allow_test_unseen,
        )
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
    runtime = SBFNavRuntime(model, backbone, config, device)
    nav = config["navigation"]
    altitude = config["altitude"]
    navigator = RecedingHorizonNavigator(
        builder,
        BoundedWaypointController(
            horizontal_step_m=nav["horizontal_step_m"],
            vertical_step_m=nav["vertical_step_m"],
            minimum_above_ground_m=altitude["min_above_ground_m"],
            maximum_above_ground_m=altitude["max_above_ground_m"],
        ),
        horizon=nav["horizon"],
        stagnation_steps=nav["stagnation_steps"],
    )
    benchmark = []
    diagnostic = []
    benchmark_by_difficulty = {
        name: [] for name in config["evaluation"]["difficulties"]
    }
    diagnostic_by_difficulty = {
        name: [] for name in config["evaluation"]["difficulties"]
    }
    candidate_sums: dict[str, float] = {}
    prediction_path = args.output_dir / "predictions.jsonl"
    evaluation = config["evaluation"]
    for episode_index, record in enumerate(dataset):
        trace = navigator.run(record, runtime.predict)
        diagnostic_metric = evaluate_episode(
            trace.poses[:, :3],
            record.target_position,
            success_distance_m=nav["success_distance_m"],
            dimensions=evaluation["diagnostic_dimensions"],
        )
        benchmark_metric = evaluate_episode(
            trace.poses[:, :3],
            record.target_position,
            success_distance_m=nav["success_distance_m"],
            dimensions=evaluation["benchmark_dimensions"],
        )
        benchmark.append(benchmark_metric)
        diagnostic.append(diagnostic_metric)
        if difficulty_lookup is not None:
            difficulty = difficulty_lookup[record.episode_id]
            benchmark_by_difficulty[difficulty].append(benchmark_metric)
            diagnostic_by_difficulty[difficulty].append(diagnostic_metric)
        final_diagnostic = trace.diagnostics[-1] if trace.diagnostics else None
        oracle = {}
        if final_diagnostic:
            candidates = [
                Candidate(np.asarray(xy), tuple(sources))
                for xy, sources in zip(
                    final_diagnostic["candidate_xy"], final_diagnostic["candidate_sources"]
                )
            ]
            oracle = dict(
                oracle_candidate_metrics(
                    candidates,
                    record.target_position[:2],
                    ks=config["evaluation"]["candidate_ks"],
                    thresholds_m=config["evaluation"]["candidate_thresholds_m"],
                )
            )
            for key, value in oracle.items():
                candidate_sums[key] = candidate_sums.get(key, 0.0) + value
        with prediction_path.open("a") as stream:
            stream.write(
                json.dumps(
                    {
                        "episode_id": list(record.episode_id),
                        "instruction": record.instruction,
                        "target_xyz": record.target_position.tolist(),
                        "trajectory_xyzyaw": trace.poses.tolist(),
                        "waypoints_xyz": trace.waypoints.tolist(),
                        "termination": trace.termination,
                        "benchmark_2d": benchmark_metric.__dict__,
                        "diagnostic_3d": diagnostic_metric.__dict__,
                        "final_candidate_oracle": oracle,
                        "final_diagnostic": final_diagnostic,
                        "step_diagnostics": list(trace.diagnostics)
                        if args.record_step_diagnostics
                        else None,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        if (episode_index + 1) % 10 == 0:
            print(
                json.dumps(
                    {"time": stamp(), "completed": episode_index + 1, "total": len(dataset),
                     "benchmark_2d": aggregate_metrics(benchmark)},
                    ensure_ascii=False,
                ),
                flush=True,
            )
    metrics = {
        "paper_based_reimplementation": True,
        "time": stamp(),
        "split": args.split,
        "difficulty": args.difficulty,
        "data_version": data["version"],
        "seed": seed,
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
        "record_step_diagnostics": args.record_step_diagnostics,
        "metric_protocol": {
            "official": evaluation["official_metric"],
            "benchmark_dimensions": evaluation["benchmark_dimensions"],
            "diagnostic_dimensions": evaluation["diagnostic_dimensions"],
        },
        "benchmark_2d": aggregate_metrics(benchmark),
        "diagnostic_3d": aggregate_metrics(diagnostic),
        "difficulty_breakdown": (
            {
                name: {
                    "benchmark_2d": aggregate_metrics(benchmark_by_difficulty[name]),
                    "diagnostic_3d": aggregate_metrics(diagnostic_by_difficulty[name]),
                }
                for name in config["evaluation"]["difficulties"]
            }
            if difficulty_lookup is not None
            else None
        ),
        "final_candidate_oracle": {
            key: value / len(dataset) for key, value in candidate_sums.items()
        },
        "command": " ".join(os.sys.argv),
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n"
    )
    if args.split == "test_unseen":
        sentinel = Path(final_manifest["single_use_sentinel"])
        completed = sentinel.with_name(sentinel.name + ".completed")
        completed.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(metrics, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
