#!/usr/bin/env python3
"""Freeze a selected checkpoint/config before the single Test-Unseen run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time

from sbfnav.checkpointing import sha256
from sbfnav.config import apply_overrides, load_config, resolved_config_sha256, validate_config


def checked_validation(
    path: Path,
    split: str,
    checkpoint_hash: str,
    effective_config_hash: str,
    overrides: list[str],
) -> dict:
    value = json.loads(path.read_text())
    if value.get("split") != split:
        raise ValueError(f"expected {split} metrics: {path}")
    if value.get("checkpoint_sha256") != checkpoint_hash:
        raise ValueError(f"validation metrics do not belong to frozen checkpoint: {path}")
    if value.get("data_version") != "processed_citynav":
        raise ValueError("final revised-data freeze requires processed_citynav metrics")
    if value.get("effective_config_sha256") != effective_config_hash:
        raise ValueError(f"validation metrics use a different effective config: {path}")
    if value.get("config_overrides") != overrides:
        raise ValueError(f"validation metrics use different config overrides: {path}")
    if value.get("metric_protocol", {}).get("official") != "benchmark_2d":
        raise ValueError(f"validation metrics do not use the CityNav benchmark protocol: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--val-seen-metrics", type=Path, required=True)
    parser.add_argument("--val-unseen-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--single-use-sentinel", type=Path, required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    if args.output.exists():
        parser.error("freeze manifest already exists")
    if args.single_use_sentinel.exists() or args.single_use_sentinel.with_name(
        args.single_use_sentinel.name + ".completed"
    ).exists():
        parser.error("single-use Test-Unseen sentinel already exists")
    config = apply_overrides(load_config(args.config), args.override)
    validate_config(config)
    checkpoint_hash = sha256(args.checkpoint)
    effective_hash = resolved_config_sha256(config)
    seen = checked_validation(
        args.val_seen_metrics, "val_seen", checkpoint_hash, effective_hash, args.override
    )
    unseen = checked_validation(
        args.val_unseen_metrics, "val_unseen", checkpoint_hash, effective_hash, args.override
    )
    root = Path(__file__).resolve().parents[1]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    value = {
        "paper_based_reimplementation": True,
        "test_unseen_authorized": True,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_commit": commit,
        "data_version": config["data"]["version"],
        "seed": config["experiment"]["seed"],
        "config": str(args.config.resolve()),
        "config_sha256": sha256(args.config),
        "config_overrides": args.override,
        "effective_config_sha256": effective_hash,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_selection": "Val-Unseen benchmark_2d SR, then SPL, then lower NE",
        "val_seen_metrics": {
            "path": str(args.val_seen_metrics.resolve()),
            "sha256": sha256(args.val_seen_metrics),
            "benchmark_2d": seen["benchmark_2d"],
            "diagnostic_3d": seen["diagnostic_3d"],
        },
        "val_unseen_metrics": {
            "path": str(args.val_unseen_metrics.resolve()),
            "sha256": sha256(args.val_unseen_metrics),
            "benchmark_2d": unseen["benchmark_2d"],
            "diagnostic_3d": unseen["diagnostic_3d"],
        },
        "single_use_sentinel": str(args.single_use_sentinel.resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(value, ensure_ascii=False))


if __name__ == "__main__":
    main()
