#!/usr/bin/env python3
"""Train the independent paper-based SBFNav reimplementation."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from sbfnav.candidate_proposal import (
    oracle_candidate_metrics,
    propose_candidates,
    propose_uniform_grid_candidates,
)
from sbfnav.checkpointing import load_checkpoint, save_checkpoint, sha256
from sbfnav.config import apply_overrides, load_config, validate_config
from sbfnav.dataset import CityNavDataset, CityReferCatalog
from sbfnav.losses import (
    LossWeights,
    altitude_loss,
    combine_losses,
    hard_nearest_loss,
    listwise_distance_loss,
    pairwise_margin_loss,
    progress_loss,
    soft_field_cross_entropy,
)
from sbfnav.model_factory import build_backbone, build_model
from sbfnav.planning_state import PlanningStateBuilder
from sbfnav.runtime import TextFeatureCache, VisionFeatureCache, build_selector_inputs
from sbfnav.training_data import SBFTrainingDataset
from sbfnav.validation import checkpoint_selection_key, validate_closed_loop


def stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def append_jsonl(path: Path, value: dict) -> None:
    with path.open("a") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        if torch.cuda.is_available():
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)


def cosine_schedule(optimizer, total_steps: int, warmup_steps: int):
    def ratio(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return max(1e-8, (step + 1) / warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, ratio)


def make_dataset(config: dict, max_records: int | None):
    data = config["data"]
    catalog = CityReferCatalog(Path(data["root"]) / "cityrefer")
    episodes = CityNavDataset(
        data["root"],
        data["train_split"],
        version=data["version"],
        max_records=max_records,
    )
    state = config["state"]
    builder = PlanningStateBuilder(
        data["root"],
        catalog,
        canvas_size=config["model"]["canvas_size"],
        field_size=config["model"]["field_size"],
        max_height_above_ground_m=state["max_height_above_ground_m"],
        trajectory_radius_m=state["trajectory_radius_m"],
    )
    altitude = config["altitude"]
    dataset = SBFTrainingDataset(
        episodes,
        builder,
        samples_per_episode=data["samples_per_episode"],
        gaussian_sigma_m=config["loss"]["gaussian_sigma_m"],
        minimum_above_ground_m=altitude["min_above_ground_m"],
        maximum_above_ground_m=altitude["max_above_ground_m"],
    )
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--epochs", type=int)
    parser.add_argument(
        "--stop-after-epoch",
        type=int,
        help="intentional interruption point without changing the planned LR schedule",
    )
    parser.add_argument("--max-train-records", type=int)
    parser.add_argument("--resume")
    parser.add_argument("--resume-optimizer", action="store_true")
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--vision-batch-size", type=int)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overfit-check", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists() and not args.resume:
        parser.error("output directory exists; use a new run or --resume")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = apply_overrides(load_config(args.config), args.override)
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.max_train_records is None:
        args.max_train_records = config["data"].get("max_train_records")
    validate_config(config)
    (args.output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
    )
    seed = int(config["experiment"]["seed"])
    seed_everything(seed, args.deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("SBFNav training requires the configured single CUDA GPU")

    dataset = make_dataset(config, args.max_train_records)
    training = config["training"]
    vision_batch_size = args.vision_batch_size or int(training["vision_batch_size"])
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=training["batch_size"],
        shuffle=True,
        num_workers=training["workers"],
        collate_fn=lambda examples: examples,
        generator=generator,
        persistent_workers=training["workers"] > 0,
    )
    backbone = build_backbone(config).to(device)
    model = build_model(config).to(device)
    optimized_parameters = list(model.parameters())
    if not backbone.freeze:
        optimized_parameters.extend(backbone.parameters())
    optimizer = torch.optim.AdamW(
        optimized_parameters, lr=training["learning_rate"], weight_decay=training["weight_decay"]
    )
    accumulation = int(training["gradient_accumulation"])
    updates_per_epoch = math.ceil(len(loader) / accumulation)
    total_updates = updates_per_epoch * int(training["epochs"])
    scheduler = cosine_schedule(
        optimizer, total_updates, round(total_updates * training["warmup_fraction"])
    )
    start_epoch = 0
    global_step = 0
    if args.resume:
        checkpoint = load_checkpoint(
            args.resume,
            model=model,
            optimizer=optimizer if args.resume_optimizer else None,
            scheduler=scheduler if args.resume_optimizer else None,
            restore_rng=args.resume_optimizer,
            map_location=device,
            backbone=backbone,
        )
        start_epoch = int(checkpoint["completed_epoch"])
        global_step = int(checkpoint["global_step"])
        append_jsonl(
            args.output_dir / "events.jsonl",
            {"time": stamp(), "event": "resumed", "checkpoint": str(Path(args.resume).resolve()),
             "sha256": sha256(args.resume), "optimizer": args.resume_optimizer,
             "completed_epoch": start_epoch, "global_step": global_step},
        )
        inherited_best = Path(args.resume).parent / "best_val_unseen.pt"
        checkpoint_dir = args.output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            inherited_best if inherited_best.is_file() else Path(args.resume),
            checkpoint_dir / "best_val_unseen.pt",
        )
        inherited_metadata = inherited_best.with_suffix(".json")
        if inherited_metadata.is_file():
            inherited = json.loads(inherited_metadata.read_text())
            inherited_metrics = dict(inherited["metrics"])
            if "benchmark_2d" not in inherited_metrics:
                inherited_metrics["benchmark_2d"] = inherited_metrics.pop("compatibility_2d")
            if "diagnostic_3d" not in inherited_metrics:
                inherited_metrics["diagnostic_3d"] = inherited_metrics.pop("primary_3d")
            inherited["metrics"] = inherited_metrics
            inherited["selection_metric"] = "benchmark_2d"
            inherited["selection_key"] = checkpoint_selection_key(inherited_metrics)
            (checkpoint_dir / "best_val_unseen.json").write_text(
                json.dumps(inherited, indent=2) + "\n"
            )
    text_cache = TextFeatureCache(backbone, device)
    precompute_started = time.time()
    if backbone.freeze:
        text_cache.precompute(
            [
                *dataset.episodes.instructions,
                *dataset.state_builder.catalog.all_named_landmark_names,
            ],
            batch_size=int(training["text_precompute_batch_size"]),
        )
    append_jsonl(
        args.output_dir / "events.jsonl",
        {
            "time": stamp(),
            "event": "frozen_text_precomputed",
            "entries": len(text_cache._cache),
            "seconds": time.time() - precompute_started,
            "disabled_trainable": not backbone.freeze,
        },
    )
    vision_cache = (
        VisionFeatureCache(training["vision_cache_entries"]) if backbone.freeze else None
    )
    loss_config = config["loss"]
    weights = LossWeights(
        field=loss_config["field_weight"],
        rank=loss_config["rank_weight"],
        margin=loss_config["margin_weight"],
        hard=loss_config["hard_weight"],
        altitude=loss_config["altitude_weight"],
        progress=loss_config["progress_weight"],
    )
    use_bf16 = training["precision"] == "bf16"
    if use_bf16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("bf16 requested but unsupported by this GPU")
    initial_epoch_loss = None
    best_validation_key = None
    if args.resume:
        saved_metrics = checkpoint.get("metrics", {})
        inherited_metadata = checkpoint_dir / "best_val_unseen.json"
        if inherited_metadata.is_file():
            inherited = json.loads(inherited_metadata.read_text())
            if inherited.get("metrics"):
                # Recompute instead of trusting a legacy key: releases before
                # this fix selected on diagnostic 3-D metrics.
                best_validation_key = checkpoint_selection_key(inherited["metrics"])
        if best_validation_key is None and saved_metrics.get("validation", {}).get("val_unseen"):
            best_validation_key = checkpoint_selection_key(
                saved_metrics["validation"]["val_unseen"]
            )
    repository_root = Path(__file__).resolve().parents[1]
    append_jsonl(
        args.output_dir / "events.jsonl",
        {"time": stamp(), "event": "training_started", "device": torch.cuda.get_device_name(0),
         "records": len(dataset.episodes), "samples": len(dataset), "start_epoch": start_epoch,
         "target_epochs": training["epochs"]},
    )

    planned_epochs = int(training["epochs"])
    stop_epoch = planned_epochs if args.stop_after_epoch is None else args.stop_after_epoch
    if stop_epoch <= start_epoch or stop_epoch > planned_epochs:
        parser.error("--stop-after-epoch must be after resumed epoch and within the plan")
    for epoch in range(start_epoch, stop_epoch):
        model.train()
        running = defaultdict(float)
        example_count = 0
        optimizer.zero_grad(set_to_none=True)
        for batch_index, examples in enumerate(loader):
            batch_size = len(examples)
            states = [example.state for example in examples]
            instructions = text_cache.get([example.record.instruction for example in examples])
            planning = torch.from_numpy(np.stack([state.values for state in states])).to(device)
            valid = torch.from_numpy(np.stack([state.valid_field for state in states])).to(device)
            field_targets = torch.stack([example.field_target for example in examples]).to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                field = model.belief_field(
                    planning,
                    instructions.tokens,
                    instructions.attention_mask,
                    valid,
                    pooled_language=instructions.pooled,
                )
            proposal_config = config["proposal"]
            candidate_batches = []
            oracle_candidate_batches = []
            for sample_index, state in enumerate(states):
                if proposal_config["mode"] == "field":
                    candidates = propose_candidates(
                        field.logits[sample_index],
                        state.valid_field,
                        state.transform,
                        state.referenced_landmarks,
                        top_k=proposal_config["top_k"],
                        nms_kernel=proposal_config["nms_kernel"],
                        dedup_radius_m=proposal_config["dedup_radius_m"],
                        add_landmarks=proposal_config["add_landmark_candidates"],
                    )
                else:
                    candidates = propose_uniform_grid_candidates(
                        state.valid_field,
                        state.transform,
                        state.referenced_landmarks,
                        top_k=proposal_config["top_k"],
                        dedup_radius_m=proposal_config["dedup_radius_m"],
                        add_landmarks=proposal_config["add_landmark_candidates"],
                    )
                oracle_candidate_batches.append(candidates)
                # Selector candidates are independent; random order prevents
                # accidental reliance on proposal rank even though no score is passed.
                permutation = torch.randperm(len(candidates), generator=generator).tolist()
                candidate_batches.append([candidates[index] for index in permutation])
            selector_config = config["selector"]
            selector_inputs = build_selector_inputs(
                states,
                candidate_batches,
                instructions,
                backbone,
                text_cache,
                device=device,
                crop_extent_m=selector_config["crop_extent_m"],
                crop_size=selector_config["crop_size"],
                vision_batch_size=vision_batch_size,
                maximum_landmarks=selector_config["max_landmarks"],
                visual_cache=vision_cache,
                visual_cache_keys=[
                    f"{example.record.episode_id}:{example.prefix_length}" for example in examples
                ],
            )
            target_xy = torch.as_tensor(
                np.stack([example.record.target_position[:2] for example in examples]),
                dtype=torch.float32,
                device=device,
            )
            distances = torch.linalg.vector_norm(
                selector_inputs.candidate_xy - target_xy[:, None], dim=-1
            )
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                selector_logits = model.selector(selector_inputs)
                auxiliary = model.auxiliary(field.spatial_features, instructions.pooled)
                field_loss = soft_field_cross_entropy(field.logits, field_targets, valid)
                list_loss = listwise_distance_loss(
                    selector_logits,
                    distances,
                    selector_inputs.candidate_mask,
                    temperature_m=loss_config["list_temperature_m"],
                )
                margin_loss = pairwise_margin_loss(
                    selector_logits,
                    distances,
                    selector_inputs.candidate_mask,
                    good_radius_m=loss_config["good_radius_m"],
                    bad_radius_m=loss_config["bad_radius_m"],
                    margin=loss_config["score_margin"],
                )
                hard_loss = hard_nearest_loss(
                    selector_logits, distances, selector_inputs.candidate_mask
                )
                altitude_target = torch.tensor(
                    [example.altitude_target_normalized for example in examples], device=device
                )
                progress_target_tensor = torch.tensor(
                    [example.progress_target for example in examples], device=device
                )
                z_loss = altitude_loss(auxiliary.altitude_normalized, altitude_target)
                p_loss = progress_loss(auxiliary.progress, progress_target_tensor)
                losses = combine_losses(
                    field=field_loss,
                    listwise=list_loss,
                    margin=margin_loss,
                    hard=hard_loss,
                    altitude=z_loss,
                    progress=p_loss,
                    weights=weights,
                )
            window_start = (batch_index // accumulation) * accumulation
            window_size = min(accumulation, len(loader) - window_start)
            (losses["total"] / window_size).backward()
            should_step = (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(loader)
            if should_step:
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    optimized_parameters,
                    training["gradient_clip_norm"],
                    error_if_nonfinite=True,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
            for name, value in losses.items():
                running[name] += float(value.detach()) * batch_size
            selected = selector_logits.argmax(dim=-1)
            nearest = distances.masked_fill(~selector_inputs.candidate_mask, torch.inf).argmin(dim=-1)
            running["selector_nearest_accuracy"] += float((selected == nearest).sum())
            for sample_index, candidates in enumerate(oracle_candidate_batches):
                metrics = oracle_candidate_metrics(
                    candidates,
                    examples[sample_index].record.target_position[:2],
                    ks=(1, 5, proposal_config["top_k"]),
                    thresholds_m=(10, 20),
                )
                for name, value in metrics.items():
                    running[f"candidate_{name}"] += value
            example_count += batch_size
            if args.log_every and (batch_index + 1) % args.log_every == 0:
                append_jsonl(
                    args.output_dir / "train_steps.jsonl",
                    {"time": stamp(), "epoch": epoch + 1, "batch": batch_index + 1,
                     "batches": len(loader), "global_step": global_step,
                     "loss": float(losses["total"].detach()),
                     "lr": optimizer.param_groups[0]["lr"],
                     "gpu_memory_allocated": torch.cuda.max_memory_allocated()},
                )
        epoch_metrics = {name: value / example_count for name, value in running.items()}
        epoch_metrics.update(
            time=stamp(), epoch=epoch + 1, global_step=global_step,
            lr=optimizer.param_groups[0]["lr"], examples=example_count,
            max_gpu_memory_bytes=torch.cuda.max_memory_allocated(),
        )
        epoch_metrics["vision_cache"] = (
            vision_cache.statistics() if vision_cache is not None else {"disabled_trainable": True}
        )
        if initial_epoch_loss is None:
            initial_epoch_loss = epoch_metrics["total"]
        validation = {}
        if (epoch + 1) % int(training["validate_every"]) == 0:
            for split in config["data"]["validation_splits"]:
                validation[split] = validate_closed_loop(
                    config,
                    model,
                    backbone,
                    device,
                    split,
                    max_episodes=training["validation_max_episodes"],
                )
            epoch_metrics["validation"] = validation
        is_best = False
        if "val_unseen" in validation:
            selection_key = checkpoint_selection_key(validation["val_unseen"])
            if best_validation_key is None or selection_key > best_validation_key:
                best_validation_key = selection_key
                is_best = True
        epoch_metrics["best_validation_key"] = best_validation_key
        append_jsonl(args.output_dir / "epochs.jsonl", epoch_metrics)
        print(json.dumps(epoch_metrics, ensure_ascii=False), flush=True)
        checkpoint_dir = args.output_dir / "checkpoints"
        save_checkpoint(
            checkpoint_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            completed_epoch=epoch + 1,
            global_step=global_step,
            config=config,
            repository_root=repository_root,
            metrics=epoch_metrics,
            backbone=None if backbone.freeze else backbone,
        )
        if (epoch + 1) % args.save_every == 0:
            shutil.copy2(checkpoint_dir / "last.pt", checkpoint_dir / f"epoch_{epoch + 1:02d}.pt")
        if is_best:
            shutil.copy2(checkpoint_dir / "last.pt", checkpoint_dir / "best_val_unseen.pt")
            (checkpoint_dir / "best_val_unseen.json").write_text(
                json.dumps(
                    {
                        "epoch": epoch + 1,
                        "selection_metric": "benchmark_2d",
                        "selection_key": best_validation_key,
                        "metrics": validation["val_unseen"],
                    },
                    indent=2,
                )
                + "\n"
            )

    final_loss = epoch_metrics["total"]
    overfit_passed = final_loss <= 0.8 * initial_epoch_loss
    summary = {
        "time": stamp(),
        "completed_epochs": stop_epoch,
        "planned_epochs": planned_epochs,
        "initial_total_loss": initial_epoch_loss,
        "final_total_loss": final_loss,
        "loss_ratio": final_loss / initial_epoch_loss,
        "overfit_check_requested": args.overfit_check,
        "overfit_check_passed": overfit_passed,
        "last_checkpoint": str((args.output_dir / "checkpoints" / "last.pt").resolve()),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if args.overfit_check and stop_epoch == planned_epochs and not overfit_passed:
        raise SystemExit("small-sample overfit criterion failed: total loss did not fall by 20%")


if __name__ == "__main__":
    main()
