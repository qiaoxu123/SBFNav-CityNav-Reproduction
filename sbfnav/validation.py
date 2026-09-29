"""Deterministic closed-loop development validation used during training."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import torch

from .candidate_proposal import Candidate, oracle_candidate_metrics
from .dataset import CityNavDataset, CityReferCatalog
from .metrics import aggregate_metrics, evaluate_episode
from .models.language_encoder import FrozenSiglipBackbone
from .models.sbfnav import SBFNav
from .navigation import BoundedWaypointController, RecedingHorizonNavigator
from .planning_state import PlanningStateBuilder
from .runtime import SBFNavRuntime


def evenly_spaced_indices(length: int, count: int | None) -> list[int]:
    """Return a deterministic, split-wide subset rather than a prefix."""
    if length < 1:
        raise ValueError("validation split is empty")
    if count is None or count >= length:
        return list(range(length))
    if count < 1:
        raise ValueError("validation episode count must be positive")
    return np.linspace(0, length - 1, count, dtype=np.int64).tolist()


@torch.inference_mode()
def validate_closed_loop(
    config: dict,
    model: SBFNav,
    backbone: FrozenSiglipBackbone,
    device: torch.device,
    split: str,
    *,
    max_episodes: int | None,
) -> dict[str, object]:
    if split not in ("val_seen", "val_unseen"):
        raise ValueError("training validation is restricted to development splits")
    data = config["data"]
    dataset = CityNavDataset(data["root"], split, version=data["version"])
    indices = evenly_spaced_indices(len(dataset), max_episodes)
    catalog = CityReferCatalog(Path(data["root"]) / "cityrefer")
    state = config["state"]
    builder = PlanningStateBuilder(
        data["root"],
        catalog,
        canvas_size=config["model"]["canvas_size"],
        field_size=config["model"]["field_size"],
        max_height_above_ground_m=state["max_height_above_ground_m"],
        trajectory_radius_m=state["trajectory_radius_m"],
    )
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
    runtime = SBFNavRuntime(model, backbone, config, device)
    evaluation = config["evaluation"]
    benchmark = []
    diagnostic_metrics = []
    oracle_sums: dict[str, float] = {}
    for index in indices:
        record = dataset[index]
        trace = navigator.run(record, runtime.predict)
        diagnostic_metrics.append(
            evaluate_episode(
                trace.poses[:, :3],
                record.target_position,
                success_distance_m=nav["success_distance_m"],
                dimensions=evaluation["diagnostic_dimensions"],
            )
        )
        benchmark.append(
            evaluate_episode(
                trace.poses[:, :3],
                record.target_position,
                success_distance_m=nav["success_distance_m"],
                dimensions=evaluation["benchmark_dimensions"],
            )
        )
        final_diagnostic = trace.diagnostics[-1] if trace.diagnostics else None
        if final_diagnostic:
            candidates = [
                Candidate(np.asarray(xy), tuple(sources))
                for xy, sources in zip(
                    final_diagnostic["candidate_xy"], final_diagnostic["candidate_sources"]
                )
            ]
            oracle = oracle_candidate_metrics(
                candidates,
                record.target_position[:2],
                ks=config["evaluation"]["candidate_ks"],
                thresholds_m=config["evaluation"]["candidate_thresholds_m"],
            )
            for name, value in oracle.items():
                oracle_sums[name] = oracle_sums.get(name, 0.0) + value
    return {
        "sampling": "evenly_spaced",
        "indices": indices,
        # CityNav's released evaluator computes all leaderboard metrics on
        # Pose.xy.  Keep the paper-literal 3-D interpretation as an explicit
        # diagnostic, but never use it to select a benchmark checkpoint.
        "benchmark_2d": aggregate_metrics(benchmark),
        "diagnostic_3d": aggregate_metrics(diagnostic_metrics),
        "final_candidate_oracle": {
            name: value / len(indices) for name, value in oracle_sums.items()
        },
    }


def checkpoint_selection_key(validation: dict[str, object]) -> tuple[float, float, float]:
    """CityNav benchmark ordering: maximize 2-D SR/SPL, then minimize NE.

    ``compatibility_2d`` is accepted only to resume checkpoints written before
    the benchmark/diagnostic naming bug was fixed.
    """
    metrics = validation.get("benchmark_2d", validation.get("compatibility_2d"))
    if metrics is None:
        raise KeyError("validation is missing benchmark_2d metrics")
    return (float(metrics["sr"]), float(metrics["spl"]), -float(metrics["ne"]))
