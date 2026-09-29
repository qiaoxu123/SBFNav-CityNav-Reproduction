"""CityNav NE/SR/OSR/SPL with explicit 3-D and compatibility 2-D modes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Mapping

import numpy as np


@dataclass(frozen=True)
class EpisodeMetrics:
    navigation_error: float
    success: float
    oracle_success: float
    spl: float
    path_length: float
    optimal_length: float
    progress: float


def evaluate_episode(
    trajectory_xyz: np.ndarray,
    target_xyz: np.ndarray,
    *,
    success_distance_m: float = 20.0,
    dimensions: int = 3,
) -> EpisodeMetrics:
    if dimensions not in (2, 3):
        raise ValueError("dimensions must be 2 or 3")
    path = np.asarray(trajectory_xyz, dtype=np.float64)
    target = np.asarray(target_xyz, dtype=np.float64)
    if path.ndim != 2 or path.shape[1] < dimensions or len(path) < 1:
        raise ValueError("trajectory must be a non-empty N×3 array")
    if target.shape[0] < dimensions or success_distance_m <= 0:
        raise ValueError("invalid target or success distance")
    path = path[:, :dimensions]
    target = target[:dimensions]
    distances = np.linalg.norm(path - target, axis=-1)
    segment_lengths = np.linalg.norm(np.diff(path, axis=0), axis=-1)
    path_length = float(segment_lengths.sum())
    optimal = float(np.linalg.norm(path[0] - target))
    error = float(distances[-1])
    success = float(error <= success_distance_m)
    oracle = float(distances.min() <= success_distance_m)
    spl = success * optimal / max(path_length, optimal, 1e-9)
    progress = 1.0 if optimal <= 1e-9 else float(np.clip(1.0 - error / optimal, 0, 1))
    return EpisodeMetrics(error, success, oracle, spl, path_length, optimal, progress)


def aggregate_metrics(metrics: Iterable[EpisodeMetrics]) -> Mapping[str, float]:
    metrics = list(metrics)
    if not metrics:
        raise ValueError("cannot aggregate an empty evaluation")
    values = {key: np.asarray([asdict(item)[key] for item in metrics]) for key in asdict(metrics[0])}
    return {
        "ne": float(values["navigation_error"].mean()),
        "sr": float(values["success"].mean() * 100.0),
        "osr": float(values["oracle_success"].mean() * 100.0),
        "spl": float(values["spl"].mean() * 100.0),
        "path_length": float(values["path_length"].mean()),
        "optimal_length": float(values["optimal_length"].mean()),
        "progress": float(values["progress"].mean() * 100.0),
        "episodes": len(metrics),
    }

