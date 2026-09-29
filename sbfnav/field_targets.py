"""Metric Gaussian field and auxiliary supervision targets."""

from __future__ import annotations

import numpy as np

from .coordinates import MapTransform


def gaussian_field_target(
    transform: MapTransform,
    target_xy: np.ndarray,
    *,
    sigma_m: float,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Return a normalized 28×28 Gaussian evaluated in metric coordinates."""
    if sigma_m <= 0:
        raise ValueError("sigma_m must be positive")
    size = transform.field_size
    if valid_mask is None:
        valid_mask = transform.valid_mask(size)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if valid_mask.shape != (size, size):
        raise ValueError(f"expected valid mask {(size, size)}, got {valid_mask.shape}")
    if not valid_mask.any():
        raise ValueError("cannot build a field target with no valid cells")
    rows, cols = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    world = transform.field_to_world(np.stack((rows, cols), axis=-1))
    target_xy = np.asarray(target_xy, dtype=np.float64)
    squared_distance = np.square(world - target_xy).sum(axis=-1)
    # Shift valid logits before exponentiation to remain finite for tiny sigma.
    log_weights = -0.5 * squared_distance / float(sigma_m) ** 2
    log_weights[~valid_mask] = -np.inf
    log_weights -= np.max(log_weights[valid_mask])
    weights = np.exp(log_weights, where=valid_mask, out=np.zeros_like(log_weights))
    normalizer = weights.sum()
    if not np.isfinite(normalizer) or normalizer <= 0:
        raise FloatingPointError("invalid Gaussian target normalizer")
    return (weights / normalizer).astype(np.float32)


def progress_target(
    current_xyz: np.ndarray,
    start_xyz: np.ndarray,
    target_xyz: np.ndarray,
) -> float:
    """Reproduction-assumption progress target in [0, 1]."""
    current_xyz = np.asarray(current_xyz, dtype=np.float64)
    start_xyz = np.asarray(start_xyz, dtype=np.float64)
    target_xyz = np.asarray(target_xyz, dtype=np.float64)
    initial = float(np.linalg.norm(start_xyz - target_xyz))
    current = float(np.linalg.norm(current_xyz - target_xyz))
    if initial <= 1e-9:
        return 1.0
    return float(np.clip(1.0 - current / initial, 0.0, 1.0))

