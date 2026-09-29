"""NMS field proposals, landmark injection, deduplication, and oracle analysis."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch

from .coordinates import MapTransform
from .dataset import Landmark


@dataclass(frozen=True)
class Candidate:
    xy: np.ndarray
    sources: tuple[str, ...]
    field_index: tuple[int, int] | None = None
    # Retained only in proposal diagnostics. Selector APIs deliberately never
    # consume this member.
    field_score: float | None = None

    def __post_init__(self) -> None:
        xy = np.asarray(self.xy, dtype=np.float64)
        if xy.shape != (2,) or not np.isfinite(xy).all():
            raise ValueError(f"candidate xy must be a finite pair, got {xy}")
        object.__setattr__(self, "xy", xy)
        if not self.sources:
            raise ValueError("candidate must retain at least one source")


def nms_topk_indices(
    field_logits: torch.Tensor | np.ndarray,
    valid_mask: torch.Tensor | np.ndarray,
    *,
    top_k: int,
    kernel_size: int = 3,
) -> list[tuple[int, int, float]]:
    """Return deterministic greedy-NMS row/col/score peaks.

    ``max_pool == input`` only identifies strict/local maxima.  A smooth
    unimodal field can therefore yield one maximum and previously caused the
    remaining slots to be filled with adjacent, *unsuppressed* pixels.  That
    silently defeated the paper's NMS step and made Top-5/Top-K effectively
    identical to Top-1.  Greedy suppression always applies the configured
    neighbourhood before selecting the next-highest valid location.
    """
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")
    logits = torch.as_tensor(field_logits, dtype=torch.float32).detach().cpu()
    valid = torch.as_tensor(valid_mask, dtype=torch.bool).detach().cpu()
    if logits.ndim != 2 or valid.shape != logits.shape:
        raise ValueError("field_logits and valid_mask must be equal 2-D arrays")
    if not valid.any():
        raise ValueError("candidate proposal has no valid field cells")
    scores = logits.masked_fill(~valid | ~torch.isfinite(logits), -torch.inf).clone()
    selected: list[tuple[int, int, float]] = []
    radius = kernel_size // 2
    for _ in range(min(top_k, int(valid.sum()))):
        flat_index = int(torch.argmax(scores).item())
        score = float(scores.flatten()[flat_index])
        if not np.isfinite(score):
            break
        row, col = divmod(flat_index, scores.shape[1])
        selected.append((row, col, score))
        row_start, row_stop = max(0, row - radius), min(scores.shape[0], row + radius + 1)
        col_start, col_stop = max(0, col - radius), min(scores.shape[1], col + radius + 1)
        scores[row_start:row_stop, col_start:col_stop] = -torch.inf
    return selected


def _merge_sources(first: tuple[str, ...], second: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(first + second))


def deduplicate_candidates(
    candidates: Iterable[Candidate], *, radius_m: float
) -> list[Candidate]:
    if radius_m < 0:
        raise ValueError("deduplication radius cannot be negative")
    result: list[Candidate] = []
    for candidate in candidates:
        duplicate = next(
            (
                index
                for index, existing in enumerate(result)
                if np.linalg.norm(existing.xy - candidate.xy) <= radius_m
            ),
            None,
        )
        if duplicate is None:
            result.append(candidate)
        else:
            existing = result[duplicate]
            result[duplicate] = replace(
                existing, sources=_merge_sources(existing.sources, candidate.sources)
            )
    return result


def propose_candidates(
    field_logits: torch.Tensor | np.ndarray,
    valid_mask: torch.Tensor | np.ndarray,
    transform: MapTransform,
    referenced_landmarks: Sequence[Landmark],
    *,
    top_k: int = 16,
    nms_kernel: int = 3,
    dedup_radius_m: float = 2.0,
    add_landmarks: bool = True,
) -> list[Candidate]:
    peaks = nms_topk_indices(
        field_logits, valid_mask, top_k=top_k, kernel_size=nms_kernel
    )
    candidates = [
        Candidate(
            xy=transform.field_to_world((row, col)),
            sources=("field_peak",),
            field_index=(row, col),
            field_score=score,
        )
        for row, col, score in peaks
    ]
    if add_landmarks:
        for landmark in referenced_landmarks:
            centroid = landmark.centroid
            # Reject materially invalid centroids. Clip only numerical noise at
            # an annotated boundary.
            if not bool(transform.contains_world(centroid)):
                clipped = transform.clip_world(centroid)
                if np.linalg.norm(clipped - centroid) > 1e-5:
                    continue
                centroid = clipped
            candidates.append(
                Candidate(
                    xy=centroid,
                    sources=("landmark_centroid",),
                    field_index=None,
                    field_score=None,
                )
            )
    return deduplicate_candidates(candidates, radius_m=dedup_radius_m)


def propose_uniform_grid_candidates(
    valid_mask: torch.Tensor | np.ndarray,
    transform: MapTransform,
    referenced_landmarks: Sequence[Landmark],
    *,
    top_k: int = 16,
    dedup_radius_m: float = 2.0,
    add_landmarks: bool = True,
) -> list[Candidate]:
    """Deterministic field-free candidates for the selector-only control."""
    if top_k < 1:
        raise ValueError("top_k must be positive")
    valid = torch.as_tensor(valid_mask, dtype=torch.bool).cpu().numpy()
    if valid.ndim != 2 or not valid.any():
        raise ValueError("uniform proposal needs a non-empty 2-D valid mask")
    positions = np.argwhere(valid)
    side = int(np.ceil(np.sqrt(top_k)))
    fractions = (np.arange(side, dtype=np.float64) + 0.5) / side
    desired = np.asarray(
        [(row * valid.shape[0], col * valid.shape[1]) for row in fractions for col in fractions]
    )[:top_k]
    chosen: list[tuple[int, int]] = []
    for point in desired:
        order = np.argsort(np.square(positions - point).sum(axis=1), kind="stable")
        selected = next(
            (tuple(int(value) for value in positions[index]) for index in order
             if tuple(int(value) for value in positions[index]) not in chosen),
            None,
        )
        if selected is not None:
            chosen.append(selected)
    candidates = [
        Candidate(
            xy=transform.field_to_world(row_col),
            sources=("uniform_grid",),
            field_index=row_col,
            field_score=None,
        )
        for row_col in chosen
    ]
    if add_landmarks:
        for landmark in referenced_landmarks:
            if bool(transform.contains_world(landmark.centroid)):
                candidates.append(
                    Candidate(landmark.centroid, ("landmark_centroid",), field_score=None)
                )
    return deduplicate_candidates(candidates, radius_m=dedup_radius_m)


def oracle_candidate_metrics(
    candidates: Sequence[Candidate],
    target_xy: np.ndarray,
    *,
    ks: Sequence[int] = (1, 5, 16),
    thresholds_m: Sequence[float] = (10.0, 20.0),
) -> Mapping[str, float]:
    target_xy = np.asarray(target_xy, dtype=np.float64)
    if not candidates:
        result = {"oracle_distance_m": float("inf")}
        for k in ks:
            for threshold in thresholds_m:
                result[f"coverage@{k}_{threshold:g}m"] = 0.0
        return result
    distances = np.asarray([np.linalg.norm(candidate.xy - target_xy) for candidate in candidates])
    result = {"oracle_distance_m": float(distances.min())}
    for k in ks:
        if k < 1:
            raise ValueError("coverage K must be positive")
        prefix = distances[: min(k, len(distances))]
        for threshold in thresholds_m:
            result[f"coverage@{k}_{threshold:g}m"] = float(prefix.min() <= threshold)
    return result
