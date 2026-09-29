"""Numerically safe SBF and selector objectives from the paper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch.nn import functional as F


def _expand_mask(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.shape == values.shape:
        return mask.bool()
    if mask.shape == values.shape[1:]:
        return mask.unsqueeze(0).expand_as(values).bool()
    raise ValueError(f"mask shape {mask.shape} incompatible with values {values.shape}")


def _require_nonempty(mask: torch.Tensor, name: str) -> None:
    if (~mask.flatten(1).any(dim=1)).any():
        raise ValueError(f"every sample needs at least one valid {name}")


def masked_log_softmax(values: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    mask = _expand_mask(values, mask)
    _require_nonempty(mask, "entry")
    masked = values.masked_fill(~mask, -torch.inf)
    output = F.log_softmax(masked, dim=dim)
    return output.masked_fill(~mask, 0.0)


def soft_field_cross_entropy(
    logits: torch.Tensor, targets: torch.Tensor, valid_mask: torch.Tensor
) -> torch.Tensor:
    if logits.shape != targets.shape:
        raise ValueError("field logits and targets must have equal shape")
    valid = _expand_mask(logits, valid_mask)
    _require_nonempty(valid, "field cell")
    flat_valid = valid.flatten(1)
    log_probabilities = masked_log_softmax(logits.flatten(1), flat_valid)
    target = targets.flatten(1).masked_fill(~flat_valid, 0.0)
    normalizer = target.sum(dim=-1, keepdim=True)
    if (normalizer <= 0).any() or not torch.isfinite(normalizer).all():
        raise ValueError("field target must have positive finite valid mass")
    target = target / normalizer
    return -(target * log_probabilities).sum(dim=-1).mean()


def listwise_distance_loss(
    logits: torch.Tensor,
    distances_m: torch.Tensor,
    candidate_mask: torch.Tensor,
    *,
    temperature_m: float,
) -> torch.Tensor:
    if temperature_m <= 0:
        raise ValueError("temperature_m must be positive")
    mask = _expand_mask(logits, candidate_mask)
    _require_nonempty(mask, "candidate")
    target_log = masked_log_softmax(-distances_m / temperature_m, mask)
    target = target_log.exp().masked_fill(~mask, 0.0)
    prediction_log = masked_log_softmax(logits, mask)
    return -(target * prediction_log).sum(dim=-1).mean()


def pairwise_margin_loss(
    logits: torch.Tensor,
    distances_m: torch.Tensor,
    candidate_mask: torch.Tensor,
    *,
    good_radius_m: float,
    bad_radius_m: float,
    margin: float,
) -> torch.Tensor:
    if good_radius_m >= bad_radius_m:
        raise ValueError("good_radius_m must be smaller than bad_radius_m")
    mask = _expand_mask(logits, candidate_mask)
    good = mask & (distances_m <= good_radius_m)
    bad = mask & (distances_m >= bad_radius_m)
    pair_mask = good.unsqueeze(2) & bad.unsqueeze(1)
    # matrix[b, i, j] = margin + r_j - r_i
    # Selector padding is represented by -inf. Subtracting two padded logits
    # yields NaN before multiplication by the false pair mask, so sanitize the
    # excluded entries before constructing all pair differences.
    safe_logits = logits.masked_fill(~mask, 0.0)
    matrix = F.relu(margin + safe_logits.unsqueeze(1) - safe_logits.unsqueeze(2))
    counts = pair_mask.sum(dim=(1, 2))
    totals = (matrix * pair_mask).sum(dim=(1, 2))
    per_sample = torch.where(counts > 0, totals / counts.clamp_min(1), totals * 0.0)
    return per_sample.mean()


def hard_nearest_loss(
    logits: torch.Tensor, distances_m: torch.Tensor, candidate_mask: torch.Tensor
) -> torch.Tensor:
    mask = _expand_mask(logits, candidate_mask)
    _require_nonempty(mask, "candidate")
    nearest = distances_m.masked_fill(~mask, torch.inf).argmin(dim=-1)
    log_probabilities = masked_log_softmax(logits, mask)
    return -log_probabilities.gather(1, nearest.unsqueeze(1)).mean()


def altitude_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.smooth_l1_loss(prediction, target)


def progress_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(prediction, target)


@dataclass(frozen=True)
class LossWeights:
    field: float = 1.0
    rank: float = 1.0
    margin: float = 0.2
    hard: float = 0.5
    altitude: float = 0.1
    progress: float = 0.1


def combine_losses(
    *,
    field: torch.Tensor,
    listwise: torch.Tensor,
    margin: torch.Tensor,
    hard: torch.Tensor,
    altitude: torch.Tensor,
    progress: torch.Tensor,
    weights: LossWeights,
) -> Mapping[str, torch.Tensor]:
    rank = listwise + weights.margin * margin + weights.hard * hard
    total = (
        weights.field * field
        + weights.rank * rank
        + weights.altitude * altitude
        + weights.progress * progress
    )
    return {
        "field": field,
        "list": listwise,
        "margin": margin,
        "hard": hard,
        "rank": rank,
        "altitude": altitude,
        "progress": progress,
        "total": total,
    }
