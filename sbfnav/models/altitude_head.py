"""Auxiliary altitude and trajectory-progress prediction heads."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class AuxiliaryOutput:
    altitude_normalized: torch.Tensor
    progress: torch.Tensor


class AltitudeProgressHead(nn.Module):
    def __init__(self, spatial_dim: int = 256, language_dim: int = 768, hidden_dim: int = 256):
        super().__init__()
        self.language_projection = nn.Linear(language_dim, hidden_dim)
        self.spatial_projection = nn.Linear(spatial_dim, hidden_dim)
        self.fusion = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.altitude = nn.Linear(hidden_dim, 1)
        self.progress = nn.Linear(hidden_dim, 1)

    def forward(
        self, spatial_features: torch.Tensor, pooled_language: torch.Tensor
    ) -> AuxiliaryOutput:
        if spatial_features.ndim != 4:
            raise ValueError("spatial_features must be B×C×H×W")
        spatial = spatial_features.mean(dim=(-2, -1))
        fused = self.fusion(
            self.spatial_projection(spatial) + self.language_projection(pooled_language)
        )
        return AuxiliaryOutput(
            torch.sigmoid(self.altitude(fused).squeeze(-1)),
            torch.sigmoid(self.progress(fused).squeeze(-1)),
        )


def altitude_to_world(
    normalized: torch.Tensor,
    ground_level: torch.Tensor,
    *,
    minimum_above_ground_m: float,
    maximum_above_ground_m: float,
) -> torch.Tensor:
    if minimum_above_ground_m >= maximum_above_ground_m:
        raise ValueError("minimum altitude must be below maximum altitude")
    normalized = normalized.clamp(0, 1)
    return ground_level + minimum_above_ground_m + normalized * (
        maximum_above_ground_m - minimum_above_ground_m
    )


def altitude_target_normalized(
    target_z: torch.Tensor,
    ground_level: torch.Tensor,
    *,
    minimum_above_ground_m: float,
    maximum_above_ground_m: float,
) -> torch.Tensor:
    span = maximum_above_ground_m - minimum_above_ground_m
    if span <= 0:
        raise ValueError("minimum altitude must be below maximum altitude")
    return ((target_z - ground_level - minimum_above_ground_m) / span).clamp(0, 1)

