"""Compact residual 224→28 map encoder (documented reproduction assumption)."""

from __future__ import annotations

import torch
from torch import nn


def _groups(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.norm1 = nn.GroupNorm(_groups(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.norm2 = nn.GroupNorm(_groups(out_channels), out_channels)
        self.activation = nn.GELU()
        self.skip = (
            nn.Identity()
            if stride == 1 and in_channels == out_channels
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.GroupNorm(_groups(out_channels), out_channels),
            )
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.skip(inputs)
        output = self.activation(self.norm1(self.conv1(inputs)))
        output = self.norm2(self.conv2(output))
        return self.activation(output + residual)


class CompactResidualCNN(nn.Module):
    """Encode B×9×224×224 into B×D×28×28."""

    def __init__(self, input_channels: int = 9, hidden_dim: int = 256):
        super().__init__()
        widths = (64, 128, hidden_dim)
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, widths[0], 5, 2, 2, bias=False),
            nn.GroupNorm(_groups(widths[0]), widths[0]),
            nn.GELU(),
        )
        self.stages = nn.Sequential(
            ResidualBlock(widths[0], widths[0]),
            ResidualBlock(widths[0], widths[1], stride=2),
            ResidualBlock(widths[1], widths[1]),
            ResidualBlock(widths[1], widths[2], stride=2),
            ResidualBlock(widths[2], widths[2]),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != self.stem[0].in_channels:
            raise ValueError(
                f"expected B×{self.stem[0].in_channels}×H×W input, got {tuple(inputs.shape)}"
            )
        output = self.stages(self.stem(inputs))
        if output.shape[-2:] != (28, 28):
            raise ValueError(f"expected a 28×28 feature grid, got {tuple(output.shape[-2:])}")
        return output

