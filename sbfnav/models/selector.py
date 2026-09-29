"""Semantic-geometric candidate selector with no SBF-score input path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np
import torch
from torch import nn

from sbfnav.candidate_proposal import Candidate
from sbfnav.planning_state import PlanningState


def relative_geometry(
    origin_xy: torch.Tensor, destination_xy: torch.Tensor, scale_m: torch.Tensor | float
) -> torch.Tensor:
    """Return [dx,dy,d,sin(beta),cos(beta)] from origin to destination."""
    delta = destination_xy - origin_xy
    distance = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
    safe_distance = distance.clamp_min(torch.finfo(delta.dtype).eps)
    sin_beta = delta[..., 1:2] / safe_distance
    cos_beta = delta[..., 0:1] / safe_distance
    zero = distance <= torch.finfo(delta.dtype).eps
    sin_beta = torch.where(zero, torch.zeros_like(sin_beta), sin_beta)
    cos_beta = torch.where(zero, torch.ones_like(cos_beta), cos_beta)
    scale = torch.as_tensor(scale_m, dtype=delta.dtype, device=delta.device)
    while scale.ndim < delta.ndim:
        scale = scale.unsqueeze(-1)
    return torch.cat((delta / scale, distance / scale, sin_beta, cos_beta), dim=-1)


@dataclass(frozen=True)
class SelectorInputs:
    candidate_visual: torch.Tensor
    candidate_xy: torch.Tensor
    candidate_mask: torch.Tensor
    instruction_tokens: torch.Tensor
    instruction_mask: torch.Tensor
    landmark_name_features: torch.Tensor
    landmark_xy: torch.Tensor
    landmark_mask: torch.Tensor
    agent_xy: torch.Tensor
    map_scale_m: torch.Tensor


def candidate_coordinates(
    batches: Sequence[Sequence[Candidate]], *, device: torch.device | str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert candidates to coordinates only; field scores are discarded."""
    if not batches or any(len(candidates) == 0 for candidates in batches):
        raise ValueError("every selector sample needs at least one candidate")
    maximum = max(len(candidates) for candidates in batches)
    coordinates = torch.zeros((len(batches), maximum, 2), dtype=torch.float32, device=device)
    mask = torch.zeros((len(batches), maximum), dtype=torch.bool, device=device)
    for batch_index, candidates in enumerate(batches):
        coordinates[batch_index, : len(candidates)] = torch.as_tensor(
            np.stack([candidate.xy for candidate in candidates]), dtype=torch.float32, device=device
        )
        mask[batch_index, : len(candidates)] = True
    return coordinates, mask


def extract_candidate_crops(
    state: PlanningState,
    candidates: Sequence[Candidate],
    *,
    extent_m: float = 100.0,
    output_size: int = 224,
) -> np.ndarray:
    """Extract north-up local accumulated-RGB crops, zero-padded at bounds."""
    if extent_m <= 0 or output_size <= 0:
        raise ValueError("crop extent and output size must be positive")
    rgb = np.moveaxis(state.values[:3], 0, -1)
    source_size = max(1.0, extent_m / state.transform.canvas_meters_per_pixel)
    crops = []
    for candidate in candidates:
        row, col = state.transform.world_to_canvas(candidate.xy)
        scale = source_size / output_size
        destination_to_source = np.asarray(
            (
                (scale, 0.0, float(col) - scale * (output_size - 1) / 2.0),
                (0.0, scale, float(row) - scale * (output_size - 1) / 2.0),
            ),
            dtype=np.float32,
        )
        crop = cv2.warpAffine(
            rgb,
            destination_to_source,
            (output_size, output_size),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        crops.append(np.clip(crop, 0, 1).astype(np.float32))
    return np.stack(crops)


class SemanticGeometricSelector(nn.Module):
    """Score candidates from vision, text, landmarks, and explicit geometry.

    There is intentionally no `field_score` argument or projection.
    """

    def __init__(
        self,
        *,
        vision_dim: int = 768,
        language_dim: int = 768,
        hidden_dim: int = 256,
        layers: int = 2,
        attention_heads: int = 8,
        dropout: float = 0.1,
        use_geometry: bool = True,
        use_landmark_text: bool = True,
    ):
        super().__init__()
        if hidden_dim % attention_heads:
            raise ValueError("hidden_dim must be divisible by attention_heads")
        self.visual_projection = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim), nn.LayerNorm(hidden_dim)
        )
        self.instruction_projection = nn.Sequential(
            nn.Linear(language_dim, hidden_dim), nn.LayerNorm(hidden_dim)
        )
        self.landmark_projection = nn.Sequential(
            nn.Linear(language_dim, hidden_dim), nn.LayerNorm(hidden_dim)
        )
        self.geometry_projection = nn.Sequential(
            nn.Linear(5, hidden_dim), nn.LayerNorm(hidden_dim)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=attention_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=layers)
        self.scr = nn.Parameter(torch.empty(hidden_dim))
        nn.init.normal_(self.scr, std=0.02)
        self.score = nn.Linear(hidden_dim, 1)
        self.use_geometry = use_geometry
        self.use_landmark_text = use_landmark_text

    def forward(self, inputs: SelectorInputs) -> torch.Tensor:
        visual = inputs.candidate_visual
        batch, candidates, _dimension = visual.shape
        if inputs.candidate_xy.shape != (batch, candidates, 2):
            raise ValueError("candidate coordinate shape mismatch")
        if inputs.candidate_mask.shape != (batch, candidates):
            raise ValueError("candidate mask shape mismatch")
        if (~inputs.candidate_mask.any(dim=1)).any():
            raise ValueError("every sample needs at least one candidate")
        instruction_length = inputs.instruction_tokens.shape[1]
        landmark_count = inputs.landmark_name_features.shape[1]

        visual_token = self.visual_projection(visual).reshape(batch * candidates, 1, -1)
        candidate_xy = inputs.candidate_xy
        agent_geometry = relative_geometry(
            inputs.agent_xy[:, None, :], candidate_xy, inputs.map_scale_m[:, None]
        )
        if not self.use_geometry:
            agent_geometry = torch.zeros_like(agent_geometry)
        agent_token = self.geometry_projection(agent_geometry).reshape(batch * candidates, 1, -1)

        instruction = self.instruction_projection(inputs.instruction_tokens)
        instruction = instruction[:, None].expand(-1, candidates, -1, -1)
        instruction = instruction.reshape(batch * candidates, instruction_length, -1)

        landmarks = inputs.landmark_xy[:, None].expand(-1, candidates, -1, -1)
        candidate_origins = candidate_xy[:, :, None].expand(-1, -1, landmark_count, -1)
        landmark_geometry = relative_geometry(
            landmarks,
            candidate_origins,
            inputs.map_scale_m[:, None, None],
        )
        if not self.use_geometry:
            landmark_geometry = torch.zeros_like(landmark_geometry)
        landmark_token = self.geometry_projection(landmark_geometry)
        if self.use_landmark_text:
            names = self.landmark_projection(inputs.landmark_name_features)
            landmark_token = landmark_token + names[:, None]
        landmark_token = landmark_token.reshape(batch * candidates, landmark_count, -1)

        scr = self.scr.view(1, 1, -1).expand(batch * candidates, -1, -1)
        tokens = torch.cat((scr, visual_token, agent_token, instruction, landmark_token), dim=1)
        base_mask = torch.zeros((batch, 3), dtype=torch.bool, device=visual.device)
        padding = torch.cat(
            (base_mask, ~inputs.instruction_mask.bool(), ~inputs.landmark_mask.bool()), dim=1
        )
        padding = padding[:, None].expand(-1, candidates, -1).reshape(batch * candidates, -1)
        encoded = self.transformer(tokens, src_key_padding_mask=padding)
        logits = self.score(encoded[:, 0]).reshape(batch, candidates)
        return logits.masked_fill(~inputs.candidate_mask.bool(), -torch.inf)
