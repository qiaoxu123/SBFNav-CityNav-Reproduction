"""Trainable SBFNav heads composed around precomputed frozen SigLIP features."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .altitude_head import AltitudeProgressHead, AuxiliaryOutput
from .belief_field import BeliefFieldOutput, SpatialBeliefField
from .selector import SelectorInputs, SemanticGeometricSelector


@dataclass(frozen=True)
class SBFNavOutput:
    field: BeliefFieldOutput
    auxiliary: AuxiliaryOutput
    selector_logits: torch.Tensor | None


class SBFNav(nn.Module):
    """Paper architecture excluding the shared frozen SigLIP feature extractor."""

    def __init__(
        self,
        *,
        hidden_dim: int = 256,
        language_dim: int = 768,
        vision_dim: int = 768,
        attention_heads: int = 8,
        selector_layers: int = 2,
        dropout: float = 0.1,
        use_cross_attention: bool = True,
        pooled_text_only: bool = False,
        use_geometry: bool = True,
        use_landmark_text: bool = True,
    ):
        super().__init__()
        self.belief_field = SpatialBeliefField(
            hidden_dim=hidden_dim,
            language_dim=language_dim,
            attention_heads=attention_heads,
            dropout=dropout,
            use_cross_attention=use_cross_attention,
            pooled_text_only=pooled_text_only,
        )
        self.selector = SemanticGeometricSelector(
            vision_dim=vision_dim,
            language_dim=language_dim,
            hidden_dim=hidden_dim,
            layers=selector_layers,
            attention_heads=attention_heads,
            dropout=dropout,
            use_geometry=use_geometry,
            use_landmark_text=use_landmark_text,
        )
        self.auxiliary = AltitudeProgressHead(hidden_dim, language_dim, hidden_dim)

    def forward(
        self,
        planning_state: torch.Tensor,
        valid_field_mask: torch.Tensor,
        instruction_tokens: torch.Tensor,
        instruction_mask: torch.Tensor,
        pooled_instruction: torch.Tensor,
        selector_inputs: SelectorInputs | None = None,
    ) -> SBFNavOutput:
        field = self.belief_field(
            planning_state,
            instruction_tokens,
            instruction_mask,
            valid_field_mask,
            pooled_language=pooled_instruction,
        )
        auxiliary = self.auxiliary(field.spatial_features, pooled_instruction)
        selector_logits = None if selector_inputs is None else self.selector(selector_inputs)
        return SBFNavOutput(field, auxiliary, selector_logits)

