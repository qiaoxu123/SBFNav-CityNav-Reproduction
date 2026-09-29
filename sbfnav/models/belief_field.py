"""Language-conditioned Spatial Belief Field predictor."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .residual_cnn import CompactResidualCNN, ResidualBlock


@dataclass(frozen=True)
class BeliefFieldOutput:
    logits: torch.Tensor
    probabilities: torch.Tensor
    spatial_features: torch.Tensor
    gate: torch.Tensor


def masked_spatial_softmax(logits: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    if valid_mask.ndim == 2:
        valid_mask = valid_mask.unsqueeze(0).expand(logits.shape[0], -1, -1)
    if logits.shape != valid_mask.shape:
        raise ValueError(f"logit/mask shape mismatch: {logits.shape} vs {valid_mask.shape}")
    valid_mask = valid_mask.bool()
    if (~valid_mask.flatten(1).any(dim=1)).any():
        raise ValueError("every sample must contain at least one valid field cell")
    flat_logits = logits.flatten(1).masked_fill(~valid_mask.flatten(1), -torch.inf)
    return torch.softmax(flat_logits, dim=-1).reshape_as(logits)


class SpatialBeliefField(nn.Module):
    def __init__(
        self,
        *,
        input_channels: int = 9,
        hidden_dim: int = 256,
        language_dim: int = 768,
        attention_heads: int = 8,
        dropout: float = 0.1,
        use_cross_attention: bool = True,
        pooled_text_only: bool = False,
        gate_init: float = 0.0,
    ):
        super().__init__()
        if hidden_dim % attention_heads:
            raise ValueError("hidden_dim must be divisible by attention_heads")
        self.encoder = CompactResidualCNN(input_channels, hidden_dim)
        self.language_projection = nn.Sequential(
            nn.Linear(language_dim, hidden_dim), nn.LayerNorm(hidden_dim)
        )
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, attention_heads, dropout=dropout, batch_first=True
        )
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.gamma = nn.Parameter(torch.tensor(float(gate_init)))
        self.decoder = nn.Sequential(
            ResidualBlock(hidden_dim, hidden_dim), nn.Conv2d(hidden_dim, 1, 1)
        )
        self.use_cross_attention = use_cross_attention
        self.pooled_text_only = pooled_text_only

    def forward(
        self,
        planning_state: torch.Tensor,
        language_tokens: torch.Tensor,
        language_mask: torch.Tensor,
        valid_field_mask: torch.Tensor,
        *,
        pooled_language: torch.Tensor | None = None,
    ) -> BeliefFieldOutput:
        features = self.encoder(planning_state)
        batch, channels, height, width = features.shape
        spatial = features.flatten(2).transpose(1, 2)
        if self.pooled_text_only:
            if pooled_language is None:
                raise ValueError("pooled_language is required for pooled_text_only")
            language_tokens = pooled_language.unsqueeze(1)
            language_mask = torch.ones(
                (batch, 1), dtype=torch.bool, device=planning_state.device
            )
        if language_tokens.shape[:2] != language_mask.shape:
            raise ValueError("language token/mask shape mismatch")
        projected_language = self.language_projection(language_tokens)
        gate = torch.tanh(self.gamma)
        if self.use_cross_attention:
            attended, _ = self.cross_attention(
                query=self.attention_norm(spatial),
                key=projected_language,
                value=projected_language,
                key_padding_mask=~language_mask.bool(),
                need_weights=False,
            )
            spatial = spatial + gate * attended
        conditioned = spatial.transpose(1, 2).reshape(batch, channels, height, width)
        logits = self.decoder(conditioned).squeeze(1)
        probabilities = masked_spatial_softmax(logits, valid_field_mask)
        return BeliefFieldOutput(logits, probabilities, conditioned, gate)

