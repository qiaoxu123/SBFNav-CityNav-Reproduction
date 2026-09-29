"""Construct SBFNav components from resolved YAML configuration."""

from __future__ import annotations

from .models.language_encoder import FrozenSiglipBackbone
from .models.sbfnav import SBFNav


def build_model(config: dict) -> SBFNav:
    model = config["model"]
    selector = config["selector"]
    return SBFNav(
        hidden_dim=model["hidden_dim"],
        language_dim=768,
        vision_dim=768,
        attention_heads=model["attention_heads"],
        selector_layers=selector["layers"],
        dropout=model["dropout"],
        use_cross_attention=model["use_cross_attention"],
        pooled_text_only=model["pooled_text_only"],
        use_geometry=selector["use_geometry"],
        use_landmark_text=selector["use_landmark_text"],
    )


def build_backbone(config: dict) -> FrozenSiglipBackbone:
    model = config["model"]
    return FrozenSiglipBackbone(
        model["siglip_name"],
        freeze=model["freeze_siglip"],
        local_files_only=model["local_files_only"],
    )

