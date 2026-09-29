"""Frozen SigLIP-B/16 token, pooled-text, and vision feature extraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel, AutoProcessor


@dataclass(frozen=True)
class SiglipFeatures:
    tokens: torch.Tensor
    pooled: torch.Tensor
    attention_mask: torch.Tensor


class FrozenSiglipBackbone(nn.Module):
    """Shared SigLIP model; frozen by default as required by the paper."""

    def __init__(
        self,
        model_name: str = "google/siglip-base-patch16-224",
        *,
        freeze: bool = True,
        local_files_only: bool = True,
    ):
        super().__init__()
        self.model_name = model_name
        self.processor = AutoProcessor.from_pretrained(
            model_name, local_files_only=local_files_only, use_fast=False
        )
        self.model = AutoModel.from_pretrained(model_name, local_files_only=local_files_only)
        self.freeze = bool(freeze)
        if self.freeze:
            self.model.requires_grad_(False)
            self.model.eval()

    @property
    def text_dim(self) -> int:
        return int(self.model.config.text_config.hidden_size)

    @property
    def vision_dim(self) -> int:
        return int(self.model.config.vision_config.hidden_size)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            self.model.eval()
        return self

    def tokenize(self, texts: Sequence[str], device: torch.device | str) -> dict[str, torch.Tensor]:
        encoded = self.processor(
            text=list(texts),
            padding="max_length",
            truncation=True,
            max_length=int(self.model.config.text_config.max_position_embeddings),
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        pad_id = int(self.model.config.text_config.pad_token_id)
        attention_mask = input_ids.ne(pad_id)
        # Empty text still needs one valid position for finite attention.
        attention_mask[:, 0] = True
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def encode_text_ids(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> SiglipFeatures:
        if input_ids.shape != attention_mask.shape:
            raise ValueError("input_ids and attention_mask must have equal shape")
        context = torch.no_grad() if self.freeze else torch.enable_grad()
        with context:
            output = self.model.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
        return SiglipFeatures(output.last_hidden_state, output.pooler_output, attention_mask.bool())

    def encode_text(self, texts: Sequence[str], device: torch.device | str) -> SiglipFeatures:
        encoded = self.tokenize(texts, device)
        return self.encode_text_ids(encoded["input_ids"], encoded["attention_mask"])

    def preprocess_images(self, images, device: torch.device | str) -> torch.Tensor:
        encoded = self.processor(images=images, return_tensors="pt")
        return encoded["pixel_values"].to(device)

    def preprocess_float_images(
        self, images: np.ndarray | torch.Tensor, device: torch.device | str
    ) -> torch.Tensor:
        """Vectorized SigLIP preprocessing for RGB images already in [0, 1]."""
        pixels = torch.as_tensor(images, dtype=torch.float32, device=device)
        if pixels.ndim != 4 or pixels.shape[-1] != 3:
            raise ValueError("images must be an NHWC RGB batch")
        pixels = pixels.permute(0, 3, 1, 2).contiguous()
        target = int(self.model.config.vision_config.image_size)
        if pixels.shape[-2:] != (target, target):
            pixels = F.interpolate(
                pixels,
                size=(target, target),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
        # google/siglip-base-patch16-224 uses image mean/std [0.5, 0.5, 0.5].
        return pixels.mul(2.0).sub(1.0)

    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        context = torch.no_grad() if self.freeze else torch.enable_grad()
        with context:
            output = self.model.vision_model(pixel_values=pixel_values, return_dict=True)
        return output.pooler_output
