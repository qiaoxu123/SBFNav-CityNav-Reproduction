"""Shared frozen-feature preparation and one-step SBFNav inference."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from multiagent.mapdata import GROUND_LEVEL

from .candidate_proposal import Candidate, propose_candidates, propose_uniform_grid_candidates
from .models.altitude_head import altitude_to_world
from .models.language_encoder import FrozenSiglipBackbone, SiglipFeatures
from .models.sbfnav import SBFNav
from .models.selector import (
    SelectorInputs,
    candidate_coordinates,
    extract_candidate_crops,
)
from .navigation import WaypointPrediction
from .planning_state import PlanningState


class TextFeatureCache:
    """In-memory CPU fp16 cache for frozen text features."""

    def __init__(self, backbone: FrozenSiglipBackbone, device: torch.device):
        self.backbone = backbone
        self.device = device
        self._cache: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    def get(self, texts: Sequence[str]) -> SiglipFeatures:
        if not self.backbone.freeze:
            return self.backbone.encode_text(texts, self.device)
        missing = list(dict.fromkeys(text for text in texts if text not in self._cache))
        if missing:
            features = self.backbone.encode_text(missing, self.device)
            for index, text in enumerate(missing):
                self._cache[text] = (
                    features.tokens[index].detach().to("cpu", dtype=torch.float16),
                    features.pooled[index].detach().to("cpu", dtype=torch.float16),
                    features.attention_mask[index].detach().cpu(),
                )
        tokens, pooled, masks = zip(*(self._cache[text] for text in texts))
        return SiglipFeatures(
            torch.stack(tokens).to(self.device, dtype=torch.float32),
            torch.stack(pooled).to(self.device, dtype=torch.float32),
            torch.stack(masks).to(self.device),
        )

    def precompute(self, texts: Sequence[str], *, batch_size: int = 256) -> None:
        if batch_size < 1:
            raise ValueError("text precompute batch size must be positive")
        unique = list(dict.fromkeys(texts))
        for start in range(0, len(unique), batch_size):
            self.get(unique[start : start + batch_size])


class VisionFeatureCache:
    """Bounded CPU-fp16 cache for frozen training-state candidate crops."""

    def __init__(self, maximum_entries: int = 200_000):
        if maximum_entries < 0:
            raise ValueError("maximum cache entries cannot be negative")
        self.maximum_entries = maximum_entries
        self.values: OrderedDict[str, torch.Tensor] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> torch.Tensor | None:
        value = self.values.get(key)
        if value is None:
            self.misses += 1
            return None
        self.values.move_to_end(key)
        self.hits += 1
        return value

    def put(self, key: str, value: torch.Tensor) -> None:
        if self.maximum_entries == 0:
            return
        self.values[key] = value.detach().to("cpu", dtype=torch.float16)
        self.values.move_to_end(key)
        while len(self.values) > self.maximum_entries:
            self.values.popitem(last=False)

    def statistics(self) -> dict[str, int]:
        return {"entries": len(self.values), "hits": self.hits, "misses": self.misses}


def encode_candidate_visuals(
    states: Sequence[PlanningState],
    candidate_batches: Sequence[Sequence[Candidate]],
    backbone: FrozenSiglipBackbone,
    *,
    device: torch.device,
    crop_extent_m: float,
    crop_size: int,
    vision_batch_size: int,
    cache: VisionFeatureCache | None = None,
    cache_keys: Sequence[str] | None = None,
) -> torch.Tensor:
    if cache_keys is not None and len(cache_keys) != len(states):
        raise ValueError("one visual cache key is required per planning state")
    flat_candidates = [
        (sample_index, candidate_index, candidate)
        for sample_index, candidates in enumerate(candidate_batches)
        for candidate_index, candidate in enumerate(candidates)
    ]
    keys = [
        (
            f"{cache_keys[sample_index]}:{candidate.xy[0]:.4f}:{candidate.xy[1]:.4f}"
            if cache is not None and cache_keys is not None
            else None
        )
        for sample_index, _candidate_index, candidate in flat_candidates
    ]
    resolved: list[torch.Tensor | None] = [
        cache.get(key) if cache is not None and key is not None else None for key in keys
    ]
    missing_positions = [index for index, value in enumerate(resolved) if value is None]
    missing_crops = []
    for position in missing_positions:
        sample_index, _candidate_index, candidate = flat_candidates[position]
        missing_crops.append(
            extract_candidate_crops(
                states[sample_index], [candidate], extent_m=crop_extent_m, output_size=crop_size
            )[0]
        )
    features = []
    missing_array = np.asarray(missing_crops, dtype=np.float32)
    for start in range(0, len(missing_array), vision_batch_size):
        pixels = backbone.preprocess_float_images(
            missing_array[start : start + vision_batch_size], device
        )
        encoded = backbone.encode_images(pixels)
        features.append(encoded.detach() if backbone.freeze else encoded)
    if features:
        encoded_missing = torch.cat(features)
        for encoded_index, position in enumerate(missing_positions):
            resolved[position] = encoded_missing[encoded_index]
            key = keys[position]
            if cache is not None and key is not None:
                cache.put(key, encoded_missing[encoded_index])
    flat_features = torch.stack(
        [value.to(device=device, dtype=torch.float32) for value in resolved]
    )
    maximum = max(len(candidates) for candidates in candidate_batches)
    padded = torch.zeros(
        (len(states), maximum, flat_features.shape[-1]),
        dtype=flat_features.dtype,
        device=device,
    )
    offset = 0
    for index, candidates in enumerate(candidate_batches):
        padded[index, : len(candidates)] = flat_features[offset : offset + len(candidates)]
        offset += len(candidates)
    return padded


def build_selector_inputs(
    states: Sequence[PlanningState],
    candidate_batches: Sequence[Sequence[Candidate]],
    instructions: SiglipFeatures,
    backbone: FrozenSiglipBackbone,
    text_cache: TextFeatureCache,
    *,
    device: torch.device,
    crop_extent_m: float,
    crop_size: int,
    vision_batch_size: int,
    maximum_landmarks: int,
    visual_cache: VisionFeatureCache | None = None,
    visual_cache_keys: Sequence[str] | None = None,
) -> SelectorInputs:
    coordinates, candidate_mask = candidate_coordinates(candidate_batches, device=device)
    visuals = encode_candidate_visuals(
        states,
        candidate_batches,
        backbone,
        device=device,
        crop_extent_m=crop_extent_m,
        crop_size=crop_size,
        vision_batch_size=vision_batch_size,
        cache=visual_cache,
        cache_keys=visual_cache_keys,
    )
    landmark_lists = [list(state.referenced_landmarks[:maximum_landmarks]) for state in states]
    landmark_count = max(1, max(map(len, landmark_lists)))
    landmark_xy = torch.zeros((len(states), landmark_count, 2), device=device)
    landmark_mask = torch.zeros((len(states), landmark_count), dtype=torch.bool, device=device)
    landmark_features = torch.zeros(
        (len(states), landmark_count, backbone.text_dim),
        dtype=instructions.pooled.dtype,
        device=device,
    )
    unique_names = [landmark.name for landmarks in landmark_lists for landmark in landmarks]
    encoded_names = text_cache.get(unique_names).pooled if unique_names else None
    offset = 0
    for index, landmarks in enumerate(landmark_lists):
        if landmarks:
            count = len(landmarks)
            landmark_xy[index, :count] = torch.as_tensor(
                np.stack([landmark.centroid for landmark in landmarks]),
                dtype=torch.float32,
                device=device,
            )
            landmark_mask[index, :count] = True
            landmark_features[index, :count] = encoded_names[offset : offset + count]
            offset += count
    return SelectorInputs(
        candidate_visual=visuals,
        candidate_xy=coordinates,
        candidate_mask=candidate_mask,
        instruction_tokens=instructions.tokens,
        instruction_mask=instructions.attention_mask,
        landmark_name_features=landmark_features,
        landmark_xy=landmark_xy,
        landmark_mask=landmark_mask,
        agent_xy=torch.as_tensor(
            np.stack([state.pose[:2] for state in states]), dtype=torch.float32, device=device
        ),
        map_scale_m=torch.as_tensor(
            [state.transform.side_m for state in states], dtype=torch.float32, device=device
        ),
    )


class SBFNavRuntime:
    def __init__(
        self,
        model: SBFNav,
        backbone: FrozenSiglipBackbone,
        config: dict,
        device: torch.device,
    ):
        self.model = model
        self.backbone = backbone
        self.config = config
        self.device = device
        self.text_cache = TextFeatureCache(backbone, device)

    @torch.no_grad()
    def predict(self, state: PlanningState, instruction: str) -> WaypointPrediction:
        self.model.eval()
        instructions = self.text_cache.get([instruction])
        planning = torch.from_numpy(state.values).unsqueeze(0).to(self.device)
        valid = torch.from_numpy(state.valid_field).unsqueeze(0).to(self.device)
        field = self.model.belief_field(
            planning,
            instructions.tokens,
            instructions.attention_mask,
            valid,
            pooled_language=instructions.pooled,
        )
        proposal = self.config["proposal"]
        if proposal["mode"] == "field":
            candidates = propose_candidates(
                field.logits[0],
                state.valid_field,
                state.transform,
                state.referenced_landmarks,
                top_k=proposal["top_k"],
                nms_kernel=proposal["nms_kernel"],
                dedup_radius_m=proposal["dedup_radius_m"],
                add_landmarks=proposal["add_landmark_candidates"],
            )
        else:
            candidates = propose_uniform_grid_candidates(
                state.valid_field,
                state.transform,
                state.referenced_landmarks,
                top_k=proposal["top_k"],
                dedup_radius_m=proposal["dedup_radius_m"],
                add_landmarks=proposal["add_landmark_candidates"],
            )
        selector_config = self.config["selector"]
        if selector_config["selection_mode"] == "selector":
            selector_inputs = build_selector_inputs(
                [state],
                [candidates],
                instructions,
                self.backbone,
                self.text_cache,
                device=self.device,
                crop_extent_m=selector_config["crop_extent_m"],
                crop_size=selector_config["crop_size"],
                vision_batch_size=max(1, len(candidates)),
                maximum_landmarks=selector_config["max_landmarks"],
            )
            selector_logits = self.model.selector(selector_inputs)[0]
            selected_index = int(selector_logits.argmax())
            selector_logit_values = selector_logits.cpu().float().tolist()
        else:
            field_indices = [
                index for index, candidate in enumerate(candidates)
                if candidate.field_score is not None
            ]
            if not field_indices:
                raise ValueError("sbf_top1 selection requires field proposals")
            selected_index = max(
                field_indices, key=lambda index: float(candidates[index].field_score)
            )
            selector_logit_values = None
        auxiliary = self.model.auxiliary(field.spatial_features, instructions.pooled)
        altitude_config = self.config["altitude"]
        ground = torch.tensor([GROUND_LEVEL[state.transform.map_name]], device=self.device)
        altitude = altitude_to_world(
            auxiliary.altitude_normalized,
            ground,
            minimum_above_ground_m=altitude_config["min_above_ground_m"],
            maximum_above_ground_m=altitude_config["max_above_ground_m"],
        )[0]
        selected = candidates[selected_index]
        return WaypointPrediction(
            np.asarray((selected.xy[0], selected.xy[1], float(altitude))),
            diagnostics={
                "selected_index": selected_index,
                "selector_logits": selector_logit_values,
                "candidate_xy": [candidate.xy.tolist() for candidate in candidates],
                "candidate_sources": [candidate.sources for candidate in candidates],
                "field_scores": [candidate.field_score for candidate in candidates],
                "progress_prediction": float(auxiliary.progress[0]),
            },
        )
