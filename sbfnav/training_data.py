"""Deterministic teacher-prefix samples for SBFNav supervised training."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from multiagent.mapdata import GROUND_LEVEL

from .dataset import CityNavDataset, CityNavRecord
from .field_targets import gaussian_field_target, progress_target
from .planning_state import PlanningState, PlanningStateBuilder


@dataclass(frozen=True)
class TrainingExample:
    record: CityNavRecord
    state: PlanningState
    field_target: torch.Tensor
    altitude_target_normalized: float
    progress_target: float
    prefix_length: int


class SBFTrainingDataset(Dataset[TrainingExample]):
    def __init__(
        self,
        episodes: CityNavDataset,
        state_builder: PlanningStateBuilder,
        *,
        samples_per_episode: int = 1,
        gaussian_sigma_m: float = 20.0,
        minimum_above_ground_m: float = 10.0,
        maximum_above_ground_m: float = 100.0,
    ):
        if samples_per_episode < 1:
            raise ValueError("samples_per_episode must be positive")
        self.episodes = episodes
        self.state_builder = state_builder
        self.samples_per_episode = samples_per_episode
        self.gaussian_sigma_m = gaussian_sigma_m
        self.minimum_above_ground_m = minimum_above_ground_m
        self.maximum_above_ground_m = maximum_above_ground_m

    def __len__(self) -> int:
        return len(self.episodes) * self.samples_per_episode

    def _prefix_length(self, record: CityNavRecord, slot: int) -> int:
        # Fixed interior quantiles make runs deterministic and cover partial to
        # near-complete histories without sampling target-derived signals.
        fraction = (slot + 1) / (self.samples_per_episode + 1)
        return int(np.clip(round(fraction * len(record.trajectory)), 1, len(record.trajectory)))

    def __getitem__(self, index: int) -> TrainingExample:
        record_index, slot = divmod(index, self.samples_per_episode)
        record = self.episodes[record_index]
        prefix_length = self._prefix_length(record, slot)
        state = self.state_builder.build(record, prefix_length=prefix_length)
        field = gaussian_field_target(
            state.transform,
            record.target_position[:2],
            sigma_m=self.gaussian_sigma_m,
            valid_mask=state.valid_field,
        )
        ground = float(GROUND_LEVEL[record.map_name])
        span = self.maximum_above_ground_m - self.minimum_above_ground_m
        altitude = np.clip(
            (record.target_position[2] - ground - self.minimum_above_ground_m) / span, 0, 1
        )
        progress = progress_target(
            state.pose[:3], record.start_pose[:3], record.target_position
        )
        return TrainingExample(
            record,
            state,
            torch.from_numpy(field),
            float(altitude),
            progress,
            prefix_length,
        )

