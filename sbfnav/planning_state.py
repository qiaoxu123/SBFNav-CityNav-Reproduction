"""Construction of the paper's 224×224, nine-channel planning state."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Sequence

import cv2
import numpy as np
import rasterio

from sbfnav.mapdata import GROUND_LEVEL

from .coordinates import MapTransform
from .dataset import CityNavRecord, CityReferCatalog, Landmark


CHANNEL_NAMES = (
    "observed_rgb_r",
    "observed_rgb_g",
    "observed_rgb_b",
    "normalized_height",
    "available_landmark_contours",
    "referenced_landmark_mask",
    "current_observation_footprint",
    "cumulatively_explored_area",
    "trajectory_history",
)


@dataclass(frozen=True)
class PlanningState:
    values: np.ndarray
    valid_canvas: np.ndarray
    valid_field: np.ndarray
    transform: MapTransform
    pose: np.ndarray
    referenced_landmarks: tuple[Landmark, ...]

    def __post_init__(self) -> None:
        expected = (9, self.transform.canvas_size, self.transform.canvas_size)
        if self.values.shape != expected:
            raise ValueError(f"expected state shape {expected}, got {self.values.shape}")
        if not np.isfinite(self.values).all():
            raise ValueError("planning state contains non-finite values")


class PlanningStateBuilder:
    def __init__(
        self,
        data_root: str | Path,
        catalog: CityReferCatalog,
        *,
        canvas_size: int = 224,
        field_size: int = 28,
        max_height_above_ground_m: float = 100.0,
        trajectory_radius_m: float = 2.0,
    ):
        self.data_root = Path(data_root)
        self.catalog = catalog
        self.canvas_size = canvas_size
        self.field_size = field_size
        self.max_height_above_ground_m = float(max_height_above_ground_m)
        self.trajectory_radius_m = float(trajectory_radius_m)
        if self.max_height_above_ground_m <= 0 or self.trajectory_radius_m <= 0:
            raise ValueError("height scale and trajectory radius must be positive")
        self._transform_cache: Dict[str, MapTransform] = {}
        self._base_cache: Dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._landmark_cache: Dict[str, np.ndarray] = {}

    def transform(self, map_name: str) -> MapTransform:
        if map_name not in self._transform_cache:
            path = self.data_root / "rgbd" / f"{map_name}.tif"
            self._transform_cache[map_name] = MapTransform.from_raster(
                path, canvas_size=self.canvas_size, field_size=self.field_size
            )
        return self._transform_cache[map_name]

    @staticmethod
    def _source_to_canvas_matrix(
        source_bounds: rasterio.coords.BoundingBox,
        transform: MapTransform,
        width: int,
        height: int,
    ) -> np.ndarray:
        source = np.asarray(((0, 0), (width, 0), (width, height), (0, height)), np.float32)
        world = np.asarray(
            (
                (source_bounds.left, source_bounds.top),
                (source_bounds.right, source_bounds.top),
                (source_bounds.right, source_bounds.bottom),
                (source_bounds.left, source_bounds.bottom),
            ),
            np.float64,
        )
        row_col = transform.world_to_grid_edges(world, transform.canvas_size)
        destination = np.stack((row_col[:, 1], row_col[:, 0]), axis=-1).astype(np.float32)
        return cv2.getPerspectiveTransform(source, destination)

    def _base_layers(self, map_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if map_name in self._base_cache:
            return self._base_cache[map_name]
        tif_path = self.data_root / "rgbd" / f"{map_name}.tif"
        png_path = self.data_root / "rgbd" / f"{map_name}.png"
        with rasterio.open(tif_path) as raster:
            world_height = raster.read(1).astype(np.float32)
            matrix = self._source_to_canvas_matrix(
                raster.bounds, self.transform(map_name), raster.width, raster.height
            )
        bgr = cv2.imread(str(png_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(png_path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        size = self.canvas_size
        warped_rgb = cv2.warpPerspective(
            rgb,
            matrix,
            (size, size),
            flags=cv2.INTER_AREA,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        ).astype(np.float32) / 255.0
        warped_height = cv2.warpPerspective(
            world_height,
            matrix,
            (size, size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=-1,
        )
        valid = self.transform(map_name).valid_mask(size)
        ground = float(GROUND_LEVEL[map_name])
        normalized_height = np.clip(
            (warped_height - ground) / self.max_height_above_ground_m, 0.0, 1.0
        )
        normalized_height[(warped_height < 0) | ~valid] = 0.0
        warped_rgb[~valid] = 0.0
        value = (warped_rgb, normalized_height.astype(np.float32), valid)
        self._base_cache[map_name] = value
        return value

    def _polygon_pixels(self, transform: MapTransform, contour: np.ndarray) -> np.ndarray:
        row_col = transform.world_to_grid_edges(contour, self.canvas_size)
        return np.rint(np.stack((row_col[:, 1], row_col[:, 0]), axis=-1)).astype(np.int32)

    def _available_landmark_contours(self, map_name: str) -> np.ndarray:
        if map_name in self._landmark_cache:
            return self._landmark_cache[map_name]
        layer = np.zeros((self.canvas_size, self.canvas_size), dtype=np.uint8)
        transform = self.transform(map_name)
        for landmark in self.catalog.all_landmarks(map_name):
            pixels = self._polygon_pixels(transform, landmark.contour)
            if len(pixels) >= 2:
                cv2.polylines(layer, [pixels], True, 1, thickness=1, lineType=cv2.LINE_8)
        result = layer.astype(np.float32)
        self._landmark_cache[map_name] = result
        return result

    def _referenced_mask(self, transform: MapTransform, landmarks: Iterable[Landmark]) -> np.ndarray:
        layer = np.zeros((self.canvas_size, self.canvas_size), dtype=np.uint8)
        for landmark in landmarks:
            pixels = self._polygon_pixels(transform, landmark.contour)
            if len(pixels) >= 3:
                cv2.fillPoly(layer, [pixels], 1, lineType=cv2.LINE_8)
        return layer.astype(np.float32)

    @staticmethod
    def view_footprint(pose: Sequence[float], ground_level: float) -> np.ndarray:
        x, y, z, yaw = (float(value) for value in pose)
        radius = max(0.0, z - ground_level)
        front = np.asarray((np.cos(yaw), np.sin(yaw)))
        left = np.asarray((-np.sin(yaw), np.cos(yaw)))
        center = np.asarray((x, y))
        return np.stack(
            (
                center + radius * (front + left),
                center + radius * (front - left),
                center + radius * (-front - left),
                center + radius * (-front + left),
            )
        )

    def _footprint_mask(self, transform: MapTransform, pose: Sequence[float]) -> np.ndarray:
        polygon = self.view_footprint(pose, GROUND_LEVEL[transform.map_name])
        pixels = self._polygon_pixels(transform, polygon)
        layer = np.zeros((self.canvas_size, self.canvas_size), dtype=np.uint8)
        cv2.fillConvexPoly(layer, pixels, 1, lineType=cv2.LINE_8)
        return layer.astype(bool)

    def _trajectory_mask(self, transform: MapTransform, poses: np.ndarray) -> np.ndarray:
        layer = np.zeros((self.canvas_size, self.canvas_size), dtype=np.uint8)
        row_col = transform.world_to_grid_edges(poses[:, :2], self.canvas_size)
        pixels = np.rint(np.stack((row_col[:, 1], row_col[:, 0]), axis=-1)).astype(np.int32)
        thickness = max(
            1,
            int(round(2.0 * self.trajectory_radius_m / transform.canvas_meters_per_pixel)),
        )
        if len(pixels) == 1:
            cv2.circle(layer, tuple(pixels[0]), max(1, thickness // 2), 1, -1)
        else:
            cv2.polylines(layer, [pixels], False, 1, thickness=thickness, lineType=cv2.LINE_8)
        return layer.astype(np.float32)

    def build(self, record: CityNavRecord, *, prefix_length: int = 1) -> PlanningState:
        if prefix_length < 1 or prefix_length > len(record.trajectory):
            raise ValueError(
                f"prefix_length must be in [1, {len(record.trajectory)}], got {prefix_length}"
            )
        return self.build_from_poses(record, record.trajectory[:prefix_length])

    def build_from_poses(
        self, record: CityNavRecord, poses: Sequence[Sequence[float]] | np.ndarray
    ) -> PlanningState:
        """Build state from a policy-generated history, without goal access."""
        poses = np.asarray(poses, dtype=np.float64)
        if poses.ndim != 2 or poses.shape[1] != 4 or len(poses) < 1:
            raise ValueError("poses must be a non-empty N×4 array")
        if not np.isfinite(poses).all():
            raise ValueError("poses contain non-finite values")
        transform = self.transform(record.map_name)
        rgb, height, valid = self._base_layers(record.map_name)
        footprints = [self._footprint_mask(transform, pose) for pose in poses]
        current = footprints[-1] & valid
        explored = np.logical_or.reduce(footprints) & valid
        referenced = tuple(self.catalog.referenced_landmarks(record))
        values = np.zeros((9, self.canvas_size, self.canvas_size), dtype=np.float32)
        values[0:3] = np.moveaxis(rgb * explored[..., None], -1, 0)
        values[3] = height * explored
        values[4] = self._available_landmark_contours(record.map_name) * valid
        values[5] = self._referenced_mask(transform, referenced) * valid
        values[6] = current.astype(np.float32)
        values[7] = explored.astype(np.float32)
        values[8] = self._trajectory_mask(transform, poses) * valid
        return PlanningState(
            values=values,
            valid_canvas=valid,
            valid_field=transform.valid_mask(self.field_size),
            transform=transform,
            pose=poses[-1].copy(),
            referenced_landmarks=referenced,
        )
