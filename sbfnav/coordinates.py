"""World/canvas/field coordinate transforms for north-up CityNav maps."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import rasterio


ArrayLike = Iterable[float] | np.ndarray


@dataclass(frozen=True)
class MapTransform:
    """Georeferencing for a symmetrically square-padded north-up map.

    Continuous canvas coordinates use the image convention ``(row, col)`` and
    place integer coordinates at pixel centers. Normalized coordinates use
    ``(u_x, u_y)`` with (0, 0) at the north-west padded-square edge.
    """

    map_name: str
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    canvas_size: int = 224
    field_size: int = 28

    def __post_init__(self) -> None:
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            raise ValueError("map bounds must have positive extent")
        if self.canvas_size <= 0 or self.field_size <= 0:
            raise ValueError("raster sizes must be positive")

    @classmethod
    def from_raster(
        cls,
        path: str | Path,
        *,
        canvas_size: int = 224,
        field_size: int = 28,
    ) -> "MapTransform":
        path = Path(path)
        with rasterio.open(path) as dataset:
            bounds = dataset.bounds
        return cls(
            path.stem,
            float(bounds.left),
            float(bounds.bottom),
            float(bounds.right),
            float(bounds.top),
            canvas_size,
            field_size,
        )

    @property
    def width_m(self) -> float:
        return self.x_max - self.x_min

    @property
    def height_m(self) -> float:
        return self.y_max - self.y_min

    @property
    def side_m(self) -> float:
        return max(self.width_m, self.height_m)

    @property
    def padded_bounds(self) -> Tuple[float, float, float, float]:
        x_pad = (self.side_m - self.width_m) / 2.0
        y_pad = (self.side_m - self.height_m) / 2.0
        return (
            self.x_min - x_pad,
            self.y_min - y_pad,
            self.x_max + x_pad,
            self.y_max + y_pad,
        )

    def world_to_normalized(self, xy: ArrayLike) -> np.ndarray:
        points = np.asarray(xy, dtype=np.float64)
        x0, _y0, _x1, y1 = self.padded_bounds
        output = np.empty_like(points, dtype=np.float64)
        output[..., 0] = (points[..., 0] - x0) / self.side_m
        output[..., 1] = (y1 - points[..., 1]) / self.side_m
        return output

    def normalized_to_world(self, uv: ArrayLike) -> np.ndarray:
        coords = np.asarray(uv, dtype=np.float64)
        x0, _y0, _x1, y1 = self.padded_bounds
        output = np.empty_like(coords, dtype=np.float64)
        output[..., 0] = x0 + coords[..., 0] * self.side_m
        output[..., 1] = y1 - coords[..., 1] * self.side_m
        return output

    def world_to_grid(self, xy: ArrayLike, size: int) -> np.ndarray:
        """Return continuous ``(row, col)`` coordinates at pixel centers."""
        uv = self.world_to_normalized(xy)
        return np.stack((uv[..., 1] * size - 0.5, uv[..., 0] * size - 0.5), axis=-1)

    def world_to_grid_edges(self, xy: ArrayLike, size: int) -> np.ndarray:
        """Return continuous ``(row, col)`` edge coordinates for rasterizing."""
        uv = self.world_to_normalized(xy)
        return np.stack((uv[..., 1] * size, uv[..., 0] * size), axis=-1)

    def grid_to_world(self, row_col: ArrayLike, size: int) -> np.ndarray:
        indices = np.asarray(row_col, dtype=np.float64)
        uv = np.stack(
            ((indices[..., 1] + 0.5) / size, (indices[..., 0] + 0.5) / size),
            axis=-1,
        )
        return self.normalized_to_world(uv)

    def world_to_canvas(self, xy: ArrayLike) -> np.ndarray:
        return self.world_to_grid(xy, self.canvas_size)

    def canvas_to_world(self, row_col: ArrayLike) -> np.ndarray:
        return self.grid_to_world(row_col, self.canvas_size)

    def world_to_field(self, xy: ArrayLike) -> np.ndarray:
        return self.world_to_grid(xy, self.field_size)

    def field_to_world(self, row_col: ArrayLike) -> np.ndarray:
        return self.grid_to_world(row_col, self.field_size)

    def field_index(self, xy: ArrayLike, *, clip: bool = False) -> np.ndarray:
        uv = self.world_to_normalized(xy)
        col = np.floor(uv[..., 0] * self.field_size).astype(np.int64)
        row = np.floor(uv[..., 1] * self.field_size).astype(np.int64)
        indices = np.stack((row, col), axis=-1)
        if clip:
            indices = np.clip(indices, 0, self.field_size - 1)
        return indices

    def contains_world(self, xy: ArrayLike, *, padded: bool = False) -> np.ndarray:
        points = np.asarray(xy, dtype=np.float64)
        if padded:
            x0, y0, x1, y1 = self.padded_bounds
        else:
            x0, y0, x1, y1 = self.x_min, self.y_min, self.x_max, self.y_max
        return (
            (points[..., 0] >= x0)
            & (points[..., 0] <= x1)
            & (points[..., 1] >= y0)
            & (points[..., 1] <= y1)
        )

    def clip_world(self, xy: ArrayLike) -> np.ndarray:
        points = np.asarray(xy, dtype=np.float64).copy()
        points[..., 0] = np.clip(points[..., 0], self.x_min, self.x_max)
        points[..., 1] = np.clip(points[..., 1], self.y_min, self.y_max)
        return points

    def valid_mask(self, size: int) -> np.ndarray:
        rows, cols = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
        world = self.grid_to_world(np.stack((rows, cols), axis=-1), size)
        return self.contains_world(world)

    @property
    def canvas_meters_per_pixel(self) -> float:
        return self.side_m / self.canvas_size

    @property
    def field_meters_per_cell(self) -> float:
        return self.side_m / self.field_size

