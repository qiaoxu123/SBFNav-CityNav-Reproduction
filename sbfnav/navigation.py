"""Ground-truth-free receding-horizon waypoint execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np

from multiagent.mapdata import GROUND_LEVEL

from .dataset import CityNavRecord
from .planning_state import PlanningState, PlanningStateBuilder


@dataclass(frozen=True)
class WaypointPrediction:
    xyz: np.ndarray
    diagnostics: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        xyz = np.asarray(self.xyz, dtype=np.float64)
        if xyz.shape != (3,) or not np.isfinite(xyz).all():
            raise ValueError("waypoint must be a finite xyz triplet")
        object.__setattr__(self, "xyz", xyz)


@dataclass(frozen=True)
class NavigationTrace:
    poses: np.ndarray
    waypoints: np.ndarray
    diagnostics: tuple[Mapping[str, object] | None, ...]
    termination: str


class BoundedWaypointController:
    def __init__(
        self,
        *,
        horizontal_step_m: float = 10.0,
        vertical_step_m: float = 5.0,
        minimum_above_ground_m: float = 10.0,
        maximum_above_ground_m: float = 100.0,
    ):
        if min(horizontal_step_m, vertical_step_m) <= 0:
            raise ValueError("controller step limits must be positive")
        if minimum_above_ground_m >= maximum_above_ground_m:
            raise ValueError("invalid legal altitude range")
        self.horizontal_step_m = horizontal_step_m
        self.vertical_step_m = vertical_step_m
        self.minimum_above_ground_m = minimum_above_ground_m
        self.maximum_above_ground_m = maximum_above_ground_m

    def step(
        self,
        pose: np.ndarray,
        waypoint_xyz: np.ndarray,
        *,
        ground_level: float,
        xy_bounds: tuple[float, float, float, float],
    ) -> np.ndarray:
        pose = np.asarray(pose, dtype=np.float64)
        waypoint = np.asarray(waypoint_xyz, dtype=np.float64).copy()
        if pose.shape != (4,) or waypoint.shape != (3,):
            raise ValueError("pose must be xyzyaw and waypoint must be xyz")
        x_min, y_min, x_max, y_max = xy_bounds
        waypoint[0] = np.clip(waypoint[0], x_min, x_max)
        waypoint[1] = np.clip(waypoint[1], y_min, y_max)
        waypoint[2] = np.clip(
            waypoint[2],
            ground_level + self.minimum_above_ground_m,
            ground_level + self.maximum_above_ground_m,
        )
        horizontal = waypoint[:2] - pose[:2]
        distance = float(np.linalg.norm(horizontal))
        if distance > self.horizontal_step_m:
            horizontal *= self.horizontal_step_m / distance
        dz = float(np.clip(waypoint[2] - pose[2], -self.vertical_step_m, self.vertical_step_m))
        yaw = pose[3] if distance <= 1e-9 else float(np.arctan2(horizontal[1], horizontal[0]))
        return np.asarray((pose[0] + horizontal[0], pose[1] + horizontal[1], pose[2] + dz, yaw))


Policy = Callable[[PlanningState, str], WaypointPrediction]


class RecedingHorizonNavigator:
    """Rebuild state and query the policy after every one bounded step."""

    def __init__(
        self,
        state_builder: PlanningStateBuilder,
        controller: BoundedWaypointController,
        *,
        horizon: int = 100,
        stagnation_steps: int = 5,
        stagnation_tolerance_m: float = 1e-4,
    ):
        if horizon < 1 or stagnation_steps < 1:
            raise ValueError("horizon and stagnation_steps must be positive")
        self.state_builder = state_builder
        self.controller = controller
        self.horizon = horizon
        self.stagnation_steps = stagnation_steps
        self.stagnation_tolerance_m = stagnation_tolerance_m

    def run(self, record: CityNavRecord, policy: Policy) -> NavigationTrace:
        poses = [record.start_pose.copy()]
        waypoints = []
        diagnostics = []
        stagnant = 0
        termination = "horizon"
        transform = self.state_builder.transform(record.map_name)
        bounds = (transform.x_min, transform.y_min, transform.x_max, transform.y_max)
        for _step in range(self.horizon):
            state = self.state_builder.build_from_poses(record, np.stack(poses))
            prediction = policy(state, record.instruction)
            next_pose = self.controller.step(
                poses[-1],
                prediction.xyz,
                ground_level=GROUND_LEVEL[record.map_name],
                xy_bounds=bounds,
            )
            waypoints.append(prediction.xyz)
            diagnostics.append(prediction.diagnostics)
            displacement = np.linalg.norm(next_pose[:3] - poses[-1][:3])
            poses.append(next_pose)
            stagnant = stagnant + 1 if displacement <= self.stagnation_tolerance_m else 0
            if stagnant >= self.stagnation_steps:
                termination = "stagnation"
                break
        return NavigationTrace(
            poses=np.stack(poses),
            waypoints=np.stack(waypoints),
            diagnostics=tuple(diagnostics),
            termination=termination,
        )

