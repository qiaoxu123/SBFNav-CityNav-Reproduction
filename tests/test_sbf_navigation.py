import os
from pathlib import Path

import numpy as np
import torch

from sbfnav.dataset import CityNavDataset, CityReferCatalog
from sbfnav.models.altitude_head import (
    AltitudeProgressHead,
    altitude_target_normalized,
    altitude_to_world,
)
from sbfnav.navigation import (
    BoundedWaypointController,
    RecedingHorizonNavigator,
    WaypointPrediction,
)
from sbfnav.planning_state import PlanningStateBuilder


DATA_ROOT = Path(
    os.environ.get("SBFNAV_TEST_DATA_ROOT", Path(__file__).resolve().parents[1] / "data")
)


def test_altitude_head_range_and_backward():
    head = AltitudeProgressHead(spatial_dim=32, language_dim=16, hidden_dim=24)
    output = head(torch.randn(2, 32, 28, 28), torch.randn(2, 16))
    assert ((0 <= output.altitude_normalized) & (output.altitude_normalized <= 1)).all()
    assert ((0 <= output.progress) & (output.progress <= 1)).all()
    (output.altitude_normalized.mean() + output.progress.mean()).backward()
    assert head.altitude.weight.grad is not None


def test_altitude_normalization_and_legal_clamp():
    ground = torch.tensor([10.0])
    normalized = altitude_target_normalized(
        torch.tensor([55.0]), ground, minimum_above_ground_m=10, maximum_above_ground_m=100
    )
    world = altitude_to_world(
        normalized, ground, minimum_above_ground_m=10, maximum_above_ground_m=100
    )
    torch.testing.assert_close(world, torch.tensor([55.0]))
    high = altitude_to_world(
        torch.tensor([2.0]), ground, minimum_above_ground_m=10, maximum_above_ground_m=100
    )
    torch.testing.assert_close(high, torch.tensor([110.0]))


def test_controller_bounds_horizontal_vertical_and_yaw():
    controller = BoundedWaypointController(horizontal_step_m=10, vertical_step_m=5)
    result = controller.step(
        np.asarray((0, 0, 50, 0)),
        np.asarray((100, 100, 100)),
        ground_level=0,
        xy_bounds=(-20, -20, 80, 80),
    )
    np.testing.assert_allclose(np.linalg.norm(result[:2]), 10)
    assert result[2] == 55
    np.testing.assert_allclose(result[3], np.pi / 4)


def test_receding_horizon_rebuilds_state_without_goal_signal():
    catalog = CityReferCatalog(DATA_ROOT / "cityrefer")
    record = CityNavDataset(DATA_ROOT, "train_seen", max_records=1)[0]
    builder = PlanningStateBuilder(DATA_ROOT, catalog)
    navigator = RecedingHorizonNavigator(
        builder,
        BoundedWaypointController(horizontal_step_m=2, vertical_step_m=1),
        horizon=3,
    )
    calls = []

    def policy(state, instruction):
        calls.append((state.pose.copy(), instruction))
        return WaypointPrediction(state.pose[:3] + np.asarray((5, 0, 0)))

    trace = navigator.run(record, policy)
    assert len(calls) == 3
    assert trace.poses.shape == (4, 4)
    assert trace.waypoints.shape == (3, 3)
    assert calls[1][0][0] != calls[0][0][0]
    assert trace.termination == "horizon"
