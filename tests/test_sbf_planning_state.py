import os
from pathlib import Path

import numpy as np

from sbfnav.dataset import CityNavDataset, CityReferCatalog
from sbfnav.planning_state import PlanningStateBuilder


DATA_ROOT = Path(
    os.environ.get("SBFNAV_TEST_DATA_ROOT", Path(__file__).resolve().parents[1] / "data")
)


def _sample_with_landmark():
    catalog = CityReferCatalog(DATA_ROOT / "cityrefer")
    dataset = CityNavDataset(DATA_ROOT, "train_seen", max_records=64)
    record = next(record for record in dataset if catalog.referenced_landmark_names(record))
    return catalog, record


def test_nine_channels_and_visibility_invariants():
    catalog, record = _sample_with_landmark()
    builder = PlanningStateBuilder(DATA_ROOT, catalog)
    state = builder.build(record, prefix_length=min(3, len(record.trajectory)))
    assert state.values.shape == (9, 224, 224)
    assert state.valid_field.shape == (28, 28)
    assert np.isfinite(state.values).all()
    explored = state.values[7].astype(bool)
    current = state.values[6].astype(bool)
    assert current.any()
    assert np.all(current <= explored)
    assert np.all(state.values[:4, ~explored] == 0)
    assert np.all(state.values[:, ~state.valid_canvas] == 0)
    assert state.values[4].sum() > 0
    assert state.values[5].sum() > 0
    assert state.values[8].sum() > 0


def test_history_accumulates_without_rotating_world_frame():
    catalog, record = _sample_with_landmark()
    builder = PlanningStateBuilder(DATA_ROOT, catalog)
    first = builder.build(record, prefix_length=1)
    later = builder.build(record, prefix_length=min(8, len(record.trajectory)))
    assert np.all(first.values[7] <= later.values[7])
    # Static geographic layers must remain identical as heading/position change.
    np.testing.assert_array_equal(first.values[4], later.values[4])
    np.testing.assert_array_equal(first.values[5], later.values[5])


def test_target_is_not_encoded_in_planning_state():
    catalog, record = _sample_with_landmark()
    builder = PlanningStateBuilder(DATA_ROOT, catalog)
    original = builder.build(record, prefix_length=1)
    altered = record.__class__(
        **{**record.__dict__, "target_position": record.target_position + np.asarray((50, 50, 20))}
    )
    changed = builder.build(altered, prefix_length=1)
    np.testing.assert_array_equal(original.values, changed.values)


def test_real_tif_coordinate_round_trip_within_half_source_pixel():
    catalog, record = _sample_with_landmark()
    transform = PlanningStateBuilder(DATA_ROOT, catalog).transform(record.map_name)
    points = np.stack((record.trajectory[:16, 0], record.trajectory[:16, 1]), axis=-1)
    reconstructed = transform.normalized_to_world(transform.world_to_normalized(points))
    assert np.max(np.linalg.norm(points - reconstructed, axis=-1)) < 0.05
