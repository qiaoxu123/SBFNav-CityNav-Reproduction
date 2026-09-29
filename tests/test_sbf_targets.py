import numpy as np
import pytest

from sbfnav.coordinates import MapTransform
from sbfnav.field_targets import gaussian_field_target, progress_target


def test_gaussian_is_normalized_masked_and_metric_centered():
    transform = MapTransform("rect", 0, 0, 400, 200)
    valid = transform.valid_mask(28)
    target_xy = np.asarray((200.0, 100.0))
    target = gaussian_field_target(transform, target_xy, sigma_m=20.0, valid_mask=valid)
    assert target.shape == (28, 28)
    assert np.isfinite(target).all()
    assert target.sum() == pytest.approx(1.0, abs=1e-6)
    assert np.all(target[~valid] == 0)
    peak = np.asarray(np.unravel_index(np.argmax(target), target.shape))
    assert np.linalg.norm(transform.field_to_world(peak) - target_xy) <= (
        np.sqrt(2) * transform.field_meters_per_cell / 2
    )


def test_gaussian_rejects_all_mask_and_bad_sigma():
    transform = MapTransform("square", 0, 0, 100, 100)
    with pytest.raises(ValueError, match="no valid"):
        gaussian_field_target(transform, (50, 50), sigma_m=10, valid_mask=np.zeros((28, 28)))
    with pytest.raises(ValueError, match="positive"):
        gaussian_field_target(transform, (50, 50), sigma_m=0)


def test_progress_target_endpoints_and_clamping():
    start = np.asarray((0, 0, 10))
    target = np.asarray((100, 0, 10))
    assert progress_target(start, start, target) == 0.0
    assert progress_target(target, start, target) == 1.0
    assert progress_target(np.asarray((-20, 0, 10)), start, target) == 0.0
    assert progress_target(np.asarray((50, 0, 10)), start, target) == pytest.approx(0.5)

