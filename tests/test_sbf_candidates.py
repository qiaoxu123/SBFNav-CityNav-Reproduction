import numpy as np
import pytest
import torch

from sbfnav.candidate_proposal import (
    Candidate,
    deduplicate_candidates,
    nms_topk_indices,
    oracle_candidate_metrics,
    propose_candidates,
    propose_uniform_grid_candidates,
)
from sbfnav.coordinates import MapTransform
from sbfnav.dataset import Landmark


def test_nms_topk_is_deterministic_and_respects_invalid_cells():
    logits = torch.zeros(5, 5)
    logits[1, 1] = 5
    logits[3, 3] = 4
    logits[0, 4] = 10
    valid = torch.ones(5, 5, dtype=torch.bool)
    valid[0, 4] = False
    peaks = nms_topk_indices(logits, valid, top_k=3, kernel_size=3)
    assert peaks[0][:2] == (1, 1)
    assert peaks[1][:2] == (3, 3)
    assert all((row, col) != (0, 4) for row, col, _score in peaks)
    assert len(peaks) == 3


def test_nms_rejects_all_invalid_and_bad_kernel():
    with pytest.raises(ValueError, match="no valid"):
        nms_topk_indices(np.zeros((2, 2)), np.zeros((2, 2), bool), top_k=1)
    with pytest.raises(ValueError, match="odd"):
        nms_topk_indices(np.zeros((2, 2)), np.ones((2, 2), bool), top_k=1, kernel_size=2)


def test_nms_greedily_suppresses_neighbours_on_smooth_field():
    # A monotone field has only one local maximum.  Top-K must still consist
    # of spatially suppressed proposals rather than adjacent fallback pixels.
    logits = torch.arange(49, dtype=torch.float32).reshape(7, 7)
    peaks = nms_topk_indices(logits, torch.ones(7, 7, dtype=torch.bool), top_k=4, kernel_size=3)
    assert len(peaks) == 4
    for index, (row, col, _score) in enumerate(peaks):
        for other_row, other_col, _other_score in peaks[:index]:
            assert max(abs(row - other_row), abs(col - other_col)) > 1


def test_nms_ties_are_row_major_deterministic():
    peaks = nms_topk_indices(
        torch.zeros(5, 5), torch.ones(5, 5, dtype=torch.bool), top_k=3, kernel_size=3
    )
    assert [item[:2] for item in peaks] == [(0, 0), (0, 2), (0, 4)]


def test_proposal_world_mapping_landmark_union_and_deduplication():
    transform = MapTransform("map", 0, 0, 280, 280)
    logits = np.zeros((28, 28), np.float32)
    logits[10, 12] = 5
    field_xy = transform.field_to_world((10, 12))
    contour = np.asarray(
        (field_xy + (-1, -1), field_xy + (1, -1), field_xy + (1, 1), field_xy + (-1, 1))
    )
    landmark = Landmark("map", 1, "same", np.r_[field_xy, 0], contour)
    candidates = propose_candidates(
        logits,
        np.ones((28, 28), bool),
        transform,
        [landmark],
        top_k=1,
        dedup_radius_m=2,
    )
    assert len(candidates) == 1
    np.testing.assert_allclose(candidates[0].xy, field_xy)
    assert candidates[0].sources == ("field_peak", "landmark_centroid")


def test_out_of_bounds_landmark_is_rejected():
    transform = MapTransform("map", 0, 0, 100, 100)
    landmark = Landmark(
        "map", 1, "outside", np.asarray((200, 200, 0)), np.asarray(((190, 190), (210, 190), (210, 210)))
    )
    candidates = propose_candidates(
        np.zeros((28, 28)),
        transform.valid_mask(28),
        transform,
        [landmark],
        top_k=1,
    )
    assert all("landmark_centroid" not in candidate.sources for candidate in candidates)


def test_dedup_and_oracle_metrics():
    candidates = deduplicate_candidates(
        [
            Candidate(np.asarray((0, 0)), ("field_peak",), (0, 0), 1),
            Candidate(np.asarray((1, 0)), ("landmark_centroid",)),
            Candidate(np.asarray((30, 0)), ("field_peak",), (0, 1), 0),
        ],
        radius_m=2,
    )
    assert len(candidates) == 2
    metrics = oracle_candidate_metrics(candidates, np.asarray((28, 0)), ks=(1, 2), thresholds_m=(5,))
    assert metrics["coverage@1_5m"] == 0
    assert metrics["coverage@2_5m"] == 1
    assert metrics["oracle_distance_m"] == pytest.approx(2)


def test_uniform_grid_proposal_is_field_free_valid_and_deterministic():
    transform = MapTransform("map", 0, 0, 280, 280)
    valid = np.ones((28, 28), dtype=bool)
    valid[:3] = False
    first = propose_uniform_grid_candidates(
        valid, transform, [], top_k=16, add_landmarks=False
    )
    second = propose_uniform_grid_candidates(
        valid, transform, [], top_k=16, add_landmarks=False
    )
    assert len(first) == 16
    assert all(item.sources == ("uniform_grid",) and item.field_score is None for item in first)
    assert all(valid[item.field_index] for item in first)
    np.testing.assert_allclose(
        np.stack([item.xy for item in first]), np.stack([item.xy for item in second])
    )
