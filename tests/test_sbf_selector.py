import inspect

import numpy as np
import torch

from sbfnav.candidate_proposal import Candidate
from sbfnav.models.selector import (
    SelectorInputs,
    SemanticGeometricSelector,
    candidate_coordinates,
    extract_candidate_crops,
    relative_geometry,
)
from sbfnav.coordinates import MapTransform
from sbfnav.planning_state import PlanningState


def _inputs(candidate_xy, candidate_mask):
    batch, candidates, _ = candidate_xy.shape
    return SelectorInputs(
        candidate_visual=torch.randn(batch, candidates, 16),
        candidate_xy=candidate_xy,
        candidate_mask=candidate_mask,
        instruction_tokens=torch.randn(batch, 6, 12),
        instruction_mask=torch.tensor([[1, 1, 1, 1, 0, 0]] * batch, dtype=torch.bool),
        landmark_name_features=torch.randn(batch, 3, 12),
        landmark_xy=torch.randn(batch, 3, 2) * 50,
        landmark_mask=torch.tensor([[1, 1, 0]] * batch, dtype=torch.bool),
        agent_xy=torch.zeros(batch, 2),
        map_scale_m=torch.full((batch,), 400.0),
    )


def test_relative_geometry_exact_values():
    geometry = relative_geometry(torch.tensor([[0.0, 0.0]]), torch.tensor([[3.0, 4.0]]), 10.0)
    torch.testing.assert_close(geometry, torch.tensor([[0.3, 0.4, 0.5, 0.8, 0.6]]))


def test_selector_forward_padding_and_backward():
    torch.manual_seed(0)
    candidate_xy = torch.randn(2, 4, 2)
    candidate_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]], dtype=torch.bool)
    inputs = _inputs(candidate_xy, candidate_mask)
    model = SemanticGeometricSelector(
        vision_dim=16,
        language_dim=12,
        hidden_dim=32,
        layers=1,
        attention_heads=4,
        dropout=0,
    )
    logits = model(inputs)
    assert logits.shape == (2, 4)
    assert torch.isneginf(logits[0, 3])
    logits[candidate_mask].mean().backward()
    assert model.score.weight.grad is not None


def test_selector_has_no_sbf_score_input_or_parameter():
    signature = inspect.signature(SemanticGeometricSelector.forward)
    assert "score" not in signature.parameters
    fields = set(SelectorInputs.__dataclass_fields__)
    assert not {"field_score", "sbf_score", "field_rank"} & fields
    model = SemanticGeometricSelector(
        vision_dim=16, language_dim=12, hidden_dim=32, layers=1, attention_heads=4
    )
    assert all("field" not in name.lower() and "sbf" not in name.lower() for name, _ in model.named_parameters())


def test_candidate_score_perturbation_cannot_change_selector_inputs_or_logits():
    first = [[
        Candidate(np.asarray((10, 20)), ("field_peak",), (1, 2), 1000),
        Candidate(np.asarray((30, 40)), ("field_peak",), (3, 4), -1000),
    ]]
    second = [[
        Candidate(np.asarray((10, 20)), ("field_peak",), (1, 2), -5),
        Candidate(np.asarray((30, 40)), ("field_peak",), (3, 4), 99999),
    ]]
    xy_a, mask_a = candidate_coordinates(first)
    xy_b, mask_b = candidate_coordinates(second)
    torch.testing.assert_close(xy_a, xy_b)
    torch.testing.assert_close(mask_a, mask_b)
    torch.manual_seed(4)
    model = SemanticGeometricSelector(
        vision_dim=16,
        language_dim=12,
        hidden_dim=32,
        layers=1,
        attention_heads=4,
        dropout=0,
    ).eval()
    inputs = _inputs(xy_a, mask_a)
    logits_a = model(inputs)
    inputs_b = SelectorInputs(**{**inputs.__dict__, "candidate_xy": xy_b, "candidate_mask": mask_b})
    logits_b = model(inputs_b)
    torch.testing.assert_close(logits_a, logits_b)


def test_candidate_crop_is_zero_padded_at_map_boundary():
    transform = MapTransform("synthetic", 0, 0, 100, 100)
    values = np.zeros((9, 224, 224), np.float32)
    values[:3] = 1.0
    state = PlanningState(
        values,
        np.ones((224, 224), bool),
        np.ones((28, 28), bool),
        transform,
        np.asarray((50, 50, 30, 0)),
        (),
    )
    crops = extract_candidate_crops(
        state, [Candidate(np.asarray((0, 100)), ("field_peak",))], extent_m=40, output_size=64
    )
    assert crops.shape == (1, 64, 64, 3)
    assert (crops[0, :20, :20] == 0).all()
    assert crops[0, 40:, 40:].mean() > 0.9
