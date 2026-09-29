import pytest
import torch

from sbfnav.losses import (
    LossWeights,
    altitude_loss,
    combine_losses,
    hard_nearest_loss,
    listwise_distance_loss,
    pairwise_margin_loss,
    progress_loss,
    soft_field_cross_entropy,
)


def test_field_loss_is_finite_masked_and_backward():
    logits = torch.randn(2, 28, 28, requires_grad=True)
    valid = torch.ones(2, 28, 28, dtype=torch.bool)
    valid[:, :, :5] = False
    targets = torch.rand(2, 28, 28) * valid
    targets /= targets.sum(dim=(1, 2), keepdim=True)
    loss = soft_field_cross_entropy(logits, targets, valid)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()
    assert torch.all(logits.grad[:, :, :5] == 0)


def test_field_loss_rejects_all_mask_without_nan():
    with pytest.raises(ValueError, match="at least one"):
        soft_field_cross_entropy(
            torch.zeros(1, 2, 2), torch.ones(1, 2, 2), torch.zeros(1, 2, 2, dtype=torch.bool)
        )


def test_listwise_prefers_metric_near_candidate():
    distances = torch.tensor([[0.0, 20.0, 50.0]])
    mask = torch.ones_like(distances, dtype=torch.bool)
    good = listwise_distance_loss(
        torch.tensor([[4.0, 0.0, -2.0]]), distances, mask, temperature_m=20
    )
    bad = listwise_distance_loss(
        torch.tensor([[-2.0, 0.0, 4.0]]), distances, mask, temperature_m=20
    )
    assert good < bad


def test_margin_zero_for_no_pairs_and_penalizes_reversal():
    logits = torch.tensor([[0.0, 1.0]], requires_grad=True)
    no_pairs = pairwise_margin_loss(
        logits,
        torch.tensor([[25.0, 30.0]]),
        torch.ones_like(logits, dtype=torch.bool),
        good_radius_m=20,
        bad_radius_m=40,
        margin=1,
    )
    assert no_pairs.item() == 0
    reversal = pairwise_margin_loss(
        logits,
        torch.tensor([[5.0, 60.0]]),
        torch.ones_like(logits, dtype=torch.bool),
        good_radius_m=20,
        bad_radius_m=40,
        margin=1,
    )
    assert reversal.item() == pytest.approx(2.0)
    reversal.backward()
    assert torch.isfinite(logits.grad).all()


def test_margin_ignores_negative_infinity_padding_without_nan():
    logits = torch.tensor([[0.0, float("-inf"), 1.0]], requires_grad=True)
    distances = torch.tensor([[25.0, float("inf"), 30.0]])
    mask = torch.tensor([[True, False, True]])
    loss = pairwise_margin_loss(
        logits,
        distances,
        mask,
        good_radius_m=20,
        bad_radius_m=40,
        margin=1,
    )
    loss.backward()
    assert loss.item() == 0
    assert torch.isfinite(logits.grad).all()


def test_hard_nearest_respects_padding():
    logits = torch.tensor([[0.0, 5.0, -1.0]], requires_grad=True)
    distances = torch.tensor([[10.0, 0.0, 30.0]])
    mask = torch.tensor([[True, False, True]])
    loss = hard_nearest_loss(logits, distances, mask)
    loss.backward()
    assert logits.grad[0, 1] == 0
    assert loss.item() > 0


def test_auxiliary_and_total_losses_are_independently_recorded():
    altitude = altitude_loss(torch.tensor([1.0]), torch.tensor([0.0]))
    progress = progress_loss(torch.tensor([0.25]), torch.tensor([0.5]))
    scalar = torch.tensor(1.0)
    losses = combine_losses(
        field=scalar,
        listwise=scalar,
        margin=scalar,
        hard=scalar,
        altitude=altitude,
        progress=progress,
        weights=LossWeights(),
    )
    assert set(losses) == {"field", "list", "margin", "hard", "rank", "altitude", "progress", "total"}
    assert torch.isfinite(losses["total"])
