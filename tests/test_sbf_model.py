import torch

from sbfnav.models.belief_field import SpatialBeliefField
from sbfnav.models.sbfnav import SBFNav
from sbfnav.models.selector import SelectorInputs


def test_belief_field_forward_masking_and_backward():
    torch.manual_seed(0)
    model = SpatialBeliefField(hidden_dim=64, attention_heads=4, dropout=0.0)
    state = torch.randn(2, 9, 224, 224)
    tokens = torch.randn(2, 12, 768)
    token_mask = torch.ones(2, 12, dtype=torch.bool)
    token_mask[0, 8:] = False
    valid = torch.ones(2, 28, 28, dtype=torch.bool)
    valid[0, :, :4] = False
    output = model(state, tokens, token_mask, valid)
    assert output.logits.shape == (2, 28, 28)
    assert output.probabilities.shape == (2, 28, 28)
    torch.testing.assert_close(output.probabilities.sum((1, 2)), torch.ones(2))
    assert torch.all(output.probabilities[0, :, :4] == 0)
    output.logits.mean().backward()
    assert model.decoder[-1].weight.grad is not None


def test_gate_starts_at_zero_but_receives_gradient():
    torch.manual_seed(1)
    model = SpatialBeliefField(hidden_dim=64, attention_heads=4, dropout=0.0, gate_init=0.0)
    output = model(
        torch.randn(1, 9, 224, 224),
        torch.randn(1, 5, 768),
        torch.ones(1, 5, dtype=torch.bool),
        torch.ones(1, 28, 28, dtype=torch.bool),
    )
    assert output.gate.item() == 0.0
    output.logits.square().mean().backward()
    assert model.gamma.grad is not None
    assert model.gamma.grad.abs().item() > 0


def test_token_language_changes_output_after_opening_gate():
    torch.manual_seed(2)
    model = SpatialBeliefField(hidden_dim=64, attention_heads=4, dropout=0.0, gate_init=0.5).eval()
    state = torch.randn(1, 9, 224, 224)
    mask = torch.ones(1, 4, dtype=torch.bool)
    valid = torch.ones(1, 28, 28, dtype=torch.bool)
    first = model(state, torch.zeros(1, 4, 768), mask, valid).logits
    second = model(state, torch.ones(1, 4, 768), mask, valid).logits
    assert not torch.allclose(first, second)


def test_composed_sbfnav_all_trainable_heads_forward_and_backward():
    torch.manual_seed(3)
    model = SBFNav(hidden_dim=64, attention_heads=4, selector_layers=1, dropout=0)
    candidate_mask = torch.tensor([[1, 1, 0]], dtype=torch.bool)
    selector = SelectorInputs(
        candidate_visual=torch.randn(1, 3, 768),
        candidate_xy=torch.randn(1, 3, 2),
        candidate_mask=candidate_mask,
        instruction_tokens=torch.randn(1, 7, 768),
        instruction_mask=torch.tensor([[1, 1, 1, 1, 1, 0, 0]], dtype=torch.bool),
        landmark_name_features=torch.randn(1, 2, 768),
        landmark_xy=torch.randn(1, 2, 2),
        landmark_mask=torch.tensor([[1, 0]], dtype=torch.bool),
        agent_xy=torch.zeros(1, 2),
        map_scale_m=torch.tensor([400.0]),
    )
    output = model(
        torch.randn(1, 9, 224, 224),
        torch.ones(1, 28, 28, dtype=torch.bool),
        selector.instruction_tokens,
        selector.instruction_mask,
        torch.randn(1, 768),
        selector,
    )
    assert output.field.logits.shape == (1, 28, 28)
    assert output.selector_logits.shape == (1, 3)
    loss = (
        output.field.logits.mean()
        + output.auxiliary.altitude_normalized.mean()
        + output.auxiliary.progress.mean()
        + output.selector_logits[candidate_mask].mean()
    )
    loss.backward()
    assert model.belief_field.decoder[-1].weight.grad is not None
    assert model.selector.score.weight.grad is not None
    assert model.auxiliary.altitude.weight.grad is not None
