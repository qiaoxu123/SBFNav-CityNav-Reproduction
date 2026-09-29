import os
from pathlib import Path

import numpy as np
import torch

from sbfnav.checkpointing import git_commit, load_checkpoint, save_checkpoint
from sbfnav.dataset import CityNavDataset, CityReferCatalog
from sbfnav.models.belief_field import SpatialBeliefField
from sbfnav.planning_state import PlanningStateBuilder
from sbfnav.runtime import VisionFeatureCache
from sbfnav.training_data import SBFTrainingDataset


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("SBFNAV_TEST_DATA_ROOT", ROOT / "data"))


def test_supervised_snapshot_commit_can_be_injected(monkeypatch, tmp_path):
    monkeypatch.setenv("SBFNAV_GIT_COMMIT", "0123456789abcdef")
    assert git_commit(tmp_path) == "0123456789abcdef"


def test_bounded_vision_cache_counts_hits_misses_and_evicts():
    cache = VisionFeatureCache(maximum_entries=2)
    assert cache.get("a") is None
    cache.put("a", torch.ones(3))
    cache.put("b", torch.ones(3) * 2)
    torch.testing.assert_close(cache.get("a"), torch.ones(3, dtype=torch.float16))
    cache.put("c", torch.ones(3) * 3)
    assert cache.get("b") is None
    assert cache.statistics() == {"entries": 2, "hits": 1, "misses": 2}


def test_training_example_real_data_is_finite_and_deterministic():
    catalog = CityReferCatalog(DATA_ROOT / "cityrefer")
    episodes = CityNavDataset(DATA_ROOT, "train_seen", max_records=1)
    builder = PlanningStateBuilder(DATA_ROOT, catalog)
    dataset = SBFTrainingDataset(episodes, builder, samples_per_episode=2)
    first = dataset[0]
    repeated = dataset[0]
    assert len(dataset) == 2
    assert first.state.values.shape == (9, 224, 224)
    assert first.field_target.shape == (28, 28)
    np.testing.assert_allclose(first.field_target.sum().item(), 1, atol=1e-6)
    np.testing.assert_array_equal(first.state.values, repeated.state.values)
    assert 0 <= first.altitude_target_normalized <= 1
    assert 0 <= first.progress_target <= 1


def test_checkpoint_model_optimizer_scheduler_round_trip(tmp_path):
    torch.manual_seed(0)
    model = SpatialBeliefField(hidden_dim=32, attention_heads=4, dropout=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    inputs = (
        torch.randn(1, 9, 224, 224),
        torch.randn(1, 3, 768),
        torch.ones(1, 3, dtype=torch.bool),
        torch.ones(1, 28, 28, dtype=torch.bool),
    )
    model(*inputs).logits.mean().backward()
    optimizer.step()
    scheduler.step()
    expected = {key: value.detach().clone() for key, value in model.state_dict().items()}
    checkpoint = tmp_path / "last.pt"
    save_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        completed_epoch=1,
        global_step=1,
        config={"test": True},
        repository_root=ROOT,
    )
    with torch.no_grad():
        next(model.parameters()).add_(10)
    metadata = load_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        restore_rng=True,
    )
    assert metadata["completed_epoch"] == 1
    assert metadata["global_step"] == 1
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, expected[key])
