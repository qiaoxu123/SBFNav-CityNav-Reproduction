import os
from pathlib import Path

import pytest

from sbfnav.dataset import (
    CityNavDataset,
    CityReferCatalog,
    assert_development_split_isolation,
    difficulty_membership,
)


DATA_ROOT = Path(
    os.environ.get("SBFNAV_TEST_DATA_ROOT", Path(__file__).resolve().parents[1] / "data")
)


def test_expected_revised_counts_and_split_isolation():
    train = CityNavDataset(DATA_ROOT, "train_seen")
    val_seen = CityNavDataset(DATA_ROOT, "val_seen")
    val_unseen = CityNavDataset(DATA_ROOT, "val_unseen")
    assert (len(train), len(val_seen), len(val_unseen)) == (21878, 2470, 2697)
    assert_development_split_isolation((train, val_seen, val_unseen))
    assert not ((train.map_names | val_seen.map_names) & val_unseen.map_names)
    assert not (train.episode_ids & val_unseen.episode_ids)


def test_test_unseen_is_sealed_by_default():
    with pytest.raises(PermissionError, match="sealed"):
        CityNavDataset(DATA_ROOT, "test_unseen", max_records=1)


def test_cityrefer_alignment_on_real_samples():
    catalog = CityReferCatalog(DATA_ROOT / "cityrefer")
    for split in ("train_seen", "val_seen", "val_unseen"):
        dataset = CityNavDataset(DATA_ROOT, split, max_records=8)
        mismatch_count = dataset.validate_against_cityrefer(catalog)
        assert 0 <= mismatch_count <= len(dataset)
        assert dataset[0].trajectory.shape[1] == 4


def test_difficulty_file_is_explicit():
    easy = CityNavDataset(DATA_ROOT, "val_unseen", difficulty="easy", max_records=5)
    assert len(easy) == 5
    assert easy.annotation_path.name == "citynav_val_unseen_easy.json"


def test_difficulty_files_exactly_partition_full_validation_split():
    full = CityNavDataset(DATA_ROOT, "val_unseen")
    membership = difficulty_membership(full)
    assert len(membership) == len(full)
    assert set(membership.values()) == {"easy", "medium", "hard"}
    assert set(membership) == full.episode_ids


def test_landmark_aliases_have_deterministic_text_only_resolution():
    catalog = CityReferCatalog(DATA_ROOT / "cityrefer")
    dataset = CityNavDataset(DATA_ROOT, "train_seen", max_records=32)
    record = next(item for item in dataset if "Wellington Rd" in catalog.referenced_landmark_names(item))
    resolutions = dict(catalog.landmark_resolutions(record))
    assert resolutions["Wellington Rd"]
    assert catalog.resolve_landmark(record.map_name, "Wellington Rd").object_id == catalog.resolve_landmark(
        record.map_name, "Wellington Rd"
    ).object_id
