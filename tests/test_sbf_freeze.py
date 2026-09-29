import json

import pytest

from scripts.freeze_sbfnav_test import checked_validation


def metric_record(**updates):
    value = {
        "split": "val_seen",
        "checkpoint_sha256": "checkpoint",
        "data_version": "processed_citynav",
        "effective_config_sha256": "config",
        "config_overrides": ["data.root=/audited"],
        "metric_protocol": {"official": "benchmark_2d"},
    }
    value.update(updates)
    return value


def test_freeze_validation_binds_checkpoint_config_and_protocol(tmp_path):
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps(metric_record()))
    assert checked_validation(
        path, "val_seen", "checkpoint", "config", ["data.root=/audited"]
    )["split"] == "val_seen"


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"checkpoint_sha256": "other"}, "checkpoint"),
        ({"effective_config_sha256": "other"}, "effective config"),
        ({"config_overrides": []}, "overrides"),
        ({"metric_protocol": {"official": "diagnostic_3d"}}, "benchmark protocol"),
    ],
)
def test_freeze_rejects_mismatched_validation_evidence(tmp_path, updates, message):
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps(metric_record(**updates)))
    with pytest.raises(ValueError, match=message):
        checked_validation(
            path, "val_seen", "checkpoint", "config", ["data.root=/audited"]
        )
