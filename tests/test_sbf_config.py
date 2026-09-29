from pathlib import Path

import pytest

from sbfnav.config import apply_overrides, load_config, resolved_config_sha256, validate_config


ROOT = Path(__file__).resolve().parents[1]


def test_main_and_debug_configs_are_valid_and_test_is_sealed():
    for name in ("main.yaml", "debug.yaml"):
        config = load_config(ROOT / "configs" / "sbfnav" / name)
        validate_config(config)
        assert config["data"]["allow_test_unseen"] is False
        assert config["selector"]["use_field_score"] is False


def test_overrides_are_typed_and_unknown_keys_rejected():
    config = load_config(ROOT / "configs" / "sbfnav" / "debug.yaml")
    changed = apply_overrides(config, ["proposal.top_k=4", "model.dropout=0.2"])
    assert changed["proposal"]["top_k"] == 4
    assert changed["model"]["dropout"] == pytest.approx(0.2)
    with pytest.raises(KeyError):
        apply_overrides(config, ["model.not_a_key=1"])


def test_all_ablation_configs_inherit_a_complete_valid_main_config():
    paths = sorted((ROOT / "configs" / "sbfnav" / "ablations").glob("*.yaml"))
    assert len(paths) >= 14
    for path in paths:
        config = load_config(path)
        validate_config(config)
        assert config["_config_path"] == str(path.resolve())
        assert config["selector"]["use_field_score"] is False


@pytest.mark.parametrize(
    ("name", "path", "expected"),
    [
        ("k4.yaml", "proposal.top_k", 4),
        ("k8.yaml", "proposal.top_k", 8),
        ("k16.yaml", "proposal.top_k", 16),
        ("k32.yaml", "proposal.top_k", 32),
        ("no_cross_attention.yaml", "model.use_cross_attention", False),
        ("pooled_text.yaml", "model.pooled_text_only", True),
        ("no_geometry.yaml", "selector.use_geometry", False),
        ("no_landmark_text.yaml", "selector.use_landmark_text", False),
        ("no_landmark_candidates.yaml", "proposal.add_landmark_candidates", False),
        ("no_ranking_loss.yaml", "loss.rank_weight", 0.0),
        ("no_hard_nearest_loss.yaml", "loss.hard_weight", 0.0),
        ("no_altitude_auxiliary_loss.yaml", "loss.altitude_weight", 0.0),
        ("sbf_only.yaml", "selector.selection_mode", "sbf_top1"),
        ("selector_only.yaml", "proposal.mode", "uniform_grid"),
    ],
)
def test_ablation_config_changes_the_named_mechanism(name, path, expected):
    config = load_config(ROOT / "configs" / "sbfnav" / "ablations" / name)
    value = config
    for component in path.split("."):
        value = value[component]
    assert value == expected


def test_effective_config_hash_tracks_values_not_source_path():
    config = load_config(ROOT / "configs" / "sbfnav" / "debug.yaml")
    same = dict(config, _config_path="/some/other/snapshot.yaml")
    assert resolved_config_sha256(config) == resolved_config_sha256(same)
    changed = apply_overrides(config, ["proposal.top_k=4"])
    assert resolved_config_sha256(config) != resolved_config_sha256(changed)
