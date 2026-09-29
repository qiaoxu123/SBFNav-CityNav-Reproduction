import pytest

from sbfnav.validation import checkpoint_selection_key, evenly_spaced_indices


def test_validation_subset_is_deterministic_and_split_wide():
    assert evenly_spaced_indices(10, 4) == [0, 3, 6, 9]
    assert evenly_spaced_indices(3, None) == [0, 1, 2]
    assert evenly_spaced_indices(3, 10) == [0, 1, 2]
    with pytest.raises(ValueError):
        evenly_spaced_indices(10, 0)


def test_checkpoint_selection_orders_sr_spl_then_ne():
    def value(sr, spl, ne):
        return {"benchmark_2d": {"sr": sr, "spl": spl, "ne": ne}}

    assert checkpoint_selection_key(value(2, 0, 100)) > checkpoint_selection_key(
        value(1, 100, 0)
    )
    assert checkpoint_selection_key(value(2, 4, 100)) > checkpoint_selection_key(
        value(2, 3, 0)
    )
    assert checkpoint_selection_key(value(2, 4, 10)) > checkpoint_selection_key(
        value(2, 4, 20)
    )


def test_checkpoint_selection_never_uses_3d_diagnostic():
    validation = {
        "benchmark_2d": {"sr": 7, "spl": 6, "ne": 5},
        "diagnostic_3d": {"sr": 100, "spl": 100, "ne": 0},
    }
    assert checkpoint_selection_key(validation) == (7.0, 6.0, -5.0)


def test_checkpoint_selection_can_resume_legacy_metric_names():
    legacy = {"compatibility_2d": {"sr": 7, "spl": 6, "ne": 5}}
    assert checkpoint_selection_key(legacy) == (7.0, 6.0, -5.0)
