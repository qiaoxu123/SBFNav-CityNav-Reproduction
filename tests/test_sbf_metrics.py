import numpy as np
import pytest

from sbfnav.metrics import aggregate_metrics, evaluate_episode


def test_success_oracle_and_spl_definitions():
    target = np.asarray((20, 0, 0))
    direct = evaluate_episode(np.asarray(((0, 0, 0), (20, 0, 0))), target)
    assert direct.navigation_error == 0
    assert direct.success == 1
    assert direct.oracle_success == 1
    assert direct.spl == pytest.approx(1)
    detour = evaluate_episode(
        np.asarray(((0, 0, 0), (0, 20, 0), (20, 20, 0), (20, 0, 0))),
        target,
        success_distance_m=1,
    )
    assert detour.spl == pytest.approx(1 / 3)


def test_three_dimensional_primary_and_two_dimensional_compatibility_differ():
    path = np.asarray(((0, 0, 100), (10, 0, 100)))
    target = np.asarray((10, 0, 0))
    primary = evaluate_episode(path, target, success_distance_m=20, dimensions=3)
    compatibility = evaluate_episode(path, target, success_distance_m=20, dimensions=2)
    assert primary.success == 0
    assert compatibility.success == 1


def test_oracle_can_succeed_when_final_fails_and_aggregate_uses_percent():
    metric = evaluate_episode(
        np.asarray(((0, 0, 0), (10, 0, 0), (100, 0, 0))),
        np.asarray((10, 0, 0)),
        success_distance_m=1,
    )
    assert metric.success == 0
    assert metric.oracle_success == 1
    aggregate = aggregate_metrics([metric, metric])
    assert aggregate["sr"] == 0
    assert aggregate["osr"] == 100
    assert aggregate["episodes"] == 2

