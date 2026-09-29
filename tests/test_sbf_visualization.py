import os
from pathlib import Path
import subprocess
import sys

from scripts.visualize_sbf_episodes import select_cases


def test_case_selection_is_metric_based_and_balanced():
    predictions = [
        {"benchmark_2d": {"success": 1.0, "spl": 0.4, "navigation_error": 10}},
        {"benchmark_2d": {"success": 1.0, "spl": 0.8, "navigation_error": 15}},
        {"benchmark_2d": {"success": 0.0, "spl": 0.0, "navigation_error": 30}},
        {"benchmark_2d": {"success": 0.0, "spl": 0.0, "navigation_error": 80}},
    ]
    selected = select_cases(predictions, 1)
    assert [label for label, _item in selected] == ["success", "failure"]
    assert selected[0][1]["benchmark_2d"]["spl"] == 0.8
    assert selected[1][1]["benchmark_2d"]["navigation_error"] == 80


def test_episode_visualizer_cannot_open_test_unseen(tmp_path):
    root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root)
    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "visualize_sbf_episodes.py"),
            "--predictions",
            str(tmp_path / "predictions.jsonl"),
            "--data-root",
            str(tmp_path),
            "--split",
            "test_unseen",
            "--output-dir",
            str(tmp_path / "output"),
        ],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "invalid choice" in result.stderr
