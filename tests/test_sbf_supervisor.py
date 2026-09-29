import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.supervise_experiment import newest_activity


def test_supervisor_uses_newest_structured_training_heartbeat(tmp_path):
    old_log = tmp_path / "training.log"
    steps = tmp_path / "train_steps.jsonl"
    old_log.write_text("startup\n")
    steps.write_text("{}\n")
    now = time.time()
    os.utime(old_log, (now - 3600, now - 3600))
    os.utime(steps, (now - 2, now - 2))
    path, age = newest_activity([old_log, steps, tmp_path / "epochs.jsonl"])
    assert path == steps
    assert 0 <= age < 10


def test_supervisor_exposes_candidate_analysis_phase():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(root / "scripts" / "supervise_experiment.py"), "--help"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "candidate-analysis" in result.stdout
    assert "--analysis-prefix-fraction" in result.stdout
