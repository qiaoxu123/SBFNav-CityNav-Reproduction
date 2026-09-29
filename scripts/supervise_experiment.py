#!/usr/bin/env python3
"""Run isolated SBFNav train/eval jobs with provenance, lock, and telemetry."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import yaml


def stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def command_output(command, cwd=None) -> str:
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=60)
    return result.stdout + result.stderr


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def append(path: Path, value: dict) -> None:
    with path.open("a") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")


def newest_activity(paths: list[Path]) -> tuple[Path, float]:
    existing = [path for path in paths if path.exists()]
    if not existing:
        raise FileNotFoundError("no heartbeat files exist")
    newest = max(existing, key=lambda path: path.stat().st_mtime)
    return newest, time.time() - newest.stat().st_mtime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--phase",
        choices=("train-eval", "eval", "candidate-analysis"),
        default="train-eval",
    )
    parser.add_argument("--checkpoint")
    parser.add_argument("--resume-from")
    parser.add_argument("--resume-optimizer", action="store_true")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--stop-after-epoch", type=int)
    parser.add_argument("--max-train-records", type=int)
    parser.add_argument("--max-eval-episodes", type=int)
    parser.add_argument("--analysis-prefix-fraction", type=float, default=0.5)
    parser.add_argument(
        "--evaluation-split",
        action="append",
        choices=("val_seen", "val_unseen", "test_unseen"),
        dest="evaluation_splits",
    )
    parser.add_argument("--allow-test-unseen", action="store_true")
    parser.add_argument("--freeze-manifest")
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--interval", type=int, default=30)
    parser.add_argument("--variant-arg", action="append", default=[])
    parser.add_argument(
        "--config-override",
        action="append",
        default=[],
        help="key=value override forwarded identically to train and evaluation",
    )
    parser.add_argument(
        "--lock-file",
        default="/home/ubuntu5/Workspace/hett-experiments/.gpu-validation.lock",
    )
    args = parser.parse_args()
    if args.phase in ("eval", "candidate-analysis") and not args.checkpoint:
        parser.error(f"--checkpoint is required for {args.phase}")
    if not 0 < args.analysis_prefix_fraction <= 1:
        parser.error("--analysis-prefix-fraction must be in (0, 1]")
    evaluation_splits = args.evaluation_splits or ["val_seen", "val_unseen"]
    if "test_unseen" in evaluation_splits and (
        not args.allow_test_unseen or not args.freeze_manifest or args.phase != "eval"
    ):
        parser.error(
            "test_unseen requires eval phase, --allow-test-unseen, and --freeze-manifest"
        )
    if args.phase == "candidate-analysis" and "test_unseen" in evaluation_splits:
        parser.error("candidate analysis is restricted to development splits")
    root = Path(__file__).resolve().parents[1]
    run = Path(args.run_dir).resolve()
    run.mkdir(parents=True, exist_ok=False)
    snapshot = run / "source"
    snapshot.mkdir()
    for directory in ("sbfnav", "scripts", "configs", "docs", "multiagent"):
        shutil.copytree(
            root / directory,
            snapshot / directory,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "checkpoints", "runs"),
        )
    for name in ("README.md", "SINGLE_GPU.md", "requirements.txt", "AGENTS.md"):
        if (root / name).exists():
            shutil.copy2(root / name, snapshot / name)
    for name in ("data", "weights"):
        source = root / name
        if source.exists():
            (snapshot / name).symlink_to(source.resolve(), target_is_directory=True)
    (run / "git.diff").write_text(command_output(["git", "diff", "HEAD"], root))
    (run / "git_status.txt").write_text(command_output(["git", "status", "--short"], root))
    (run / "pip_freeze.txt").write_text(command_output([args.python, "-m", "pip", "freeze"]))
    (run / "gpu_initial.txt").write_text(command_output(["nvidia-smi"]))
    source_hashes = {
        str(path.relative_to(snapshot)): digest(path)
        for path in snapshot.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    config_source = Path(args.config).resolve()
    config_snapshot = snapshot / "configs" / "sbfnav" / config_source.name
    if not config_snapshot.exists():
        shutil.copy2(config_source, snapshot / "configs" / "sbfnav" / "supplied.yaml")
        config_snapshot = snapshot / "configs" / "sbfnav" / "supplied.yaml"
    raw_config = yaml.safe_load(config_source.read_text())
    configured_data_root = raw_config["data"]["root"]
    for expression in args.config_override:
        if expression.startswith("data.root="):
            configured_data_root = expression.split("=", 1)[1]
    data_root = Path(configured_data_root).resolve()
    snapshot_data = snapshot / "data"
    if snapshot_data.is_symlink():
        snapshot_data.unlink()
    elif snapshot_data.exists():
        raise RuntimeError(f"refusing to replace non-symlink snapshot path: {snapshot_data}")
    snapshot_data.symlink_to(data_root, target_is_directory=True)
    inputs = []
    for relative in (
        "cityrefer/objects.json",
        "cityrefer/processed_descriptions.json",
        "processed_citynav/citynav_train_seen.json",
        "processed_citynav/citynav_val_seen.json",
        "processed_citynav/citynav_val_unseen.json",
    ):
        path = data_root / relative
        inputs.append(
            {"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)}
        )
    if "test_unseen" in evaluation_splits:
        path = data_root / "processed_citynav" / "citynav_test_unseen.json"
        inputs.append({"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)})
    raster_inventory = [
        {"path": str(path), "bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
        for path in sorted((data_root / "rgbd").iterdir())
        if path.is_file()
    ]
    git_head = command_output(["git", "rev-parse", "HEAD"], root).strip()
    atomic_json(
        run / "provenance.json",
        {
            "started": stamp(),
            "paper_based_reimplementation": True,
            "git_head": git_head,
            "phase": args.phase,
            "source_sha256": source_hashes,
            "input_sha256": inputs,
            "raster_inventory": raster_inventory,
            "config_path": str(config_source),
            "config_sha256": digest(config_source),
            "variant_args": args.variant_arg,
            "config_overrides": args.config_override,
            "resume_from": (
                {"path": str(Path(args.resume_from).resolve()), "sha256": digest(Path(args.resume_from))}
                if args.resume_from
                else None
            ),
        },
    )
    environment = os.environ.copy()
    environment.update(
        CUDA_VISIBLE_DEVICES="0",
        PYTHONUNBUFFERED="1",
        PYTHONPATH=str(snapshot),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
        SBFNAV_GIT_COMMIT=git_head,
    )
    training_dir = run / "training"
    train = [
        args.python,
        str(snapshot / "scripts" / "train_sbfnav.py"),
        "--config",
        str(config_snapshot),
        "--output-dir",
        str(training_dir),
        "--save-every",
        str(args.save_every),
    ]
    if args.epochs is not None:
        train += ["--epochs", str(args.epochs)]
    if args.stop_after_epoch is not None:
        train += ["--stop-after-epoch", str(args.stop_after_epoch)]
    if args.max_train_records is not None:
        train += ["--max-train-records", str(args.max_train_records)]
    if args.resume_from:
        train += ["--resume", str(Path(args.resume_from).resolve())]
        if args.resume_optimizer:
            train += ["--resume-optimizer"]
    train += args.variant_arg
    for override in args.config_override:
        train += ["--override", override]
    checkpoint = (
        Path(args.checkpoint).resolve()
        if args.checkpoint
        else training_dir / "checkpoints" / "best_val_unseen.pt"
    )
    evaluations = []
    for split in evaluation_splits:
        output = run / "evaluation" / split
        command = [
            args.python,
            str(snapshot / "scripts" / "evaluate_sbfnav.py"),
            "--config",
            str(config_snapshot),
            "--checkpoint",
            str(checkpoint),
            "--split",
            split,
            "--output-dir",
            str(output),
        ]
        if args.max_eval_episodes is not None:
            command += ["--max-episodes", str(args.max_eval_episodes)]
        for override in args.config_override:
            command += ["--override", override]
        if split == "test_unseen":
            command += [
                "--allow-test-unseen",
                "--freeze-manifest",
                str(Path(args.freeze_manifest).resolve()),
            ]
        evaluations.append((f"evaluation_{split}", command))
    candidate_analyses = []
    if args.phase == "candidate-analysis":
        for split in evaluation_splits:
            output = run / "candidate_analysis" / split
            command = [
                args.python,
                str(snapshot / "scripts" / "analyze_sbf_candidates.py"),
                "--config",
                str(config_snapshot),
                "--checkpoint",
                str(checkpoint),
                "--split",
                split,
                "--output-dir",
                str(output),
                "--prefix-fraction",
                str(args.analysis_prefix_fraction),
            ]
            if args.max_eval_episodes is not None:
                command += ["--max-episodes", str(args.max_eval_episodes)]
            for override in args.config_override:
                command += ["--override", override]
            candidate_analyses.append((f"candidate_analysis_{split}", command))
    atomic_json(
        run / "commands.json",
        {
            "train": train,
            "evaluations": [command for _name, command in evaluations],
            "candidate_analyses": [command for _name, command in candidate_analyses],
            "environment_overrides": {
                key: environment[key]
                for key in (
                    "CUDA_VISIBLE_DEVICES",
                    "PYTHONUNBUFFERED",
                    "PYTHONPATH",
                    "HF_HUB_OFFLINE",
                    "TRANSFORMERS_OFFLINE",
                    "CUBLAS_WORKSPACE_CONFIG",
                    "SBFNAV_GIT_COMMIT",
                )
            },
        },
    )
    lock_path = Path(args.lock_file)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_stream = lock_path.open("w")
    atomic_json(run / "status.json", {"time": stamp(), "phase": "waiting_gpu"})
    fcntl.flock(lock_stream, fcntl.LOCK_EX)
    lock_stream.write(f"{os.getpid()} {run}\n")
    lock_stream.flush()

    stopped = False
    child = None

    def stop(_signum, _frame):
        nonlocal stopped
        stopped = True
        if child is not None and child.poll() is None:
            child.terminate()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    if args.phase == "eval":
        phases = evaluations
    elif args.phase == "candidate-analysis":
        phases = candidate_analyses
    else:
        phases = [("training", train), *evaluations]
    for phase_name, command in phases:
        if stopped:
            break
        if phase_name.startswith(("evaluation", "candidate_analysis")) and not checkpoint.exists():
            raise FileNotFoundError(f"checkpoint missing for {phase_name}: {checkpoint}")
        log_path = run / f"{phase_name}.log"
        with log_path.open("w") as log:
            child = subprocess.Popen(
                command, cwd=snapshot, env=environment, stdout=log, stderr=subprocess.STDOUT
            )
            append(
                run / "events.jsonl",
                {"time": stamp(), "event": "started", "phase": phase_name, "pid": child.pid},
            )
            while True:
                code = child.poll()
                heartbeat_paths = [log_path]
                if phase_name == "training":
                    heartbeat_paths.extend(
                        (training_dir / "train_steps.jsonl", training_dir / "epochs.jsonl")
                    )
                heartbeat_path, age = newest_activity(heartbeat_paths)
                status = {
                    "time": stamp(),
                    "phase": phase_name,
                    "supervisor_pid": os.getpid(),
                    "child_pid": child.pid,
                    "exit_code": code,
                    "log_bytes": log_path.stat().st_size,
                    "log_age_seconds": age,
                    "heartbeat_path": str(heartbeat_path),
                    "disk_free_gib": shutil.disk_usage(run).free / 2**30,
                    "gpu": command_output(
                        [
                            "nvidia-smi",
                            "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                            "--format=csv,noheader,nounits",
                        ]
                    ).strip(),
                    "alerts": [],
                }
                if age > 1800 and code is None:
                    status["alerts"].append("no log update for 30 minutes")
                if status["disk_free_gib"] < 20:
                    status["alerts"].append("less than 20 GiB free disk")
                if code not in (None, 0):
                    status["alerts"].append("child exited with error")
                atomic_json(run / "status.json", status)
                append(run / "telemetry.jsonl", status)
                if status["alerts"]:
                    append(run / "alerts.jsonl", status)
                if code is not None:
                    append(
                        run / "events.jsonl",
                        {"time": stamp(), "event": "exited", "phase": phase_name, "exit_code": code},
                    )
                    if code != 0:
                        raise SystemExit(code if code > 0 else 1)
                    break
                time.sleep(args.interval)
    atomic_json(
        run / "status.json",
        {"time": stamp(), "phase": "stopped" if stopped else "complete", "supervisor_pid": os.getpid()},
    )


if __name__ == "__main__":
    main()
