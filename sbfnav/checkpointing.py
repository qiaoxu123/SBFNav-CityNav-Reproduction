"""Checkpoint save/resume helpers including RNG state."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import random
import subprocess
from typing import Any

import numpy as np
import torch


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit(root: str | Path) -> str:
    injected = os.environ.get("SBFNAV_GIT_COMMIT")
    if injected:
        return injected
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def rng_state() -> dict[str, Any]:
    value = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        value["cuda"] = torch.cuda.get_rng_state_all()
    return value


def restore_rng_state(value: dict[str, Any]) -> None:
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch"].cpu())
    if "cuda" in value and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([state.cpu() for state in value["cuda"]])


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    completed_epoch: int,
    global_step: int,
    config: dict,
    repository_root: str | Path,
    metrics: dict | None = None,
    backbone: torch.nn.Module | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "format_version": 1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "completed_epoch": completed_epoch,
        "global_step": global_step,
        "config": config,
        "git_commit": git_commit(repository_root),
        "rng_state": rng_state(),
        "metrics": metrics or {},
    }
    if backbone is not None:
        payload["backbone"] = backbone.state_dict()
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    restore_rng: bool = False,
    map_location: str | torch.device = "cpu",
    backbone: torch.nn.Module | None = None,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    if "backbone" in checkpoint:
        if backbone is None:
            raise ValueError("checkpoint contains a trainable backbone but none was supplied")
        backbone.load_state_dict(checkpoint["backbone"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if restore_rng:
        restore_rng_state(checkpoint["rng_state"])
    return checkpoint
