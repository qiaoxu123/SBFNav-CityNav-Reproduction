"""YAML configuration loading with strict recursive overrides."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, MutableMapping

import yaml


def _merge(base: dict[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in update.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, Mapping):
            result[key] = _merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path: str | Path, _stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    path = Path(path)
    path = path.resolve()
    if path in _stack:
        raise ValueError(f"cyclic configuration inheritance: {path}")
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"configuration root must be a mapping: {path}")
    value = copy.deepcopy(value)
    base_path = value.pop("_base_", None)
    if base_path is not None:
        parent = Path(base_path)
        if not parent.is_absolute():
            parent = path.parent / parent
        value = _merge(load_config(parent, _stack + (path,)), value)
    value["_config_path"] = str(path)
    return value


def apply_overrides(config: Mapping[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply `a.b=value` overrides while rejecting unknown keys."""
    result = copy.deepcopy(dict(config))
    for expression in overrides:
        if "=" not in expression:
            raise ValueError(f"override must have key=value form: {expression!r}")
        dotted, raw = expression.split("=", 1)
        parts = dotted.split(".")
        cursor: MutableMapping[str, Any] = result
        for part in parts[:-1]:
            if part not in cursor or not isinstance(cursor[part], MutableMapping):
                raise KeyError(f"unknown configuration path {dotted!r}")
            cursor = cursor[part]
        if parts[-1] not in cursor:
            raise KeyError(f"unknown configuration key {dotted!r}")
        cursor[parts[-1]] = yaml.safe_load(raw)
    return result


def resolved_config_sha256(config: Mapping[str, Any]) -> str:
    """Hash effective values while excluding the machine-local source path."""
    value = copy.deepcopy(dict(config))
    value.pop("_config_path", None)
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_config(config: Mapping[str, Any]) -> None:
    required = ("data", "model", "loss", "proposal", "selector", "training", "navigation")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"missing configuration sections: {missing}")
    if config["model"]["input_channels"] != 9:
        raise ValueError("the paper-defined planning state must have 9 channels")
    if config["model"]["canvas_size"] != 224 or config["model"]["field_size"] != 28:
        raise ValueError("the paper main geometry is fixed at 224→28")
    if config["proposal"]["top_k"] < 1:
        raise ValueError("proposal.top_k must be positive")
    if config["proposal"]["mode"] not in ("field", "uniform_grid"):
        raise ValueError("proposal.mode must be field or uniform_grid")
    if config["selector"]["selection_mode"] not in ("selector", "sbf_top1"):
        raise ValueError("selector.selection_mode must be selector or sbf_top1")
    if config["loss"]["good_radius_m"] >= config["loss"]["bad_radius_m"]:
        raise ValueError("good_radius_m must be less than bad_radius_m")
    training = config["training"]
    if training["batch_size"] * training["gradient_accumulation"] != training["effective_batch_size"]:
        raise ValueError("batch_size × gradient_accumulation must equal effective_batch_size")
    if training["vision_batch_size"] < 1 or training["vision_cache_entries"] < 0:
        raise ValueError("invalid frozen-vision batching/cache configuration")
    if training["text_precompute_batch_size"] < 1:
        raise ValueError("text_precompute_batch_size must be positive")
    evaluation = config.get("evaluation", {})
    if evaluation.get("official_metric") != "benchmark_2d":
        raise ValueError("the CityNav leaderboard metric must be benchmark_2d")
    if evaluation.get("benchmark_dimensions") != 2:
        raise ValueError("the released CityNav benchmark evaluates Pose.xy")
    if evaluation.get("diagnostic_dimensions") != 3:
        raise ValueError("the additional Euclidean diagnostic must be 3-D")
