#!/usr/bin/env python3
"""Audit SBFNav closed-loop failures from evaluator prediction JSONL files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def selected_candidate_xy(step: dict[str, Any] | None) -> np.ndarray | None:
    if not step:
        return None
    candidates = step.get("candidate_xy")
    index = step.get("selected_index")
    if candidates is None or index is None:
        return None
    if not 0 <= int(index) < len(candidates):
        return None
    value = np.asarray(candidates[int(index)], dtype=np.float64)
    return value if value.shape == (2,) and np.isfinite(value).all() else None


def selector_margin(step: dict[str, Any] | None) -> float | None:
    if not step:
        return None
    logits = step.get("selector_logits")
    if not logits:
        return None
    finite = np.asarray([value for value in logits if np.isfinite(value)], dtype=np.float64)
    if finite.size < 2:
        return None
    top = np.partition(finite, -2)[-2:]
    return float(top.max() - top.min())


def analyze_record(
    record: dict[str, Any],
    *,
    success_distance_m: float,
    low_spl_threshold: float,
    switch_distance_m: float,
) -> dict[str, Any]:
    trajectory = np.asarray(record["trajectory_xyzyaw"], dtype=np.float64)
    target = np.asarray(record["target_xyz"], dtype=np.float64)
    if trajectory.ndim != 2 or trajectory.shape[1] < 2:
        raise ValueError("trajectory_xyzyaw must be an N x >=2 array")
    distances = np.linalg.norm(trajectory[:, :2] - target[:2], axis=1)
    inside = distances <= success_distance_m
    hits = np.flatnonzero(inside)
    first_hit = int(hits[0]) if hits.size else None
    last_hit = int(hits[-1]) if hits.size else None
    leaves_after_first_hit = bool(
        first_hit is not None and np.any(distances[first_hit + 1 :] > success_distance_m)
    )

    benchmark = record["benchmark_2d"]
    success = bool(benchmark["success"])
    oracle_success = bool(benchmark["oracle_success"])
    spl = float(benchmark["spl"])
    if success:
        category = "A_success_high_spl" if spl >= low_spl_threshold else "B_success_low_spl"
    elif oracle_success:
        category = "C_oracle_only_final_failure"
    else:
        category = "D_never_reached"

    diagnostics = record.get("step_diagnostics") or []
    chosen = [selected_candidate_xy(step) for step in diagnostics]
    switch_distances = [
        float(np.linalg.norm(second - first))
        for first, second in zip(chosen, chosen[1:])
        if first is not None and second is not None
    ]
    switch_count = sum(distance > switch_distance_m for distance in switch_distances)
    margins = [margin for step in diagnostics if (margin := selector_margin(step)) is not None]

    optimal = float(benchmark["optimal_length"])
    path_length = float(benchmark["path_length"])
    path_ratio = path_length / max(optimal, 1e-9)
    final_distance = float(distances[-1])
    min_distance = float(distances.min())
    departure_m = (
        max(0.0, final_distance - success_distance_m)
        if first_hit is not None and not success
        else 0.0
    )

    return {
        "episode_id": record["episode_id"],
        "instruction": record.get("instruction"),
        "category": category,
        "success": success,
        "oracle_success": oracle_success,
        "spl": spl,
        "path_length_m": path_length,
        "optimal_length_m": optimal,
        "path_ratio": path_ratio,
        "min_goal_distance_m": min_distance,
        "final_goal_distance_m": final_distance,
        "first_success_step": first_hit,
        "last_success_step": last_hit,
        "left_success_region_after_first_hit": leaves_after_first_hit,
        "departure_beyond_success_radius_m": departure_m,
        "candidate_switch_count": int(switch_count),
        "mean_candidate_switch_distance_m": (
            float(np.mean(switch_distances)) if switch_distances else None
        ),
        "max_candidate_switch_distance_m": (
            float(np.max(switch_distances)) if switch_distances else None
        ),
        "mean_selector_margin": float(np.mean(margins)) if margins else None,
        "min_selector_margin": float(np.min(margins)) if margins else None,
        "steps_with_diagnostics": len(diagnostics),
        "termination": record.get("termination"),
    }


def finite_mean(values: list[float | None]) -> float | None:
    values = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(values)) if values else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--success-distance-m", type=float, default=20.0)
    parser.add_argument(
        "--low-spl-threshold",
        type=float,
        default=0.5,
        help="successful episodes below this per-episode SPL are category B",
    )
    parser.add_argument(
        "--switch-distance-m",
        type=float,
        default=20.0,
        help="count consecutive selected waypoints farther apart than this as a candidate switch",
    )
    parser.add_argument("--top-cases", type=int, default=50)
    args = parser.parse_args()

    if args.output_dir.exists():
        parser.error("output directory already exists")
    if args.success_distance_m <= 0 or args.switch_distance_m < 0:
        parser.error("distance thresholds must be positive/non-negative")
    if not 0 <= args.low_spl_threshold <= 1:
        parser.error("--low-spl-threshold must be in [0, 1]")
    args.output_dir.mkdir(parents=True)

    analyses = []
    with args.predictions.open() as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON on line {line_number}") from error
            analyses.append(
                analyze_record(
                    record,
                    success_distance_m=args.success_distance_m,
                    low_spl_threshold=args.low_spl_threshold,
                    switch_distance_m=args.switch_distance_m,
                )
            )
    if not analyses:
        raise ValueError("prediction file is empty")

    categories: dict[str, list[dict[str, Any]]] = {}
    for item in analyses:
        categories.setdefault(item["category"], []).append(item)

    summary_categories = {}
    for name, items in sorted(categories.items()):
        summary_categories[name] = {
            "episodes": len(items),
            "fraction": len(items) / len(analyses),
            "mean_spl": finite_mean([item["spl"] for item in items]),
            "mean_path_ratio": finite_mean([item["path_ratio"] for item in items]),
            "mean_min_goal_distance_m": finite_mean(
                [item["min_goal_distance_m"] for item in items]
            ),
            "mean_final_goal_distance_m": finite_mean(
                [item["final_goal_distance_m"] for item in items]
            ),
            "left_success_region_fraction": (
                sum(item["left_success_region_after_first_hit"] for item in items) / len(items)
            ),
            "mean_candidate_switch_count": finite_mean(
                [item["candidate_switch_count"] for item in items]
            ),
            "mean_selector_margin": finite_mean(
                [item["mean_selector_margin"] for item in items]
            ),
        }

    oracle_only = categories.get("C_oracle_only_final_failure", [])
    summary = {
        "predictions": str(args.predictions.resolve()),
        "episodes": len(analyses),
        "success_distance_m": args.success_distance_m,
        "low_spl_threshold": args.low_spl_threshold,
        "switch_distance_m": args.switch_distance_m,
        "step_diagnostics_available": any(item["steps_with_diagnostics"] for item in analyses),
        "categories": summary_categories,
        "oracle_only_diagnostics": {
            "episodes": len(oracle_only),
            "left_success_region_after_first_hit": sum(
                item["left_success_region_after_first_hit"] for item in oracle_only
            ),
            "mean_departure_beyond_success_radius_m": finite_mean(
                [item["departure_beyond_success_radius_m"] for item in oracle_only]
            ),
            "mean_first_success_step": finite_mean(
                [item["first_success_step"] for item in oracle_only]
            ),
            "mean_candidate_switch_count": finite_mean(
                [item["candidate_switch_count"] for item in oracle_only]
            ),
        },
    }

    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    with (args.output_dir / "cases.jsonl").open("w") as stream:
        for item in analyses:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")

    ranked = sorted(
        analyses,
        key=lambda item: (
            item["category"] != "C_oracle_only_final_failure",
            -item["departure_beyond_success_radius_m"],
            -item["candidate_switch_count"],
            -item["path_ratio"],
        ),
    )[: args.top_cases]
    (args.output_dir / "priority_cases.json").write_text(
        json.dumps(ranked, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
