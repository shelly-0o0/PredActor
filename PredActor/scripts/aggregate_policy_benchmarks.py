#!/usr/bin/env python3
"""Aggregate independent checkpoint means using sample standard deviation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


METRICS = {
    "text_survival": ("text_command", "survival", "mean"),
    "text_retrieval": ("text_command", "retrieval", "accuracy"),
    "straight_abs_lateral_m": (
        "navigation", "summary", "walk_straight",
        "abs_lateral_displacement_m", "mean",
    ),
    "straight_forward_m": (
        "navigation", "summary", "walk_straight", "forward_distance_m", "mean",
    ),
    "straight_abs_yaw_rad": (
        "navigation", "summary", "walk_straight",
        "cumulative_abs_delta_yaw_rad", "mean",
    ),
    "command_balanced_accuracy": (
        "navigation", "summary", "command_distinguishability", "balanced_accuracy",
    ),
}


def _get(value, path):
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    reports = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.reports]
    suite_names = {report["suite"]["name"] for report in reports}
    if len(suite_names) != 1:
        raise ValueError(f"reports use different suites: {sorted(suite_names)}")
    metrics = {}
    for name, path in METRICS.items():
        values = [_get(report, path) for report in reports]
        values = [float(value) for value in values if value is not None]
        metrics[name] = {
            "checkpoint_means": values,
            "mean": float(np.mean(values)) if values else None,
            "sample_sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
            "checkpoints": len(values),
        }
    result = {
        "schema": "predactor-policy-benchmark-aggregate-v1",
        "suite": next(iter(suite_names)),
        "reports": [str(Path(path).expanduser().resolve()) for path in args.reports],
        "metrics": metrics,
        "uncertainty_unit": "independently trained checkpoint mean",
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
