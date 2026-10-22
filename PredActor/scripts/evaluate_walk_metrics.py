"""Evaluate saved pi0/pi1 walk rollouts with the same acceptance metrics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dagger.metrics import (
    WalkMetrics,
    command_distinguishability,
    straight_walk_cost,
    straight_walk_metrics,
)
from dagger.text_commands import canonicalize_command


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def evaluate(path: Path, dt: float) -> tuple[dict, str | None]:
    archive = np.load(path, allow_pickle=False)
    file_dt = float(np.asarray(archive["dt"])) if "dt" in archive else dt
    kwargs = {}
    for key in ("terminated", "left_foot_rot", "right_foot_rot"):
        if key in archive:
            kwargs[key] = archive[key]
    metrics = straight_walk_metrics(
        archive["root_pos"], archive["root_rot"], dt=file_dt, **kwargs
    )
    command = None
    if "command" in archive:
        values = np.asarray(archive["command"]).reshape(-1)
        if len(values):
            command = str(values[0])
    return {
        **metrics.to_dict(),
        "abs_net_delta_yaw_rad": abs(metrics.net_delta_yaw_rad),
        "cost": straight_walk_cost(metrics),
    }, command


def summarize(records: list[dict]) -> dict:
    if not records:
        return {"count": 0}
    numeric_keys = [
        key for key, value in records[0].items()
        if key != "command" and isinstance(value, (int, float))
    ]
    summary = {"count": len(records)}
    for key in numeric_keys:
        values = np.asarray([record[key] for record in records], dtype=np.float64)
        finite = values[np.isfinite(values)]
        summary[key] = None if not len(finite) else {
            "mean": float(np.mean(finite)),
            "std": float(np.std(finite)),
            "min": float(np.min(finite)),
            "max": float(np.max(finite)),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("rollouts", nargs="+")
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--output")
    parser.add_argument("--yaw-threshold-rad", type=float, default=0.15)
    parser.add_argument(
        "--right-positive-yaw",
        action="store_true",
        help="Use this if the simulator yaw convention is positive to the right",
    )
    args = parser.parse_args()
    results = {}
    labeled_metrics = []
    records_by_command = {}
    for value in args.rollouts:
        path = Path(value).expanduser().resolve()
        metrics, command = evaluate(path, args.dt)
        if command is not None:
            metrics["command"] = command
            canonical = canonicalize_command(command)
            records_by_command.setdefault(canonical, []).append(metrics)
            labeled_metrics.append((command, WalkMetrics(**{
                key: metrics[key]
                for key in WalkMetrics.__dataclass_fields__
            })))
        results[str(path)] = metrics

    straight_records = [
        record
        for command in ("walk", "walk straight")
        for record in records_by_command.get(command, [])
    ]
    report = {
        "rollouts": results,
        "summary": {
            "all": summarize(list(results.values())),
            "straight": summarize(straight_records),
            "by_command": {
                command: summarize(records)
                for command, records in sorted(records_by_command.items())
            },
        },
    }
    if labeled_metrics:
        try:
            report["command_distinguishability"] = {
                "available": True,
                **command_distinguishability(
                    labeled_metrics,
                    yaw_threshold_rad=args.yaw_threshold_rad,
                    left_positive_yaw=not args.right_positive_yaw,
                ).to_dict(),
            }
        except ValueError as error:
            report["command_distinguishability"] = {
                "available": False,
                "reason": str(error),
            }
    text = json.dumps(_json_safe(report), indent=2, allow_nan=False) + "\n"
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
