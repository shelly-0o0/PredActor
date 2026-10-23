#!/usr/bin/env python3
"""Score fixed-command rollout archives for one policy checkpoint."""

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

from dagger.motionclip_retrieval import encode_and_retrieve
from dagger.policy_benchmark import (
    PolicyBenchmarkSuite,
    archive_command,
    first_complete_motion_window,
    navigation_distinguishability,
    navigation_metrics,
    rollout_path,
    rollout_survival,
    scalar_summary,
)


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _latency(archive) -> dict:
    values = (
        np.asarray(archive["inference_time_ms"], dtype=np.float64)
        if "inference_time_ms" in archive else np.empty((0,))
    )
    values = values[np.isfinite(values)]
    return {
        "mean_ms": float(np.mean(values)) if len(values) else None,
        "p95_ms": float(np.percentile(values, 95)) if len(values) else None,
        "samples": int(len(values)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True)
    parser.add_argument("--rollout-dir", required=True)
    parser.add_argument("--motionclip-checkpoint")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--checkpoint", help="Policy checkpoint label stored in the report")
    parser.add_argument("--output", required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    suite = PolicyBenchmarkSuite.load(args.suite)
    rollout_root = Path(args.rollout_dir).expanduser().resolve()
    requested = len(suite.all_commands) * len(suite.seeds)
    missing: list[str] = []
    records: dict[tuple[str, int], dict] = {}
    archives = {}

    for case in suite.all_commands:
        for seed in suite.seeds:
            path = rollout_path(rollout_root, case, seed)
            if not path.exists():
                missing.append(str(path))
                continue
            with np.load(path, allow_pickle=False) as loaded:
                archive = {key: loaded[key] for key in loaded.files}
            actual_command = archive_command(archive)
            if actual_command != case.prompt:
                raise ValueError(
                    f"{path}: stored command {actual_command!r} != {case.prompt!r}"
                )
            survival = rollout_survival(archive, suite.duration_seconds)
            records[(case.id, seed)] = {
                "command_id": case.id,
                "prompt": case.prompt,
                "seed": seed,
                "path": str(path),
                **survival,
                "latency": _latency(archive),
            }
            archives[(case.id, seed)] = archive

    if args.require_complete and missing:
        raise FileNotFoundError(
            f"benchmark is incomplete: {len(missing)}/{requested} rollout cells missing; "
            f"first missing: {missing[0]}"
        )

    text_ids = {case.id for case in suite.text_commands}
    text_records = [
        record for key, record in records.items() if key[0] in text_ids
    ]
    retrieval_keys = []
    retrieval_windows = []
    for case in suite.text_commands:
        for seed in suite.seeds:
            key = (case.id, seed)
            archive = archives.get(key)
            if archive is None:
                continue
            window = first_complete_motion_window(
                archive,
                activation_delay_seconds=suite.activation_delay_seconds,
                window_frames=suite.window_frames,
                target_fps=suite.target_fps,
            )
            if window is not None:
                retrieval_keys.append(key)
                retrieval_windows.append(window)

    retrieval = {
        "available": False,
        "reason": "--motionclip-checkpoint was not provided",
        "eligible": len(retrieval_windows),
        "requested": len(suite.text_commands) * len(suite.seeds),
    }
    if args.motionclip_checkpoint:
        prompts = [case.prompt for case in suite.text_commands]
        result = encode_and_retrieve(
            retrieval_windows,
            prompts,
            checkpoint=Path(args.motionclip_checkpoint).expanduser().resolve(),
            device=args.device,
        )
        prompt_index = {case.id: index for index, case in enumerate(suite.text_commands)}
        correct = 0
        per_command_correct = {case.id: [] for case in suite.text_commands}
        for row, key in enumerate(retrieval_keys):
            predicted_index = int(result["predicted_indices"][row])
            expected_index = prompt_index[key[0]]
            is_correct = predicted_index == expected_index
            correct += int(is_correct)
            per_command_correct[key[0]].append(float(is_correct))
            records[key]["retrieval"] = {
                "eligible": True,
                "correct": is_correct,
                "predicted_prompt": prompts[predicted_index],
                "target_score": float(result["scores"][row, expected_index]),
                "top1_score": float(result["scores"][row, predicted_index]),
            }
        for key in records:
            if key[0] in text_ids and "retrieval" not in records[key]:
                records[key]["retrieval"] = {"eligible": False}
        retrieval = {
            "available": True,
            "accuracy": (
                float(correct / len(retrieval_keys)) if retrieval_keys else None
            ),
            "correct": correct,
            "eligible": len(retrieval_keys),
            "requested": len(suite.text_commands) * len(suite.seeds),
            "per_command": {
                case.id: scalar_summary(per_command_correct[case.id])
                for case in suite.text_commands
            },
        }

    navigation_records = []
    navigation_rows = []
    for case in suite.navigation_commands:
        for seed in suite.seeds:
            key = (case.id, seed)
            archive = archives.get(key)
            if archive is None:
                continue
            metrics = navigation_metrics(archive)
            navigation_records.append((case.prompt, metrics))
            row = metrics.to_dict()
            row.update({"command_id": case.id, "prompt": case.prompt, "seed": seed})
            navigation_rows.append(row)

    nav_keys = [
        "survival_rate", "cumulative_abs_delta_yaw_rad", "net_delta_yaw_rad",
        "abs_lateral_displacement_m", "forward_distance_m",
        "mean_abs_foot_yaw_difference_rad", "duration_s",
    ]
    navigation_summary = {
        key: scalar_summary(row[key] for row in navigation_rows)
        for key in nav_keys
    }
    straight_rows = [row for row in navigation_rows if row["command_id"] == "walk_straight"]
    navigation_summary["walk_straight"] = {
        key: scalar_summary(row[key] for row in straight_rows)
        for key in nav_keys
    }
    navigation_summary["command_distinguishability"] = (
        navigation_distinguishability(navigation_records)
    )

    per_text_command = {}
    for case in suite.text_commands:
        selected = [record for record in text_records if record["command_id"] == case.id]
        per_text_command[case.id] = {
            "prompt": case.prompt,
            "cells": len(selected),
            "survival": scalar_summary(row["survival_fraction"] for row in selected),
            "completion_rate": scalar_summary(
                float(row["completed_horizon"]) for row in selected
            ),
        }

    report = {
        "schema": "predactor-policy-benchmark-report-v1",
        "policy_checkpoint": args.checkpoint,
        "suite": {
            "name": suite.name,
            "path": str(Path(args.suite).expanduser().resolve()),
            "seeds": list(suite.seeds),
            "duration_seconds": suite.duration_seconds,
            "activation_delay_seconds": suite.activation_delay_seconds,
            "window_frames": suite.window_frames,
            "target_fps": suite.target_fps,
        },
        "coverage": {
            "all_cells": {"evaluated": len(records), "requested": requested},
            "missing": missing,
        },
        "text_command": {
            "survival": scalar_summary(
                record["survival_fraction"] for record in text_records
            ),
            "completion_rate": scalar_summary(
                float(record["completed_horizon"]) for record in text_records
            ),
            "retrieval": retrieval,
            "per_command": per_text_command,
        },
        "navigation": {
            "cells": len(navigation_rows),
            "summary": navigation_summary,
            "rollouts": navigation_rows,
        },
        "runtime": {
            "rollout_inference_mean_ms": scalar_summary(
                record["latency"]["mean_ms"]
                for record in records.values()
                if record["latency"]["mean_ms"] is not None
            ),
            "note": (
                "Rollout timing is diagnostic only; it is not the paper's synchronized "
                "RTX 4090 B=5 actor-latency protocol."
            ),
        },
        "rollouts": list(records.values()),
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(_json_safe(report), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(_json_safe({
        "output": str(output),
        "coverage": report["coverage"]["all_cells"],
        "text_survival": report["text_command"]["survival"]["mean"],
        "text_retrieval": retrieval.get("accuracy"),
        "retrieval_eligible": f"{retrieval['eligible']}/{retrieval['requested']}",
    }), indent=2))


if __name__ == "__main__":
    main()
