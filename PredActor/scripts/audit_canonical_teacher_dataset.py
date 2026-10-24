#!/usr/bin/env python3
"""Audit RLTracker rollouts collected from canonical navigation references."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import zarr

from dagger.canonical_dataset import TEACHER_AUDIT_SCHEMA
from dagger.canonical_reference import (
    REFERENCE_SCHEMA,
    quaternion_yaw_wxyz,
    reference_path_metrics,
    sha256_file,
)


def _angle_difference(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(left - right), np.cos(left - right))


def _episode_metrics(data, start: int, end: int, fps: float) -> dict:
    root_pos = np.asarray(data["root_pos"][start:end], dtype=np.float64)
    root_rot = np.asarray(data["root_rot"][start:end], dtype=np.float64)
    actual = reference_path_metrics(
        root_pos[:, None, :], root_rot[:, None, :], fps
    )
    result = {"actual": actual}
    if "motion_ref_root_pos" in data and "motion_ref_root_rot" in data:
        ref_pos = np.asarray(
            data["motion_ref_root_pos"][start:end], dtype=np.float64
        )
        ref_rot = np.asarray(
            data["motion_ref_root_rot"][start:end], dtype=np.float64
        )
        reference = reference_path_metrics(
            ref_pos[:, None, :], ref_rot[:, None, :], fps
        )
        actual_rel = root_pos - root_pos[0]
        reference_rel = ref_pos - ref_pos[0]
        position_error = actual_rel - reference_rel
        actual_yaw = quaternion_yaw_wxyz(root_rot)
        reference_yaw = quaternion_yaw_wxyz(ref_rot)
        yaw_error = _angle_difference(
            actual_yaw - actual_yaw[0],
            reference_yaw - reference_yaw[0],
        )
        result.update({
            "reference": reference,
            "root_position_tracking_rmse_m": float(np.sqrt(np.mean(
                np.sum(position_error**2, axis=1)
            ))),
            "root_yaw_tracking_rmse_rad": float(np.sqrt(np.mean(yaw_error**2))),
        })
    action = np.asarray(data["act"][start:end], dtype=np.float64)
    result["max_abs_action"] = float(np.max(np.abs(action)))
    result["finite"] = bool(
        np.isfinite(root_pos).all()
        and np.isfinite(root_rot).all()
        and np.isfinite(action).all()
    )
    return result


def audit(
    *,
    zarr_path: Path,
    reference_manifest: Path,
    expected_samples_per_motion: int,
    min_frames: int,
    min_pass_rate: float,
    min_passed_episodes_per_command: int,
    max_position_rmse_m: float,
    max_yaw_rmse_rad: float,
    max_action_abs: float,
) -> dict:
    manifest = json.loads(reference_manifest.read_text(encoding="utf-8"))
    if manifest.get("schema") != REFERENCE_SCHEMA:
        raise ValueError("unsupported reference manifest schema")
    command_by_motion = {
        int(entry["motion_idx"]): str(entry["command"])
        for entry in manifest["commands"]
    }
    root = zarr.open(str(zarr_path), mode="r")
    if "data" not in root or "meta" not in root:
        raise ValueError("teacher zarr must contain data/ and meta/")
    data = root["data"]
    required = {"root_pos", "root_rot", "act", "motion_idx"}
    missing = required - set(data.keys())
    if missing:
        raise KeyError(f"teacher zarr lacks fields: {sorted(missing)}")
    ends = np.asarray(root["meta/episode_ends"][:], dtype=np.int64)
    starts = np.r_[0, ends[:-1]]
    motion_idx = np.asarray(data["motion_idx"][:]).reshape(-1)
    expected_episodes = expected_samples_per_motion * len(command_by_motion)
    if not 1 <= min_passed_episodes_per_command <= expected_samples_per_motion:
        raise ValueError(
            "min_passed_episodes_per_command must be within collected samples"
        )
    if len(ends) != expected_episodes:
        raise ValueError(
            f"teacher zarr has {len(ends)} episodes, expected {expected_episodes}"
        )
    fps = float(manifest["target_fps"])
    episodes = []
    per_command_passes = {command: [] for command in command_by_motion.values()}
    per_command_counts = {command: 0 for command in command_by_motion.values()}
    for episode, (start, end) in enumerate(zip(starts, ends)):
        ids = np.unique(np.rint(motion_idx[start:end]).astype(np.int64))
        if len(ids) != 1 or int(ids[0]) not in command_by_motion:
            raise ValueError(f"episode {episode} has invalid motion IDs {ids.tolist()}")
        motion_id = int(ids[0])
        command = command_by_motion[motion_id]
        per_command_counts[command] += 1
        metrics = _episode_metrics(data, int(start), int(end), fps)
        actual = metrics["actual"]
        reasons = []
        if end - start < min_frames:
            reasons.append(f"frames<{min_frames}")
        if not metrics["finite"]:
            reasons.append("non_finite")
        if metrics["max_abs_action"] > max_action_abs:
            reasons.append(f"max_abs_action>{max_action_abs}")
        if metrics.get("root_position_tracking_rmse_m", np.inf) > max_position_rmse_m:
            reasons.append(f"position_rmse>{max_position_rmse_m}")
        if metrics.get("root_yaw_tracking_rmse_rad", np.inf) > max_yaw_rmse_rad:
            reasons.append(f"yaw_rmse>{max_yaw_rmse_rad}")
        if command == "walk straight":
            if actual["forward_displacement_m"] < 4.0:
                reasons.append("straight_forward<4m")
            if abs(actual["lateral_displacement_m"]) > 0.75:
                reasons.append("straight_abs_lateral>0.75m")
            if abs(actual["net_yaw_rad"]) > 0.35:
                reasons.append("straight_abs_yaw>0.35rad")
        elif command == "turn left while walking":
            if actual["forward_displacement_m"] < 3.0:
                reasons.append("left_forward<3m")
            if actual["net_yaw_rad"] < 0.5:
                reasons.append("left_net_yaw<0.5rad")
            if actual["lateral_displacement_m"] < 1.0:
                reasons.append("left_lateral<1m")
        elif command == "turn right while walking":
            if actual["forward_displacement_m"] < 3.0:
                reasons.append("right_forward<3m")
            if actual["net_yaw_rad"] > -0.5:
                reasons.append("right_net_yaw>-0.5rad")
            if actual["lateral_displacement_m"] > -1.0:
                reasons.append("right_lateral>-1m")
        passed = not reasons
        per_command_passes[command].append(passed)
        episodes.append({
            "episode": episode,
            "motion_idx": motion_id,
            "command": command,
            "start": int(start),
            "end": int(end),
            "frames": int(end - start),
            "passed": passed,
            "failure_reasons": reasons,
            **metrics,
        })
    if any(count != expected_samples_per_motion for count in per_command_counts.values()):
        raise ValueError(f"unbalanced teacher episodes: {per_command_counts}")
    pass_rates = {
        command: float(np.mean(values))
        for command, values in per_command_passes.items()
    }
    pass_counts = {
        command: int(np.sum(values))
        for command, values in per_command_passes.items()
    }
    status = "passed" if all(
        pass_rates[command] >= min_pass_rate
        and pass_counts[command] >= min_passed_episodes_per_command
        for command in pass_rates
    ) else "failed"
    return {
        "schema": TEACHER_AUDIT_SCHEMA,
        "status": status,
        "zarr": str(zarr_path.resolve()),
        "reference_manifest": str(reference_manifest.resolve()),
        "reference_manifest_sha256": sha256_file(reference_manifest),
        "expected_samples_per_motion": expected_samples_per_motion,
        "episodes": len(episodes),
        "frames": int(ends[-1]),
        "pass_rate_by_command": pass_rates,
        "passed_count_by_command": pass_counts,
        "count_by_command": per_command_counts,
        "thresholds": {
            "min_frames": min_frames,
            "min_pass_rate": min_pass_rate,
            "min_passed_episodes_per_command": min_passed_episodes_per_command,
            "max_position_rmse_m": max_position_rmse_m,
            "max_yaw_rmse_rad": max_yaw_rmse_rad,
            "max_action_abs": max_action_abs,
        },
        "episode_reports": episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarr", required=True)
    parser.add_argument("--reference-manifest", required=True)
    parser.add_argument("--expected-samples-per-motion", type=int, required=True)
    parser.add_argument("--min-frames", type=int, default=450)
    parser.add_argument("--min-pass-rate", type=float, default=0.9)
    parser.add_argument("--min-passed-episodes-per-command", type=int, default=1)
    parser.add_argument("--max-position-rmse-m", type=float, default=0.75)
    parser.add_argument("--max-yaw-rmse-rad", type=float, default=0.5)
    parser.add_argument("--max-action-abs", type=float, default=10.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = audit(
        zarr_path=Path(args.zarr).expanduser().resolve(),
        reference_manifest=Path(args.reference_manifest).expanduser().resolve(),
        expected_samples_per_motion=args.expected_samples_per_motion,
        min_frames=args.min_frames,
        min_pass_rate=args.min_pass_rate,
        min_passed_episodes_per_command=args.min_passed_episodes_per_command,
        max_position_rmse_m=args.max_position_rmse_m,
        max_yaw_rmse_rad=args.max_yaw_rmse_rad,
        max_action_abs=args.max_action_abs,
    )
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "output": str(output),
        "pass_rate_by_command": report["pass_rate_by_command"],
    }, indent=2))
    if report["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
