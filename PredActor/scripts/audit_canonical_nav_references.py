#!/usr/bin/env python3
"""Fail-closed audit for generated canonical navigation references."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dagger.canonical_reference import (
    CANONICAL_NAVIGATION_SPECS,
    REFERENCE_SCHEMA,
    reference_path_metrics,
    sha256_file,
)


REQUIRED_FIELDS = (
    "fps", "joint_pos", "joint_vel", "body_pos_w", "body_quat_w",
    "body_lin_vel_w", "body_ang_vel_w",
)


def _max_velocity_error(value: np.ndarray, velocity: np.ndarray, dt: float) -> float:
    expected = np.gradient(value.astype(np.float64), dt, axis=0, edge_order=2)
    return float(np.max(np.abs(expected - velocity.astype(np.float64))))


def audit(manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != REFERENCE_SCHEMA:
        raise ValueError(f"unsupported manifest schema: {manifest.get('schema')!r}")
    entries = manifest.get("commands") or []
    if [entry.get("command") for entry in entries] != [
        spec.command for spec in CANONICAL_NAVIGATION_SPECS
    ]:
        raise ValueError("manifest command order is not canonical")

    reports = []
    archives = []
    expected_frames = int(manifest["frames"])
    expected_fps = float(manifest["target_fps"])
    for spec, entry in zip(CANONICAL_NAVIGATION_SPECS, entries):
        path = Path(entry["motion"]).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        if sha256_file(path) != entry["sha256"]:
            raise ValueError(f"motion hash changed after generation: {path}")
        with np.load(path, allow_pickle=False) as archive:
            missing = set(REQUIRED_FIELDS) - set(archive.files)
            if missing:
                raise KeyError(f"{path}: missing fields {sorted(missing)}")
            values = {key: np.asarray(archive[key]) for key in REQUIRED_FIELDS}
        fps = float(values["fps"].reshape(-1)[0])
        if fps != expected_fps:
            raise ValueError(f"{path}: fps={fps}, expected {expected_fps}")
        shapes = {
            "joint_pos": (expected_frames, 29),
            "joint_vel": (expected_frames, 29),
            "body_pos_w": (expected_frames, 30, 3),
            "body_quat_w": (expected_frames, 30, 4),
            "body_lin_vel_w": (expected_frames, 30, 3),
            "body_ang_vel_w": (expected_frames, 30, 3),
        }
        for key, shape in shapes.items():
            if values[key].shape != shape:
                raise ValueError(f"{path}: {key} shape={values[key].shape}, expected={shape}")
            if not np.isfinite(values[key]).all():
                raise ValueError(f"{path}: {key} contains NaN/Inf")
        quaternion_norm_error = float(np.max(np.abs(
            np.linalg.norm(values["body_quat_w"], axis=-1) - 1.0
        )))
        if quaternion_norm_error > 2.0e-4:
            raise ValueError(f"{path}: quaternion norm error {quaternion_norm_error}")
        dt = 1.0 / fps
        joint_velocity_error = _max_velocity_error(
            values["joint_pos"], values["joint_vel"], dt
        )
        body_velocity_error = _max_velocity_error(
            values["body_pos_w"], values["body_lin_vel_w"], dt
        )
        if joint_velocity_error > 2.0e-4 or body_velocity_error > 2.0e-4:
            raise ValueError(
                f"{path}: inconsistent finite differences: joint={joint_velocity_error}, "
                f"body={body_velocity_error}"
            )
        metrics = reference_path_metrics(
            values["body_pos_w"], values["body_quat_w"], fps
        )
        distance = float(manifest["distance_m"])
        turn = float(entry["turn_angle_rad"])
        if spec.turn_sign == 0:
            if metrics["forward_displacement_m"] < 0.95 * distance:
                raise ValueError(f"straight reference does not advance: {metrics}")
            if abs(metrics["lateral_displacement_m"]) > 0.03:
                raise ValueError(f"straight reference has lateral drift: {metrics}")
            if abs(metrics["net_yaw_rad"]) > 0.03:
                raise ValueError(f"straight reference has net yaw: {metrics}")
            if metrics["path_efficiency"] < 0.999:
                raise ValueError(f"straight root path is not linear: {metrics}")
        else:
            if np.sign(metrics["net_yaw_rad"]) != spec.turn_sign:
                raise ValueError(f"turn direction contradicts command: {metrics}")
            if abs(metrics["net_yaw_rad"] - turn) > 0.03:
                raise ValueError(f"turn magnitude differs from manifest: {metrics}")
            if metrics["path_length_m"] < 0.98 * distance:
                raise ValueError(f"turning arc is too short: {metrics}")
        reports.append({
            "motion_idx": spec.motion_idx,
            "command": spec.command,
            "motion": str(path),
            "metrics": metrics,
            "quaternion_norm_error_max": quaternion_norm_error,
            "joint_velocity_error_max": joint_velocity_error,
            "body_velocity_error_max": body_velocity_error,
        })
        archives.append(values)

    straight, left, right = archives
    for key in ("joint_pos", "joint_vel"):
        if not np.array_equal(straight[key], left[key]) or not np.array_equal(
            straight[key], right[key]
        ):
            raise ValueError(f"{key} differs across canonical commands")
    left_xy = left["body_pos_w"][:, 0, :2]
    right_xy = right["body_pos_w"][:, 0, :2]
    symmetry_error = float(max(
        np.max(np.abs(left_xy[:, 0] - right_xy[:, 0])),
        np.max(np.abs(left_xy[:, 1] + right_xy[:, 1])),
    ))
    if symmetry_error > 2.0e-4:
        raise ValueError(f"left/right root paths are not symmetric: {symmetry_error}")
    return {
        "schema": "predactor-canonical-reference-audit-v1",
        "status": "passed",
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "left_right_root_symmetry_error_max": symmetry_error,
        "commands": reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = audit(Path(args.manifest).expanduser().resolve())
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
