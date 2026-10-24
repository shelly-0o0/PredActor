"""Build kinematically consistent canonical G1 navigation references.

The released AMASS-derived motions contain useful gait joint trajectories, but
their world-frame root paths are not command canonical.  This module keeps the
retargeted joint motion, assigns an explicit straight/left/right root path, and
recomputes all body transforms from the G1 URDF.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from diffusion_policy.utils.g1_reindex import (
    ISAACLAB_BODY_NAMES,
    ISAACLAB_DOF_NAMES,
)


REFERENCE_SCHEMA = "predactor-canonical-navigation-v1"


@dataclass(frozen=True)
class CanonicalCommandSpec:
    motion_idx: int
    directory: str
    command: str
    turn_sign: int


CANONICAL_NAVIGATION_SPECS = (
    CanonicalCommandSpec(0, "00_walk_straight", "walk straight", 0),
    CanonicalCommandSpec(
        1, "01_turn_left_while_walking", "turn left while walking", 1
    ),
    CanonicalCommandSpec(
        2, "02_turn_right_while_walking", "turn right while walking", -1
    ),
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quaternion_yaw_wxyz(quaternion: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.unwrap(np.arctan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    ))


def reference_path_metrics(
    body_pos_w: np.ndarray,
    body_quat_w: np.ndarray,
    fps: float,
) -> dict[str, float | int]:
    position = np.asarray(body_pos_w, dtype=np.float64)[:, 0]
    yaw = quaternion_yaw_wxyz(
        np.asarray(body_quat_w, dtype=np.float64)[:, 0]
    )
    displacement = position[-1, :2] - position[0, :2]
    chord = float(np.linalg.norm(displacement))
    path_length = float(np.linalg.norm(
        np.diff(position[:, :2], axis=0), axis=1
    ).sum())
    initial_yaw = float(yaw[0])
    forward_axis = np.array([np.cos(initial_yaw), np.sin(initial_yaw)])
    lateral_axis = np.array([-np.sin(initial_yaw), np.cos(initial_yaw)])
    if chord > 1.0e-9:
        path_axis = displacement / chord
        cross_axis = np.array([-path_axis[1], path_axis[0]])
        cross_track = (position[:, :2] - position[0, :2]) @ cross_axis
        travel_yaw = float(np.arctan2(displacement[1], displacement[0]))
        heading_offset = float(np.arctan2(
            np.sin(travel_yaw - initial_yaw),
            np.cos(travel_yaw - initial_yaw),
        ))
    else:
        cross_track = np.zeros(len(position), dtype=np.float64)
        heading_offset = 0.0
    return {
        "frames": int(len(position)),
        "fps": float(fps),
        "duration_s": float((len(position) - 1) / fps),
        "forward_displacement_m": float(displacement @ forward_axis),
        "lateral_displacement_m": float(displacement @ lateral_axis),
        "net_yaw_rad": float(yaw[-1] - yaw[0]),
        "cumulative_abs_delta_yaw_rad": float(np.abs(np.diff(yaw)).sum()),
        "chord_m": chord,
        "path_length_m": path_length,
        "path_efficiency": float(chord / max(path_length, 1.0e-9)),
        "cross_track_rms_m": float(np.sqrt(np.mean(cross_track**2))),
        "cross_track_max_m": float(np.max(np.abs(cross_track))),
        "travel_heading_offset_rad": heading_offset,
    }


def _resample_linear(array: np.ndarray, frames: int) -> np.ndarray:
    value = np.asarray(array, dtype=np.float64)
    old_t = np.linspace(0.0, 1.0, len(value), dtype=np.float64)
    new_t = np.linspace(0.0, 1.0, frames, dtype=np.float64)
    flat = value.reshape(len(value), -1)
    result = np.empty((frames, flat.shape[1]), dtype=np.float64)
    for column in range(flat.shape[1]):
        result[:, column] = np.interp(new_t, old_t, flat[:, column])
    return result.reshape((frames, *value.shape[1:]))


def canonical_root_path(
    *,
    frames: int,
    duration_s: float,
    distance_m: float,
    turn_angle_rad: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return XY root positions and tangent yaw for a line or circular arc."""
    if frames < 2 or duration_s <= 0.0 or distance_m <= 0.0:
        raise ValueError("canonical path needs frames>=2, duration>0, distance>0")
    progress = np.linspace(0.0, 1.0, frames, dtype=np.float64)
    arc_length = distance_m * progress
    yaw = turn_angle_rad * progress
    if abs(turn_angle_rad) < 1.0e-12:
        xy = np.stack([arc_length, np.zeros_like(arc_length)], axis=-1)
    else:
        curvature = turn_angle_rad / distance_m
        radius = 1.0 / curvature
        xy = np.stack([
            radius * np.sin(yaw),
            radius * (1.0 - np.cos(yaw)),
        ], axis=-1)
    return xy, yaw


def _rpy_to_quaternion_wxyz(
    roll: np.ndarray,
    pitch: np.ndarray,
    yaw: np.ndarray,
) -> np.ndarray:
    half_roll = roll * 0.5
    half_pitch = pitch * 0.5
    half_yaw = yaw * 0.5
    cr, sr = np.cos(half_roll), np.sin(half_roll)
    cp, sp = np.cos(half_pitch), np.sin(half_pitch)
    cy, sy = np.cos(half_yaw), np.sin(half_yaw)
    return np.stack([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ], axis=-1)


def _make_quaternion_sign_continuous(quaternion_wxyz: np.ndarray) -> np.ndarray:
    result = np.asarray(quaternion_wxyz, dtype=np.float64).copy()
    for body in range(result.shape[1]):
        for frame in range(1, result.shape[0]):
            if np.dot(result[frame - 1, body], result[frame, body]) < 0.0:
                result[frame, body] *= -1.0
    result /= np.linalg.norm(result, axis=-1, keepdims=True)
    return result


def _finite_difference(value: np.ndarray, dt: float) -> np.ndarray:
    return np.gradient(
        np.asarray(value, dtype=np.float64), dt, axis=0, edge_order=2
    )


def quaternion_angular_velocity_wxyz(
    quaternion_wxyz: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Finite-difference world-frame angular velocity for [T,B,4] wxyz."""
    q = _make_quaternion_sign_continuous(quaternion_wxyz)
    xyzw = q[..., [1, 2, 3, 0]]
    frames, bodies = q.shape[:2]
    output = np.zeros((frames, bodies, 3), dtype=np.float64)
    rotations = [Rotation.from_quat(xyzw[:, body]) for body in range(bodies)]
    for body, rotation in enumerate(rotations):
        output[0, body] = (rotation[1] * rotation[0].inv()).as_rotvec() / dt
        output[-1, body] = (
            rotation[-1] * rotation[-2].inv()
        ).as_rotvec() / dt
        if frames > 2:
            output[1:-1, body] = (
                rotation[2:] * rotation[:-2].inv()
            ).as_rotvec() / (2.0 * dt)
    return output


def _fk_body_transforms(
    *,
    urdf_path: Path,
    joint_pos: np.ndarray,
    root_pos: np.ndarray,
    root_quat_wxyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        import torch
        import pytorch_kinematics as pk
    except ImportError as exc:  # pragma: no cover - dependency preflight
        raise ImportError(
            "canonical reference generation requires torch and pytorch-kinematics"
        ) from exc

    with urdf_path.open("rb") as stream:
        chain = pk.build_chain_from_urdf(stream.read())
    chain = chain.to(device="cpu", dtype=torch.float64)
    chain_joint_names = chain.get_joint_parameter_names()
    missing = sorted(set(chain_joint_names) - set(ISAACLAB_DOF_NAMES))
    if missing:
        raise ValueError(f"URDF contains unmapped actuated joints: {missing}")
    q = torch.from_numpy(np.asarray(joint_pos, dtype=np.float64))
    q_dict = {
        name: q[:, ISAACLAB_DOF_NAMES.index(name)]
        for name in chain_joint_names
    }
    with torch.no_grad():
        transforms = chain.forward_kinematics(q_dict)

    base_matrices = []
    for body_name in ISAACLAB_BODY_NAMES:
        if body_name not in transforms:
            raise KeyError(f"URDF FK lacks required body {body_name!r}")
        base_matrices.append(
            transforms[body_name].get_matrix().detach().cpu().numpy()
        )
    base = np.stack(base_matrices, axis=1)
    base_position = base[..., :3, 3]
    base_rotation = base[..., :3, :3]

    root_xyzw = np.asarray(root_quat_wxyz, dtype=np.float64)[..., [1, 2, 3, 0]]
    root_rotation = Rotation.from_quat(root_xyzw).as_matrix()
    body_pos_w = (
        np.einsum("tij,tbj->tbi", root_rotation, base_position)
        + root_pos[:, None, :]
    )
    body_rotation_w = np.einsum(
        "tij,tbjk->tbik", root_rotation, base_rotation
    )
    body_xyzw = Rotation.from_matrix(
        body_rotation_w.reshape(-1, 3, 3)
    ).as_quat().reshape(len(joint_pos), len(ISAACLAB_BODY_NAMES), 4)
    body_quat_w = body_xyzw[..., [3, 0, 1, 2]]
    return body_pos_w, _make_quaternion_sign_continuous(body_quat_w)


def build_canonical_reference_set(
    *,
    source_motion: str | Path,
    urdf_path: str | Path,
    output_dir: str | Path,
    duration_s: float = 10.0,
    target_fps: float = 50.0,
    turn_angle_rad: float = 1.0,
    distance_m: float | None = None,
    source_start: int = 0,
    source_end: int | None = None,
) -> dict:
    source_path = Path(source_motion).expanduser().resolve()
    urdf = Path(urdf_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    with np.load(source_path, allow_pickle=False) as archive:
        required = {
            "joint_pos", "joint_vel", "body_pos_w", "body_quat_w",
            "body_lin_vel_w", "body_ang_vel_w", "fps",
        }
        missing = required - set(archive.files)
        if missing:
            raise KeyError(f"source motion lacks fields: {sorted(missing)}")
        source = {key: np.asarray(archive[key]) for key in required}
    original_frames = len(source["joint_pos"])
    end = original_frames if source_end is None else int(source_end)
    start = int(source_start)
    if not 0 <= start < end <= original_frames or end - start < 3:
        raise ValueError(
            f"invalid source slice [{start}:{end}] for {original_frames} frames"
        )
    joint_pos_source = np.asarray(source["joint_pos"][start:end], dtype=np.float64)
    root_pos_source = np.asarray(source["body_pos_w"][start:end, 0], dtype=np.float64)
    root_quat_source = np.asarray(
        source["body_quat_w"][start:end, 0], dtype=np.float64
    )
    source_fps = float(np.asarray(source["fps"]).reshape(-1)[0])
    if source_fps <= 0.0 or target_fps <= 0.0 or duration_s <= 0.0:
        raise ValueError("fps and duration must be positive")

    source_metrics = reference_path_metrics(
        source["body_pos_w"][start:end],
        source["body_quat_w"][start:end],
        source_fps,
    )
    target_distance = (
        float(distance_m) if distance_m is not None
        else float(source_metrics["chord_m"])
    )
    if target_distance <= 0.0:
        raise ValueError("target distance must be positive")
    frames = int(round(duration_s * target_fps)) + 1
    joint_pos = _resample_linear(joint_pos_source, frames)

    source_xyzw = root_quat_source[..., [1, 2, 3, 0]]
    source_rpy = Rotation.from_quat(source_xyzw).as_euler("xyz")
    source_rpy = np.unwrap(source_rpy, axis=0)
    roll_pitch = _resample_linear(source_rpy[:, :2], frames)
    root_z = _resample_linear(root_pos_source[:, 2:3], frames)[:, 0]
    dt = 1.0 / target_fps
    joint_vel = _finite_difference(joint_pos, dt)

    commands = []
    for spec in CANONICAL_NAVIGATION_SPECS:
        signed_turn = float(spec.turn_sign) * float(turn_angle_rad)
        xy, yaw = canonical_root_path(
            frames=frames,
            duration_s=duration_s,
            distance_m=target_distance,
            turn_angle_rad=signed_turn,
        )
        root_pos = np.column_stack([xy, root_z])
        root_quat = _rpy_to_quaternion_wxyz(
            roll_pitch[:, 0], roll_pitch[:, 1], yaw
        )
        body_pos_w, body_quat_w = _fk_body_transforms(
            urdf_path=urdf,
            joint_pos=joint_pos,
            root_pos=root_pos,
            root_quat_wxyz=root_quat,
        )
        body_lin_vel_w = _finite_difference(body_pos_w, dt)
        body_ang_vel_w = quaternion_angular_velocity_wxyz(body_quat_w, dt)
        motion_dir = output / spec.directory
        motion_dir.mkdir(parents=True, exist_ok=False)
        motion_path = motion_dir / "motion.npz"
        np.savez_compressed(
            motion_path,
            fps=np.asarray([target_fps], dtype=np.float32),
            joint_pos=joint_pos.astype(np.float32),
            joint_vel=joint_vel.astype(np.float32),
            body_pos_w=body_pos_w.astype(np.float32),
            body_quat_w=body_quat_w.astype(np.float32),
            body_lin_vel_w=body_lin_vel_w.astype(np.float32),
            body_ang_vel_w=body_ang_vel_w.astype(np.float32),
        )
        commands.append({
            **asdict(spec),
            "turn_angle_rad": signed_turn,
            "motion": str(motion_path),
            "sha256": sha256_file(motion_path),
            "metrics": reference_path_metrics(
                body_pos_w, body_quat_w, target_fps
            ),
        })

    manifest = {
        "schema": REFERENCE_SCHEMA,
        "source_motion": str(source_path),
        "source_motion_sha256": sha256_file(source_path),
        "source_slice": [start, end],
        "source_fps": source_fps,
        "source_metrics": source_metrics,
        "urdf": str(urdf),
        "urdf_sha256": sha256_file(urdf),
        "duration_s": float(duration_s),
        "target_fps": float(target_fps),
        "frames": frames,
        "distance_m": target_distance,
        "turn_angle_magnitude_rad": float(turn_angle_rad),
        "commands": commands,
    }
    manifest_path = output / "canonical_nav_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
