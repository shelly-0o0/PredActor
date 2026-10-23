"""Reviewed motion-index to exact canonical-command mappings."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from dagger.metrics import quaternion_yaw_wxyz
from dagger.text_commands import CANONICAL_COMMANDS


SCHEMA = "predactor-motion-command-map-v1"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class MotionCommandEntry:
    motion_idx: int
    motion_name: str
    commands: tuple[str, ...]
    reviewed: bool
    review_basis: str
    reason: str


@dataclass(frozen=True)
class MotionCommandMap:
    path: Path
    entries: tuple[MotionCommandEntry, ...]
    sha256: str

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        require_reviewed: bool = True,
    ) -> "MotionCommandMap":
        resolved = Path(path).expanduser().resolve()
        with resolved.open("r", encoding="utf-8") as stream:
            raw: dict[str, Any] = yaml.safe_load(stream) or {}
        if raw.get("schema") != SCHEMA:
            raise ValueError(f"command map schema must be {SCHEMA!r}")

        entries = []
        seen_indices: set[int] = set()
        for item in raw.get("motions", []):
            commands = tuple(str(value) for value in item.get("commands", ()))
            entry = MotionCommandEntry(
                motion_idx=int(item["motion_idx"]),
                motion_name=str(item["motion_name"]),
                commands=commands,
                reviewed=bool(item.get("reviewed", False)),
                review_basis=str(item.get("review_basis", "")),
                reason=str(item.get("reason", "")),
            )
            if entry.motion_idx < 0:
                raise ValueError("motion_idx must be non-negative")
            if entry.motion_idx in seen_indices:
                raise ValueError(f"duplicate motion_idx: {entry.motion_idx}")
            if not entry.motion_name:
                raise ValueError(f"motion {entry.motion_idx} has an empty motion_name")
            if not entry.commands:
                raise ValueError(f"motion {entry.motion_idx} has no commands")
            if len(set(entry.commands)) != len(entry.commands):
                raise ValueError(f"motion {entry.motion_idx} repeats a command")
            unsupported = set(entry.commands) - set(CANONICAL_COMMANDS)
            if unsupported:
                raise ValueError(
                    f"motion {entry.motion_idx} uses non-canonical commands: "
                    f"{sorted(unsupported)}"
                )
            if require_reviewed and not entry.reviewed:
                raise ValueError(f"motion {entry.motion_idx} has not been reviewed")
            if entry.reviewed and not entry.review_basis:
                raise ValueError(
                    f"motion {entry.motion_idx} is reviewed but has no review_basis"
                )
            seen_indices.add(entry.motion_idx)
            entries.append(entry)

        if not entries:
            raise ValueError("command map must contain at least one motion")
        return cls(path=resolved, entries=tuple(entries), sha256=sha256_file(resolved))

    @property
    def commands_by_motion_idx(self) -> dict[int, tuple[str, ...]]:
        return {entry.motion_idx: entry.commands for entry in self.entries}


def reference_motion_metrics(archive_path: str | Path) -> dict[str, float | int]:
    """Measure displacement and heading in the motion's initial root frame."""
    with np.load(archive_path, allow_pickle=False) as archive:
        body_pos = np.asarray(archive["body_pos_w"], dtype=np.float64)
        body_quat = np.asarray(archive["body_quat_w"], dtype=np.float64)
        fps = float(np.asarray(archive["fps"]).reshape(-1)[0])
    if body_pos.ndim != 3 or body_pos.shape[1:] != (30, 3):
        raise ValueError(f"invalid body_pos_w shape: {body_pos.shape}")
    if body_quat.ndim != 3 or body_quat.shape[1:] != (30, 4):
        raise ValueError(f"invalid body_quat_w shape: {body_quat.shape}")
    if len(body_pos) < 2 or len(body_pos) != len(body_quat):
        raise ValueError("reference motion must contain at least two aligned frames")
    if fps <= 0:
        raise ValueError("reference motion fps must be positive")

    yaw = np.unwrap(quaternion_yaw_wxyz(body_quat[:, 0]))
    initial_yaw = yaw[0]
    displacement = body_pos[-1, 0, :2] - body_pos[0, 0, :2]
    forward_axis = np.asarray([np.cos(initial_yaw), np.sin(initial_yaw)])
    lateral_axis = np.asarray([-np.sin(initial_yaw), np.cos(initial_yaw)])
    return {
        "frames": int(len(body_pos)),
        "fps": fps,
        "duration_s": float((len(body_pos) - 1) / fps),
        "forward_displacement_m": float(displacement @ forward_axis),
        "lateral_displacement_m": float(displacement @ lateral_axis),
        "net_yaw_rad": float(yaw[-1] - yaw[0]),
        "cumulative_abs_delta_yaw_rad": float(np.abs(np.diff(yaw)).sum()),
    }


def validate_command_geometry(command: str, metrics: dict[str, float | int]) -> None:
    """Reject reference motions whose measured direction contradicts the label."""
    forward = float(metrics["forward_displacement_m"])
    lateral = float(metrics["lateral_displacement_m"])
    yaw = float(metrics["net_yaw_rad"])
    if command in {"walk", "walk straight"}:
        valid = forward > 0.3 and abs(yaw) < 0.35
    elif command == "walk backward":
        valid = forward < -0.3
    elif command == "walk left":
        valid = lateral > 0.3
    elif command == "walk right":
        valid = lateral < -0.3
    elif command == "turn left while walking":
        valid = yaw > 0.3
    elif command == "turn right while walking":
        valid = yaw < -0.3
    elif command == "walk in a circle":
        valid = abs(yaw) > 1.5 and abs(forward) > 0.3
    else:  # guarded by MotionCommandMap.load, retained for direct callers
        raise ValueError(f"unsupported canonical command: {command!r}")
    if not valid:
        raise ValueError(
            f"reference kinematics contradict {command!r}: "
            f"forward={forward:.3f}, lateral={lateral:.3f}, yaw={yaw:.3f}"
        )
