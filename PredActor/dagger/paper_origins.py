"""Immutable learner-origin records for process-isolated paper DAgger."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class PaperDaggerOrigin:
    student_prefix: Mapping[str, np.ndarray]
    student_actions: np.ndarray
    root_state_w: np.ndarray
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    interval_event_timers: np.ndarray
    motion_idx: int
    reference_phase: int
    metadata: Mapping[str, object]


def save_origin(path: str | Path, origin: PaperDaggerOrigin) -> None:
    target = Path(path)
    payload = {
        f"prefix__{key}": np.asarray(value)
        for key, value in origin.student_prefix.items()
    }
    payload.update({
        "student_actions": np.asarray(origin.student_actions),
        "root_state_w": np.asarray(origin.root_state_w),
        "joint_pos": np.asarray(origin.joint_pos),
        "joint_vel": np.asarray(origin.joint_vel),
        "interval_event_timers": np.asarray(origin.interval_event_timers),
        "motion_idx": np.asarray(origin.motion_idx, dtype=np.int64),
        "reference_phase": np.asarray(origin.reference_phase, dtype=np.int64),
        "metadata_json": np.asarray(json.dumps(dict(origin.metadata), sort_keys=True)),
    })
    np.savez_compressed(target, **payload)


def load_origin(path: str | Path) -> PaperDaggerOrigin:
    with np.load(path, allow_pickle=False) as archive:
        prefix = {
            key.removeprefix("prefix__"): np.asarray(archive[key]).copy()
            for key in archive.files if key.startswith("prefix__")
        }
        metadata = json.loads(str(archive["metadata_json"]))
        return PaperDaggerOrigin(
            student_prefix=prefix,
            student_actions=np.asarray(archive["student_actions"]).copy(),
            root_state_w=np.asarray(archive["root_state_w"]).copy(),
            joint_pos=np.asarray(archive["joint_pos"]).copy(),
            joint_vel=np.asarray(archive["joint_vel"]).copy(),
            interval_event_timers=np.asarray(archive["interval_event_timers"]).copy(),
            motion_idx=int(archive["motion_idx"]),
            reference_phase=int(archive["reference_phase"]),
            metadata=metadata,
        )
