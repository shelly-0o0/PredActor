"""Data contract for learner-origin, isolated-teacher DAgger branches."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping

import numpy as np

@dataclass(frozen=True)
class BranchWindow:
    """One fixed-horizon supervised window.

    ``student_prefix`` ends at the learner-visited origin. ``teacher_states``
    begins with that exact origin and then follows an isolated teacher branch.
    Therefore the origin appears once in the assembled state trajectory.
    """

    student_prefix: Mapping[str, np.ndarray]
    student_actions: np.ndarray
    teacher_states: Mapping[str, np.ndarray]
    teacher_actions: np.ndarray
    command_id: int
    role: str = "origin"

    def assemble(self, *, horizon: int, n_obs_steps: int) -> dict[str, np.ndarray]:
        if self.role not in {"origin", "recovery", "disturbance", "counterfactual"}:
            raise ValueError(f"unsupported strict DAgger role: {self.role!r}")
        expected_teacher = horizon - n_obs_steps + 1
        actions = np.asarray(self.teacher_actions)
        student_actions = np.asarray(self.student_actions)
        if actions.shape[0] != expected_teacher:
            raise ValueError(
                f"teacher branch must contain {expected_teacher} actions, got {actions.shape[0]}"
            )
        if student_actions.shape[0] != n_obs_steps - 1:
            raise ValueError(
                "student action prefix must contain n_obs_steps - 1 actions, got "
                f"{student_actions.shape[0]}"
            )
        if student_actions.shape[1:] != actions.shape[1:]:
            raise ValueError("student/teacher action feature shapes differ")
        shared = set(self.student_prefix) & set(self.teacher_states)
        if not shared:
            raise ValueError("student prefix and teacher branch share no state fields")

        output: dict[str, np.ndarray] = {}
        for key in sorted(shared):
            prefix = np.asarray(self.student_prefix[key])
            teacher = np.asarray(self.teacher_states[key])
            if prefix.shape[0] != n_obs_steps:
                raise ValueError(f"{key}: student prefix length must be {n_obs_steps}")
            if teacher.shape[0] != expected_teacher:
                raise ValueError(f"{key}: teacher state length must be {expected_teacher}")
            if prefix.shape[1:] != teacher.shape[1:]:
                raise ValueError(f"{key}: student/teacher feature shapes differ")
            if np.issubdtype(prefix.dtype, np.floating):
                if not np.allclose(prefix[-1], teacher[0], atol=1.0e-5, rtol=1.0e-5):
                    raise ValueError(f"{key}: teacher branch does not start at learner origin")
            elif not np.array_equal(prefix[-1], teacher[0]):
                raise ValueError(f"{key}: teacher branch does not start at learner origin")
            output[key] = np.concatenate([prefix[:-1], teacher], axis=0)

        invalid_prefix = np.zeros(n_obs_steps - 1, dtype=np.bool_)
        valid_teacher = np.ones(expected_teacher, dtype=np.bool_)
        # Learner actions are retained as model context, but action_valid below
        # excludes them from every supervised loss. Only the isolated expert
        # continuation, beginning at the learner-visited origin, is a target.
        output["act"] = np.concatenate([student_actions, actions], axis=0)
        output["action_valid"] = np.concatenate([invalid_prefix, valid_teacher])
        output["command_id"] = np.full(horizon, int(self.command_id), dtype=np.int64)
        if any(value.shape[0] != horizon for value in output.values()):
            bad = {key: value.shape[0] for key, value in output.items() if value.shape[0] != horizon}
            raise RuntimeError(f"assembled DAgger window has bad lengths: {bad}")
        return output


def load_branch_npz(path: str | Path) -> BranchWindow:
    """Load the auditable branch interchange format produced by a simulator adapter."""
    archive = np.load(path, allow_pickle=False)
    student = {
        key[len("student__"):]: archive[key]
        for key in archive.files
        if key.startswith("student__") and key != "student__act"
    }
    teacher = {
        key[len("teacher__"):]: archive[key]
        for key in archive.files
        if key.startswith("teacher__") and key != "teacher__act"
    }
    role = str(archive["role"].item()) if "role" in archive else "origin"
    return BranchWindow(
        student_prefix=student,
        student_actions=archive["student__act"],
        teacher_states=teacher,
        teacher_actions=archive["teacher__act"],
        command_id=int(archive["command_id"].item()),
        role=role,
    )


def save_branch_npz(path: str | Path, window: BranchWindow, metadata: dict | None = None) -> None:
    payload = {
        **{f"student__{key}": value for key, value in window.student_prefix.items()},
        "student__act": window.student_actions,
        **{f"teacher__{key}": value for key, value in window.teacher_states.items()},
        "teacher__act": window.teacher_actions,
        "command_id": np.asarray(window.command_id, dtype=np.int64),
        "role": np.asarray(window.role),
        "metadata_json": np.asarray(json.dumps(metadata or {}, sort_keys=True)),
    }
    np.savez_compressed(Path(path), **payload)
