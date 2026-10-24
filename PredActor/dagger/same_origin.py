"""Same-origin counterfactual branch contracts and audits.

The simulator-facing collector writes three branch archives for every learner
origin.  Physical history must be identical across the triplet; only the text
condition and expert continuation are allowed to differ.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from dagger.branching import load_branch_npz


NAVIGATION_COMMAND_IDS = (1, 5, 6)
CONDITION_KEYS = frozenset({"motion_latent", "text_cond", "command_id"})


def _array_sha256(values: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in sorted(values):
        if key in CONDITION_KEYS:
            continue
        array = np.ascontiguousarray(values[key])
        digest.update(key.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def branch_metadata(path: str | Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        if "metadata_json" not in archive:
            raise ValueError(f"{path}: missing metadata_json")
        value = json.loads(str(archive["metadata_json"].item()))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: metadata_json must decode to an object")
    return value


@dataclass(frozen=True)
class PairAudit:
    pair_id: str
    commands: tuple[int, ...]
    prefix_sha256: str
    min_teacher_action_rmse: float
    live_restore_max_abs: float
    files: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "pair_id": self.pair_id,
            "commands": list(self.commands),
            "prefix_sha256": self.prefix_sha256,
            "min_teacher_action_rmse": self.min_teacher_action_rmse,
            "live_restore_max_abs": self.live_restore_max_abs,
            "files": list(self.files),
        }


def audit_same_origin_group(
    paths: Iterable[str | Path],
    *,
    expected_command_ids: tuple[int, ...] = NAVIGATION_COMMAND_IDS,
    horizon: int = 20,
    n_obs_steps: int = 4,
    min_teacher_action_rmse: float = 1.0e-4,
    max_live_restore_error: float = 1.0e-5,
) -> PairAudit:
    paths = tuple(Path(path).resolve() for path in paths)
    if len(paths) != len(expected_command_ids):
        raise ValueError(
            f"same-origin group requires {len(expected_command_ids)} branches, got {len(paths)}"
        )
    records = [(path, load_branch_npz(path), branch_metadata(path)) for path in paths]
    pair_ids = {str(metadata.get("pair_id", "")) for _, _, metadata in records}
    if len(pair_ids) != 1 or not next(iter(pair_ids)):
        raise ValueError(f"branches do not share one non-empty pair_id: {sorted(pair_ids)}")
    pair_id = next(iter(pair_ids))

    by_command = {branch.command_id: (path, branch, metadata) for path, branch, metadata in records}
    if set(by_command) != set(expected_command_ids) or len(by_command) != len(records):
        raise ValueError(
            f"{pair_id}: expected command IDs {expected_command_ids}, got {sorted(by_command)}"
        )

    ordered = [by_command[command_id] for command_id in expected_command_ids]
    reference_prefix = ordered[0][1].student_prefix
    reference_actions = np.asarray(ordered[0][1].student_actions)
    prefix_hash = _array_sha256(reference_prefix)
    for path, branch, metadata in ordered:
        if branch.role not in {"origin", "counterfactual"}:
            raise ValueError(f"{path}: same-origin branch has invalid role {branch.role!r}")
        if _array_sha256(branch.student_prefix) != prefix_hash:
            raise ValueError(f"{path}: learner physical prefix differs within {pair_id}")
        if not np.array_equal(np.asarray(branch.student_actions), reference_actions):
            raise ValueError(f"{path}: learner action prefix differs within {pair_id}")
        assembled = branch.assemble(horizon=horizon, n_obs_steps=n_obs_steps)
        expected_mask = np.r_[
            np.zeros(n_obs_steps - 1, dtype=bool),
            np.ones(horizon - n_obs_steps + 1, dtype=bool),
        ]
        if not np.array_equal(assembled["action_valid"], expected_mask):
            raise ValueError(f"{path}: invalid strict action mask")
        if int(metadata.get("live_teacher_commits", -1)) != 0:
            raise ValueError(f"{path}: teacher action was committed to the live trajectory")
        if int(metadata.get("live_learner_commits", 0)) < 0:
            raise ValueError(f"{path}: invalid learner commit count")

    action_distances = []
    for left in range(len(ordered)):
        for right in range(left + 1, len(ordered)):
            a = np.asarray(ordered[left][1].teacher_actions, dtype=np.float64)
            b = np.asarray(ordered[right][1].teacher_actions, dtype=np.float64)
            action_distances.append(float(np.sqrt(np.mean((a - b) ** 2))))
    minimum_distance = min(action_distances)
    if minimum_distance <= min_teacher_action_rmse:
        raise ValueError(
            f"{pair_id}: teacher targets are not command-distinct; "
            f"minimum RMSE={minimum_distance:.6g}"
        )

    restore_errors = [
        float(metadata.get("live_restore_max_abs", float("inf")))
        for _, _, metadata in ordered
    ]
    restore_error = max(restore_errors)
    if not np.isfinite(restore_error) or restore_error > max_live_restore_error:
        raise ValueError(
            f"{pair_id}: live state changed across teacher transaction; "
            f"max_abs={restore_error:.6g}"
        )
    return PairAudit(
        pair_id=pair_id,
        commands=expected_command_ids,
        prefix_sha256=prefix_hash,
        min_teacher_action_rmse=minimum_distance,
        live_restore_max_abs=restore_error,
        files=tuple(str(path) for path, _, _ in ordered),
    )


def audit_same_origin_directory(
    directory: str | Path,
    **kwargs,
) -> list[PairAudit]:
    directory = Path(directory).expanduser().resolve()
    groups: dict[str, list[Path]] = {}
    for path in sorted(directory.rglob("branch_*.npz")):
        pair_id = str(branch_metadata(path).get("pair_id", ""))
        if not pair_id:
            raise ValueError(f"{path}: missing pair_id")
        groups.setdefault(pair_id, []).append(path)
    if not groups:
        raise FileNotFoundError(f"no branch_*.npz files under {directory}")
    return [audit_same_origin_group(groups[key], **kwargs) for key in sorted(groups)]
