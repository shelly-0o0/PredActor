"""Audit canonical motion labels against reference kinematics and immutable D0."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import zarr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dagger.command_map import (
    MotionCommandMap,
    reference_motion_metrics,
    validate_command_geometry,
)
from dagger.text_commands import CANONICAL_COMMANDS


def _motion_archives(motions_dir: Path) -> list[Path]:
    return sorted(motions_dir.glob("*/motion.npz"), key=lambda path: str(path))


def _open_d0(path: Path):
    root = zarr.open(str(path), mode="r")
    if "data" not in root or "meta" not in root:
        raise ValueError("D0 zarr must contain data/ and meta/ groups")
    if "motion_idx" not in root["data"]:
        raise ValueError("D0 zarr is missing data/motion_idx")
    if "episode_ends" not in root["meta"]:
        raise ValueError("D0 zarr is missing meta/episode_ends")
    return root


def audit(
    motions_dir: Path,
    d0_zarr: Path,
    command_map_path: Path,
    *,
    horizon: int,
    expected_motions: int,
    expected_episodes_per_motion: int,
) -> dict:
    command_map = MotionCommandMap.load(command_map_path, require_reviewed=True)
    archives = _motion_archives(motions_dir)
    if len(archives) != expected_motions:
        raise ValueError(
            f"expected {expected_motions} reference motions, found {len(archives)}"
        )
    if horizon < 1:
        raise ValueError("horizon must be positive")

    motion_names = [path.parent.name for path in archives]
    motion_list_hash = hashlib.sha256(
        ("\n".join(motion_names) + "\n").encode("utf-8")
    ).hexdigest()
    reviewed = []
    for entry in command_map.entries:
        if entry.motion_idx >= len(archives):
            raise ValueError(f"motion_idx {entry.motion_idx} is outside the reference set")
        archive = archives[entry.motion_idx]
        if archive.parent.name != entry.motion_name:
            raise ValueError(
                f"motion_idx {entry.motion_idx} name mismatch: map={entry.motion_name!r}, "
                f"sorted_reference={archive.parent.name!r}"
            )
        metrics = reference_motion_metrics(archive)
        for command in entry.commands:
            validate_command_geometry(command, metrics)
        reviewed.append({
            "motion_idx": entry.motion_idx,
            "motion_name": entry.motion_name,
            "commands": list(entry.commands),
            "reviewed": entry.reviewed,
            "review_basis": entry.review_basis,
            "reason": entry.reason,
            "reference_metrics": metrics,
        })

    d0 = _open_d0(d0_zarr)
    motion_idx = np.asarray(d0["data"]["motion_idx"][:]).reshape(-1)
    if not np.all(np.isfinite(motion_idx)):
        raise ValueError("D0 motion_idx contains non-finite values")
    rounded = np.rint(motion_idx).astype(np.int64)
    if not np.allclose(motion_idx, rounded):
        raise ValueError("D0 motion_idx must contain integer-valued IDs")
    episode_ends = np.asarray(d0["meta"]["episode_ends"][:], dtype=np.int64)
    if len(episode_ends) == 0 or episode_ends[-1] != len(rounded):
        raise ValueError("D0 episode_ends do not match data length")

    episode_motion_ids = []
    episode_lengths = []
    start = 0
    for episode, end in enumerate(episode_ends):
        values = rounded[start:end]
        if len(values) == 0:
            raise ValueError(f"D0 episode {episode} is empty")
        unique = np.unique(values)
        if len(unique) != 1:
            raise ValueError(
                f"D0 episode {episode} crosses motion IDs: {unique.tolist()}"
            )
        episode_motion_ids.append(int(unique[0]))
        episode_lengths.append(int(end - start))
        start = int(end)

    observed = Counter(episode_motion_ids)
    expected_ids = set(range(expected_motions))
    if set(observed) != expected_ids:
        missing = sorted(expected_ids - set(observed))
        extra = sorted(set(observed) - expected_ids)
        raise ValueError(f"D0 motion ID coverage mismatch: missing={missing}, extra={extra}")
    bad_episode_counts = {
        motion: count
        for motion, count in sorted(observed.items())
        if count != expected_episodes_per_motion
    }
    if bad_episode_counts:
        raise ValueError(
            "D0 episode count per motion differs from expected "
            f"{expected_episodes_per_motion}: {bad_episode_counts}"
        )

    commands_by_motion = command_map.commands_by_motion_idx
    command_windows = Counter()
    mapped_windows = 0
    for motion, length in zip(episode_motion_ids, episode_lengths):
        windows = max(0, length - horizon + 1)
        commands = commands_by_motion.get(motion, ())
        if commands:
            mapped_windows += windows
        for command in commands:
            command_windows[command] += windows

    missing_mapped_ids = sorted(set(commands_by_motion) - set(observed))
    if missing_mapped_ids:
        raise ValueError(f"mapped motion IDs are absent from D0: {missing_mapped_ids}")

    return {
        "schema": "predactor-motion-command-audit-v1",
        "status": "passed",
        "command_map": str(command_map.path),
        "command_map_sha256": command_map.sha256,
        "motions_dir": str(motions_dir),
        "motion_count": len(archives),
        "motion_list_sha256": motion_list_hash,
        "d0_zarr": str(d0_zarr),
        "d0_steps": int(len(rounded)),
        "d0_episodes": int(len(episode_ends)),
        "episodes_per_motion": expected_episodes_per_motion,
        "horizon": horizon,
        "reviewed_motion_count": len(reviewed),
        "reviewed_motion_command_pairs": int(sum(len(x["commands"]) for x in reviewed)),
        "mapped_d0_windows_before_command_aliases": int(mapped_windows),
        "calibration_pairs_by_command": {
            command: int(command_windows.get(command, 0))
            for command in CANONICAL_COMMANDS
        },
        "commands_without_reviewed_motion": [
            command for command in CANONICAL_COMMANDS
            if command_windows.get(command, 0) == 0
        ],
        "reviewed_motions": reviewed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--motions-dir", required=True)
    parser.add_argument("--d0-zarr", required=True)
    parser.add_argument("--map", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--expected-motions", type=int, default=152)
    parser.add_argument("--expected-episodes-per-motion", type=int, default=50)
    args = parser.parse_args()

    result = audit(
        Path(args.motions_dir).expanduser().resolve(),
        Path(args.d0_zarr).expanduser().resolve(),
        Path(args.map).expanduser().resolve(),
        horizon=args.horizon,
        expected_motions=args.expected_motions,
        expected_episodes_per_motion=args.expected_episodes_per_motion,
    )
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        f"PASS: {result['reviewed_motion_count']} reviewed motions, "
        f"{result['reviewed_motion_command_pairs']} motion-command pairs; "
        f"audit saved to {output}"
    )


if __name__ == "__main__":
    main()
