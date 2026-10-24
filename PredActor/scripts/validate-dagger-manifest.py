"""Fail-fast validation for strict DAgger manifests and zarr schemas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dagger.manifest import DaggerManifest
from diffusion_policy.utils.replay_buffer import ReplayBuffer


REQUIRED_STATE_FIELDS = {
    "act", "body_pos", "body_rot", "body_lin_vel", "body_ang_vel",
    "joint_pos", "joint_vel", "root_pos", "root_rot", "projected_gravity",
    "base_lin_vel", "base_ang_vel", "last_actions", "motion_latent",
    "motion_idx",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--n-obs-steps", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.n_obs_steps <= args.horizon:
        parser.error("--n-obs-steps must be in [1, horizon]")
    manifest = DaggerManifest.load(args.manifest)
    sources = []
    for source in manifest.sources:
        buffer = ReplayBuffer.create_from_path(str(source.path), mode="r")
        fields = set(buffer.keys())
        missing = REQUIRED_STATE_FIELDS - fields
        if missing:
            raise ValueError(f"{source.name}: missing fields {sorted(missing)}")
        if source.role != "d0":
            missing_labels = {"action_valid", "command_id"} - fields
            if missing_labels:
                raise ValueError(
                    f"{source.name}: strict DAgger shard lacks labels "
                    f"{sorted(missing_labels)}"
                )
        if source.role != "d0":
            valid_raw = np.asarray(buffer["action_valid"][:])
            if valid_raw.ndim not in {1, 2}:
                raise ValueError(f"{source.name}: action_valid rank must be 1 or 2")
            valid = valid_raw.reshape(-1)
        episode_ends = buffer.episode_ends[:]
        starts = [0, *episode_ends[:-1].tolist()]
        lengths = [int(end - start) for start, end in zip(starts, episode_ends)]
        if source.role != "d0" and any(length != args.horizon for length in lengths):
            raise ValueError(f"{source.name}: every strict window must have horizon={args.horizon}")
        if source.role != "d0":
            expected_valid = np.asarray(
                [False] * (args.n_obs_steps - 1)
                + [True] * (args.horizon - args.n_obs_steps + 1),
                dtype=bool,
            )
            windows = valid.reshape(-1, args.horizon).astype(bool)
            if not np.all(windows == expected_valid[None, :]):
                raise ValueError(
                    f"{source.name}: action_valid must mask exactly the "
                    f"{args.n_obs_steps - 1} learner-history actions"
                )
            command_ids = np.asarray(buffer["command_id"][:]).reshape(-1, args.horizon)
            if np.any(command_ids != command_ids[:, :1]):
                raise ValueError(f"{source.name}: command_id must be constant per window")
            if np.any(command_ids < 0):
                raise ValueError(f"{source.name}: command_id must be non-negative")
            if manifest.text_commands and np.any(command_ids >= len(manifest.text_commands)):
                raise ValueError(f"{source.name}: command_id exceeds text command table")
        sources.append({
            "name": source.name,
            "role": source.role,
            "weight": source.weight,
            "episodes": buffer.n_episodes,
            "steps": int(buffer.n_steps),
        })
    print(json.dumps({
        "schema": "predactor-strict-dagger-validation-v1",
        "manifest": str(manifest.path),
        "normalizer_checkpoint": str(manifest.normalizer_checkpoint),
        "d0_weight": manifest.d0_weight,
        "dagger_weight": manifest.dagger_weight,
        "sources": sources,
        "status": "PASS",
    }, indent=2))


if __name__ == "__main__":
    main()
