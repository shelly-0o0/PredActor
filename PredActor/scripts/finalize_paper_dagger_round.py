#!/usr/bin/env python3
"""Audit paper-style learner-origin branches and build one immutable D_i."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dagger.branching import load_branch_npz
from dagger.same_origin import branch_metadata
from diffusion_policy.utils.replay_buffer import ReplayBuffer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branches", required=True)
    parser.add_argument("--round-dir", required=True)
    parser.add_argument("--d0-zarr", required=True)
    parser.add_argument("--normalizer-checkpoint", required=True)
    parser.add_argument("--expected-motion-count", type=int, default=152)
    parser.add_argument(
        "--expected-motion-ids", default="",
        help="Comma-separated audited subset for a canary; formal rounds leave this empty",
    )
    parser.add_argument("--expected-origins-per-motion", type=int, required=True)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--n-obs-steps", type=int, default=4)
    parser.add_argument("--max-live-restore-error", type=float, default=1.0e-5)
    args = parser.parse_args()

    branches = Path(args.branches).expanduser().resolve()
    round_dir = Path(args.round_dir).expanduser().resolve()
    round_dir.mkdir(parents=True, exist_ok=True)
    output = round_dir / "d1_origins.zarr"
    audit_path = round_dir / "paper_dagger_audit.json"
    manifest_path = round_dir / "manifest.yaml"
    for path in (output, output.with_suffix(".receipt.json"), audit_path, manifest_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite finalized artifact: {path}")

    paths = sorted(branches.glob("branch_*.npz"))
    expected_ids = (
        {int(value) for value in args.expected_motion_ids.split(",")}
        if args.expected_motion_ids else set(range(args.expected_motion_count))
    )
    if not expected_ids or min(expected_ids) < 0 or max(expected_ids) >= args.expected_motion_count:
        raise ValueError("invalid --expected-motion-ids")
    expected_total = len(expected_ids) * args.expected_origins_per_motion
    if len(paths) != expected_total:
        raise ValueError(f"expected {expected_total} branches, got {len(paths)}")
    counts: Counter[int] = Counter()
    checkpoint = None
    maximum_restore = 0.0
    buffer = ReplayBuffer.create_empty_numpy()
    results = []
    for path in paths:
        branch = load_branch_npz(path)
        metadata = branch_metadata(path)
        if metadata.get("schema") != "predactor-paper-dagger-branch-v1":
            raise ValueError(f"{path}: invalid paper DAgger schema")
        if metadata.get("collection_schema") != "process-isolated-v2":
            raise ValueError(f"{path}: teacher branch was not process-isolated")
        motion_idx = int(metadata.get("motion_idx", -1))
        if branch.role != "origin" or branch.command_id != motion_idx:
            raise ValueError(f"{path}: role/task identity mismatch")
        if motion_idx not in expected_ids:
            raise ValueError(f"{path}: unexpected motion_idx {motion_idx}")
        phases = [int(value) for value in metadata.get("teacher_phases", [])]
        expected_steps = args.horizon - args.n_obs_steps + 1
        if len(phases) != expected_steps or phases != list(range(phases[0], phases[0] + expected_steps)):
            raise ValueError(f"{path}: teacher reference phases are not contiguous")
        if phases[0] != int(metadata.get("reference_phase", -1)):
            raise ValueError(f"{path}: teacher does not start at learner reference phase")
        if int(metadata.get("live_teacher_commits", -1)) != 0:
            raise ValueError(f"{path}: teacher action entered live simulator")
        restore = float(metadata.get("live_restore_max_abs", float("inf")))
        if not np.isfinite(restore) or restore > args.max_live_restore_error:
            raise ValueError(f"{path}: live restore error {restore} exceeds tolerance")
        assembled = branch.assemble(horizon=args.horizon, n_obs_steps=args.n_obs_steps)
        for key, value in assembled.items():
            array = np.asarray(value)
            if np.issubdtype(array.dtype, np.floating) and not np.all(np.isfinite(array)):
                raise ValueError(f"{path}: {key} contains NaN/Inf")
        expected_mask = np.r_[
            np.zeros(args.n_obs_steps - 1, dtype=bool),
            np.ones(expected_steps, dtype=bool),
        ]
        if not np.array_equal(assembled["action_valid"], expected_mask):
            raise ValueError(f"{path}: learner action prefix is not masked")
        current_checkpoint = str(metadata.get("learner_checkpoint", ""))
        checkpoint = checkpoint or current_checkpoint
        if not current_checkpoint or current_checkpoint != checkpoint:
            raise ValueError("branches were collected by different learner checkpoints")
        buffer.add_episode(assembled)
        counts[motion_idx] += 1
        maximum_restore = max(maximum_restore, restore)
        results.append({"file": str(path), "motion_idx": motion_idx, "phase": phases[0]})

    if set(counts) != expected_ids or any(
        counts[index] != args.expected_origins_per_motion for index in expected_ids
    ):
        raise ValueError(f"motion coverage/count mismatch: {dict(sorted(counts.items()))}")
    output.parent.mkdir(parents=True, exist_ok=True)
    buffer.save_to_path(str(output))
    receipt = {
        "schema": "predactor-strict-dagger-shard-v1",
        "source_directory": str(branches),
        "output": str(output),
        "windows": len(paths),
        "horizon": args.horizon,
        "n_obs_steps": args.n_obs_steps,
        "role_counts": {"origin": len(paths)},
        # Equation-2 replay keeps the observed learner-history prefix.  The
        # action at the branch origin and every later action are teacher
        # labels; the three pre-origin actions remain the actions that produced
        # their corresponding observed states.  No DAgger-specific loss mask
        # is introduced.
        "learner_actions_used_as_targets": True,
        "motion_counts": {str(key): counts[key] for key in sorted(counts)},
    }
    output.with_suffix(".receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    audit_path.write_text(json.dumps({
        "schema": "predactor-paper-dagger-audit-v1",
        "passed": True,
        "windows": len(paths),
        "motions": len(counts),
        "origins_per_motion": args.expected_origins_per_motion,
        "learner_checkpoint": checkpoint,
        "maximum_live_restore_abs": maximum_restore,
        "results": results,
    }, indent=2) + "\n", encoding="utf-8")

    d0 = Path(args.d0_zarr).expanduser().resolve()
    normalizer = Path(args.normalizer_checkpoint).expanduser().resolve()
    if not d0.exists() or not normalizer.is_file():
        raise FileNotFoundError("D0 or normalizer checkpoint is missing")
    d0_buffer = ReplayBuffer.create_from_path(str(d0), mode="r")
    d0_ends = np.asarray(d0_buffer.episode_ends[:], dtype=np.int64)
    d0_starts = np.r_[0, d0_ends[:-1]]
    d0_windows = int(np.maximum(0, d0_ends - d0_starts - args.horizon + 1).sum())
    # D0 uses one padded start and end position, matching the normal BC config.
    d0_windows += 2 * int(len(d0_ends))
    d1_windows = len(paths)
    total_windows = d0_windows + d1_windows
    d0_weight = d0_windows / total_windows
    d1_weight = d1_windows / total_windows
    manifest_path.write_text("\n".join([
        "schema: predactor-strict-dagger-v1",
        "sampling_mode: aggregate",
        f"normalizer_checkpoint: {normalizer}",
        "normalizer_replica: ema_agent",
        "sources:",
        f"  - {{name: d0, path: {d0}, role: d0, weight: {d0_weight:.12f}}}",
        f"  - {{name: d1_origins, path: {output}, role: origin, weight: {d1_weight:.12f}}}",
        "",
    ]), encoding="utf-8")
    print(json.dumps({
        "manifest": str(manifest_path), "audit": str(audit_path),
        "d1": str(output), "windows": len(paths),
        "aggregate_windows": total_windows, "status": "PASS",
    }, indent=2))


if __name__ == "__main__":
    main()
