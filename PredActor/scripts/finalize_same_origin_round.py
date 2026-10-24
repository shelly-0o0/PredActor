#!/usr/bin/env python3
"""Audit same-origin branches, build immutable shards, and write a manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dagger.branching import load_branch_npz
from dagger.same_origin import audit_same_origin_directory
from diffusion_policy.utils.replay_buffer import ReplayBuffer


def _build_role(input_dir: Path, output: Path, role: str, horizon: int, n_obs: int) -> dict:
    if output.exists() or output.with_suffix(".receipt.json").exists():
        raise FileExistsError(f"refusing to overwrite strict shard: {output}")
    paths = sorted(input_dir.glob("branch_*.npz"))
    if not paths:
        raise FileNotFoundError(f"no {role} branches under {input_dir}")
    buffer = ReplayBuffer.create_empty_numpy()
    for path in paths:
        branch = load_branch_npz(path)
        if branch.role != role:
            raise ValueError(f"{path}: expected role {role!r}, got {branch.role!r}")
        buffer.add_episode(branch.assemble(horizon=horizon, n_obs_steps=n_obs))
    output.parent.mkdir(parents=True, exist_ok=True)
    buffer.save_to_path(str(output))
    receipt = {
        "schema": "predactor-strict-dagger-shard-v1",
        "source_directory": str(input_dir),
        "output": str(output),
        "windows": len(paths),
        "horizon": horizon,
        "n_obs_steps": n_obs,
        "role_counts": {role: len(paths)},
        "learner_actions_used_as_targets": False,
    }
    output.with_suffix(".receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branches", required=True)
    parser.add_argument("--round-dir", required=True)
    parser.add_argument("--d0-zarr", required=True)
    parser.add_argument("--pi0-checkpoint", required=True)
    parser.add_argument("--text-embeddings", required=True)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--n-obs-steps", type=int, default=4)
    args = parser.parse_args()
    branches = Path(args.branches).expanduser().resolve()
    round_dir = Path(args.round_dir).expanduser().resolve()
    round_dir.mkdir(parents=True, exist_ok=True)
    audit_path = round_dir / "same_origin_audit.json"
    manifest_path = round_dir / "manifest.yaml"
    if audit_path.exists() or manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite finalized round: {round_dir}")
    audits = audit_same_origin_directory(
        branches, horizon=args.horizon, n_obs_steps=args.n_obs_steps
    )
    audit = {
        "schema": "predactor-same-origin-audit-v1",
        "passed": True,
        "pairs": len(audits),
        "branches": len(audits) * 3,
        "minimum_teacher_action_rmse": min(
            value.min_teacher_action_rmse for value in audits
        ),
        "maximum_live_restore_abs": max(
            value.live_restore_max_abs for value in audits
        ),
        "results": [value.to_dict() for value in audits],
    }
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    origin = _build_role(
        branches / "origin", round_dir / "origins.zarr", "origin",
        args.horizon, args.n_obs_steps,
    )
    counterfactual = _build_role(
        branches / "counterfactual", round_dir / "counterfactuals.zarr",
        "counterfactual", args.horizon, args.n_obs_steps,
    )
    d0 = Path(args.d0_zarr).expanduser().resolve()
    pi0 = Path(args.pi0_checkpoint).expanduser().resolve()
    embeddings = Path(args.text_embeddings).expanduser().resolve()
    for value in (d0, pi0, embeddings):
        if not value.exists():
            raise FileNotFoundError(value)
    manifest_path.write_text(
        "\n".join([
            "schema: predactor-strict-dagger-v1",
            f"normalizer_checkpoint: {pi0}",
            "normalizer_replica: ema_agent",
            "sources:",
            f"  - {{name: d0, path: {d0}, role: d0, weight: 0.60}}",
            f"  - {{name: round1_origins, path: {round_dir / 'origins.zarr'}, role: origin, weight: 0.20}}",
            f"  - {{name: round1_counterfactuals, path: {round_dir / 'counterfactuals.zarr'}, role: counterfactual, weight: 0.20}}",
            "text_conditioning:",
            f"  embeddings: {embeddings}",
            "  commands:",
            "    - walk",
            "    - walk straight",
            "    - walk backward",
            "    - walk left",
            "    - walk right",
            "    - turn left while walking",
            "    - turn right while walking",
            "    - walk in a circle",
            "",
        ]),
        encoding="utf-8",
    )
    print(json.dumps({
        "manifest": str(manifest_path),
        "audit": str(audit_path),
        "pairs": len(audits),
        "origin_windows": origin["windows"],
        "counterfactual_windows": counterfactual["windows"],
    }, indent=2))


if __name__ == "__main__":
    main()
