"""Build one immutable DAgger zarr shard from isolated branch archives."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dagger.branching import load_branch_npz
from diffusion_policy.utils.replay_buffer import ReplayBuffer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Directory containing branch_*.npz")
    parser.add_argument("--output", required=True, help="New immutable .zarr shard")
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--n-obs-steps", type=int, default=4)
    args = parser.parse_args()

    input_dir = Path(args.input).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite DAgger shard: {output}")
    paths = sorted(input_dir.glob("branch_*.npz"))
    if not paths:
        raise FileNotFoundError(f"no branch_*.npz files under {input_dir}")

    buffer = ReplayBuffer.create_empty_numpy()
    role_counts: dict[str, int] = {}
    for path in paths:
        branch = load_branch_npz(path)
        buffer.add_episode(
            branch.assemble(horizon=args.horizon, n_obs_steps=args.n_obs_steps)
        )
        role_counts[branch.role] = role_counts.get(branch.role, 0) + 1
    output.parent.mkdir(parents=True, exist_ok=True)
    buffer.save_to_path(str(output))
    receipt = {
        "schema": "predactor-strict-dagger-shard-v1",
        "source_directory": str(input_dir),
        "output": str(output),
        "windows": len(paths),
        "horizon": args.horizon,
        "n_obs_steps": args.n_obs_steps,
        "role_counts": role_counts,
        "learner_actions_used_as_targets": False,
    }
    output.with_suffix(".receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
