#!/usr/bin/env python3
"""Build D0 latent prototypes for policy-vs-text-mapping diagnosis."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dagger.latent_prototypes import build_command_latent_prototypes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d0-zarr", required=True)
    parser.add_argument("--command-map", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    payload = build_command_latent_prototypes(args.d0_zarr, args.command_map)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)
    print(f"saved {len(payload['commands'])} D0 command prototypes to {output}")
    for command, count, source_ids in zip(
        payload["commands"], payload["row_counts"], payload["source_motion_ids"]
    ):
        print(f"  {command}: rows={int(count)} motion_idx={source_ids}")


if __name__ == "__main__":
    main()
