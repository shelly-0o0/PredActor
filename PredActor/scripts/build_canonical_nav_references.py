#!/usr/bin/env python3
"""Generate straight and symmetric turning G1 reference motions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dagger.canonical_reference import build_canonical_reference_set


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-motion", required=True)
    parser.add_argument(
        "--urdf",
        default=str(ROOT / "PredActor/assets/unitree_description/urdf/g1/main.urdf"),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--duration-s", type=float, default=10.0)
    parser.add_argument("--target-fps", type=float, default=50.0)
    parser.add_argument("--turn-angle-rad", type=float, default=1.0)
    parser.add_argument("--distance-m", type=float)
    parser.add_argument("--source-start", type=int, default=0)
    parser.add_argument("--source-end", type=int)
    args = parser.parse_args()
    manifest = build_canonical_reference_set(
        source_motion=args.source_motion,
        urdf_path=args.urdf,
        output_dir=args.output_dir,
        duration_s=args.duration_s,
        target_fps=args.target_fps,
        turn_angle_rad=args.turn_angle_rad,
        distance_m=args.distance_m,
        source_start=args.source_start,
        source_end=args.source_end,
    )
    print(json.dumps({
        "manifest": str(Path(args.output_dir).resolve() / "canonical_nav_manifest.json"),
        "frames": manifest["frames"],
        "distance_m": manifest["distance_m"],
        "commands": [entry["command"] for entry in manifest["commands"]],
    }, indent=2))


if __name__ == "__main__":
    main()
