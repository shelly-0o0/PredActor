#!/usr/bin/env python3
"""Copy an audited teacher zarr and attach immutable exact-text labels."""

from __future__ import annotations

import argparse
import json

from dagger.canonical_dataset import build_exact_text_teacher_zarr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-zarr", required=True)
    parser.add_argument("--reference-manifest", required=True)
    parser.add_argument("--teacher-audit", required=True)
    parser.add_argument("--text-embeddings", required=True)
    parser.add_argument("--output-zarr", required=True)
    args = parser.parse_args()
    receipt = build_exact_text_teacher_zarr(
        raw_zarr=args.raw_zarr,
        reference_manifest=args.reference_manifest,
        teacher_audit=args.teacher_audit,
        text_embeddings=args.text_embeddings,
        output_zarr=args.output_zarr,
    )
    print(json.dumps({
        "status": "built",
        "output": receipt["output"],
        "raw_episodes": receipt["raw_episodes"],
        "retained_episodes": receipt["retained_episodes"],
        "dropped_episode_count": len(receipt["dropped_episodes"]),
        "frames": receipt["frames"],
        "frames_by_command": receipt["frames_by_command"],
    }, indent=2))


if __name__ == "__main__":
    main()
