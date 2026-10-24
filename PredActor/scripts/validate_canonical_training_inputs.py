#!/usr/bin/env python3
"""Instantiate and sample the canonical-teacher mixture before formal training."""

from __future__ import annotations

import argparse
import json

import torch

from dagger.canonical_teacher import CanonicalTeacherDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d0-zarr", required=True)
    parser.add_argument("--canonical-zarr", required=True)
    parser.add_argument("--command-map", required=True)
    parser.add_argument("--pi0-checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--epoch-size", type=int, default=30)
    args = parser.parse_args()
    dataset = CanonicalTeacherDataset(
        zarr_path=args.d0_zarr,
        canonical_zarr_path=args.canonical_zarr,
        d0_command_map_path=args.command_map,
        normalizer_checkpoint=args.pi0_checkpoint,
        data_profile="g1_prdp_standard_cond",
        horizon=20,
        n_obs_steps=4,
        n_task_steps=4,
        d0_weight=0.60,
        canonical_weight=0.40,
        epoch_size=args.epoch_size,
        load_to_memory=False,
        symm_aug=False,
        val_ratio=0.0,
        dropout_config_train_overrides={"text_cond": {"enabled": False}},
    )
    indices = list(range(min(args.batch_size, len(dataset))))
    batch = dataset.collate_fn([dataset[index] for index in indices])
    normalizer = dataset.get_normalizer()
    normalized = normalizer.normalize({
        key: batch[key] for key in ("obs", "action", "cond")
    })
    report = {
        "status": "passed",
        "dataset_length": len(dataset),
        "source_counts": dataset.source_counts,
        "excluded_d0_motion_ids": list(dataset.excluded_d0_motion_ids),
        "batch_shapes": {
            key: list(value.shape)
            for key, value in batch.items()
            if torch.is_tensor(value)
        },
        "normalized_finite": {
            key: bool(torch.isfinite(value).all())
            for key, value in normalized.items()
        },
    }
    if not all(report["normalized_finite"].values()):
        raise ValueError("normalizer produced non-finite canonical training data")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
