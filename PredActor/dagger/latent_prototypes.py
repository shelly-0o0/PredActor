"""Build diagnostic command prototypes from the conditions stored in D0."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr

from dagger.command_map import MotionCommandMap
from dagger.text_commands import CANONICAL_COMMANDS


def build_command_latent_prototypes(
    d0_zarr: str | Path,
    command_map_path: str | Path,
) -> dict[str, object]:
    """Average D0 latents for each reviewed command and L2-normalize them.

    This is a diagnostic representation of what the policy actually saw.  It
    is not a replacement for exact CLIP text embeddings and must not be used to
    claim text-command recognition.
    """

    source = Path(d0_zarr).expanduser().resolve()
    command_map = MotionCommandMap.load(command_map_path, require_reviewed=True)
    root = zarr.open_group(str(source), mode="r")
    data = root["data"]
    ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)
    starts = np.concatenate(([0], ends[:-1]))
    motion_idx = np.asarray(data["motion_idx"][:]).reshape(-1)
    if not len(ends) or int(ends[-1]) != len(motion_idx):
        raise ValueError("D0 episode_ends do not match motion_idx length")

    latent_array = data["motion_latent"]
    if len(latent_array.shape) != 2:
        raise ValueError("D0 motion_latent must be a two-dimensional array")
    latent_dim = int(latent_array.shape[1])
    sums = {
        command: np.zeros(latent_dim, dtype=np.float64)
        for command in CANONICAL_COMMANDS
    }
    counts = {command: 0 for command in CANONICAL_COMMANDS}
    source_ids = {command: set() for command in CANONICAL_COMMANDS}
    commands_by_motion = command_map.commands_by_motion_idx

    for start, end in zip(starts, ends):
        start, end = int(start), int(end)
        values = np.rint(motion_idx[start:end]).astype(np.int64)
        if len(values) == 0 or np.any(values != values[0]):
            raise ValueError("each D0 episode must contain one constant motion_idx")
        motion = int(values[0])
        commands = commands_by_motion.get(motion, ())
        if not commands:
            continue
        latent = np.asarray(latent_array[start:end], dtype=np.float64)
        if not np.all(np.isfinite(latent)):
            raise ValueError(f"motion_idx {motion} contains non-finite latents")
        latent_sum = latent.sum(axis=0)
        for command in commands:
            sums[command] += latent_sum
            counts[command] += len(latent)
            source_ids[command].add(motion)

    supported = tuple(command for command in CANONICAL_COMMANDS if counts[command])
    if not supported:
        raise ValueError("reviewed command map produced no D0 prototype rows")
    embeddings = []
    for command in supported:
        prototype = sums[command] / float(counts[command])
        norm = float(np.linalg.norm(prototype))
        if norm <= 1.0e-8:
            raise ValueError(f"prototype for {command!r} has zero norm")
        embeddings.append((prototype / norm).astype(np.float32))
    return {
        "commands": np.asarray(supported),
        "embeddings": np.stack(embeddings),
        "row_counts": np.asarray([counts[command] for command in supported], dtype=np.int64),
        "source_motion_ids": np.asarray([
            ",".join(str(value) for value in sorted(source_ids[command]))
            for command in supported
        ]),
        "d0_zarr": np.asarray(str(source)),
        "command_map": np.asarray(str(command_map.path)),
        "command_map_sha256": np.asarray(command_map.sha256),
        "schema": np.asarray("predactor-d0-command-latent-prototypes-v1"),
    }
