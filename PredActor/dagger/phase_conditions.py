"""Phase-conditioned MotionCLIP latents extracted from immutable D0."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


SCHEMA = "predactor-d0-phase-conditions-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_phase_condition_bank(
    d0_zarr: str | Path,
    motion_root: str | Path,
    output: str | Path,
    *,
    expected_motions: int = 152,
) -> dict:
    import zarr

    d0_zarr = Path(d0_zarr).expanduser().resolve()
    motion_root = Path(motion_root).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    motion_files = sorted(motion_root.glob("*/motion.npz"))
    if len(motion_files) != expected_motions:
        raise ValueError(
            f"expected {expected_motions} motion files, got {len(motion_files)}"
        )
    root = zarr.open(str(d0_zarr), mode="r")
    data = root["data"]
    ends = np.asarray(root["meta/episode_ends"][:], dtype=np.int64)
    starts = np.r_[0, ends[:-1]]
    episodes: dict[int, list[tuple[int, int]]] = {}
    for start, end in zip(starts.tolist(), ends.tolist()):
        values = np.asarray(data["motion_idx"][start:end], dtype=np.int64)
        if len(values) == 0 or np.any(values != values[0]):
            raise ValueError(f"D0 episode [{start}:{end}] has mixed motion_idx")
        episodes.setdefault(int(values[0]), []).append((start, end))
    if set(episodes) != set(range(expected_motions)):
        raise ValueError("D0 motion_idx coverage is not exactly 0..151")

    payload: dict[str, np.ndarray] = {}
    records = []
    for motion_idx, path in enumerate(motion_files):
        slices = episodes[motion_idx]
        lengths = {end - start for start, end in slices}
        if len(lengths) != 1:
            raise ValueError(f"motion {motion_idx} has inconsistent D0 episode lengths")
        start, end = slices[0]
        latent = np.asarray(data["motion_latent"][start:end], dtype=np.float32)
        reference_length = int(np.load(path, allow_pickle=False)["joint_pos"].shape[0])
        # The source collector stores transitions and intentionally omits the
        # terminal reference state, so a dyed episode is normally L-1 frames
        # for an L-frame motion.  Accept L as well for legacy complete episodes.
        if len(latent) not in {reference_length - 1, reference_length}:
            raise ValueError(
                f"motion {motion_idx}: D0 length {len(latent)} != reference {reference_length}"
            )
        if not np.all(np.isfinite(latent)):
            raise ValueError(f"motion {motion_idx} latent contains NaN/Inf")
        norms = np.linalg.norm(latent, axis=1)
        if np.max(np.abs(norms - 1.0)) > 1.0e-3:
            raise ValueError(f"motion {motion_idx} latent is not unit-normalized")
        # Dyeing is reference-derived, so all 50 episodes must have the same
        # phase latent. Validate the last episode as an independent sentinel.
        other_start, other_end = slices[-1]
        other = np.asarray(data["motion_latent"][other_start:other_end], dtype=np.float32)
        if not np.allclose(latent, other, atol=1.0e-6, rtol=1.0e-6):
            raise ValueError(f"motion {motion_idx} latent differs across D0 episodes")
        payload[f"motion_{motion_idx:03d}"] = latent
        records.append({
            "motion_idx": motion_idx,
            "motion_name": path.parent.name,
            "motion_path": str(path),
            "motion_sha256": _sha256(path),
            "frames": len(latent),
            "reference_frames": reference_length,
            "d0_episodes": len(slices),
        })
    payload["metadata_json"] = np.asarray(json.dumps({
        "schema": SCHEMA,
        "d0_zarr": str(d0_zarr),
        "motions": records,
    }, sort_keys=True))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)
    return {"output": str(output), "motions": len(records), "episodes": len(ends)}


class PhaseConditionBank:
    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        archive = np.load(self.path, allow_pickle=False)
        self.metadata = json.loads(str(archive["metadata_json"].item()))
        if self.metadata.get("schema") != SCHEMA:
            raise ValueError(f"unsupported phase-condition schema: {self.metadata.get('schema')}")
        self.values = tuple(
            np.asarray(archive[f"motion_{index:03d}"], dtype=np.float32)
            for index in range(len(self.metadata["motions"]))
        )

    def at(self, motion_idx: int, phase: int) -> np.ndarray:
        values = self.values[int(motion_idx)]
        phase = min(max(int(phase), 0), len(values) - 1)
        return values[phase]

    def length(self, motion_idx: int) -> int:
        return len(self.values[int(motion_idx)])
