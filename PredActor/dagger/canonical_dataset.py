"""Construction and validation helpers for canonical teacher datasets."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid

import numpy as np
import zarr

from dagger.canonical_reference import REFERENCE_SCHEMA, sha256_file
from dagger.text_commands import CANONICAL_COMMANDS, COMMAND_TO_ID


TEACHER_AUDIT_SCHEMA = "predactor-canonical-teacher-audit-v1"
CANONICAL_DATASET_RECEIPT_SCHEMA = "predactor-canonical-teacher-zarr-v1"


def sha256_tree(path: str | Path) -> str:
    root = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        digest.update(str(item.relative_to(root)).encode("utf-8"))
        with item.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _load_embedding_archive(path: Path) -> tuple[tuple[str, ...], np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        commands = tuple(str(value) for value in archive["commands"].tolist())
        embeddings = np.asarray(archive["embeddings"], dtype=np.float32)
    if commands != CANONICAL_COMMANDS:
        raise ValueError("embedding archive command order is not canonical")
    if embeddings.ndim != 2 or embeddings.shape[0] != len(commands):
        raise ValueError(f"invalid embedding shape: {embeddings.shape}")
    if not np.isfinite(embeddings).all():
        raise ValueError("text embeddings contain NaN/Inf")
    if not np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1.0e-4):
        raise ValueError("text embeddings must be L2-normalized")
    return commands, embeddings


def _episode_motion_ids(
    motion_idx: np.ndarray,
    episode_ends: np.ndarray,
) -> np.ndarray:
    starts = np.r_[0, episode_ends[:-1]]
    result = np.empty(len(episode_ends), dtype=np.int64)
    for episode, (start, end) in enumerate(zip(starts, episode_ends)):
        ids = np.unique(np.rint(motion_idx[start:end]).astype(np.int64))
        if len(ids) != 1:
            raise ValueError(f"episode {episode} crosses motion IDs: {ids.tolist()}")
        result[episode] = ids[0]
    return result


def _audited_episode_slices(
    audit: dict,
    episode_ends: np.ndarray,
    command_by_motion: dict[int, str],
) -> tuple[list[tuple[int, int, int]], list[dict]]:
    """Return only episode slices explicitly accepted by the teacher audit."""
    reports = audit.get("episode_reports")
    if not isinstance(reports, list) or len(reports) != len(episode_ends):
        raise ValueError("teacher audit does not cover every raw episode")
    starts = np.r_[0, episode_ends[:-1]]
    selected: list[tuple[int, int, int]] = []
    dropped: list[dict] = []
    for episode, (start, end, report) in enumerate(
        zip(starts, episode_ends, reports)
    ):
        if int(report.get("episode", -1)) != episode:
            raise ValueError(f"teacher audit episode order differs at {episode}")
        if int(report.get("start", -1)) != int(start) or int(
            report.get("end", -1)
        ) != int(end):
            raise ValueError(f"teacher audit slice differs for episode {episode}")
        motion_id = int(report.get("motion_idx", -1))
        if motion_id not in command_by_motion:
            raise ValueError(f"teacher audit has unknown motion ID {motion_id}")
        if str(report.get("command", "")) != command_by_motion[motion_id]:
            raise ValueError(f"teacher audit command differs for episode {episode}")
        if report.get("passed") is True:
            selected.append((int(start), int(end), motion_id))
        else:
            dropped.append({
                "episode": episode,
                "motion_idx": motion_id,
                "command": command_by_motion[motion_id],
                "failure_reasons": list(report.get("failure_reasons", [])),
            })
    retained_commands = {
        command_by_motion[motion_id] for _, _, motion_id in selected
    }
    missing = sorted(set(command_by_motion.values()) - retained_commands)
    if missing:
        raise ValueError(f"audit retained no teacher episode for commands: {missing}")
    return selected, dropped


def _create_array_like(group, name: str, source, shape: tuple[int, ...]):
    chunks = tuple(
        min(int(chunk), int(size)) if int(size) > 0 else int(chunk)
        for chunk, size in zip(source.chunks, shape)
    )
    return group.create_dataset(
        name,
        shape=shape,
        chunks=chunks,
        dtype=source.dtype,
        compressor=source.compressor,
        fill_value=source.fill_value,
        order=source.order,
        filters=source.filters,
    )


def _copy_accepted_episodes(
    raw,
    destination,
    selected: list[tuple[int, int, int]],
) -> np.ndarray:
    """Stream audited episode rows into a fresh Zarr without mutating raw data."""
    total_frames = sum(end - start for start, end, _ in selected)
    data_out = destination.create_group("data")
    meta_out = destination.create_group("meta")
    destination.attrs.update(dict(raw.attrs))
    data_out.attrs.update(dict(raw["data"].attrs))
    meta_out.attrs.update(dict(raw["meta"].attrs))

    for name, source in raw["data"].arrays():
        if source.shape[0] != int(raw["meta/episode_ends"][-1]):
            raise ValueError(f"data/{name} is not frame-aligned")
        target = _create_array_like(
            data_out, name, source, (total_frames, *source.shape[1:])
        )
        cursor = 0
        for start, end, _ in selected:
            length = end - start
            target[cursor:cursor + length] = source[start:end]
            cursor += length

    lengths = np.asarray([end - start for start, end, _ in selected], dtype=np.int64)
    retained_ends = np.cumsum(lengths, dtype=np.int64)
    source_ends = raw["meta/episode_ends"]
    ends_out = _create_array_like(
        meta_out, "episode_ends", source_ends, retained_ends.shape
    )
    ends_out[:] = retained_ends
    extra_meta = set(raw["meta"].array_keys()) - {"episode_ends"}
    if extra_meta:
        raise ValueError(f"unsupported extra meta arrays: {sorted(extra_meta)}")
    return retained_ends


def build_exact_text_teacher_zarr(
    *,
    raw_zarr: str | Path,
    reference_manifest: str | Path,
    teacher_audit: str | Path,
    text_embeddings: str | Path,
    output_zarr: str | Path,
) -> dict:
    raw_path = Path(raw_zarr).expanduser().resolve()
    manifest_path = Path(reference_manifest).expanduser().resolve()
    audit_path = Path(teacher_audit).expanduser().resolve()
    embeddings_path = Path(text_embeddings).expanduser().resolve()
    output_path = Path(output_zarr).expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite canonical dataset: {output_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != REFERENCE_SCHEMA:
        raise ValueError("reference manifest has an unsupported schema")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("schema") != TEACHER_AUDIT_SCHEMA or audit.get("status") != "passed":
        raise ValueError("teacher dataset audit did not pass")
    if Path(audit.get("zarr", "")).resolve() != raw_path:
        raise ValueError("teacher audit covers a different raw zarr")
    if Path(audit.get("reference_manifest", "")).resolve() != manifest_path:
        raise ValueError("teacher audit covers a different reference manifest")
    commands, embeddings = _load_embedding_archive(embeddings_path)

    raw = zarr.open(str(raw_path), mode="r")
    if "data" not in raw or "meta" not in raw:
        raise ValueError("raw teacher zarr must contain data/ and meta/")
    data = raw["data"]
    if "motion_idx" not in data or "act" not in data:
        raise KeyError("raw teacher zarr lacks motion_idx or act")
    episode_ends = np.asarray(raw["meta/episode_ends"][:], dtype=np.int64)
    motion_idx = np.asarray(data["motion_idx"][:]).reshape(-1)
    if len(motion_idx) != int(episode_ends[-1]):
        raise ValueError("motion_idx length differs from final episode end")
    command_by_motion = {
        int(entry["motion_idx"]): str(entry["command"])
        for entry in manifest["commands"]
    }
    episode_ids = _episode_motion_ids(motion_idx, episode_ends)
    if set(episode_ids.tolist()) != set(command_by_motion):
        raise ValueError(
            f"teacher motion coverage={sorted(set(episode_ids.tolist()))}, "
            f"manifest={sorted(command_by_motion)}"
        )
    selected, dropped = _audited_episode_slices(
        audit, episode_ends, command_by_motion
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.parent / f".{output_path.name}.tmp-{uuid.uuid4().hex}"
    try:
        destination = zarr.open(str(temporary), mode="w")
        retained_ends = _copy_accepted_episodes(raw, destination, selected)
        destination_data = destination["data"]
        retained_motion = np.asarray(
            destination_data["motion_idx"][:]
        ).reshape(-1)
        rounded_motion = np.rint(retained_motion).astype(np.int64)
        frame_command_id = np.empty(len(retained_motion), dtype=np.int64)
        motion_latent = np.empty(
            (len(retained_motion), embeddings.shape[1]), dtype=np.float32
        )
        for motion_id, command in command_by_motion.items():
            command_id = COMMAND_TO_ID[command]
            mask = rounded_motion == motion_id
            frame_command_id[mask] = command_id
            motion_latent[mask] = embeddings[command_id]
        action_valid = np.ones(len(retained_motion), dtype=np.bool_)
        for field in ("motion_latent", "command_id", "action_valid"):
            if field in destination_data:
                raise ValueError(f"raw teacher zarr unexpectedly contains {field!r}")
        chunk_rows = min(4096, len(retained_motion))
        compressor = data["motion_idx"].compressor
        destination_data.array(
            "motion_latent", motion_latent,
            chunks=(chunk_rows, embeddings.shape[1]), compressor=compressor,
        )
        destination_data.array(
            "command_id", frame_command_id,
            chunks=(chunk_rows,), compressor=compressor,
        )
        destination_data.array(
            "action_valid", action_valid,
            chunks=(chunk_rows,), compressor=compressor,
        )
        os.replace(temporary, output_path)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    counts = {
        command: int(np.sum(frame_command_id == COMMAND_TO_ID[command]))
        for command in command_by_motion.values()
    }
    receipt = {
        "schema": CANONICAL_DATASET_RECEIPT_SCHEMA,
        "output": str(output_path),
        "output_sha256": sha256_tree(output_path),
        "raw_teacher_zarr": str(raw_path),
        "raw_teacher_zarr_sha256": sha256_tree(raw_path),
        "reference_manifest": str(manifest_path),
        "reference_manifest_sha256": sha256_file(manifest_path),
        "teacher_audit": str(audit_path),
        "teacher_audit_sha256": sha256_file(audit_path),
        "text_embeddings": str(embeddings_path),
        "text_embeddings_sha256": sha256_file(embeddings_path),
        "commands": list(commands),
        "frames_by_command": counts,
        "raw_episodes": int(len(episode_ends)),
        "retained_episodes": int(len(retained_ends)),
        "dropped_episodes": dropped,
        "frames": int(len(frame_command_id)),
        "labels_are_exact_text_embeddings": True,
        "actions_are_rltracker_teacher_targets": True,
        "learner_actions_used_as_targets": False,
    }
    receipt_path = output_path.with_suffix(".receipt.json")
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt
