"""Pure protocol and aggregation utilities for checkpoint policy benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

import numpy as np
import yaml

from dagger.metrics import WalkMetrics, command_distinguishability, straight_walk_metrics


_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


@dataclass(frozen=True)
class CommandCase:
    id: str
    prompt: str

    @classmethod
    def from_dict(cls, value: dict) -> "CommandCase":
        case = cls(id=str(value["id"]), prompt=str(value["prompt"]).strip())
        if not _ID_RE.fullmatch(case.id):
            raise ValueError(f"invalid command id {case.id!r}")
        if not case.prompt:
            raise ValueError(f"command {case.id!r} has an empty prompt")
        return case


@dataclass(frozen=True)
class PolicyBenchmarkSuite:
    name: str
    seeds: tuple[int, ...]
    duration_seconds: float
    activation_delay_seconds: float
    window_frames: int
    target_fps: float
    text_commands: tuple[CommandCase, ...]
    navigation_commands: tuple[CommandCase, ...]

    @property
    def all_commands(self) -> tuple[CommandCase, ...]:
        ordered: dict[str, CommandCase] = {}
        for case in (*self.text_commands, *self.navigation_commands):
            previous = ordered.get(case.id)
            if previous is not None and previous.prompt != case.prompt:
                raise ValueError(
                    f"command id {case.id!r} maps to two prompts: "
                    f"{previous.prompt!r} and {case.prompt!r}"
                )
            ordered[case.id] = case
        return tuple(ordered.values())

    @classmethod
    def load(cls, path: str | Path) -> "PolicyBenchmarkSuite":
        source = Path(path).expanduser().resolve()
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
        if raw.get("schema") != "predactor-policy-benchmark-v1":
            raise ValueError("unsupported policy benchmark schema")
        retrieval = raw.get("retrieval", {})
        suite = cls(
            name=str(raw.get("name", source.stem)),
            seeds=tuple(int(seed) for seed in raw["seeds"]),
            duration_seconds=float(raw.get("duration_seconds", 10.0)),
            activation_delay_seconds=float(
                retrieval.get("activation_delay_seconds", 3.0)
            ),
            window_frames=int(retrieval.get("window_frames", 60)),
            target_fps=float(retrieval.get("target_fps", 30.0)),
            text_commands=tuple(
                CommandCase.from_dict(value) for value in raw.get("text_commands", [])
            ),
            navigation_commands=tuple(
                CommandCase.from_dict(value)
                for value in raw.get("navigation_commands", [])
            ),
        )
        if not suite.seeds or len(set(suite.seeds)) != len(suite.seeds):
            raise ValueError("seeds must be a non-empty unique list")
        if suite.duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive")
        if suite.activation_delay_seconds < 0:
            raise ValueError("activation_delay_seconds cannot be negative")
        if suite.window_frames < 2 or suite.target_fps <= 0:
            raise ValueError("retrieval window_frames/target_fps are invalid")
        if not suite.text_commands:
            raise ValueError("text_commands cannot be empty")
        # Materialize to validate duplicate ids before any rollout is launched.
        suite.all_commands
        return suite


def rollout_path(root: str | Path, case: CommandCase, seed: int) -> Path:
    return Path(root) / "rollouts" / case.id / f"seed-{int(seed)}.npz"


def scalar_summary(values: Iterable[float]) -> dict:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"count": 0, "mean": None, "sample_sd": None}
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "sample_sd": (float(np.std(array, ddof=1)) if len(array) > 1 else None),
    }


def archive_command(archive) -> str:
    values = np.asarray(archive["command"]).reshape(-1)
    if not len(values):
        raise ValueError("rollout command is empty")
    return str(values[0])


def rollout_survival(archive, requested_duration_s: float) -> dict:
    dt = float(np.asarray(archive["dt"]))
    observed_duration_s = max(0.0, (len(archive["root_pos"]) - 1) * dt)
    requested = (
        float(np.asarray(archive["requested_duration_s"]))
        if "requested_duration_s" in archive
        else float(requested_duration_s)
    )
    terminated = bool(np.asarray(archive["terminated"], dtype=bool).any())
    return {
        "observed_duration_s": observed_duration_s,
        "requested_duration_s": requested,
        "survival_fraction": float(min(1.0, observed_duration_s / requested)),
        "completed_horizon": bool(
            observed_duration_s + dt * 0.5 >= requested and not terminated
        ),
        "terminated": terminated,
    }


def first_complete_motion_window(
    archive,
    *,
    activation_delay_seconds: float,
    window_frames: int,
    target_fps: float,
) -> np.ndarray | None:
    """Return the first complete post-delay body-position window.

    The returned layout is ``[window_frames, 30, 3]`` at ``target_fps``.
    No boundary padding is used: an early terminal is intentionally
    non-eligible for retrieval rather than converted into a synthetic window.
    """
    if "body_pos" not in archive:
        return None
    body_pos = np.asarray(archive["body_pos"], dtype=np.float32)
    if body_pos.ndim != 3 or body_pos.shape[1:] != (30, 3):
        raise ValueError(f"body_pos must be [T, 30, 3], got {body_pos.shape}")
    dt = float(np.asarray(archive["dt"]))
    source_time = np.arange(len(body_pos), dtype=np.float64) * dt
    target_time = (
        float(activation_delay_seconds)
        + np.arange(window_frames, dtype=np.float64) / float(target_fps)
    )
    if not len(source_time) or target_time[-1] > source_time[-1] + dt * 1e-4:
        return None
    flat = body_pos.reshape(len(body_pos), -1)
    sampled = np.empty((window_frames, flat.shape[1]), dtype=np.float32)
    for column in range(flat.shape[1]):
        sampled[:, column] = np.interp(target_time, source_time, flat[:, column])
    return sampled.reshape(window_frames, 30, 3)


def cosine_top1(
    motion_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    motion = np.asarray(motion_embeddings, dtype=np.float64)
    text = np.asarray(text_embeddings, dtype=np.float64)
    if motion.ndim != 2 or text.ndim != 2 or motion.shape[1] != text.shape[1]:
        raise ValueError("motion/text embeddings must be compatible matrices")
    motion /= np.maximum(np.linalg.norm(motion, axis=1, keepdims=True), 1e-12)
    text /= np.maximum(np.linalg.norm(text, axis=1, keepdims=True), 1e-12)
    scores = motion @ text.T
    return np.argmax(scores, axis=1), scores


def navigation_metrics(archive) -> WalkMetrics:
    kwargs = {
        key: archive[key]
        for key in ("terminated", "left_foot_rot", "right_foot_rot")
        if key in archive
    }
    requested = float(np.asarray(archive.get("requested_duration_s", 10.0)))
    return straight_walk_metrics(
        archive["root_pos"], archive["root_rot"],
        dt=float(np.asarray(archive["dt"])), window_seconds=requested, **kwargs,
    )


def navigation_distinguishability(records: list[tuple[str, WalkMetrics]]) -> dict:
    try:
        return {"available": True, **command_distinguishability(records).to_dict()}
    except ValueError as error:
        return {"available": False, "reason": str(error)}
