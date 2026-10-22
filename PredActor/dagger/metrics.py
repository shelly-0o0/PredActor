"""Straight-walk acceptance metrics and branch risk scoring."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np

from dagger.text_commands import canonicalize_command


def quaternion_yaw_wxyz(quaternion: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64)
    if q.shape[-1] != 4:
        raise ValueError("quaternion must use [..., 4] wxyz layout")
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


@dataclass(frozen=True)
class WalkMetrics:
    survival_rate: float
    cumulative_abs_delta_yaw_rad: float
    net_delta_yaw_rad: float
    abs_lateral_displacement_m: float
    forward_distance_m: float
    mean_abs_foot_yaw_difference_rad: float
    duration_s: float

    def to_dict(self) -> dict:
        return asdict(self)


def straight_walk_metrics(
    root_pos: np.ndarray,
    root_rot: np.ndarray,
    *,
    dt: float,
    terminated: np.ndarray | None = None,
    left_foot_rot: np.ndarray | None = None,
    right_foot_rot: np.ndarray | None = None,
    window_seconds: float = 10.0,
) -> WalkMetrics:
    positions = np.asarray(root_pos, dtype=np.float64)
    rotations = np.asarray(root_rot, dtype=np.float64)
    count = min(len(positions), len(rotations), int(round(window_seconds / dt)) + 1)
    if count < 2:
        raise ValueError("at least two trajectory samples are required")
    positions = positions[:count]
    yaw = np.unwrap(quaternion_yaw_wxyz(rotations[:count]))
    initial_yaw = yaw[0]
    displacement = positions[-1, :2] - positions[0, :2]
    forward_axis = np.array([np.cos(initial_yaw), np.sin(initial_yaw)])
    lateral_axis = np.array([-np.sin(initial_yaw), np.cos(initial_yaw)])

    foot_difference = float("nan")
    if left_foot_rot is not None and right_foot_rot is not None:
        left = np.unwrap(quaternion_yaw_wxyz(np.asarray(left_foot_rot)[:count]))
        right = np.unwrap(quaternion_yaw_wxyz(np.asarray(right_foot_rot)[:count]))
        foot_difference = float(np.mean(np.abs(np.angle(np.exp(1j * (left - right))))))

    survived = 1.0
    if terminated is not None:
        survived = float(not np.asarray(terminated, dtype=bool)[:count].any())
    return WalkMetrics(
        survival_rate=survived,
        cumulative_abs_delta_yaw_rad=float(np.abs(np.diff(yaw)).sum()),
        net_delta_yaw_rad=float(yaw[-1] - yaw[0]),
        abs_lateral_displacement_m=float(abs(displacement @ lateral_axis)),
        forward_distance_m=float(displacement @ forward_axis),
        mean_abs_foot_yaw_difference_rad=foot_difference,
        duration_s=float((count - 1) * dt),
    )


@dataclass(frozen=True)
class CommandDistinguishability:
    samples: int
    yaw_threshold_rad: float
    accuracy: float
    balanced_accuracy: float
    per_class_accuracy: dict[str, float]
    mean_net_yaw_rad: dict[str, float]
    ordering_margin_rad: float

    def to_dict(self) -> dict:
        return asdict(self)


def _turn_class(command: str) -> str | None:
    command = canonicalize_command(command)
    if command in {"walk", "walk straight"}:
        return "straight"
    if command == "turn left while walking":
        return "left"
    if command == "turn right while walking":
        return "right"
    return None


def command_distinguishability(
    records: Iterable[tuple[str, WalkMetrics]],
    *,
    yaw_threshold_rad: float = 0.15,
    left_positive_yaw: bool = True,
) -> CommandDistinguishability:
    """Measure whether straight/left-turn/right-turn rollouts are separable.

    The classifier intentionally uses only signed net yaw. This prevents a
    large oscillatory yaw trace from being counted as a successful turn.
    ``ordering_margin_rad > 0`` means all three class centroids are correctly
    ordered under the configured yaw convention.
    """
    if yaw_threshold_rad <= 0:
        raise ValueError("yaw_threshold_rad must be positive")
    grouped: dict[str, list[float]] = {"straight": [], "left": [], "right": []}
    correct: dict[str, int] = {key: 0 for key in grouped}
    total = 0
    for command, metrics in records:
        expected = _turn_class(command)
        if expected is None:
            continue
        signed_yaw = metrics.net_delta_yaw_rad
        if not left_positive_yaw:
            signed_yaw = -signed_yaw
        grouped[expected].append(float(signed_yaw))
        if signed_yaw > yaw_threshold_rad:
            predicted = "left"
        elif signed_yaw < -yaw_threshold_rad:
            predicted = "right"
        else:
            predicted = "straight"
        correct[expected] += int(predicted == expected)
        total += 1

    missing = [key for key, values in grouped.items() if not values]
    if missing:
        raise ValueError(
            "distinguishability requires straight, left-turn, and right-turn "
            f"rollouts; missing {missing}"
        )
    per_class = {
        key: correct[key] / len(grouped[key])
        for key in grouped
    }
    means = {key: float(np.mean(values)) for key, values in grouped.items()}
    margin = min(
        means["left"] - means["straight"],
        means["straight"] - means["right"],
    )
    return CommandDistinguishability(
        samples=total,
        yaw_threshold_rad=float(yaw_threshold_rad),
        accuracy=sum(correct.values()) / total,
        balanced_accuracy=float(np.mean(list(per_class.values()))),
        per_class_accuracy=per_class,
        mean_net_yaw_rad=means,
        ordering_margin_rad=float(margin),
    )


def straight_walk_cost(metrics: WalkMetrics) -> float:
    """Cost-to-go used for risk ranking (lower is better)."""
    foot = 0.0 if np.isnan(metrics.mean_abs_foot_yaw_difference_rad) else metrics.mean_abs_foot_yaw_difference_rad
    forward_shortfall = max(0.0, 1.0 - metrics.forward_distance_m)
    return float(
        20.0 * (1.0 - metrics.survival_rate)
        + 3.0 * metrics.cumulative_abs_delta_yaw_rad
        + 4.0 * metrics.abs_lateral_displacement_m
        + 1.5 * forward_shortfall
        + foot
    )
