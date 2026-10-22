"""Closed-loop trajectory export for strict DAgger acceptance tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from dagger.metrics import straight_walk_metrics
from diffusion_policy.dataset.g1_dataset import BODY_NAMES


def _latest_first_env(value: Any) -> np.ndarray:
    """Return the latest sample for environment zero as a detached array."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.ndim < 2:
        raise ValueError(f"expected [env, time, ...] metric term, got {array.shape}")
    return np.asarray(array[0, -1]).copy()


class WalkRolloutCollector:
    """Collect one fixed-command rollout and save a metrics-compatible NPZ.

    The runner calls :meth:`start` once with the reset state and :meth:`step`
    after each control step. Returning ``False`` from ``step`` asks the runner
    to stop on the first termination instead of silently evaluating a reset
    robot as part of the same ten-second trajectory.
    """

    def __init__(
        self,
        output_path: str | Path,
        command: str,
        *,
        dt: float,
        seed: int | None = None,
    ) -> None:
        if dt <= 0:
            raise ValueError("dt must be positive")
        self.output_path = Path(output_path).expanduser().resolve()
        self.command = str(command)
        self.dt = float(dt)
        self.seed = None if seed is None else int(seed)
        self.left_foot_index = BODY_NAMES.index("left_ankle_roll_link")
        self.right_foot_index = BODY_NAMES.index("right_ankle_roll_link")
        self._root_pos: list[np.ndarray] = []
        self._root_rot: list[np.ndarray] = []
        self._left_foot_rot: list[np.ndarray] = []
        self._right_foot_rot: list[np.ndarray] = []
        self._terminated: list[bool] = []
        self._finalized = False

    def _sample(self, raw_history: dict, *, terminated: bool) -> dict:
        missing = {"root_pos", "root_rot"} - set(raw_history)
        if missing:
            raise KeyError(f"walk metric terms are missing: {sorted(missing)}")
        sample = {
            "root_pos": _latest_first_env(raw_history["root_pos"]),
            "root_rot": _latest_first_env(raw_history["root_rot"]),
            "terminated": bool(terminated),
        }
        if "body_rot" in raw_history:
            body_rot = _latest_first_env(raw_history["body_rot"])
            if body_rot.ndim == 1:
                body_rot = body_rot.reshape(len(BODY_NAMES), 4)
            sample["left_foot_rot"] = body_rot[self.left_foot_index]
            sample["right_foot_rot"] = body_rot[self.right_foot_index]
        return sample

    def _append(self, sample: dict) -> None:
        self._root_pos.append(sample["root_pos"])
        self._root_rot.append(sample["root_rot"])
        self._terminated.append(sample["terminated"])
        if "left_foot_rot" in sample:
            self._left_foot_rot.append(sample["left_foot_rot"])
            self._right_foot_rot.append(sample["right_foot_rot"])

    def start(self, raw_history: dict) -> None:
        if self._root_pos:
            raise RuntimeError("walk rollout collector was started more than once")
        self._append(self._sample(raw_history, terminated=False))

    def collect_step_data(
        self,
        *,
        raw_history: dict,
        actions,
        inference_time_ms: float,
        obs_time_ms: float,
        step_time_ms: float,
        reward,
        done: bool,
    ) -> dict:
        del actions, inference_time_ms, obs_time_ms, step_time_ms, reward
        return self._sample(raw_history, terminated=done)

    def step(self, step_data: dict) -> bool:
        self._append(step_data)
        return not step_data["terminated"]

    def flush_episode(self) -> None:
        # The trajectory remains intact for final export. Evaluation stops at
        # the first terminal state, so there is never a reset discontinuity.
        return None

    def finalize(self) -> dict:
        if self._finalized:
            raise RuntimeError("walk rollout collector was finalized more than once")
        self._finalized = True
        if len(self._root_pos) < 2:
            raise RuntimeError("walk rollout contains fewer than two samples")

        payload = {
            "root_pos": np.asarray(self._root_pos),
            "root_rot": np.asarray(self._root_rot),
            "terminated": np.asarray(self._terminated, dtype=bool),
            "command": np.asarray([self.command]),
            "dt": np.asarray(self.dt),
        }
        if self.seed is not None:
            payload["seed"] = np.asarray(self.seed)
        if len(self._left_foot_rot) == len(self._root_pos):
            payload["left_foot_rot"] = np.asarray(self._left_foot_rot)
            payload["right_foot_rot"] = np.asarray(self._right_foot_rot)

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.output_path, **payload)
        metrics = straight_walk_metrics(
            payload["root_pos"],
            payload["root_rot"],
            dt=self.dt,
            terminated=payload["terminated"],
            left_foot_rot=payload.get("left_foot_rot"),
            right_foot_rot=payload.get("right_foot_rot"),
        )
        print(
            f"[WalkRolloutCollector] saved {len(self._root_pos)} samples to "
            f"{self.output_path}"
        )
        return {
            "metric_results": [metrics],
            "wandb": {},
            "table": [],
        }
