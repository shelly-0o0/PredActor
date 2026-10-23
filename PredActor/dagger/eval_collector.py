"""Closed-loop trajectory export for strict DAgger and policy benchmarks."""

from __future__ import annotations

from pathlib import Path
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from diffusion_policy.dataset.g1_dataset import BODY_NAMES


def _latest_first_env(value: Any) -> np.ndarray:
    """Return the latest sample for environment zero as a detached array."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.ndim < 2:
        raise ValueError(f"expected [env, time, ...] metric term, got {array.shape}")
    return np.asarray(array[0, -1]).copy()


@dataclass(frozen=True)
class RolloutSummary:
    """Small JSON-safe summary returned through :class:`EnvRunner`."""

    survival_fraction: float
    completed_horizon: bool
    observed_duration_s: float
    requested_duration_s: float
    inference_time_mean_ms: float
    inference_time_p95_ms: float

    def to_dict(self) -> dict:
        return asdict(self)


class PolicyRolloutCollector:
    """Collect one fixed-command rollout and save a benchmark-compatible NPZ.

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
        requested_duration_s: float | None = None,
        paper_termination_gates: bool = False,
    ) -> None:
        if dt <= 0:
            raise ValueError("dt must be positive")
        self.output_path = Path(output_path).expanduser().resolve()
        self.command = str(command)
        self.dt = float(dt)
        self.seed = None if seed is None else int(seed)
        self.requested_duration_s = (
            None if requested_duration_s is None else float(requested_duration_s)
        )
        if self.requested_duration_s is not None and self.requested_duration_s <= 0:
            raise ValueError("requested_duration_s must be positive")
        self.paper_termination_gates = bool(paper_termination_gates)
        self.left_foot_index = BODY_NAMES.index("left_ankle_roll_link")
        self.right_foot_index = BODY_NAMES.index("right_ankle_roll_link")
        self._root_pos: list[np.ndarray] = []
        self._root_rot: list[np.ndarray] = []
        self._left_foot_rot: list[np.ndarray] = []
        self._right_foot_rot: list[np.ndarray] = []
        self._body_pos: list[np.ndarray] = []
        self._body_rot: list[np.ndarray] = []
        self._base_ang_vel: list[np.ndarray] = []
        self._terminated: list[bool] = []
        self._termination_reason: list[str] = []
        self._actions: list[np.ndarray] = []
        self._inference_time_ms: list[float] = []
        self._obs_time_ms: list[float] = []
        self._step_time_ms: list[float] = []
        self._rewards: list[float] = []
        self._finalized = False

    @staticmethod
    def _paper_terminal_reason(raw_history: dict) -> str:
        """Return the first matching Table-2 terminal gate, or an empty string."""
        root_pos = _latest_first_env(raw_history["root_pos"])
        root_rot = _latest_first_env(raw_history["root_rot"])
        # For a wxyz quaternion, world-z dot body-z is 1 - 2(x^2 + y^2).
        _, x, y, _ = np.asarray(root_rot, dtype=np.float64)
        pelvis_tilt = float(np.arccos(np.clip(1.0 - 2.0 * (x * x + y * y), -1.0, 1.0)))
        angular_speed = 0.0
        if "base_ang_vel" in raw_history:
            angular_speed = float(np.linalg.norm(
                _latest_first_env(raw_history["base_ang_vel"])
            ))
        if float(root_pos[2]) < 0.55:
            return "root_height"
        if pelvis_tilt >= 0.3:
            return "pelvis_tilt"
        if angular_speed >= 1.0:
            return "base_angular_speed"
        return ""

    @classmethod
    def _paper_terminal(cls, raw_history: dict) -> bool:
        """Apply the Table-2 architecture-study terminal gates."""
        return bool(cls._paper_terminal_reason(raw_history))

    def _sample(
        self, raw_history: dict, *, terminated: bool,
        termination_reason: str = "",
    ) -> dict:
        missing = {"root_pos", "root_rot"} - set(raw_history)
        if missing:
            raise KeyError(f"walk metric terms are missing: {sorted(missing)}")
        sample = {
            "root_pos": _latest_first_env(raw_history["root_pos"]),
            "root_rot": _latest_first_env(raw_history["root_rot"]),
            "terminated": bool(terminated),
            "termination_reason": str(termination_reason),
        }
        if "base_ang_vel" in raw_history:
            sample["base_ang_vel"] = _latest_first_env(raw_history["base_ang_vel"])
        if "body_rot" in raw_history:
            body_rot = _latest_first_env(raw_history["body_rot"])
            if body_rot.ndim == 1:
                body_rot = body_rot.reshape(len(BODY_NAMES), 4)
            sample["left_foot_rot"] = body_rot[self.left_foot_index]
            sample["right_foot_rot"] = body_rot[self.right_foot_index]
            sample["body_rot"] = body_rot
        if "body_pos" in raw_history:
            body_pos = _latest_first_env(raw_history["body_pos"])
            if body_pos.ndim == 1:
                body_pos = body_pos.reshape(len(BODY_NAMES), 3)
            sample["body_pos"] = body_pos
        return sample

    def _append(self, sample: dict) -> None:
        self._root_pos.append(sample["root_pos"])
        self._root_rot.append(sample["root_rot"])
        self._terminated.append(sample["terminated"])
        self._termination_reason.append(sample["termination_reason"])
        if "base_ang_vel" in sample:
            self._base_ang_vel.append(sample["base_ang_vel"])
        if "left_foot_rot" in sample:
            self._left_foot_rot.append(sample["left_foot_rot"])
            self._right_foot_rot.append(sample["right_foot_rot"])
        if "body_pos" in sample:
            self._body_pos.append(sample["body_pos"])
        if "body_rot" in sample:
            self._body_rot.append(sample["body_rot"])

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
        terminated = bool(done)
        termination_reason = "environment" if terminated else ""
        if self.paper_termination_gates:
            paper_reason = self._paper_terminal_reason(raw_history)
            if paper_reason:
                terminated = True
                termination_reason = paper_reason
        sample = self._sample(
            raw_history, terminated=terminated,
            termination_reason=termination_reason,
        )
        if hasattr(actions, "detach"):
            actions = actions.detach().cpu().numpy()
        action = np.asarray(actions)
        if action.ndim >= 2 and action.shape[0] == 1:
            action = action[0]
        sample.update({
            "action": np.asarray(action).copy(),
            "inference_time_ms": float(inference_time_ms),
            "obs_time_ms": float(obs_time_ms),
            "step_time_ms": float(step_time_ms),
            "reward": float(np.asarray(reward).reshape(-1)[0]),
        })
        return sample

    def step(self, step_data: dict) -> bool:
        self._append(step_data)
        self._actions.append(step_data["action"])
        self._inference_time_ms.append(step_data["inference_time_ms"])
        self._obs_time_ms.append(step_data["obs_time_ms"])
        self._step_time_ms.append(step_data["step_time_ms"])
        self._rewards.append(step_data["reward"])
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
            "termination_reason": np.asarray(self._termination_reason),
            "command": np.asarray([self.command]),
            "dt": np.asarray(self.dt),
            "actions": np.asarray(self._actions),
            "inference_time_ms": np.asarray(self._inference_time_ms),
            "obs_time_ms": np.asarray(self._obs_time_ms),
            "step_time_ms": np.asarray(self._step_time_ms),
            "rewards": np.asarray(self._rewards),
        }
        if self.seed is not None:
            payload["seed"] = np.asarray(self.seed)
        if len(self._left_foot_rot) == len(self._root_pos):
            payload["left_foot_rot"] = np.asarray(self._left_foot_rot)
            payload["right_foot_rot"] = np.asarray(self._right_foot_rot)
        if len(self._body_pos) == len(self._root_pos):
            payload["body_pos"] = np.asarray(self._body_pos)
        if len(self._body_rot) == len(self._root_pos):
            payload["body_rot"] = np.asarray(self._body_rot)
        if len(self._base_ang_vel) == len(self._root_pos):
            payload["base_ang_vel"] = np.asarray(self._base_ang_vel)

        observed_duration_s = float((len(self._root_pos) - 1) * self.dt)
        requested_duration_s = (
            observed_duration_s
            if self.requested_duration_s is None
            else self.requested_duration_s
        )
        payload["requested_duration_s"] = np.asarray(requested_duration_s)

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.output_path, **payload)
        inference = payload["inference_time_ms"]
        summary = RolloutSummary(
            survival_fraction=float(min(1.0, observed_duration_s / requested_duration_s)),
            completed_horizon=bool(
                observed_duration_s + self.dt * 0.5 >= requested_duration_s
                and not payload["terminated"].any()
            ),
            observed_duration_s=observed_duration_s,
            requested_duration_s=requested_duration_s,
            inference_time_mean_ms=(float(np.mean(inference)) if len(inference) else float("nan")),
            inference_time_p95_ms=(float(np.percentile(inference, 95)) if len(inference) else float("nan")),
        )
        print(
            f"[WalkRolloutCollector] saved {len(self._root_pos)} samples to "
            f"{self.output_path}"
        )
        return {
            "metric_results": [summary],
            "wandb": {},
            "table": [],
        }


# Backward-compatible name used by the original walk-only CLI and tests.
WalkRolloutCollector = PolicyRolloutCollector
