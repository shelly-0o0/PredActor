"""Simulator-agnostic strict DAgger collection loop.

The adapter contract deliberately makes an isolated branch explicit. A live
simulator can only be advanced by the learner; expert continuation is executed
on the object returned by ``fork``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol

import numpy as np

from dagger.branching import BranchWindow


State = Mapping[str, np.ndarray]


class BranchableSimulator(Protocol):
    def reset(self) -> State: ...
    def observe(self) -> State: ...
    def snapshot(self) -> Any: ...
    def fork(self, snapshot: Any) -> "BranchableSimulator": ...
    def step(self, action: np.ndarray) -> State: ...
    @property
    def terminated(self) -> bool: ...


Policy = Callable[[State], np.ndarray]
RiskTrigger = Callable[[tuple[State, ...]], bool]


@dataclass(frozen=True)
class CollectionConfig:
    horizon: int = 20
    n_obs_steps: int = 4
    cooldown_steps: int = 10
    max_branches_per_episode: int = 32

    @property
    def teacher_steps(self) -> int:
        return self.horizon - self.n_obs_steps + 1


def _copy_state(state: State) -> dict[str, np.ndarray]:
    return {key: np.asarray(value).copy() for key, value in state.items()}


class StrictDaggerCollector:
    def __init__(
        self,
        config: CollectionConfig,
        learner: Policy,
        teacher: Policy,
        risk_trigger: RiskTrigger,
    ):
        if config.n_obs_steps < 1 or config.horizon < config.n_obs_steps:
            raise ValueError("invalid DAgger history/horizon")
        self.config = config
        self.learner = learner
        self.teacher = teacher
        self.risk_trigger = risk_trigger

    def _teacher_branch(
        self,
        live: BranchableSimulator,
        prefix: tuple[State, ...],
        student_actions: tuple[np.ndarray, ...],
        *,
        command_id: int,
        role: str,
    ) -> BranchWindow | None:
        branch = live.fork(live.snapshot())
        teacher_states = []
        teacher_actions = []
        current = _copy_state(branch.observe())
        shared_origin_fields = set(prefix[-1]) & set(current)
        if not shared_origin_fields:
            raise RuntimeError("teacher branch exposes no fields from learner origin")
        for key in shared_origin_fields:
            learner_value = np.asarray(prefix[-1][key])
            branch_value = np.asarray(current[key])
            if learner_value.shape != branch_value.shape or not np.allclose(
                learner_value, branch_value, atol=1.0e-5, rtol=1.0e-5
            ):
                raise RuntimeError(
                    f"isolated teacher branch does not reproduce learner origin field {key!r}"
                )
        for step in range(self.config.teacher_steps):
            teacher_states.append(current)
            action = np.asarray(self.teacher(current)).copy()
            teacher_actions.append(action)
            if step + 1 < self.config.teacher_steps:
                current = _copy_state(branch.step(action))
                if branch.terminated:
                    return None
        state_keys = set.intersection(*(set(item) for item in prefix), *(set(item) for item in teacher_states))
        student_data = {
            key: np.stack([item[key] for item in prefix]) for key in state_keys
        }
        teacher_data = {
            key: np.stack([item[key] for item in teacher_states]) for key in state_keys
        }
        teacher_actions_array = np.stack(teacher_actions)
        if student_actions:
            student_actions_array = np.stack(student_actions)
        else:
            student_actions_array = np.empty(
                (0, *teacher_actions_array.shape[1:]),
                dtype=teacher_actions_array.dtype,
            )
        return BranchWindow(
            student_prefix=student_data,
            student_actions=student_actions_array,
            teacher_states=teacher_data,
            teacher_actions=teacher_actions_array,
            command_id=command_id,
            role=role,
        )

    def collect_episode(
        self,
        live: BranchableSimulator,
        *,
        command_id: int,
        max_steps: int,
        role: str = "origin",
    ) -> list[BranchWindow]:
        state = _copy_state(live.reset())
        history: deque[State] = deque(maxlen=self.config.n_obs_steps)
        action_history: deque[np.ndarray] = deque(
            maxlen=max(0, self.config.n_obs_steps - 1)
        )
        history.append(state)
        branches = []
        cooldown = 0
        for _ in range(max_steps):
            prefix = tuple(history)
            if (
                len(prefix) == self.config.n_obs_steps
                and len(action_history) == self.config.n_obs_steps - 1
                and cooldown == 0
                and len(branches) < self.config.max_branches_per_episode
                and self.risk_trigger(prefix)
            ):
                window = self._teacher_branch(
                    live,
                    prefix,
                    tuple(action_history),
                    command_id=command_id,
                    role=role,
                )
                if window is not None:
                    branches.append(window)
                    cooldown = self.config.cooldown_steps

            # Only the learner advances the live simulator. The expert branch
            # above cannot mutate this object by contract.
            learner_action = np.asarray(self.learner(state))
            state = _copy_state(live.step(learner_action))
            action_history.append(learner_action.copy())
            history.append(state)
            cooldown = max(0, cooldown - 1)
            if live.terminated:
                break
        return branches
