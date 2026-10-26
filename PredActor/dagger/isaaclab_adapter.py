"""Transactional vector-lane adapter for strict DAgger in IsaacLab.

IsaacLab owns one global simulation context, so creating a second independent
``ManagerBasedRLEnv`` is not supported.  This adapter reserves physical vector
lanes as teacher shadows.  A teacher step may advance the global physics clock,
but the protected live lane is restored immediately from a complete row-level
snapshot.  No teacher transition is ever committed to the live trajectory.

The module deliberately imports no IsaacLab package.  It operates on the
stable articulation/manager interface and can therefore be unit-tested with a
small fake environment before starting Omniverse.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np
import torch


State = Mapping[str, np.ndarray]
ObserveFn = Callable[[], Mapping[str, Any]]
CommandInitializer = Callable[[int, int, int], None]
TerminationFn = Callable[[int], bool]


def _row(value: Any, env_id: int) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.ndim == 0:
        raise ValueError("lane observation must have an environment dimension")
    return np.asarray(array[env_id]).copy()


def _copy_state(state: Mapping[str, Any], env_id: int) -> dict[str, np.ndarray]:
    return {key: _row(value, env_id) for key, value in state.items()}


def state_max_abs(left: State, right: State) -> float:
    if set(left) != set(right):
        missing = sorted(set(left) ^ set(right))
        raise ValueError(f"state fields differ: {missing}")
    maximum = 0.0
    for key in left:
        a = np.asarray(left[key])
        b = np.asarray(right[key])
        if a.shape != b.shape:
            raise ValueError(f"state field {key!r} shapes differ: {a.shape} != {b.shape}")
        if np.issubdtype(a.dtype, np.floating):
            if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
                raise ValueError(f"state field {key!r} contains non-finite values")
            maximum = max(maximum, float(np.max(np.abs(a.astype(np.float64) - b))))
        elif not np.array_equal(a, b):
            return float("inf")
    return maximum


def state_abs_errors(left: State, right: State) -> dict[str, float]:
    """Per-field diagnostic companion to the fail-closed aggregate check."""
    if set(left) != set(right):
        raise ValueError(f"state fields differ: {sorted(set(left) ^ set(right))}")
    result = {}
    for key in left:
        a = np.asarray(left[key])
        b = np.asarray(right[key])
        if a.shape != b.shape:
            result[key] = float("inf")
        elif np.issubdtype(a.dtype, np.floating):
            result[key] = (
                float(np.max(np.abs(a.astype(np.float64) - b)))
                if a.size else 0.0
            )
        else:
            result[key] = 0.0 if np.array_equal(a, b) else float("inf")
    return result


@dataclass
class IsaacLabLaneSnapshot:
    env_id: int
    root_state_w: torch.Tensor
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    manager_rows: dict[tuple[str, str], torch.Tensor]
    interval_event_rows: tuple[torch.Tensor, ...]
    observable: dict[str, np.ndarray]
    reference_phase: int


class IsaacLabVectorForkAdapter:
    """Protect one live environment row and expose transactional shadow forks."""

    def __init__(
        self,
        env,
        *,
        observe_fn: ObserveFn,
        command_initializer: CommandInitializer,
        shadow_by_command: Mapping[int, int],
        live_env_id: int = 0,
        termination_fn: TerminationFn | None = None,
        restore_atol: float = 1.0e-5,
    ) -> None:
        self.env = env
        self.observe_fn = observe_fn
        self.command_initializer = command_initializer
        self.shadow_by_command = {int(k): int(v) for k, v in shadow_by_command.items()}
        self.live_env_id = int(live_env_id)
        self.termination_fn = termination_fn or (lambda env_id: False)
        self.restore_atol = float(restore_atol)
        shadow_ids = set(self.shadow_by_command.values())
        if self.live_env_id in shadow_ids:
            raise ValueError("live and shadow environment IDs must be distinct")
        ids = [self.live_env_id, *sorted(shadow_ids)]
        self.robot = env.scene["robot"]
        self.device = self.robot.data.joint_pos.device
        self.num_envs = int(self.robot.data.joint_pos.shape[0])
        if min(ids) < 0 or max(ids) >= self.num_envs:
            raise ValueError("live/shadow environment ID is outside the vector environment")
        self._owners = self._discover_tensor_owners()
        self.live_learner_commits = 0
        self.live_teacher_commits = 0
        self.live_auto_resets_prevented = 0
        self.max_restore_error = 0.0

    def _interval_event_timers(self) -> tuple[torch.Tensor, ...]:
        """Return per-environment interval timers owned by IsaacLab.

        ``EventManager`` keeps these tensors inside a Python list, so the
        generic manager-row discovery below cannot see them.  They are part of
        the simulator state: allowing a teacher-only step to decrement the
        live timer would make the learner receive pushes earlier than it would
        on an independently advanced rollout.
        """
        manager = getattr(self.env, "event_manager", None)
        timers = getattr(manager, "_interval_term_time_left", ())
        result = []
        for timer in timers:
            if not torch.is_tensor(timer):
                raise TypeError("IsaacLab interval-event timer must be a tensor")
            if timer.ndim < 1 or int(timer.shape[0]) != self.num_envs:
                raise RuntimeError(
                    "global interval events cannot be isolated by vector lane"
                )
            result.append(timer)
        return tuple(result)

    def _copy_interval_event_rows(self, env_id: int) -> tuple[torch.Tensor, ...]:
        return tuple(timer[env_id].detach().clone() for timer in self._interval_event_timers())

    def _restore_interval_event_rows(
        self, rows: tuple[torch.Tensor, ...], env_id: int,
    ) -> None:
        timers = self._interval_event_timers()
        if len(rows) != len(timers):
            raise RuntimeError("interval-event timer layout changed during transaction")
        for timer, row in zip(timers, rows):
            timer[env_id].copy_(row.to(device=timer.device, dtype=timer.dtype))

    def _suspend_live_interval_events(self) -> tuple[torch.Tensor, ...]:
        """Make interval events ineligible for the protected live lane."""
        saved = self._copy_interval_event_rows(self.live_env_id)
        for timer in self._interval_event_timers():
            if not torch.is_floating_point(timer):
                raise TypeError("IsaacLab interval-event timer must be floating point")
            timer[self.live_env_id] = torch.inf
        return saved

    def _suspend_live_auto_reset(self):
        """Prevent a teacher-only global step from resetting the live lane.

        IsaacLab advances all vector lanes in one ``env.step`` and immediately
        resets any lane whose termination manager fires.  A live lane close to
        a failure boundary can therefore be reset while only the shadow teacher
        is supposed to advance.  Returning ``False`` for the protected row
        blocks that reset; the termination-manager row is restored from the
        origin snapshot after the transaction.
        """
        manager = getattr(self.env, "termination_manager", None)
        original = getattr(manager, "compute", None)
        if original is None:
            raise RuntimeError("environment has no termination-manager compute method")

        def compute_without_live_reset(*args, **kwargs):
            reset = original(*args, **kwargs)
            if bool(reset[self.live_env_id].item()):
                self.live_auto_resets_prevented += 1
            reset[self.live_env_id] = False
            return reset

        manager.compute = compute_without_live_reset
        return manager, original

    def _discover_tensor_owners(self) -> dict[str, Any]:
        owners: dict[str, Any] = {"env": self.env}
        for name in ("action_manager", "termination_manager", "reward_manager"):
            owner = getattr(self.env, name, None)
            if owner is not None:
                owners[name] = owner
                for index, term in enumerate(getattr(owner, "_terms", ())):
                    owners[f"{name}.term{index}"] = term
        command_manager = getattr(self.env, "command_manager", None)
        if command_manager is not None:
            owners["command_manager"] = command_manager
            for name, term in getattr(command_manager, "_terms", {}).items():
                owners[f"command_manager.{name}"] = term
        return owners

    def _manager_rows(self, env_id: int) -> dict[tuple[str, str], torch.Tensor]:
        result: dict[tuple[str, str], torch.Tensor] = {}
        for owner_name, owner in self._owners.items():
            namespace = getattr(owner, "__dict__", None)
            if namespace is None:
                continue
            for name, value in namespace.items():
                if (
                    torch.is_tensor(value)
                    and value.ndim >= 1
                    and int(value.shape[0]) == self.num_envs
                ):
                    result[(owner_name, name)] = value[env_id].detach().clone()
        return result

    def snapshot(self, *, reference_phase: int) -> IsaacLabLaneSnapshot:
        env_id = self.live_env_id
        return IsaacLabLaneSnapshot(
            env_id=env_id,
            root_state_w=self.robot.data.root_state_w[env_id].detach().clone(),
            joint_pos=self.robot.data.joint_pos[env_id].detach().clone(),
            joint_vel=self.robot.data.joint_vel[env_id].detach().clone(),
            manager_rows=self._manager_rows(env_id),
            interval_event_rows=self._copy_interval_event_rows(env_id),
            observable=_copy_state(self.observe_fn(), env_id),
            reference_phase=int(reference_phase),
        )

    def _translate_root(self, root_state: torch.Tensor, source: int, target: int) -> torch.Tensor:
        result = root_state.detach().clone()
        origins = self.env.scene.env_origins
        result[:3] += origins[target] - origins[source]
        return result

    def restore(self, snapshot: IsaacLabLaneSnapshot, *, env_id: int | None = None) -> None:
        target = snapshot.env_id if env_id is None else int(env_id)
        target_ids = torch.as_tensor([target], dtype=torch.long, device=self.device)
        root_state = self._translate_root(snapshot.root_state_w, snapshot.env_id, target)
        self.robot.write_root_state_to_sim(root_state.unsqueeze(0), env_ids=target_ids)
        self.robot.write_joint_state_to_sim(
            snapshot.joint_pos.unsqueeze(0), snapshot.joint_vel.unsqueeze(0),
            env_ids=target_ids,
        )
        for (owner_name, name), row in snapshot.manager_rows.items():
            owner = self._owners.get(owner_name)
            value = getattr(owner, name, None) if owner is not None else None
            if torch.is_tensor(value) and value.ndim >= 1 and value.shape[0] == self.num_envs:
                value[target].copy_(row.to(device=value.device, dtype=value.dtype))
        self._restore_interval_event_rows(snapshot.interval_event_rows, target)
        scene = getattr(self.env, "scene", None)
        if scene is not None and hasattr(scene, "write_data_to_sim"):
            scene.write_data_to_sim()
        simulation = getattr(self.env, "sim", None)
        if simulation is not None and hasattr(simulation, "forward"):
            simulation.forward()

    def verify_live_restored(self, snapshot: IsaacLabLaneSnapshot) -> float:
        current = _copy_state(self.observe_fn(), self.live_env_id)
        error = state_max_abs(snapshot.observable, current)
        self.max_restore_error = max(self.max_restore_error, error)
        if not np.isfinite(error) or error > self.restore_atol:
            fields = state_abs_errors(snapshot.observable, current)
            worst = sorted(fields.items(), key=lambda item: item[1], reverse=True)[:5]
            raise RuntimeError(
                "teacher transaction changed protected live state: "
                f"max_abs={error:.6g}, tolerance={self.restore_atol:.6g}, "
                f"worst_fields={worst}"
            )
        return error

    def fork(self, snapshot: IsaacLabLaneSnapshot, command_id: int) -> "IsaacLabShadowFork":
        command_id = int(command_id)
        if command_id not in self.shadow_by_command:
            raise KeyError(f"no shadow lane reserved for command_id={command_id}")
        shadow = self.shadow_by_command[command_id]
        self.restore(snapshot, env_id=shadow)
        self.command_initializer(shadow, command_id, snapshot.reference_phase)
        return IsaacLabShadowFork(self, snapshot, command_id, shadow)

    def _action_batch(self, env_id: int, action: np.ndarray | torch.Tensor) -> torch.Tensor:
        value = torch.as_tensor(
            action, dtype=torch.float32, device=self.device
        ).detach().reshape(-1)
        action_dim = int(value.numel())
        batch = torch.zeros((self.num_envs, action_dim), dtype=value.dtype, device=self.device)
        batch[env_id] = value
        return batch

    def _step_shadow(
        self, fork: "IsaacLabShadowFork", action: np.ndarray | torch.Tensor,
    ) -> dict[str, np.ndarray]:
        self.restore(fork.origin)
        # ``env.step`` advances every vector lane and applies interval events
        # after physics.  Keep the live lane ineligible while the isolated
        # teacher lane advances, then put its timer back exactly.  Without
        # this guard, push_robot can alter cached live body velocities even
        # though root/joint state is subsequently restored.
        live_event_rows = self._suspend_live_interval_events()
        termination_manager, original_termination_compute = self._suspend_live_auto_reset()
        try:
            self.env.step(self._action_batch(fork.env_id, action))
        finally:
            termination_manager.compute = original_termination_compute
            self._restore_interval_event_rows(live_event_rows, self.live_env_id)
        shadow_state = _copy_state(self.observe_fn(), fork.env_id)
        fork._terminated = bool(self.termination_fn(fork.env_id))
        # Roll back every live-row tensor and physical state before returning.
        self.restore(fork.origin)
        fork.max_restore_error = max(
            fork.max_restore_error, self.verify_live_restored(fork.origin)
        )
        return shadow_state

    def commit_learner(
        self, action: np.ndarray | torch.Tensor,
    ) -> dict[str, np.ndarray]:
        self.env.step(self._action_batch(self.live_env_id, action))
        self.live_learner_commits += 1
        return _copy_state(self.observe_fn(), self.live_env_id)


class IsaacLabShadowFork:
    """One command-specific teacher continuation on a reserved vector lane."""

    def __init__(
        self,
        adapter: IsaacLabVectorForkAdapter,
        origin: IsaacLabLaneSnapshot,
        command_id: int,
        env_id: int,
    ) -> None:
        self.adapter = adapter
        self.origin = origin
        self.command_id = int(command_id)
        self.env_id = int(env_id)
        self._terminated = False
        self.max_restore_error = 0.0

    def observe(self) -> dict[str, np.ndarray]:
        return _copy_state(self.adapter.observe_fn(), self.env_id)

    def step(self, action: np.ndarray | torch.Tensor) -> dict[str, np.ndarray]:
        return self.adapter._step_shadow(self, action)

    @property
    def terminated(self) -> bool:
        return self._terminated
