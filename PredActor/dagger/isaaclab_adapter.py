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


@dataclass
class IsaacLabLaneSnapshot:
    env_id: int
    root_state_w: torch.Tensor
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    manager_rows: dict[tuple[str, str], torch.Tensor]
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
        ids = [self.live_env_id, *self.shadow_by_command.values()]
        if len(ids) != len(set(ids)):
            raise ValueError("live and shadow environment IDs must be distinct")
        self.robot = env.scene["robot"]
        self.device = self.robot.data.joint_pos.device
        self.num_envs = int(self.robot.data.joint_pos.shape[0])
        if min(ids) < 0 or max(ids) >= self.num_envs:
            raise ValueError("live/shadow environment ID is outside the vector environment")
        self._owners = self._discover_tensor_owners()
        self.live_learner_commits = 0
        self.live_teacher_commits = 0
        self.max_restore_error = 0.0

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
            raise RuntimeError(
                "teacher transaction changed protected live state: "
                f"max_abs={error:.6g}, tolerance={self.restore_atol:.6g}"
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
        self.env.step(self._action_batch(fork.env_id, action))
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
