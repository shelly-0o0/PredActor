#!/usr/bin/env python3
"""Collect strict same-origin navigation branches in IsaacLab.

One vector row is the committed learner trajectory.  Three reserved rows track
straight/left/right counterfactual teacher continuations from the exact same
learner origin.  Teacher transactions are rolled back from the live row before
the next learner action is committed.
"""

from __future__ import annotations

import argparse
from collections import deque
import json
import math
import os
from pathlib import Path
import random
import sys
import traceback

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PREDACTOR_DIR = SCRIPT_DIR.parents[2]
if str(PREDACTOR_DIR) not in sys.path:
    sys.path.insert(0, str(PREDACTOR_DIR))

from dagger.branching import BranchWindow, save_branch_npz
from dagger.isaaclab_adapter import IsaacLabVectorForkAdapter
from dagger.policy_runtime import ExactTextPolicyRuntime
from dagger.same_origin import NAVIGATION_COMMAND_IDS


SPLIT_691 = [90, 120, 90, 90, 29, 29, 3, 4, 1, 3, 3, 3, 29,
             3, 4, 3, 3, 29, 29, 42, 42, 42]
POLICY_STATE_FIELDS = (
    "body_pos", "body_rot", "body_lin_vel", "body_ang_vel",
    "joint_pos", "joint_vel", "root_pos", "root_rot",
    "projected_gravity", "base_lin_vel", "base_ang_vel", "last_actions",
)
COMMAND_TO_MOTION = {1: 0, 5: 1, 6: 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pi0-checkpoint", required=True)
    parser.add_argument("--text-embeddings", required=True)
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--canonical-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pairs", type=int, required=True)
    parser.add_argument("--max-live-steps", type=int, default=5000)
    parser.add_argument("--branch-every", type=int, default=12)
    parser.add_argument("--episode-steps", type=int, default=220)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--n-obs-steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--restore-atol", type=float, default=1.0e-5)
    return parser.parse_args()


def _canonical_motion_files(root: Path) -> list[str]:
    manifest_path = root / "canonical_nav_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_command = {entry["command"]: entry for entry in manifest["commands"]}
    commands = (
        "walk straight", "turn left while walking", "turn right while walking"
    )
    files = []
    for command in commands:
        entry = by_command[command]
        candidate = root / entry["directory"] / "motion.npz"
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        files.append(str(candidate))
    return files


def _embedding(runtime: ExactTextPolicyRuntime, command_id: int) -> np.ndarray:
    return runtime.embeddings[int(command_id)].numpy().astype(np.float32, copy=True)


def _conditioned_state(
    state: dict[str, np.ndarray], runtime: ExactTextPolicyRuntime, command_id: int,
) -> dict[str, np.ndarray]:
    result = {key: np.asarray(state[key]).copy() for key in POLICY_STATE_FIELDS}
    result["motion_latent"] = _embedding(runtime, command_id)
    return result


def main() -> None:
    args = parse_args()
    if args.pairs < 1 or args.branch_every < 1 or args.episode_steps < args.n_obs_steps:
        raise ValueError("invalid collection counts")
    teacher_steps = args.horizon - args.n_obs_steps + 1
    if teacher_steps < 1:
        raise ValueError("horizon must be >= n_obs_steps")
    output = Path(args.output).expanduser().resolve()
    if output.exists() and any(output.rglob("branch_*.npz")):
        raise FileExistsError(f"refusing to append to non-empty branch directory: {output}")
    (output / "origin").mkdir(parents=True, exist_ok=True)
    (output / "counterfactual").mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if "ISAAC_PATH" not in os.environ:
        import importlib.util
        spec = importlib.util.find_spec("isaacsim")
        if spec and spec.origin:
            os.environ["ISAAC_PATH"] = str(Path(spec.origin).parent)
    os.environ.setdefault(
        "OMNI_KIT_APP_DATA", str(Path.home() / ".cache/isaacsim/kit-data")
    )
    Path(os.environ["OMNI_KIT_APP_DATA"]).mkdir(parents=True, exist_ok=True)

    from isaaclab.app import AppLauncher
    launcher_args = argparse.Namespace(
        headless=bool(args.headless),
        livestream=0,
        device=args.device,
        enable_cameras=False,
        experience="",
    )
    launcher = AppLauncher(launcher_args)
    simulation_app = launcher.app

    env = None
    try:
        import gymnasium as gym
        import rl_tracker.tasks  # noqa: F401
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from isaaclab_tasks.utils.hydra import register_task_to_hydra
        from rsl_rl.runners import OnPolicyRunner
        from rl_tracker.tasks.tracking.mdp.commands_collection import (
            MotionCollectionCommandCfg,
        )
        from data_collection import _parse_diffusion_obs

        task_name = "Isaac-PegasusMoDye-DataCollection-G1-v0"
        env_cfg, agent_cfg = register_task_to_hydra(
            task_name, "rsl_rl_cfg_entry_point"
        )
        env_cfg.seed = args.seed
        env_cfg.scene.num_envs = 4
        source_cfg = env_cfg.commands.motion
        zeros = {key: (0.0, 0.0) for key in ("x", "y", "z", "roll", "pitch", "yaw")}
        env_cfg.commands.motion = MotionCollectionCommandCfg(
            asset_name=source_cfg.asset_name,
            anchor_body_name=source_cfg.anchor_body_name,
            body_names=source_cfg.body_names,
            motion_files=_canonical_motion_files(
                Path(args.canonical_root).expanduser().resolve()
            ),
            samples_per_motion=2,
            default_height=getattr(source_cfg, "default_height", 2.0),
            pose_range=zeros,
            velocity_range=zeros,
            joint_position_range=(0.0, 0.0),
            future_steps=getattr(source_cfg, "future_steps", 1),
            direction_range=(0.0, 0.0),
            resampling_time_range=(1.0e9, 1.0e9),
            termination_grace_steps=0,
            record_only_termination=False,
        )
        env_cfg.rewards = None
        env_cfg.episode_length_s = max(float(args.episode_steps) * 0.02 + 1.0, 10.0)
        env = gym.make(task_name, cfg=env_cfg)
        unwrapped = env.unwrapped
        wrapped = RslRlVecEnvWrapper(env)

        runner = OnPolicyRunner(
            wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
        )
        runner.load(str(Path(args.teacher_checkpoint).expanduser().resolve()))
        teacher_policy = runner.get_inference_policy(device=unwrapped.device)
        learner = ExactTextPolicyRuntime(
            args.pi0_checkpoint,
            args.text_embeddings,
            device=args.device,
            replica="ema_agent",
            n_obs_steps=args.n_obs_steps,
            n_task_steps=4,
        )
        motion_term = unwrapped.command_manager._terms["motion"]

        def observe_all():
            concat = unwrapped.observation_manager.compute_group("diffusion")
            states, _ = _parse_diffusion_obs(concat, SPLIT_691)
            return {key: states[key] for key in POLICY_STATE_FIELDS}

        def init_command(env_id: int, command_id: int, phase: int) -> None:
            motion_id = COMMAND_TO_MOTION[int(command_id)]
            length = int(motion_term.motion.file_lengths[motion_id].item())
            safe_phase = min(int(phase), length - teacher_steps - 1)
            motion_term.dagger_set_canonical_phase(env_id, motion_id, safe_phase)
            unwrapped.scene.write_data_to_sim()
            unwrapped.sim.forward()

        def terminated(env_id: int) -> bool:
            values = unwrapped.termination_manager.terminated
            return bool(values[int(env_id)].item())

        adapter = IsaacLabVectorForkAdapter(
            unwrapped,
            observe_fn=observe_all,
            command_initializer=init_command,
            shadow_by_command={1: 1, 5: 2, 6: 3},
            live_env_id=0,
            termination_fn=terminated,
            restore_atol=args.restore_atol,
        )

        def reset_live(command_id: int) -> dict[str, np.ndarray]:
            motion_id = COMMAND_TO_MOTION[command_id]
            motion_term.dagger_set_canonical_phase(0, motion_id, 0)
            ids = torch.tensor([0], dtype=torch.long, device=unwrapped.device)
            motion_term._apply_initialization_randomization(ids)
            unwrapped.episode_length_buf[0] = 0
            unwrapped.scene.write_data_to_sim()
            unwrapped.sim.forward()
            return {key: np.asarray(value[0].detach().cpu()).copy()
                    for key, value in observe_all().items()}

        pairs_written = 0
        total_live_steps = 0
        episode = 0
        while pairs_written < args.pairs and total_live_steps < args.max_live_steps:
            live_command = NAVIGATION_COMMAND_IDS[episode % len(NAVIGATION_COMMAND_IDS)]
            state = reset_live(live_command)
            learner.reset(state, live_command)
            action_history: deque[np.ndarray] = deque(maxlen=args.n_obs_steps - 1)
            for local_step in range(args.episode_steps):
                if pairs_written >= args.pairs or total_live_steps >= args.max_live_steps:
                    break
                if (
                    local_step >= args.n_obs_steps - 1
                    and local_step % args.branch_every == 0
                    and len(action_history) == args.n_obs_steps - 1
                ):
                    phase = local_step % max(
                        1,
                        min(int(value.item()) for value in motion_term.motion.file_lengths)
                        - teacher_steps - 1,
                    )
                    origin = adapter.snapshot(reference_phase=phase)
                    raw_prefix = learner.history_numpy(live_command)
                    learner_actions = np.stack(tuple(action_history))
                    staged = []
                    pair_id = f"pair-{pairs_written:06d}"
                    pair_failed = False
                    for branch_command in NAVIGATION_COMMAND_IDS:
                        prefix = {
                            key: np.asarray(raw_prefix[key]).copy()
                            for key in POLICY_STATE_FIELDS
                        }
                        prefix["motion_latent"] = np.repeat(
                            _embedding(learner, branch_command)[None, :],
                            args.n_obs_steps,
                            axis=0,
                        )
                        fork = adapter.fork(origin, branch_command)
                        teacher_states = []
                        teacher_actions = []
                        for teacher_step in range(teacher_steps):
                            current = _conditioned_state(
                                fork.observe(), learner, branch_command
                            )
                            teacher_states.append(current)
                            policy_obs, _ = wrapped.get_observations()
                            with torch.inference_mode():
                                action = teacher_policy(policy_obs)[fork.env_id].detach()
                            teacher_actions.append(action.detach().cpu().numpy().copy())
                            if teacher_step + 1 < teacher_steps:
                                fork.step(action)
                                if fork.terminated:
                                    pair_failed = True
                                    break
                        if pair_failed:
                            break
                        teacher_dict = {
                            key: np.stack([item[key] for item in teacher_states])
                            for key in teacher_states[0]
                        }
                        role = (
                            "origin" if branch_command == live_command
                            else "counterfactual"
                        )
                        staged.append((branch_command, role, BranchWindow(
                            student_prefix=prefix,
                            student_actions=learner_actions.copy(),
                            teacher_states=teacher_dict,
                            teacher_actions=np.stack(teacher_actions),
                            command_id=branch_command,
                            role=role,
                        ), fork.max_restore_error))
                    adapter.restore(origin)
                    adapter.verify_live_restored(origin)
                    if not pair_failed and len(staged) == 3:
                        for branch_command, role, window, branch_error in staged:
                            target = output / role / (
                                f"branch_{pairs_written:06d}_cmd{branch_command}.npz"
                            )
                            save_branch_npz(target, window, metadata={
                                "schema": "predactor-same-origin-branch-v1",
                                "pair_id": pair_id,
                                "live_command_id": int(live_command),
                                "branch_command_id": int(branch_command),
                                "reference_phase": int(phase),
                                "live_teacher_commits": 0,
                                "live_learner_commits": int(adapter.live_learner_commits),
                                "live_restore_max_abs": float(max(
                                    branch_error, adapter.max_restore_error
                                )),
                            })
                        pairs_written += 1
                        print(
                            f"[SameOrigin] pair={pairs_written}/{args.pairs} "
                            f"live_command={live_command} phase={phase} "
                            f"restore_max={adapter.max_restore_error:.3e}"
                        )

                action = learner.act(live_command)
                state = adapter.commit_learner(action)
                total_live_steps += 1
                action_history.append(action.copy())
                learner.append(state, live_command)
                if terminated(0):
                    break
            episode += 1

        if pairs_written != args.pairs:
            raise RuntimeError(
                f"collection stopped with {pairs_written}/{args.pairs} pairs after "
                f"{total_live_steps} committed learner steps"
            )
        receipt = {
            "schema": "predactor-same-origin-collection-v1",
            "pairs": pairs_written,
            "branches": pairs_written * 3,
            "live_learner_commits": adapter.live_learner_commits,
            "live_teacher_commits": adapter.live_teacher_commits,
            "maximum_live_restore_abs": adapter.max_restore_error,
            "pi0_checkpoint": str(Path(args.pi0_checkpoint).expanduser().resolve()),
            "teacher_checkpoint": str(Path(args.teacher_checkpoint).expanduser().resolve()),
            "seed": args.seed,
        }
        (output / "collection_receipt.json").write_text(
            json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(receipt, indent=2))
    except BaseException:
        # SimulationApp.close() may terminate Kit before Python prints a
        # pending exception, so emit it explicitly first.
        traceback.print_exc()
        raise
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
