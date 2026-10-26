#!/usr/bin/env python3
"""Label learner origins in a teacher-only IsaacLab process."""

from __future__ import annotations

import argparse
from collections import Counter
import glob
import json
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
from dagger.paper_origins import load_origin
from dagger.phase_conditions import PhaseConditionBank


SPLIT_691 = [90, 120, 90, 90, 29, 29, 3, 4, 1, 3, 3, 3, 29,
             3, 4, 3, 3, 29, 29, 42, 42, 42]
POLICY_STATE_FIELDS = (
    "body_pos", "body_rot", "body_lin_vel", "body_ang_vel",
    "joint_pos", "joint_vel", "root_pos", "root_rot",
    "projected_gravity", "base_lin_vel", "base_ang_vel", "last_actions",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--origins", required=True)
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--motion-root", required=True)
    parser.add_argument("--phase-conditions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--origins-per-motion", type=int, default=5)
    parser.add_argument("--motion-ids", default="")
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--n-obs-steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--restore-atol", type=float, default=1.0e-5)
    return parser.parse_args()


def _conditioned_state(state, bank, motion_idx, phase):
    result = {key: np.asarray(state[key]).copy() for key in POLICY_STATE_FIELDS}
    result["motion_latent"] = bank.at(motion_idx, phase).copy()
    result["motion_idx"] = np.asarray(motion_idx, dtype=np.int64)
    return result


def main() -> None:
    args = parse_args()
    teacher_steps = args.horizon - args.n_obs_steps + 1
    origins_dir = Path(args.origins).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if output.exists() and any(output.glob("branch_*.npz")):
        raise FileExistsError(f"refusing to append to non-empty directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    origin_paths = sorted(origins_dir.glob("origin_*.npz"))
    if not origin_paths:
        raise FileNotFoundError(f"no origin files in {origins_dir}")

    motion_root = Path(args.motion_root).expanduser().resolve()
    motion_files = sorted(glob.glob(str(motion_root / "*" / "motion.npz")))
    bank = PhaseConditionBank(args.phase_conditions)
    selected = (
        tuple(int(value) for value in args.motion_ids.split(","))
        if args.motion_ids else tuple(range(len(motion_files)))
    )
    expected_ids = set(selected)
    available = Counter(load_origin(path).motion_idx for path in origin_paths)
    missing = sorted(index for index in expected_ids
                     if available[index] < args.origins_per_motion)
    if missing:
        raise ValueError(f"insufficient origin candidates for motions: {missing}")

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
    os.environ.setdefault("OMNI_KIT_APP_DATA", str(Path.home() / ".cache/isaacsim/kit-data"))
    Path(os.environ["OMNI_KIT_APP_DATA"]).mkdir(parents=True, exist_ok=True)

    from isaaclab.app import AppLauncher
    launcher = AppLauncher(argparse.Namespace(
        headless=bool(args.headless), livestream=0, device=args.device,
        enable_cameras=False, experience="",
    ))
    simulation_app = launcher.app
    env = None
    try:
        import gymnasium as gym
        import rl_tracker.tasks  # noqa: F401
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from isaaclab_tasks.utils.hydra import register_task_to_hydra
        from rsl_rl.runners import OnPolicyRunner
        from rl_tracker.tasks.tracking.mdp.commands_collection import MotionCollectionCommandCfg
        from data_collection import _parse_diffusion_obs

        task_name = "Isaac-PegasusMoDye-DataCollection-G1-v0"
        env_cfg, agent_cfg = register_task_to_hydra(task_name, "rsl_rl_cfg_entry_point")
        env_cfg.seed = args.seed
        env_cfg.scene.num_envs = 1
        source = env_cfg.commands.motion
        env_cfg.commands.motion = MotionCollectionCommandCfg(
            asset_name=source.asset_name, anchor_body_name=source.anchor_body_name,
            body_names=source.body_names, motion_files=motion_files,
            samples_per_motion=1, default_height=getattr(source, "default_height", 2.0),
            pose_range=getattr(source, "pose_range", {}),
            velocity_range=getattr(source, "velocity_range", {}),
            joint_position_range=getattr(source, "joint_position_range", (-0.52, 0.52)),
            future_steps=getattr(source, "future_steps", 1), direction_range=(0.0, 0.0),
            resampling_time_range=(1.0e9, 1.0e9), termination_grace_steps=0,
            record_only_termination=False,
        )
        env_cfg.rewards = None
        env_cfg.episode_length_s = max(
            max(bank.length(index) for index in selected) * 0.02 + 2.0, 10.0
        )
        env = gym.make(task_name, cfg=env_cfg)
        unwrapped = env.unwrapped
        wrapped = RslRlVecEnvWrapper(env)
        runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None,
                                device=agent_cfg.device)
        runner.load(str(Path(args.teacher_checkpoint).expanduser().resolve()))
        teacher_policy = runner.get_inference_policy(device=unwrapped.device)
        motion_term = unwrapped.command_manager._terms["motion"]
        robot = unwrapped.scene["robot"]
        ids = torch.tensor([0], dtype=torch.long, device=unwrapped.device)

        def observe():
            concat = unwrapped.observation_manager.compute_group("diffusion")
            states, _ = _parse_diffusion_obs(concat, SPLIT_691)
            return {key: np.asarray(states[key][0].detach().cpu()).copy()
                    for key in POLICY_STATE_FIELDS}

        def initialize(origin):
            motion_term.dagger_set_canonical_phase(
                0, origin.motion_idx, origin.reference_phase
            )
            root = torch.as_tensor(origin.root_state_w, device=unwrapped.device).reshape(1, -1)
            joint_pos = torch.as_tensor(origin.joint_pos, device=unwrapped.device).reshape(1, -1)
            joint_vel = torch.as_tensor(origin.joint_vel, device=unwrapped.device).reshape(1, -1)
            robot.write_root_state_to_sim(root, env_ids=ids)
            robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=ids)
            unwrapped.action_manager.reset(ids)
            last_action = torch.as_tensor(
                origin.student_prefix["last_actions"][-1],
                dtype=torch.float32, device=unwrapped.device,
            ).reshape(1, -1)
            unwrapped.action_manager.process_action(last_action)
            timers = getattr(unwrapped.event_manager, "_interval_term_time_left", ())
            if len(timers) != len(origin.interval_event_timers):
                raise RuntimeError("origin/teacher interval-event timer layouts differ")
            for timer, value in zip(timers, origin.interval_event_timers):
                timer[0] = float(value)
            unwrapped.episode_length_buf[0] = 0
            unwrapped.scene.write_data_to_sim()
            unwrapped.sim.forward()
            for name in ("_body_state_w", "_body_link_state_w"):
                cache = getattr(robot.data, name, None)
                if cache is not None:
                    cache.timestamp = -1.0
            motion_term._dagger_update_relative_targets_selected(ids)
            unwrapped.obs_buf = unwrapped.observation_manager.compute()
            root_error = float(torch.max(torch.abs(robot.data.root_state_w[0] - root[0])).item())
            joint_error = max(
                float(torch.max(torch.abs(robot.data.joint_pos[0] - joint_pos[0])).item()),
                float(torch.max(torch.abs(robot.data.joint_vel[0] - joint_vel[0])).item()),
            )
            error = max(root_error, joint_error)
            if not np.isfinite(error) or error > args.restore_atol:
                raise RuntimeError(f"teacher origin restore error {error} > {args.restore_atol}")
            return error

        accepted = Counter()
        rejected = Counter()
        written = 0
        maximum_restore = 0.0
        for path in origin_paths:
            origin = load_origin(path)
            motion_idx = origin.motion_idx
            if motion_idx not in expected_ids or accepted[motion_idx] >= args.origins_per_motion:
                continue
            restore_error = initialize(origin)
            maximum_restore = max(maximum_restore, restore_error)
            teacher_states = []
            teacher_actions = []
            teacher_phases = []
            failed = False
            for step in range(teacher_steps):
                phase = int(motion_term.time_steps[0].item())
                teacher_phases.append(phase)
                state = (
                    {key: np.asarray(value[-1]).copy()
                     for key, value in origin.student_prefix.items()}
                    if step == 0 else _conditioned_state(
                        observe(), bank, motion_idx, phase
                    )
                )
                state["motion_idx"] = np.asarray(motion_idx, dtype=np.int64)
                state["motion_latent"] = bank.at(motion_idx, phase).copy()
                teacher_states.append(state)
                policy_obs, _ = wrapped.get_observations()
                with torch.inference_mode():
                    action = teacher_policy(policy_obs)[0].detach()
                teacher_actions.append(action.cpu().numpy().copy())
                if step + 1 < teacher_steps:
                    unwrapped.step(action.reshape(1, -1))
                    if bool(unwrapped.termination_manager.terminated[0].item()):
                        failed = True
                        rejected[motion_idx] += 1
                        break
            if failed:
                continue
            teacher_dict = {
                key: np.stack([item[key] for item in teacher_states])
                for key in teacher_states[0]
            }
            window = BranchWindow(
                student_prefix=origin.student_prefix,
                student_actions=origin.student_actions,
                teacher_states=teacher_dict,
                teacher_actions=np.stack(teacher_actions),
                command_id=motion_idx, role="origin",
            )
            target = output / f"branch_{written:06d}_motion{motion_idx:03d}.npz"
            save_branch_npz(target, window, metadata={
                "schema": "predactor-paper-dagger-branch-v1",
                "collection_schema": "process-isolated-v2",
                "origin_file": str(path), "motion_idx": motion_idx,
                "motion_name": origin.metadata.get("motion_name", ""),
                "reference_phase": origin.reference_phase,
                "teacher_phases": teacher_phases,
                "live_teacher_commits": 0,
                "live_learner_commits": int(origin.metadata.get("learner_commits", 0)),
                "live_restore_max_abs": restore_error,
                "learner_checkpoint": str(origin.metadata.get("learner_checkpoint", "")),
                "teacher_checkpoint": str(Path(args.teacher_checkpoint).resolve()),
            })
            accepted[motion_idx] += 1
            written += 1
            print(
                f"[PaperTeacher] motion={motion_idx:03d} "
                f"accepted={accepted[motion_idx]}/{args.origins_per_motion} "
                f"total={written} restore={restore_error:.3e}"
            )

        incomplete = {index: accepted[index] for index in selected
                      if accepted[index] != args.origins_per_motion}
        if incomplete:
            raise RuntimeError(f"insufficient accepted teacher branches: {incomplete}")
        receipt = {
            "schema": "predactor-paper-dagger-teacher-label-v2",
            "process_isolated": True, "windows": written,
            "selected_motion_ids": list(selected),
            "origins_per_motion": args.origins_per_motion,
            "per_motion": {str(index): accepted[index] for index in selected},
            "teacher_terminated_rejections": sum(rejected.values()),
            "maximum_origin_restore_abs": maximum_restore,
            "live_teacher_commits": 0, "seed": args.seed,
        }
        (output / "collection_receipt.json").write_text(
            json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(receipt, indent=2))
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
