#!/usr/bin/env python3
"""Collect learner-visited origins without running a teacher simulator.

This is stage 1 of process-isolated paper DAgger.  Only pi(i-1) advances the
simulator.  Candidate origins are immutable files consumed by a separate
teacher process in stage 2.
"""

from __future__ import annotations

import argparse
from collections import deque
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

from dagger.paper_origins import PaperDaggerOrigin, is_origin_step, save_origin
from dagger.phase_conditions import PhaseConditionBank
from dagger.policy_runtime import ExactTextPolicyRuntime


SPLIT_691 = [90, 120, 90, 90, 29, 29, 3, 4, 1, 3, 3, 3, 29,
             3, 4, 3, 3, 29, 29, 42, 42, 42]
POLICY_STATE_FIELDS = (
    "body_pos", "body_rot", "body_lin_vel", "body_ang_vel",
    "joint_pos", "joint_vel", "root_pos", "root_rot",
    "projected_gravity", "base_lin_vel", "base_ang_vel", "last_actions",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--learner-checkpoint", required=True)
    parser.add_argument("--motion-root", required=True)
    parser.add_argument("--phase-conditions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--candidates-per-motion", type=int, default=8)
    parser.add_argument("--motion-ids", default="")
    parser.add_argument("--branch-every", type=int, default=12)
    parser.add_argument("--max-episode-steps", type=int, default=240)
    parser.add_argument("--max-attempts-per-motion", type=int, default=24)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--n-obs-steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.candidates_per_motion < 1 or args.branch_every < 1:
        raise ValueError("collection counts must be positive")
    teacher_steps = args.horizon - args.n_obs_steps + 1
    output = Path(args.output).expanduser().resolve()
    if output.exists() and any(output.glob("origin_*.npz")):
        raise FileExistsError(f"refusing to append to non-empty directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    motion_root = Path(args.motion_root).expanduser().resolve()
    motion_files = sorted(glob.glob(str(motion_root / "*" / "motion.npz")))
    bank = PhaseConditionBank(args.phase_conditions)
    if len(motion_files) != len(bank.values):
        raise ValueError("motion file count differs from phase-condition bank")
    selected = (
        tuple(int(value) for value in args.motion_ids.split(","))
        if args.motion_ids else tuple(range(len(motion_files)))
    )
    if not selected or min(selected) < 0 or max(selected) >= len(motion_files):
        raise ValueError("invalid --motion-ids")

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
        from isaaclab_tasks.utils.hydra import register_task_to_hydra
        from rl_tracker.tasks.tracking.mdp.commands_collection import MotionCollectionCommandCfg
        from data_collection import _parse_diffusion_obs

        task_name = "Isaac-PegasusMoDye-DataCollection-G1-v0"
        env_cfg, _ = register_task_to_hydra(task_name, "rsl_rl_cfg_entry_point")
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
        longest = max(bank.length(index) for index in selected)
        env_cfg.episode_length_s = max(longest * 0.02 + 2.0, 10.0)
        env = gym.make(task_name, cfg=env_cfg)
        unwrapped = env.unwrapped
        learner = ExactTextPolicyRuntime(
            args.learner_checkpoint, args.phase_conditions, device=args.device,
            replica="ema_agent", n_obs_steps=args.n_obs_steps, n_task_steps=4,
        )
        motion_term = unwrapped.command_manager._terms["motion"]
        robot = unwrapped.scene["robot"]

        def observe_all():
            concat = unwrapped.observation_manager.compute_group("diffusion")
            states, _ = _parse_diffusion_obs(concat, SPLIT_691)
            return {key: states[key] for key in POLICY_STATE_FIELDS}

        def reset_live(motion_idx: int):
            ids = torch.tensor([0], dtype=torch.long, device=unwrapped.device)
            motion_term.dagger_set_canonical_phase(0, motion_idx, 0)
            motion_term._apply_initialization_randomization(ids)
            unwrapped.action_manager.reset(ids)
            unwrapped.episode_length_buf[0] = 0
            unwrapped.scene.write_data_to_sim()
            unwrapped.sim.forward()
            return {key: np.asarray(value[0].detach().cpu()).copy()
                    for key, value in observe_all().items()}

        written = 0
        learner_commits = 0
        per_motion = {}
        for motion_idx in selected:
            accepted = 0
            attempts = 0
            while accepted < args.candidates_per_motion:
                attempts += 1
                if attempts > args.max_attempts_per_motion:
                    raise RuntimeError(
                        f"motion {motion_idx} reached {accepted}/{args.candidates_per_motion} "
                        f"origins after {args.max_attempts_per_motion} attempts"
                    )
                state = reset_live(motion_idx)
                learner.reset_condition(state, bank.at(motion_idx, 0))
                action_history: deque[np.ndarray] = deque(maxlen=args.n_obs_steps - 1)
                for local_step in range(args.max_episode_steps):
                    if accepted >= args.candidates_per_motion:
                        break
                    phase = int(motion_term.time_steps[0].item())
                    can_save = (
                        len(action_history) == args.n_obs_steps - 1
                        and is_origin_step(
                            local_step, args.n_obs_steps, args.branch_every
                        )
                        and phase + teacher_steps <= bank.length(motion_idx)
                    )
                    if can_save:
                        prefix = learner.conditioned_history_numpy()
                        prefix["motion_idx"] = np.full(
                            args.n_obs_steps, motion_idx, dtype=np.int64
                        )
                        timers = getattr(
                            unwrapped.event_manager, "_interval_term_time_left", ()
                        )
                        timer_values = np.asarray([
                            float(timer[0].detach().cpu()) for timer in timers
                        ], dtype=np.float32)
                        origin = PaperDaggerOrigin(
                            student_prefix=prefix,
                            student_actions=np.stack(tuple(action_history)),
                            root_state_w=robot.data.root_state_w[0].detach().cpu().numpy().copy(),
                            joint_pos=robot.data.joint_pos[0].detach().cpu().numpy().copy(),
                            joint_vel=robot.data.joint_vel[0].detach().cpu().numpy().copy(),
                            interval_event_timers=timer_values,
                            motion_idx=motion_idx, reference_phase=phase,
                            metadata={
                                "schema": "predactor-paper-dagger-origin-v1",
                                "motion_idx": motion_idx,
                                "motion_name": Path(motion_files[motion_idx]).parent.name,
                                "reference_phase": phase,
                                "learner_checkpoint": str(Path(args.learner_checkpoint).resolve()),
                                "learner_commits": learner_commits,
                            },
                        )
                        save_origin(
                            output / f"origin_{written:06d}_motion{motion_idx:03d}.npz",
                            origin,
                        )
                        accepted += 1
                        written += 1
                        print(
                            f"[PaperOrigin] motion={motion_idx:03d} "
                            f"candidate={accepted}/{args.candidates_per_motion} "
                            f"phase={phase} total={written}"
                        )
                    action = learner.act_conditioned()
                    batch = torch.as_tensor(action, dtype=torch.float32,
                                            device=unwrapped.device).reshape(1, -1)
                    # Step the manager environment directly.  The outer
                    # Gymnasium OrderEnforcing wrapper is intentionally not
                    # used because state initialization is explicit rather
                    # than performed through its reset() API.
                    unwrapped.step(batch)
                    learner_commits += 1
                    action_history.append(action.copy())
                    state = {key: np.asarray(value[0].detach().cpu()).copy()
                             for key, value in observe_all().items()}
                    next_phase = int(motion_term.time_steps[0].item())
                    learner.append_condition(state, bank.at(motion_idx, next_phase))
                    if bool(unwrapped.termination_manager.terminated[0].item()):
                        break
            per_motion[str(motion_idx)] = accepted

        receipt = {
            "schema": "predactor-paper-dagger-origin-collection-v1",
            "origins": written, "selected_motion_ids": list(selected),
            "candidates_per_motion": args.candidates_per_motion,
            "per_motion": per_motion, "learner_commits": learner_commits,
            "teacher_steps_in_live_simulator": 0,
            "learner_checkpoint": str(Path(args.learner_checkpoint).resolve()),
            "seed": args.seed,
        }
        (output / "origin_receipt.json").write_text(
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
