#!/usr/bin/env python3
"""Paper-style DAgger collection over the original 152-motion task set.

The learner alone advances env 0.  At learner-visited origins, env 1 receives
the exact physical state and the same motion/reference phase; RLTracker then
generates an isolated expert continuation.  No beta mixing or counterfactual
command switch is performed.
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

from dagger.branching import BranchWindow, save_branch_npz
from dagger.isaaclab_adapter import IsaacLabVectorForkAdapter
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
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--motion-root", required=True)
    parser.add_argument("--phase-conditions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--origins-per-motion", type=int, required=True)
    parser.add_argument("--motion-ids", default="", help="Comma-separated canary subset")
    parser.add_argument("--branch-every", type=int, default=12)
    parser.add_argument("--max-episode-steps", type=int, default=240)
    parser.add_argument("--max-attempts-per-motion", type=int, default=8)
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
    if args.origins_per_motion < 1 or args.branch_every < 1:
        raise ValueError("collection counts must be positive")
    teacher_steps = args.horizon - args.n_obs_steps + 1
    if teacher_steps < 1:
        raise ValueError("horizon must be >= n_obs_steps")
    output = Path(args.output).expanduser().resolve()
    if output.exists() and any(output.glob("branch_*.npz")):
        raise FileExistsError(f"refusing to append to non-empty directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    motion_root = Path(args.motion_root).expanduser().resolve()
    motion_files = sorted(glob.glob(str(motion_root / "*" / "motion.npz")))
    bank = PhaseConditionBank(args.phase_conditions)
    if len(motion_files) != len(bank.values):
        raise ValueError("motion file count differs from phase-condition bank")
    if args.motion_ids:
        selected = tuple(int(value) for value in args.motion_ids.split(","))
    else:
        selected = tuple(range(len(motion_files)))
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
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from isaaclab_tasks.utils.hydra import register_task_to_hydra
        from rsl_rl.runners import OnPolicyRunner
        from rl_tracker.tasks.tracking.mdp.commands_collection import MotionCollectionCommandCfg
        from data_collection import _parse_diffusion_obs

        task_name = "Isaac-PegasusMoDye-DataCollection-G1-v0"
        env_cfg, agent_cfg = register_task_to_hydra(task_name, "rsl_rl_cfg_entry_point")
        env_cfg.seed = args.seed
        env_cfg.scene.num_envs = 2
        source = env_cfg.commands.motion
        env_cfg.commands.motion = MotionCollectionCommandCfg(
            asset_name=source.asset_name,
            anchor_body_name=source.anchor_body_name,
            body_names=source.body_names,
            motion_files=motion_files,
            samples_per_motion=1,
            default_height=getattr(source, "default_height", 2.0),
            pose_range=getattr(source, "pose_range", {}),
            velocity_range=getattr(source, "velocity_range", {}),
            joint_position_range=getattr(source, "joint_position_range", (-0.52, 0.52)),
            future_steps=getattr(source, "future_steps", 1),
            direction_range=(0.0, 0.0),
            resampling_time_range=(1.0e9, 1.0e9),
            termination_grace_steps=0,
            record_only_termination=False,
        )
        env_cfg.rewards = None
        longest = max(bank.length(index) for index in selected)
        env_cfg.episode_length_s = max(longest * 0.02 + 2.0, 10.0)
        env = gym.make(task_name, cfg=env_cfg)
        unwrapped = env.unwrapped
        wrapped = RslRlVecEnvWrapper(env)

        runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(str(Path(args.teacher_checkpoint).expanduser().resolve()))
        teacher_policy = runner.get_inference_policy(device=unwrapped.device)
        # ExactTextPolicyRuntime now also supports arbitrary phase conditions;
        # the archive argument is only needed for construction compatibility.
        learner = ExactTextPolicyRuntime(
            args.learner_checkpoint, args.phase_conditions,
            device=args.device, replica="ema_agent",
            n_obs_steps=args.n_obs_steps, n_task_steps=4,
        )
        motion_term = unwrapped.command_manager._terms["motion"]

        def observe_all():
            concat = unwrapped.observation_manager.compute_group("diffusion")
            states, _ = _parse_diffusion_obs(concat, SPLIT_691)
            return {key: states[key] for key in POLICY_STATE_FIELDS}

        def init_reference(env_id: int, motion_idx: int, phase: int) -> None:
            safe = min(int(phase), bank.length(motion_idx) - teacher_steps)
            motion_term.dagger_set_canonical_phase(env_id, motion_idx, safe)
            unwrapped.scene.write_data_to_sim()
            unwrapped.sim.forward()

        def terminated(env_id: int) -> bool:
            return bool(unwrapped.termination_manager.terminated[int(env_id)].item())

        adapter = IsaacLabVectorForkAdapter(
            unwrapped, observe_fn=observe_all, command_initializer=init_reference,
            shadow_by_command={motion_idx: 1 for motion_idx in selected},
            live_env_id=0, termination_fn=terminated, restore_atol=args.restore_atol,
        )

        def reset_live(motion_idx: int):
            motion_term.dagger_set_canonical_phase(0, motion_idx, 0)
            ids = torch.tensor([0], dtype=torch.long, device=unwrapped.device)
            motion_term._apply_initialization_randomization(ids)
            unwrapped.episode_length_buf[0] = 0
            unwrapped.scene.write_data_to_sim()
            unwrapped.sim.forward()
            return {
                key: np.asarray(value[0].detach().cpu()).copy()
                for key, value in observe_all().items()
            }

        written = 0
        per_motion = {}
        rejected_terminated = 0
        for motion_idx in selected:
            accepted = 0
            attempts = 0
            while accepted < args.origins_per_motion:
                attempts += 1
                if attempts > args.max_attempts_per_motion:
                    raise RuntimeError(
                        f"motion {motion_idx} reached {accepted}/{args.origins_per_motion} "
                        f"origins after {args.max_attempts_per_motion} attempts"
                    )
                state = reset_live(motion_idx)
                learner.reset_condition(state, bank.at(motion_idx, 0))
                action_history: deque[np.ndarray] = deque(maxlen=args.n_obs_steps - 1)
                for local_step in range(args.max_episode_steps):
                    if accepted >= args.origins_per_motion:
                        break
                    phase = int(motion_term.time_steps[0].item())
                    can_branch = (
                        len(action_history) == args.n_obs_steps - 1
                        and local_step % args.branch_every == 0
                        and phase + teacher_steps <= bank.length(motion_idx)
                    )
                    if can_branch:
                        origin = adapter.snapshot(reference_phase=phase)
                        prefix = learner.conditioned_history_numpy()
                        prefix["motion_idx"] = np.full(
                            args.n_obs_steps, motion_idx, dtype=np.int64
                        )
                        learner_actions = np.stack(tuple(action_history))
                        fork = adapter.fork(origin, motion_idx)
                        teacher_states = []
                        teacher_actions = []
                        failed = False
                        teacher_phases = []
                        for teacher_step in range(teacher_steps):
                            shadow_phase = int(motion_term.time_steps[fork.env_id].item())
                            teacher_phases.append(shadow_phase)
                            teacher_states.append(_conditioned_state(
                                fork.observe(), bank, motion_idx, shadow_phase
                            ))
                            policy_obs, _ = wrapped.get_observations()
                            with torch.inference_mode():
                                action = teacher_policy(policy_obs)[fork.env_id].detach()
                            teacher_actions.append(action.cpu().numpy().copy())
                            if teacher_step + 1 < teacher_steps:
                                fork.step(action)
                                if fork.terminated:
                                    failed = True
                                    rejected_terminated += 1
                                    break
                        adapter.restore(origin)
                        restore_error = adapter.verify_live_restored(origin)
                        if not failed:
                            teacher_dict = {
                                key: np.stack([item[key] for item in teacher_states])
                                for key in teacher_states[0]
                            }
                            window = BranchWindow(
                                student_prefix=prefix,
                                student_actions=learner_actions,
                                teacher_states=teacher_dict,
                                teacher_actions=np.stack(teacher_actions),
                                command_id=motion_idx,
                                role="origin",
                            )
                            target = output / f"branch_{written:06d}_motion{motion_idx:03d}.npz"
                            save_branch_npz(target, window, metadata={
                                "schema": "predactor-paper-dagger-branch-v1",
                                "motion_idx": motion_idx,
                                "motion_name": Path(motion_files[motion_idx]).parent.name,
                                "reference_phase": phase,
                                "teacher_phases": teacher_phases,
                                "live_teacher_commits": 0,
                                "live_learner_commits": adapter.live_learner_commits,
                                "live_auto_resets_prevented": adapter.live_auto_resets_prevented,
                                "live_restore_max_abs": max(restore_error, fork.max_restore_error),
                                "learner_checkpoint": str(Path(args.learner_checkpoint).resolve()),
                            })
                            accepted += 1
                            written += 1
                            print(
                                f"[PaperDAgger] motion={motion_idx:03d} "
                                f"origin={accepted}/{args.origins_per_motion} phase={phase} "
                                f"total={written} restore={restore_error:.3e}"
                            )
                    action = learner.act_conditioned()
                    state = adapter.commit_learner(action)
                    action_history.append(action.copy())
                    next_phase = int(motion_term.time_steps[0].item())
                    learner.append_condition(state, bank.at(motion_idx, next_phase))
                    if terminated(0):
                        break
            per_motion[str(motion_idx)] = accepted

        receipt = {
            "schema": "predactor-paper-dagger-collection-v1",
            "windows": written,
            "selected_motion_ids": list(selected),
            "origins_per_motion": args.origins_per_motion,
            "per_motion": per_motion,
            "live_learner_commits": adapter.live_learner_commits,
            "live_teacher_commits": adapter.live_teacher_commits,
            "live_auto_resets_prevented": adapter.live_auto_resets_prevented,
            "maximum_live_restore_abs": adapter.max_restore_error,
            "teacher_terminated_rejections": rejected_terminated,
            "learner_checkpoint": str(Path(args.learner_checkpoint).resolve()),
            "teacher_checkpoint": str(Path(args.teacher_checkpoint).resolve()),
            "phase_conditions": str(Path(args.phase_conditions).resolve()),
            "seed": args.seed,
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
