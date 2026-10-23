#!/usr/bin/env python3
"""Run the fixed policy benchmark for one checkpoint, with resume support."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dagger.policy_benchmark import PolicyBenchmarkSuite, rollout_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--motionclip-checkpoint")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--checkpoint-replica", choices=("agent", "ema_agent"), default="agent")
    parser.add_argument("--precision", choices=("fp32", "fp16"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    config = Path(args.config).expanduser().resolve()
    suite_path = Path(args.suite).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    suite = PolicyBenchmarkSuite.load(suite_path)

    commands = []
    for case in suite.all_commands:
        for seed in suite.seeds:
            trajectory = rollout_path(output, case, seed)
            if trajectory.exists() and not args.overwrite:
                print(f"[resume] keep {trajectory}")
                continue
            runtime_dir = output / "runtime" / case.id / f"seed-{seed}"
            command = [
                sys.executable, str(ROOT / "cond_eval.py"),
                "--checkpoint", str(checkpoint),
                "--config", str(config),
                "--output_dir", str(runtime_dir),
                "--device", args.device,
                "--checkpoint-replica", args.checkpoint_replica,
                "--text-command", case.prompt,
                "--seed", str(seed),
                "--trajectory-rollout", str(trajectory),
                "--rollout-seconds", str(suite.duration_seconds),
                "--headless", "--no-web-ui",
            ]
            if args.precision:
                command.extend(("--precision", args.precision))
            commands.append(command)

    for index, command in enumerate(commands, start=1):
        print(f"[{index}/{len(commands)}] {' '.join(command)}")
        if not args.dry_run:
            subprocess.run(command, cwd=REPOSITORY, check=True)

    report = output / "policy_benchmark.json"
    evaluate = [
        sys.executable, str(ROOT / "scripts" / "evaluate_policy_benchmark.py"),
        "--suite", str(suite_path),
        "--rollout-dir", str(output),
        "--device", args.device,
        "--checkpoint", str(checkpoint),
        "--output", str(report),
        "--require-complete",
    ]
    if args.motionclip_checkpoint:
        evaluate.extend((
            "--motionclip-checkpoint",
            str(Path(args.motionclip_checkpoint).expanduser().resolve()),
        ))
    print(f"[score] {' '.join(evaluate)}")
    if not args.dry_run:
        subprocess.run(evaluate, cwd=REPOSITORY, check=True)


if __name__ == "__main__":
    main()
