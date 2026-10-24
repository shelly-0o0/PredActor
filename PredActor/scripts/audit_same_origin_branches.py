#!/usr/bin/env python3
"""Fail-closed audit for three-command same-origin DAgger archives."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dagger.same_origin import audit_same_origin_directory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Directory containing branch_*.npz triplets")
    parser.add_argument("--output", help="Write an immutable JSON audit report")
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--n-obs-steps", type=int, default=4)
    parser.add_argument("--min-teacher-action-rmse", type=float, default=1.0e-4)
    parser.add_argument("--max-live-restore-error", type=float, default=1.0e-5)
    args = parser.parse_args()
    audits = audit_same_origin_directory(
        args.input,
        horizon=args.horizon,
        n_obs_steps=args.n_obs_steps,
        min_teacher_action_rmse=args.min_teacher_action_rmse,
        max_live_restore_error=args.max_live_restore_error,
    )
    report = {
        "schema": "predactor-same-origin-audit-v1",
        "input": str(Path(args.input).expanduser().resolve()),
        "passed": True,
        "pairs": len(audits),
        "branches": len(audits) * 3,
        "minimum_teacher_action_rmse": min(
            item.min_teacher_action_rmse for item in audits
        ),
        "maximum_live_restore_abs": max(item.live_restore_max_abs for item in audits),
        "results": [item.to_dict() for item in audits],
    }
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        output = Path(args.output).expanduser().resolve()
        if output.exists():
            raise FileExistsError(f"refusing to overwrite audit report: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
