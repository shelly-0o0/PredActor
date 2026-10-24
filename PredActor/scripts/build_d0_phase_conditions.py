#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dagger.phase_conditions import build_phase_condition_bank


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d0-zarr", required=True)
    parser.add_argument("--motion-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(build_phase_condition_bank(
        args.d0_zarr, args.motion_root, args.output
    ), indent=2))


if __name__ == "__main__":
    main()
