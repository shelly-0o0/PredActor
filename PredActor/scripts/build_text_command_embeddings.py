"""Encode canonical commands with the same normalized CLIP text tower used online."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dagger.text_commands import CANONICAL_COMMANDS


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="MotionCLIP checkpoint used by CLIPTeleop during deployment",
    )
    args = parser.parse_args()
    import clip
    from diffusion_policy.task_provider.clip_teleop import CLIPTeleop

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    provider = CLIPTeleop(
        device=args.device,
        cond_dim=512,
        checkpoint_path=str(checkpoint),
        interactive_input=False,
        default_text=CANONICAL_COMMANDS[0],
    )
    with torch.no_grad():
        tokens = clip.tokenize(list(CANONICAL_COMMANDS)).to(args.device)
        embeddings = provider.clip_model.encode_text(tokens).float()
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
    checkpoint_sha256 = sha256_file(checkpoint)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        commands=np.asarray(CANONICAL_COMMANDS),
        embeddings=embeddings.cpu().numpy().astype(np.float32),
        encoder_checkpoint=np.asarray(str(checkpoint)),
        encoder_checkpoint_sha256=np.asarray(checkpoint_sha256),
    )
    print(
        f"saved {len(CANONICAL_COMMANDS)} exact MotionCLIP text embeddings "
        f"to {output} (checkpoint_sha256={checkpoint_sha256})"
    )


if __name__ == "__main__":
    main()
