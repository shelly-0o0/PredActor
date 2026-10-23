"""MotionCLIP retrieval adapter used by the policy benchmark."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch

from dagger.policy_benchmark import cosine_top1
from diffusion_policy.dataset.g1_dataset import BODY_NAMES


MOTIONCLIP_BODY_NAMES = [
    "pelvis",
    "left_hip_pitch_link", "left_hip_roll_link", "left_hip_yaw_link",
    "left_knee_link", "left_ankle_pitch_link", "left_ankle_roll_link",
    "right_hip_pitch_link", "right_hip_roll_link", "right_hip_yaw_link",
    "right_knee_link", "right_ankle_pitch_link", "right_ankle_roll_link",
    "waist_yaw_link", "waist_roll_link", "torso_link",
    "left_shoulder_pitch_link", "left_shoulder_roll_link",
    "left_shoulder_yaw_link", "left_elbow_link", "left_wrist_roll_link",
    "left_wrist_pitch_link", "left_wrist_yaw_link",
    "right_shoulder_pitch_link", "right_shoulder_roll_link",
    "right_shoulder_yaw_link", "right_elbow_link", "right_wrist_roll_link",
    "right_wrist_pitch_link", "right_wrist_yaw_link",
]


def _runtime_loader():
    dye_dir = (
        Path(__file__).resolve().parents[1]
        / "collection" / "scripts" / "data_dyeing"
    )
    if str(dye_dir) not in sys.path:
        sys.path.insert(0, str(dye_dir))
    from motionclip_runtime import get_motion_clip

    return get_motion_clip


def encode_and_retrieve(
    windows: list[np.ndarray],
    prompts: list[str],
    *,
    checkpoint: str | Path,
    device: str,
) -> dict:
    """Encode complete motion windows and return raw top-1 prompt retrieval."""
    if not windows:
        return {
            "predicted_indices": np.empty((0,), dtype=np.int64),
            "scores": np.empty((0, len(prompts)), dtype=np.float32),
        }
    import clip

    get_motion_clip = _runtime_loader()
    model, model_cfg = get_motion_clip(str(checkpoint), device=device)
    expected_frames = int(model_cfg.model.num_frames)
    if any(np.asarray(window).shape != (expected_frames, 30, 3) for window in windows):
        raise ValueError(
            f"MotionCLIP expects every window to be [{expected_frames}, 30, 3]"
        )
    mapping = [BODY_NAMES.index(name) for name in MOTIONCLIP_BODY_NAMES]
    motion = np.stack(windows).astype(np.float32, copy=False)[:, :, mapping, :]
    motion = motion - motion[:, :1, :1, :]
    tensor = torch.from_numpy(motion).to(device).permute(0, 2, 3, 1).contiguous()
    lengths = torch.full(
        (len(windows),), expected_frames, dtype=torch.long, device=device
    )
    with torch.no_grad():
        encoded = model.encoder({
            "x": tensor,
            "y": torch.zeros(len(windows), dtype=torch.long, device=device),
            "mask": model.lengths_to_mask(lengths),
        })["mu"].float()
        tokens = clip.tokenize(prompts).to(device)
        text = model.clip_model.encode_text(tokens).float()
    predicted, scores = cosine_top1(
        encoded.cpu().numpy(), text.cpu().numpy()
    )
    return {"predicted_indices": predicted, "scores": scores}
