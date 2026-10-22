"""Checkpoint helpers shared by DAgger training and evaluation."""

from __future__ import annotations

from pathlib import Path

import torch

from diffusion_policy.trainer.normalizer import LinearNormalizer


def load_checkpoint_payload(path: str | Path) -> dict:
    return torch.load(Path(path), map_location="cpu", weights_only=False)


def checkpoint_replica_state(payload: dict, replica: str) -> dict:
    states = payload.get("state_dicts", {})
    if replica not in states:
        raise KeyError(f"checkpoint has no {replica!r} state; available={sorted(states)}")
    return states[replica]


def normalizer_from_checkpoint(path: str | Path, replica: str = "agent") -> LinearNormalizer:
    state = checkpoint_replica_state(load_checkpoint_payload(path), replica)
    prefix = "normalizer."
    normalizer_state = {
        key[len(prefix):]: value
        for key, value in state.items()
        if key.startswith(prefix)
    }
    if not normalizer_state:
        raise ValueError(f"checkpoint {path} does not contain an embedded normalizer")
    normalizer = LinearNormalizer()
    normalizer.load_state_dict(normalizer_state, strict=True)
    required = {"obs", "action", "cond"}
    missing = required - set(normalizer.params_dict)
    if missing:
        raise ValueError(f"checkpoint normalizer is missing fields: {sorted(missing)}")
    return normalizer
