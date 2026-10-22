"""Strict DAgger checkpoint loader used by cond_eval and collectors."""

from __future__ import annotations

import copy

import hydra
import torch
from omegaconf import OmegaConf

from dagger.checkpoint import checkpoint_replica_state, load_checkpoint_payload


def load_student_agent(checkpoint, device, actor_config=None):
    payload = load_checkpoint_payload(checkpoint)
    cfg = OmegaConf.create(OmegaConf.to_container(payload["cfg"], resolve=True))
    policy_cfg = copy.deepcopy(cfg.policy)
    if actor_config is not None:
        checkpoint_target = OmegaConf.select(policy_cfg, "actor._target_")
        requested_target = OmegaConf.select(actor_config, "_target_")
        if checkpoint_target != requested_target:
            raise ValueError(
                f"actor mismatch: checkpoint={checkpoint_target!r}, requested={requested_target!r}"
            )
        policy_cfg.actor = copy.deepcopy(actor_config)
    agent = hydra.utils.instantiate(policy_cfg)
    agent.load_state_dict(checkpoint_replica_state(payload, "agent"), strict=True)
    agent.to(torch.device(device))
    agent.eval()
    return agent
