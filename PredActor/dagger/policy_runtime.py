"""Small exact-text PredActor runtime used by IsaacLab DAgger collection."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

from dagger.checkpoint import (
    checkpoint_replica_state,
    load_checkpoint_payload,
    normalizer_from_checkpoint,
)
from diffusion_policy.dataset.g1_dataset import DataProfile
from diffusion_policy.dataset.term_compose import TermComposer
from diffusion_policy.utils.action_selection import (
    configure_actor_action_selection,
    select_action_for_agent,
)


class ExactTextPolicyRuntime:
    """Run one checkpoint replica from raw G1 terms and a fixed text archive."""

    def __init__(
        self,
        checkpoint: str | Path,
        embedding_archive: str | Path,
        *,
        device: str = "cuda:0",
        replica: str = "ema_agent",
        n_obs_steps: int = 4,
        n_task_steps: int = 4,
        delay_action_step: float = 0.0,
    ) -> None:
        payload = load_checkpoint_payload(checkpoint)
        cfg = OmegaConf.create(OmegaConf.to_container(payload["cfg"], resolve=True))
        self.agent = hydra.utils.instantiate(cfg.policy)
        self.agent.set_normalizer(normalizer_from_checkpoint(checkpoint, replica))
        self.agent.load_state_dict(checkpoint_replica_state(payload, replica), strict=True)
        self.device = torch.device(device)
        self.agent.to(self.device).eval()
        self.n_obs_steps = int(n_obs_steps)
        self.delay_action_step = float(delay_action_step)
        profile_name = OmegaConf.select(
            cfg, "dataset.data_profile", default="g1_prdp_standard_cond"
        )
        profile = DataProfile.from_yaml(profile_name)
        self.composer = TermComposer(
            profile, self.n_obs_steps, n_task_steps=int(n_task_steps)
        )
        if hasattr(self.agent, "set_task_mask_from_profile"):
            self.agent.set_task_mask_from_profile(profile.task_condition)
        if hasattr(self.agent, "set_dropout_from_profile"):
            self.agent.set_dropout_from_profile(profile.task_condition)
        if hasattr(self.agent, "set_task_mask_boundaries"):
            self.agent.set_task_mask_boundaries(self.composer.cond_term_boundaries)
        configure_actor_action_selection(
            self.agent.actor, self.n_obs_steps, self.delay_action_step
        )
        archive = np.load(Path(embedding_archive).expanduser().resolve(), allow_pickle=False)
        if "commands" in archive and "embeddings" in archive:
            commands = tuple(str(value) for value in archive["commands"].tolist())
            embeddings = np.asarray(archive["embeddings"], dtype=np.float32)
            if embeddings.ndim != 2 or len(commands) != len(embeddings):
                raise ValueError("text archive requires commands[N] and embeddings[N,D]")
            if not np.all(np.isfinite(embeddings)):
                raise ValueError("text archive contains non-finite embeddings")
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            if np.any(norms <= 1.0e-8):
                raise ValueError("text archive contains zero-norm embeddings")
            self.embeddings = torch.from_numpy(embeddings / norms)
            self.commands = commands
        else:
            # Paper-style DAgger supplies the phase-conditioned D0 latent via
            # reset_condition/append_condition and needs no fixed text table.
            self.embeddings = torch.empty((0, 0), dtype=torch.float32)
            self.commands = ()
        self.history: dict[str, torch.Tensor] | None = None

    @staticmethod
    def _one_step(state: Mapping[str, np.ndarray]) -> dict[str, torch.Tensor]:
        result = {}
        for key, value in state.items():
            if key == "motion_latent":
                continue
            tensor = torch.as_tensor(np.asarray(value))
            if tensor.dtype == torch.float64:
                tensor = tensor.float()
            result[key] = tensor.unsqueeze(0).unsqueeze(1)
        return result

    def _embedding(self, command_id: int, batch: int = 1) -> torch.Tensor:
        command_id = int(command_id)
        if not 0 <= command_id < len(self.embeddings):
            raise IndexError(f"command_id {command_id} has no text embedding")
        return self.embeddings[command_id].reshape(1, 1, -1).expand(batch, 1, -1)

    def reset(self, state: Mapping[str, np.ndarray], command_id: int) -> None:
        terms = self._one_step(state)
        terms["motion_latent"] = self._embedding(command_id)
        self.history = TermComposer.init_history(terms, self.n_obs_steps)

    def reset_condition(
        self, state: Mapping[str, np.ndarray], condition: np.ndarray,
    ) -> None:
        """Reset with one phase-conditioned MotionCLIP latent from D0."""
        terms = self._one_step(state)
        value = torch.as_tensor(np.asarray(condition), dtype=torch.float32)
        terms["motion_latent"] = value.reshape(1, 1, -1)
        self.history = TermComposer.init_history(terms, self.n_obs_steps)

    def append(self, state: Mapping[str, np.ndarray], command_id: int) -> None:
        if self.history is None:
            self.reset(state, command_id)
            return
        terms = self._one_step(state)
        terms["motion_latent"] = self._embedding(command_id)
        self.history = TermComposer.shift_append(self.history, terms)

    def append_condition(
        self, state: Mapping[str, np.ndarray], condition: np.ndarray,
    ) -> None:
        if self.history is None:
            self.reset_condition(state, condition)
            return
        terms = self._one_step(state)
        value = torch.as_tensor(np.asarray(condition), dtype=torch.float32)
        terms["motion_latent"] = value.reshape(1, 1, -1)
        self.history = TermComposer.shift_append(self.history, terms)

    def act(self, command_id: int) -> np.ndarray:
        if self.history is None:
            raise RuntimeError("policy runtime must be reset before act")
        # Ensure a command switch rewrites the complete condition history, not
        # only the newest row.
        embedding = self._embedding(command_id).to(self.history["motion_latent"].device)
        self.history["motion_latent"] = embedding.expand(
            1, self.n_obs_steps, -1
        ).clone()
        observation = self.composer.compose(self.history).to(self.device)
        condition = self.composer.compose_cond(self.history).to(self.device)
        with torch.no_grad():
            result = self.agent.act({"obs": observation, "cond": condition})
            trajectory = result[0] if isinstance(result, tuple) else result
            action = select_action_for_agent(
                trajectory, self.agent, self.n_obs_steps, self.delay_action_step
            )
        return action[0].detach().cpu().numpy().astype(np.float32, copy=False)

    def act_conditioned(self) -> np.ndarray:
        """Act from the phase-conditioned history without rewriting it."""
        if self.history is None:
            raise RuntimeError("policy runtime must be reset before act")
        observation = self.composer.compose(self.history).to(self.device)
        condition = self.composer.compose_cond(self.history).to(self.device)
        with torch.no_grad():
            result = self.agent.act({"obs": observation, "cond": condition})
            trajectory = result[0] if isinstance(result, tuple) else result
            action = select_action_for_agent(
                trajectory, self.agent, self.n_obs_steps, self.delay_action_step
            )
        return action[0].detach().cpu().numpy().astype(np.float32, copy=False)

    def history_numpy(self, command_id: int) -> dict[str, np.ndarray]:
        if self.history is None:
            raise RuntimeError("policy runtime must be reset before reading history")
        result = {}
        for key, value in self.history.items():
            array = value[0].detach().cpu().numpy().copy()
            if key == "motion_latent":
                array[:] = self.embeddings[int(command_id)].numpy()
            result[key] = array
        return result

    def conditioned_history_numpy(self) -> dict[str, np.ndarray]:
        if self.history is None:
            raise RuntimeError("policy runtime must be reset before reading history")
        return {
            key: value[0].detach().cpu().numpy().copy()
            for key, value in self.history.items()
        }
