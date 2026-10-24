"""Fixed task-condition embeddings for controlled policy diagnostics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from diffusion_policy.task_provider.base import TaskCondProvider


class FixedEmbeddingProvider(TaskCondProvider):
    """Inject one named embedding from a safe ``.npz`` archive.

    The archive must contain ``commands`` (one-dimensional strings) and
    ``embeddings`` (two-dimensional, finite vectors).  This provider bypasses
    the text encoder intentionally, so it is suitable for diagnosing whether a
    failure lives in the policy or in the text-to-condition mapping.
    """

    def __init__(self, archive_path: str, command: str):
        source = Path(archive_path).expanduser().resolve()
        with np.load(source, allow_pickle=False) as archive:
            commands = tuple(str(value) for value in archive["commands"].tolist())
            embeddings = np.asarray(archive["embeddings"], dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[0] != len(commands):
            raise ValueError(
                "fixed embedding archive requires commands[N] and embeddings[N,D]"
            )
        if len(set(commands)) != len(commands):
            raise ValueError("fixed embedding archive contains duplicate commands")
        if command not in commands:
            raise ValueError(
                f"command {command!r} is absent from {source}; available={commands}"
            )
        embedding = embeddings[commands.index(command)]
        if not np.all(np.isfinite(embedding)):
            raise ValueError(f"embedding for {command!r} contains non-finite values")
        norm = float(np.linalg.norm(embedding))
        if norm <= 1.0e-8:
            raise ValueError(f"embedding for {command!r} has zero norm")
        self.archive_path = source
        self.command = str(command)
        self.embedding = torch.from_numpy(embedding / norm)

    def _expanded(self, batch_size: int, steps: int, device=None) -> torch.Tensor:
        value = self.embedding
        if device is not None:
            value = value.to(device)
        return value.reshape(1, 1, -1).expand(batch_size, steps, -1)

    def inject_raw_keys(self, next_terms: dict, batch_size: int) -> None:
        device = next(iter(next_terms.values())).device if next_terms else None
        next_terms["motion_latent"] = self._expanded(batch_size, 1, device)

    def get_task_cond(self, raw_history, batch_size, n_task_steps, device=None):
        del raw_history
        return self._expanded(batch_size, n_task_steps, device)
