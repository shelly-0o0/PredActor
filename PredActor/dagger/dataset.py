"""Stratified D0/DAgger dataset without copying or merging zarr archives."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from dagger.checkpoint import normalizer_from_checkpoint
from dagger.manifest import DaggerManifest
from diffusion_policy.dataset.g1_dataset import G1Dataset
from diffusion_policy.dataset.offline_dataset import OfflineDataset
from diffusion_policy.utils.replay_buffer import ReplayBuffer
from diffusion_policy.utils.sampler import SequenceSampler


# Raw fields consumed by the active G1 profile plus auditable task identity.
# D0 also stores motion_ref_* telemetry that is not an input to this policy;
# requiring those fields in D_i would copy teacher-only diagnostics into every
# DAgger shard without affecting the model.
MODEL_RAW_KEYS = {
    "body_pos", "body_rot", "body_lin_vel", "body_ang_vel",
    "joint_pos", "joint_vel", "root_pos", "root_rot",
    "projected_gravity", "base_lin_vel", "base_ang_vel", "last_actions",
    "motion_latent", "motion_idx", "action",
}


def _weighted_counts(weights: list[float], total: int) -> list[int]:
    if total < len(weights):
        raise ValueError("epoch_size must be at least the number of sources")
    raw = np.asarray(weights, dtype=np.float64) * total
    counts = np.floor(raw).astype(np.int64)
    remainder = total - int(counts.sum())
    order = np.argsort(-(raw - counts), kind="stable")
    counts[order[:remainder]] += 1
    if np.any(counts == 0):
        raise ValueError("epoch_size gives a positive-weight source zero samples")
    return counts.tolist()


class StrictDaggerDataset(OfflineDataset):
    """Weighted view over immutable source zarrs described by a manifest.

    D0 and new DAgger data are sampled at fixed proportions. No aggregate zarr
    is created, so another round only adds one source entry to the manifest.
    """

    def __init__(
        self,
        zarr_path: str,
        data_profile=None,
        horizon: int = 20,
        n_obs_steps: int = 4,
        n_task_steps: int = 4,
        pad_before: int = 0,
        pad_after: int = 0,
        seed: int = 42,
        val_ratio: float = 0.0,
        load_to_memory: bool = False,
        symm_aug: bool = False,
        epoch_size: int | None = None,
        sampling_mode: str = "weighted",
        use_action_valid_mask: bool = True,
        text_replacement_probability: float = 0.35,
        **kwargs,
    ):
        self.manifest = DaggerManifest.load(zarr_path)
        self._zarr_path = Path(zarr_path)
        self._load_to_memory = bool(load_to_memory)
        self.horizon = int(horizon)
        self.n_obs_steps = int(n_obs_steps)
        self.n_task_steps = int(n_task_steps)
        self.symm_aug = bool(symm_aug)
        if self.symm_aug:
            raise ValueError(
                "StrictDaggerDataset forbids runtime symmetry augmentation: CLIP "
                "left/right semantics cannot be reflected by a matrix. Store "
                "mirrored samples with swapped command_id in a counterfactual shard."
            )
        self.seed = int(seed)
        if sampling_mode not in {"weighted", "aggregate"}:
            raise ValueError("sampling_mode must be 'weighted' or 'aggregate'")
        self.sampling_mode = sampling_mode
        self.use_action_valid_mask = bool(use_action_valid_mask)
        probability = float(text_replacement_probability)
        if not 0.0 <= probability <= 1.0:
            raise ValueError("text_replacement_probability must be in [0, 1]")
        self.text_replacement_probability = probability

        child_kwargs = dict(kwargs)
        child_kwargs.update(
            data_profile=data_profile,
            horizon=horizon,
            n_obs_steps=n_obs_steps,
            n_task_steps=n_task_steps,
            seed=seed,
            load_to_memory=load_to_memory,
            symm_aug=False,
            use_normalizer_cache=False,
        )
        self.datasets = [
            G1Dataset(
                zarr_path=str(source.path),
                pad_before=pad_before if source.role == "d0" else 0,
                pad_after=pad_after if source.role == "d0" else 0,
                val_ratio=val_ratio if source.role == "d0" else 0.0,
                **child_kwargs,
            )
            for source in self.manifest.sources
        ]
        self._d0_index = next(
            index
            for index, source in enumerate(self.manifest.sources)
            if source.role == "d0"
        )
        if any(len(dataset) == 0 for dataset in self.datasets):
            empty = [
                self.manifest.sources[index].name
                for index, dataset in enumerate(self.datasets) if len(dataset) == 0
            ]
            raise ValueError(f"DAgger sources contain no training windows: {empty}")

        primary = self.datasets[self._d0_index]
        self.data_profile = primary.data_profile
        self.term_composer = primary.term_composer
        self.obs_reflect_op = primary.obs_reflect_op
        self.action_reflect_op = primary.action_reflect_op
        self.cond_reflect_op = primary.cond_reflect_op
        self.ee_idxs = primary.ee_idxs
        available = (set(primary.replay_buffer.keys()) - {"act"}) | {"action"}
        missing_model_fields = MODEL_RAW_KEYS - available
        if missing_model_fields:
            raise ValueError(
                f"D0 lacks model-required raw fields: {sorted(missing_model_fields)}"
            )
        self._base_keys = set(MODEL_RAW_KEYS)

        if self.sampling_mode == "aggregate":
            if epoch_size is not None:
                raise ValueError("aggregate sampling uses each available window once")
            self.source_counts = [len(dataset) for dataset in self.datasets]
            total = sum(self.source_counts)
        else:
            total = int(epoch_size) if epoch_size is not None else sum(map(len, self.datasets))
            self.source_counts = _weighted_counts(
                [source.weight for source in self.manifest.sources], total
            )
        self._ends = np.cumsum(self.source_counts)

        self._text_embeddings = None
        if self.manifest.text_embeddings is not None:
            archive = np.load(self.manifest.text_embeddings, allow_pickle=False)
            commands = tuple(str(value) for value in archive["commands"].tolist())
            embeddings = np.asarray(archive["embeddings"], dtype=np.float32)
            if self.manifest.text_commands and commands != self.manifest.text_commands:
                raise ValueError("text command order differs between manifest and embedding archive")
            if embeddings.ndim != 2 or embeddings.shape[0] != len(commands):
                raise ValueError("text embedding archive has inconsistent shapes")
            self._text_embeddings = torch.from_numpy(embeddings)

        print(
            f"[StrictDaggerDataset] sampling_mode={self.sampling_mode}; "
            + ", ".join(
                f"{source.name}:{count}/{total} ({count / total:.4%}, {source.role})"
                for source, count in zip(self.manifest.sources, self.source_counts)
            )
        )

    def __len__(self) -> int:
        return int(self._ends[-1])

    def _locate(self, index: int) -> tuple[int, int]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        source_index = int(np.searchsorted(self._ends, index, side="right"))
        start = 0 if source_index == 0 else int(self._ends[source_index - 1])
        local = (index - start) % len(self.datasets[source_index])
        return source_index, local

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        source_index, local_index = self._locate(index)
        item = self.datasets[source_index][local_index]
        role = self.manifest.sources[source_index].role
        result = {key: value for key, value in item.items() if key in self._base_keys}
        if role != "d0" and self.use_action_valid_mask:
            if "action_valid" not in item:
                raise KeyError(
                    f"strict DAgger source {self.manifest.sources[source_index].name!r} "
                    "is missing required 'action_valid' labels"
                )
            result["action_valid"] = item["action_valid"].bool().reshape(self.horizon)
        command_id = item.get("command_id")
        if role != "d0" and command_id is None:
            raise KeyError(
                f"strict DAgger source {self.manifest.sources[source_index].name!r} "
                "is missing required 'command_id' labels"
            )
        if command_id is None:
            command_id = torch.full((self.horizon,), -1, dtype=torch.long)
        result["command_id"] = command_id.long().reshape(self.horizon)
        result["_source_index"] = torch.tensor(source_index, dtype=torch.long)
        return result

    def _replace_text_conditions(self, batch: list[dict[str, torch.Tensor]]) -> None:
        if self._text_embeddings is None or self.text_replacement_probability <= 0:
            return
        origin = self.n_obs_steps - 1
        for item in batch:
            if "motion_latent" not in item or torch.rand(()).item() >= self.text_replacement_probability:
                continue
            command_id = int(item["command_id"][origin].item())
            if command_id < 0:
                continue
            if command_id >= len(self._text_embeddings):
                raise IndexError(f"command_id {command_id} has no exact text embedding")
            embedding = self._text_embeddings[command_id].to(item["motion_latent"].dtype)
            item["motion_latent"] = embedding.expand_as(item["motion_latent"]).clone()

    def collate_fn(self, batch: list[dict[str, torch.Tensor]], normalize: bool = False) -> dict:
        items = [dict(item) for item in batch]
        source_ids = torch.stack([item.pop("_source_index") for item in items])
        self._replace_text_conditions(items)
        result = self.datasets[self._d0_index].collate_fn(items, normalize=normalize)
        result["source_id"] = source_ids
        return result

    def get_normalizer(self, **kwargs):
        del kwargs
        print(
            "[StrictDaggerDataset] Loading frozen pi0 normalizer from "
            f"{self.manifest.normalizer_checkpoint} ({self.manifest.normalizer_replica})"
        )
        return normalizer_from_checkpoint(
            self.manifest.normalizer_checkpoint,
            self.manifest.normalizer_replica,
        )

    def get_validation_dataset(self):
        # Validation remains an immutable D0 holdout, not a changing mixture.
        return self.datasets[self._d0_index].get_validation_dataset()

    def get_all_actions(self):
        return self.datasets[self._d0_index].get_all_actions()

    @staticmethod
    def worker_init_fn(worker_id: int) -> None:
        del worker_id
        info = torch.utils.data.get_worker_info()
        if info is None:
            return
        dataset = info.dataset
        for child in dataset.datasets:
            replay = (
                ReplayBuffer.copy_from_path(str(child._zarr_path))
                if child._load_to_memory
                else ReplayBuffer.create_from_path(str(child._zarr_path), mode="r")
            )
            child.replay_buffer = replay
            child.sampler = SequenceSampler(
                replay_buffer=replay,
                sequence_length=child.horizon,
                pad_before=child.pad_before,
                pad_after=child.pad_after,
                episode_mask=child.train_mask,
            )
