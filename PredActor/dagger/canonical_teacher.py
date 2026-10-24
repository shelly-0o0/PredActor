"""Clean canonical-teacher replay mixture and conservative pi0 warm start."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch

from dagger.canonical_dataset import CANONICAL_DATASET_RECEIPT_SCHEMA
from dagger.checkpoint import (
    checkpoint_replica_state,
    load_checkpoint_payload,
    normalizer_from_checkpoint,
)
from dagger.command_map import MotionCommandMap, sha256_file
from dagger.dataset import _weighted_counts
from diffusion_policy.dataset.g1_dataset import G1Dataset
from diffusion_policy.dataset.offline_dataset import OfflineDataset
from diffusion_policy.trainer.bc_training_step import nested_state_sha256
from diffusion_policy.trainer.offline_trainer import OfflineTrainer
from diffusion_policy.utils.pytorch_util import dict_apply
from diffusion_policy.utils.replay_buffer import ReplayBuffer
from diffusion_policy.utils.sampler import SequenceSampler


TARGET_NAVIGATION_COMMANDS = frozenset({
    "walk straight",
    "turn left while walking",
    "turn right while walking",
})


def excluded_motion_ids_for_commands(
    command_map: MotionCommandMap,
    commands=TARGET_NAVIGATION_COMMANDS,
) -> tuple[int, ...]:
    selected = set(commands)
    return tuple(sorted(
        entry.motion_idx
        for entry in command_map.entries
        if selected.intersection(entry.commands)
    ))


class CanonicalTeacherDataset(OfflineDataset):
    """Fixed D0/clean-teacher mixture without rewriting either zarr.

    D0 windows from reviewed straight/turn motions are removed from the replay
    view because their old dyed conditions conflict with the new exact-text
    supervision.  D0 itself remains immutable.
    """

    def __init__(
        self,
        zarr_path: str,
        canonical_zarr_path: str,
        d0_command_map_path: str,
        normalizer_checkpoint: str,
        normalizer_replica: str = "ema_agent",
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
        d0_weight: float = 0.60,
        canonical_weight: float = 0.40,
        epoch_size: int = 100_000,
        **kwargs,
    ):
        if pad_before != 0 or pad_after != 0:
            raise ValueError("canonical teacher training requires complete windows")
        if val_ratio != 0.0:
            raise ValueError("canonical teacher training requires val_ratio=0")
        if symm_aug:
            raise ValueError("runtime reflection would corrupt exact left/right text")
        if not 0.50 <= float(d0_weight) <= 0.70:
            raise ValueError("D0 weight must be in [0.50, 0.70]")
        if not 0.30 <= float(canonical_weight) <= 0.50:
            raise ValueError("canonical weight must be in [0.30, 0.50]")
        if abs(float(d0_weight) + float(canonical_weight) - 1.0) > 1.0e-6:
            raise ValueError("D0 and canonical weights must sum to 1")
        if normalizer_replica not in {"agent", "ema_agent"}:
            raise ValueError("normalizer_replica must be agent or ema_agent")

        self._zarr_path = Path(zarr_path).expanduser().resolve()
        self.canonical_zarr_path = Path(canonical_zarr_path).expanduser().resolve()
        self._load_to_memory = bool(load_to_memory)
        self.horizon = int(horizon)
        self.n_obs_steps = int(n_obs_steps)
        self.n_task_steps = int(n_task_steps)
        self.symm_aug = False
        self.seed = int(seed)
        self.normalizer_checkpoint = Path(
            normalizer_checkpoint
        ).expanduser().resolve()
        self.normalizer_replica = str(normalizer_replica)
        self.command_map = MotionCommandMap.load(
            d0_command_map_path, require_reviewed=True
        )
        self.excluded_d0_motion_ids = excluded_motion_ids_for_commands(
            self.command_map
        )
        self.canonical_receipt_path = self.canonical_zarr_path.with_suffix(
            ".receipt.json"
        )
        self._validate_canonical_receipt()

        child_kwargs = dict(kwargs)
        child_kwargs.pop("data_sample_rate", None)
        child_kwargs.update(
            data_profile=data_profile,
            horizon=horizon,
            n_obs_steps=n_obs_steps,
            n_task_steps=n_task_steps,
            pad_before=0,
            pad_after=0,
            seed=seed,
            val_ratio=0.0,
            load_to_memory=load_to_memory,
            symm_aug=False,
            use_normalizer_cache=False,
        )
        self.d0 = G1Dataset(zarr_path=str(self._zarr_path), **child_kwargs)
        self.canonical = G1Dataset(
            zarr_path=str(self.canonical_zarr_path), **child_kwargs
        )
        self.datasets = (self.d0, self.canonical)
        self.primary = self.d0
        self.data_profile = self.d0.data_profile
        self.term_composer = self.d0.term_composer
        self.obs_reflect_op = self.d0.obs_reflect_op
        self.action_reflect_op = self.d0.action_reflect_op
        self.cond_reflect_op = self.d0.cond_reflect_op
        self.ee_idxs = self.d0.ee_idxs

        d0_origin_rows = (
            np.asarray(self.d0.sampler.indices[:, 0], dtype=np.int64)
            + self.n_obs_steps - 1
        )
        d0_motion_ids = np.rint(np.asarray(
            self.d0.replay_buffer["motion_idx"][:]
        ).reshape(-1)).astype(np.int64)
        self._eligible_d0_indices = np.flatnonzero(~np.isin(
            d0_motion_ids[d0_origin_rows], self.excluded_d0_motion_ids
        )).astype(np.int64)
        if not len(self._eligible_d0_indices):
            raise ValueError("D0 filtering removed every complete window")
        if not len(self.canonical):
            raise ValueError("canonical dataset contains no complete window")

        total = int(epoch_size)
        self.source_counts = _weighted_counts(
            [float(d0_weight), float(canonical_weight)], total
        )
        rng = np.random.default_rng(self.seed)
        self._d0_indices = rng.choice(
            self._eligible_d0_indices,
            size=self.source_counts[0],
            replace=self.source_counts[0] > len(self._eligible_d0_indices),
        ).astype(np.int64)
        self._canonical_indices = self._balanced_canonical_indices(
            rng, self.source_counts[1]
        )
        self._ends = np.cumsum(self.source_counts)
        self._base_keys = set(self.d0[0].keys())
        print(
            "[CanonicalTeacherDataset] "
            f"D0={self.source_counts[0]}/{total} ({float(d0_weight):.1%}), "
            f"canonical={self.source_counts[1]}/{total} "
            f"({float(canonical_weight):.1%}), excluded_d0_motion_ids="
            f"{list(self.excluded_d0_motion_ids)}"
        )

    def _validate_canonical_receipt(self) -> None:
        if not self.canonical_receipt_path.is_file():
            raise FileNotFoundError(
                f"canonical dataset lacks receipt: {self.canonical_receipt_path}"
            )
        receipt = json.loads(
            self.canonical_receipt_path.read_text(encoding="utf-8")
        )
        if receipt.get("schema") != CANONICAL_DATASET_RECEIPT_SCHEMA:
            raise ValueError("unsupported canonical dataset receipt")
        if Path(receipt.get("output", "")).resolve() != self.canonical_zarr_path:
            raise ValueError("canonical receipt output does not match dataset")
        if receipt.get("labels_are_exact_text_embeddings") is not True:
            raise ValueError("canonical labels are not certified exact text embeddings")
        if receipt.get("actions_are_rltracker_teacher_targets") is not True:
            raise ValueError("canonical actions are not certified teacher targets")
        if receipt.get("learner_actions_used_as_targets") is not False:
            raise ValueError("learner actions must never be canonical targets")

    def _balanced_canonical_indices(
        self, rng: np.random.Generator, total: int
    ) -> np.ndarray:
        origin_rows = (
            np.asarray(self.canonical.sampler.indices[:, 0], dtype=np.int64)
            + self.n_obs_steps - 1
        )
        command_ids = np.asarray(
            self.canonical.replay_buffer["command_id"][:]
        ).reshape(-1).astype(np.int64)
        window_commands = command_ids[origin_rows]
        unique = sorted(set(window_commands.tolist()))
        if len(unique) != 3:
            raise ValueError(
                f"canonical dataset must have exactly three commands, got {unique}"
            )
        counts = _weighted_counts([1.0 / len(unique)] * len(unique), total)
        selected = []
        for command_id, count in zip(unique, counts):
            candidates = np.flatnonzero(window_commands == command_id)
            selected.append(rng.choice(
                candidates, size=count, replace=count > len(candidates)
            ))
        result = np.concatenate(selected).astype(np.int64)
        return result[rng.permutation(len(result))]

    def __len__(self) -> int:
        return int(self._ends[-1])

    @staticmethod
    def _sample(dataset: G1Dataset, index: int) -> dict[str, torch.Tensor]:
        sample = dataset.sampler.sample_sequence(int(index))
        return dict_apply(dataset._sample_to_data(sample), torch.from_numpy)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        if index < self.source_counts[0]:
            item = self._sample(self.d0, self._d0_indices[index])
            source_id = 0
            command_id = torch.full((self.horizon,), -1, dtype=torch.long)
        else:
            local = index - self.source_counts[0]
            item = self._sample(
                self.canonical, self._canonical_indices[local]
            )
            source_id = 1
            command_id = item["command_id"].long().reshape(self.horizon)
        result = {key: value for key, value in item.items() if key in self._base_keys}
        result["action_valid"] = torch.ones(self.horizon, dtype=torch.bool)
        result["_source_index"] = torch.tensor(source_id, dtype=torch.long)
        result["_command_id"] = command_id
        return result

    def collate_fn(self, batch: list[dict[str, torch.Tensor]], normalize: bool = False):
        items = [dict(item) for item in batch]
        source_ids = torch.stack([item.pop("_source_index") for item in items])
        command_ids = torch.stack([item.pop("_command_id") for item in items])
        result = self.d0.collate_fn(items, normalize=normalize)
        result["source_id"] = source_ids
        result["command_id"] = command_ids
        return result

    def get_normalizer(self, **kwargs):
        del kwargs
        print(
            "[CanonicalTeacherDataset] Loading frozen pi0 normalizer from "
            f"{self.normalizer_checkpoint} ({self.normalizer_replica})"
        )
        return normalizer_from_checkpoint(
            self.normalizer_checkpoint, self.normalizer_replica
        )

    def get_validation_dataset(self):
        return self.d0.get_validation_dataset()

    def get_all_actions(self):
        return self.d0.get_all_actions()

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
            old_sampler = child.sampler
            child.replay_buffer = replay
            child.sampler = SequenceSampler(
                replay_buffer=replay,
                sequence_length=child.horizon,
                pad_before=child.pad_before,
                pad_after=child.pad_after,
                keys=old_sampler.keys,
                key_first_k=dict(old_sampler.key_first_k),
                episode_mask=child.train_mask,
            )


class CanonicalTeacherTrainer(OfflineTrainer):
    """Warm start from EMA pi0 and update only the condition/cross-attention path."""

    @staticmethod
    def _condition_parameter(name: str) -> bool:
        return (
            name == "backbone.cond_pos_emb"
            or name.startswith("backbone.cond_obs_emb.")
            or name.startswith("backbone.encoder.")
            or (
                name.startswith("backbone.decoder.layers.")
                and (".multihead_attn." in name or ".norm2." in name)
            )
        )

    def _after_training_data_ready(self, cfg, dataset, normalizer) -> None:
        if not isinstance(dataset, CanonicalTeacherDataset):
            raise TypeError("CanonicalTeacherTrainer requires CanonicalTeacherDataset")
        if cfg.training.resume:
            raise ValueError("canonical teacher training uses init_checkpoint, not resume")
        epochs = int(cfg.training.num_epochs)
        learning_rate = float(cfg.optimizer.learning_rate)
        if not 1 <= epochs <= 4:
            raise ValueError("canonical teacher training requires 1-4 epochs")
        if not 1.0e-5 <= learning_rate <= 2.0e-5:
            raise ValueError("canonical learning rate must be in [1e-5, 2e-5]")
        if cfg.training.use_ema:
            raise ValueError("canonical short training must publish one student")

        checkpoint = Path(str(cfg.canonical.init_checkpoint)).expanduser().resolve()
        if checkpoint != dataset.normalizer_checkpoint:
            raise ValueError("init and normalizer checkpoints must be identical")
        replica = str(cfg.canonical.get("init_replica", dataset.normalizer_replica))
        if replica != dataset.normalizer_replica:
            raise ValueError("weight and normalizer replicas must be identical")
        state = checkpoint_replica_state(load_checkpoint_payload(checkpoint), replica)
        self.agent.load_state_dict(state, strict=True)
        expected_normalizer_hash = nested_state_sha256(normalizer.state_dict())
        loaded_normalizer_hash = nested_state_sha256(
            self.agent.normalizer.state_dict()
        )
        if loaded_normalizer_hash != expected_normalizer_hash:
            raise RuntimeError("pi0 normalizer changed during weight initialization")

        # The frozen pi0 normalizer is attached after the optimizer is created
        # and stores statistics as nn.Parameters.  Freeze the complete agent
        # first so those statistics (and any future auxiliary modules) cannot
        # accidentally enter the trainable set.
        for parameter in self.agent.parameters():
            parameter.requires_grad_(False)

        trainable_names = []
        for name, parameter in self.agent.actor.named_parameters():
            trainable = self._condition_parameter(name)
            parameter.requires_grad_(trainable)
            if trainable:
                trainable_names.append(f"actor.{name}")
        if self.agent.cond_compressor is not None:
            for name, parameter in self.agent.cond_compressor.named_parameters():
                parameter.requires_grad_(True)
                trainable_names.append(f"cond_compressor.{name}")
        if not trainable_names:
            raise RuntimeError("condition/cross-attention scope selected no parameters")

        trainable_ids = {
            id(parameter)
            for parameter in self.agent.parameters()
            if parameter.requires_grad
        }
        for optimizer in self.optimizer.values():
            for group in optimizer.param_groups:
                group["params"] = [
                    parameter for parameter in group["params"]
                    if id(parameter) in trainable_ids
                ]
        optimizer_ids = {
            id(parameter)
            for optimizer in self.optimizer.values()
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        if optimizer_ids != trainable_ids:
            raise RuntimeError("optimizer scope differs from trainable parameter scope")
        normalizer_trainable = [
            name for name, parameter in self.agent.normalizer.named_parameters()
            if parameter.requires_grad
        ]
        if normalizer_trainable:
            raise RuntimeError(f"pi0 normalizer is trainable: {normalizer_trainable}")

        source_command_ids = np.asarray(
            dataset.canonical.replay_buffer["command_id"][:]
        ).reshape(-1).astype(np.int64)
        audit = {
            "schema": "predactor-canonical-teacher-train-v1",
            "init_checkpoint": str(checkpoint),
            "init_replica": replica,
            "normalizer_sha256": loaded_normalizer_hash,
            "normalizer_frozen": True,
            "optimizer_state_loaded": False,
            "trainable_scope": "condition_encoder_and_decoder_cross_attention",
            "trainable_parameter_count": int(sum(
                parameter.numel()
                for parameter in self.agent.parameters()
                if parameter.requires_grad
            )),
            "trainable_parameters": trainable_names,
            "d0_zarr": str(dataset._zarr_path),
            "canonical_zarr": str(dataset.canonical_zarr_path),
            "canonical_receipt": str(dataset.canonical_receipt_path),
            "canonical_receipt_sha256": sha256_file(dataset.canonical_receipt_path),
            "d0_command_map": str(dataset.command_map.path),
            "d0_command_map_sha256": dataset.command_map.sha256,
            "excluded_d0_motion_ids": list(dataset.excluded_d0_motion_ids),
            "source_counts": {
                "d0": dataset.source_counts[0],
                "canonical_teacher": dataset.source_counts[1],
            },
            "canonical_frames_by_command_id": {
                str(command_id): count
                for command_id, count in Counter(source_command_ids.tolist()).items()
            },
            "learning_rate": learning_rate,
            "num_epochs": epochs,
        }
        output = Path(self.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "canonical_teacher_initialization.json").write_text(
            json.dumps(audit, indent=2) + "\n", encoding="utf-8"
        )
        print(
            "[CanonicalTeacherTrainer] Initialized condition/cross-attention "
            f"path from pi0 ({replica}); {audit['trainable_parameter_count']} "
            "parameters trainable; normalizer frozen."
        )
