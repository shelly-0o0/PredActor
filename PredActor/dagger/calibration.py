"""Exact-text calibration dataset and low-rate warm-start trainer."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch

from dagger.checkpoint import (
    checkpoint_replica_state,
    load_checkpoint_payload,
    normalizer_from_checkpoint,
)
from dagger.command_map import MotionCommandMap, sha256_file
from dagger.dataset import _weighted_counts
from dagger.text_commands import CANONICAL_COMMANDS, COMMAND_TO_ID
from diffusion_policy.dataset.g1_dataset import G1Dataset
from diffusion_policy.dataset.offline_dataset import OfflineDataset
from diffusion_policy.trainer.bc_training_step import nested_state_sha256
from diffusion_policy.trainer.offline_trainer import OfflineTrainer
from diffusion_policy.utils.pytorch_util import dict_apply
from diffusion_policy.utils.replay_buffer import ReplayBuffer
from diffusion_policy.utils.sampler import SequenceSampler


class ExactTextCalibrationDataset(OfflineDataset):
    """A fixed D0/exact-text mixture backed by one immutable D0 zarr.

    The exact-text view changes only ``motion_latent`` in memory. It does not
    duplicate, rewrite, or aggregate D0 on disk. One motion window may appear
    under more than one exact prompt when the reviewed map explicitly permits
    it (currently ``walk`` and ``walk straight``).
    """

    def __init__(
        self,
        zarr_path: str,
        command_map_path: str,
        text_embeddings_path: str,
        normalizer_checkpoint: str,
        audit_report_path: str,
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
        d0_weight: float = 0.65,
        calibration_weight: float = 0.35,
        epoch_size: int = 200_000,
        **kwargs,
    ):
        if pad_before != 0 or pad_after != 0:
            raise ValueError("exact-text calibration requires unpadded complete windows")
        if val_ratio != 0:
            raise ValueError("exact-text calibration uses immutable D0 with val_ratio=0")
        if symm_aug:
            raise ValueError(
                "runtime reflection would leave CLIP left/right text unchanged; "
                "exact-text calibration requires symm_aug=false"
            )
        if not 0.50 <= float(d0_weight) <= 0.70:
            raise ValueError("D0 weight must be in [0.50, 0.70]")
        if not 0.30 <= float(calibration_weight) <= 0.50:
            raise ValueError("calibration weight must be in [0.30, 0.50]")
        if abs(float(d0_weight) + float(calibration_weight) - 1.0) > 1.0e-6:
            raise ValueError("D0 and calibration weights must sum to 1.0")
        if normalizer_replica not in {"agent", "ema_agent"}:
            raise ValueError("normalizer_replica must be 'agent' or 'ema_agent'")

        self._zarr_path = Path(zarr_path).expanduser().resolve()
        self._load_to_memory = bool(load_to_memory)
        self.horizon = int(horizon)
        self.n_obs_steps = int(n_obs_steps)
        self.n_task_steps = int(n_task_steps)
        self.symm_aug = False
        self.seed = int(seed)
        self.normalizer_checkpoint = Path(normalizer_checkpoint).expanduser().resolve()
        self.normalizer_replica = str(normalizer_replica)
        self.command_map = MotionCommandMap.load(command_map_path, require_reviewed=True)
        self.audit_report_path = Path(audit_report_path).expanduser().resolve()
        self._validate_audit_report()

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
        self.primary = G1Dataset(zarr_path=str(self._zarr_path), **child_kwargs)
        if len(self.primary) == 0:
            raise ValueError("D0 contains no complete calibration windows")

        self.data_profile = self.primary.data_profile
        self.term_composer = self.primary.term_composer
        self.obs_reflect_op = self.primary.obs_reflect_op
        self.action_reflect_op = self.primary.action_reflect_op
        self.cond_reflect_op = self.primary.cond_reflect_op
        self.ee_idxs = self.primary.ee_idxs

        self.text_embeddings_path = Path(text_embeddings_path).expanduser().resolve()
        with np.load(self.text_embeddings_path, allow_pickle=False) as archive:
            commands = tuple(str(value) for value in archive["commands"].tolist())
            embeddings = np.asarray(archive["embeddings"], dtype=np.float32)
            if "encoder_checkpoint_sha256" not in archive:
                raise ValueError("text embedding archive lacks encoder checkpoint hash")
            self.encoder_checkpoint_sha256 = str(
                np.asarray(archive["encoder_checkpoint_sha256"]).reshape(-1)[0]
            )
        if commands != CANONICAL_COMMANDS:
            raise ValueError(
                "text embedding command order must exactly equal CANONICAL_COMMANDS"
            )
        expected_dim = int(self.primary.replay_buffer["motion_latent"].shape[-1])
        if embeddings.shape != (len(CANONICAL_COMMANDS), expected_dim):
            raise ValueError(
                f"text embeddings have shape {embeddings.shape}, expected "
                f"{(len(CANONICAL_COMMANDS), expected_dim)}"
            )
        if not np.all(np.isfinite(embeddings)):
            raise ValueError("text embeddings contain non-finite values")
        norms = np.linalg.norm(embeddings, axis=1)
        if not np.allclose(norms, 1.0, atol=1.0e-4):
            raise ValueError("text embeddings must be L2-normalized")
        self._text_embeddings = torch.from_numpy(embeddings)

        candidate_windows, candidate_commands = self._build_calibration_pairs()
        total = int(epoch_size)
        self.source_counts = _weighted_counts(
            [float(d0_weight), float(calibration_weight)], total
        )
        self._d0_count, self._calibration_count = self.source_counts
        rng = np.random.default_rng(self.seed)
        self._d0_indices = rng.choice(
            len(self.primary),
            size=self._d0_count,
            replace=self._d0_count > len(self.primary),
        ).astype(np.int64)
        available_command_ids = sorted(set(candidate_commands.tolist()))
        per_command_counts = _weighted_counts(
            [1.0 / len(available_command_ids)] * len(available_command_ids),
            self._calibration_count,
        )
        selected_windows = []
        selected_commands = []
        for command_id, count in zip(available_command_ids, per_command_counts):
            candidates = np.flatnonzero(candidate_commands == command_id)
            selected = rng.choice(
                candidates,
                size=count,
                replace=count > len(candidates),
            )
            selected_windows.append(candidate_windows[selected])
            selected_commands.append(np.full(count, command_id, dtype=np.int64))
        self._calibration_indices = np.concatenate(selected_windows).astype(np.int64)
        self._calibration_commands = np.concatenate(selected_commands).astype(np.int64)
        permutation = rng.permutation(self._calibration_count)
        self._calibration_indices = self._calibration_indices[permutation]
        self._calibration_commands = self._calibration_commands[permutation]
        self.calibration_pair_count = int(len(candidate_windows))
        selected_counts = Counter(self._calibration_commands.tolist())
        self.selected_command_counts = {
            command: int(selected_counts.get(COMMAND_TO_ID[command], 0))
            for command in CANONICAL_COMMANDS
        }
        print(
            "[ExactTextCalibrationDataset] "
            f"D0={self._d0_count}/{total} ({float(d0_weight):.1%}), "
            f"exact_text={self._calibration_count}/{total} "
            f"({float(calibration_weight):.1%}), "
            f"available_exact_pairs={self.calibration_pair_count}"
        )

    def _validate_audit_report(self) -> None:
        report = json.loads(self.audit_report_path.read_text(encoding="utf-8"))
        if report.get("schema") != "predactor-motion-command-audit-v1":
            raise ValueError("motion-command audit has an unsupported schema")
        if report.get("status") != "passed":
            raise ValueError("motion-command audit did not pass")
        if report.get("command_map_sha256") != self.command_map.sha256:
            raise ValueError("motion-command map changed after its audit")
        audited_d0 = Path(str(report.get("d0_zarr", ""))).expanduser().resolve()
        if audited_d0 != self._zarr_path:
            raise ValueError(
                f"motion-command audit covers {audited_d0}, not {self._zarr_path}"
            )

    def _build_calibration_pairs(self) -> tuple[np.ndarray, np.ndarray]:
        origin_rows = (
            np.asarray(self.primary.sampler.indices[:, 0], dtype=np.int64)
            + self.n_obs_steps - 1
        )
        motion_ids = np.asarray(
            self.primary.replay_buffer["motion_idx"][:]
        ).reshape(-1)
        window_motion_ids = np.rint(motion_ids[origin_rows]).astype(np.int64)
        all_windows = []
        all_commands = []
        for entry in self.command_map.entries:
            window_indices = np.flatnonzero(window_motion_ids == entry.motion_idx)
            if len(window_indices) == 0:
                raise ValueError(
                    f"reviewed motion_idx {entry.motion_idx} has no complete D0 windows"
                )
            for command in entry.commands:
                all_windows.append(window_indices)
                all_commands.append(np.full(
                    len(window_indices), COMMAND_TO_ID[command], dtype=np.int64
                ))
        return np.concatenate(all_windows), np.concatenate(all_commands)

    def __len__(self) -> int:
        return self._d0_count + self._calibration_count

    def _exact_item(self, primary_index: int) -> dict[str, torch.Tensor]:
        sample = self.primary.sampler.sample_sequence(int(primary_index))
        return dict_apply(self.primary._sample_to_data(sample), torch.from_numpy)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        if index < self._d0_count:
            item = self._exact_item(self._d0_indices[index])
            command_id = -1
            source_id = 0
        else:
            offset = index - self._d0_count
            item = self._exact_item(self._calibration_indices[offset])
            command_id = int(self._calibration_commands[offset])
            source_id = 1
        item["_calibration_command_id"] = torch.tensor(command_id, dtype=torch.long)
        item["_source_index"] = torch.tensor(source_id, dtype=torch.long)
        return item

    def collate_fn(self, batch: list[dict[str, torch.Tensor]], normalize: bool = False):
        items = [dict(item) for item in batch]
        command_ids = torch.stack([
            item.pop("_calibration_command_id") for item in items
        ])
        source_ids = torch.stack([item.pop("_source_index") for item in items])
        for item, command_id in zip(items, command_ids.tolist()):
            if command_id < 0:
                continue
            embedding = self._text_embeddings[command_id].to(
                item["motion_latent"].dtype
            )
            item["motion_latent"] = embedding.expand_as(
                item["motion_latent"]
            ).clone()
        result = self.primary.collate_fn(items, normalize=normalize)
        result["source_id"] = source_ids
        result["command_id"] = command_ids
        return result

    def get_normalizer(self, **kwargs):
        del kwargs
        print(
            "[ExactTextCalibrationDataset] Loading frozen pi0 normalizer from "
            f"{self.normalizer_checkpoint} ({self.normalizer_replica})"
        )
        return normalizer_from_checkpoint(
            self.normalizer_checkpoint, self.normalizer_replica
        )

    def get_validation_dataset(self):
        return self.primary.get_validation_dataset()

    def get_all_actions(self):
        return self.primary.get_all_actions()

    @staticmethod
    def worker_init_fn(worker_id: int) -> None:
        del worker_id
        info = torch.utils.data.get_worker_info()
        if info is None:
            return
        dataset = info.dataset
        child = dataset.primary
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


class CalibrationTrainer(OfflineTrainer):
    """Warm-start from selected pi0 weights; keep optimizer state fresh."""

    def _after_training_data_ready(self, cfg, dataset, normalizer) -> None:
        if not isinstance(dataset, ExactTextCalibrationDataset):
            raise TypeError("CalibrationTrainer requires ExactTextCalibrationDataset")
        if cfg.training.resume:
            raise ValueError("exact-text calibration uses init_checkpoint, not resume")
        epochs = int(cfg.training.num_epochs)
        learning_rate = float(cfg.optimizer.learning_rate)
        if not 1 <= epochs <= 4:
            raise ValueError(f"exact-text calibration requires 1-4 epochs, got {epochs}")
        if not 1.0e-5 <= learning_rate <= 2.0e-5:
            raise ValueError(
                "exact-text calibration learning rate must be in [1e-5, 2e-5], "
                f"got {learning_rate}"
            )
        if cfg.training.use_ema:
            raise ValueError(
                "short calibration must publish one unambiguous student; use_ema=false"
            )

        checkpoint = Path(str(cfg.calibration.init_checkpoint)).expanduser().resolve()
        if checkpoint != dataset.normalizer_checkpoint:
            raise ValueError(
                "init checkpoint and frozen-normalizer checkpoint must be identical: "
                f"{checkpoint} != {dataset.normalizer_checkpoint}"
            )
        replica = str(cfg.calibration.get(
            "init_replica", dataset.normalizer_replica
        ))
        if replica != dataset.normalizer_replica:
            raise ValueError("weight and normalizer replicas must be identical")
        state = checkpoint_replica_state(load_checkpoint_payload(checkpoint), replica)
        self.agent.load_state_dict(state, strict=True)

        expected_normalizer_hash = nested_state_sha256(normalizer.state_dict())
        loaded_normalizer_hash = nested_state_sha256(
            self.agent.normalizer.state_dict()
        )
        if loaded_normalizer_hash != expected_normalizer_hash:
            raise RuntimeError("pi0 normalizer changed while loading calibration weights")
        policy_parameters = [
            (name, parameter)
            for name, parameter in self.agent.actor.named_parameters(prefix="actor")
            if not name.startswith("actor.normalizer.")
        ]
        if self.agent.cond_compressor is not None:
            policy_parameters.extend(
                self.agent.cond_compressor.named_parameters(prefix="cond_compressor")
            )
        frozen_policy_parameters = [
            name for name, parameter in policy_parameters
            if not parameter.requires_grad
        ]
        if frozen_policy_parameters:
            raise RuntimeError(
                "calibration expects the complete policy to remain trainable; frozen="
                f"{frozen_policy_parameters}"
            )
        trainable_optimizer_ids = {
            id(parameter)
            for optimizer in self.optimizer.values()
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        missing_from_optimizer = [
            name for name, parameter in policy_parameters
            if id(parameter) not in trainable_optimizer_ids
        ]
        if missing_from_optimizer:
            raise RuntimeError(
                "trainable policy parameters are missing from the optimizer: "
                f"{missing_from_optimizer}"
            )
        normalizer_trainable = [
            name for name, parameter in self.agent.normalizer.named_parameters()
            if parameter.requires_grad
        ]
        if normalizer_trainable:
            raise RuntimeError(
                f"pi0 normalizer must remain frozen: {normalizer_trainable}"
            )

        audit = {
            "schema": "predactor-exact-text-calibration-train-v1",
            "init_checkpoint": str(checkpoint),
            "init_replica": replica,
            "initial_policy_sha256": nested_state_sha256(self.agent.state_dict()),
            "normalizer_sha256": loaded_normalizer_hash,
            "normalizer_frozen": True,
            "optimizer_state_loaded": False,
            "trainable_scope": "actor_and_condition_path",
            "trainable_parameter_count": int(sum(
                parameter.numel() for _, parameter in policy_parameters
            )),
            "d0_zarr": str(dataset._zarr_path),
            "command_map": str(dataset.command_map.path),
            "command_map_sha256": dataset.command_map.sha256,
            "command_audit": str(dataset.audit_report_path),
            "command_audit_sha256": sha256_file(dataset.audit_report_path),
            "text_embeddings": str(dataset.text_embeddings_path),
            "text_embeddings_sha256": sha256_file(dataset.text_embeddings_path),
            "encoder_checkpoint_sha256": dataset.encoder_checkpoint_sha256,
            "source_counts": {
                "d0": dataset.source_counts[0],
                "exact_text": dataset.source_counts[1],
            },
            "available_exact_pairs": dataset.calibration_pair_count,
            "selected_exact_pairs_by_command": dataset.selected_command_counts,
            "exact_text_sampling": "uniform_over_supported_canonical_commands",
            "learning_rate": learning_rate,
            "num_epochs": epochs,
        }
        output = Path(self.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "exact_text_calibration_initialization.json").write_text(
            json.dumps(audit, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"[CalibrationTrainer] Initialized all policy weights from pi0 "
            f"({replica}); optimizer/global step start fresh; normalizer is frozen."
        )
