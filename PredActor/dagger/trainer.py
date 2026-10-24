"""DAgger trainer that warm-starts weights but never optimizer/data state."""

from __future__ import annotations

import json
from pathlib import Path

from dagger.checkpoint import checkpoint_replica_state, load_checkpoint_payload
from dagger.dataset import StrictDaggerDataset
from diffusion_policy.trainer.bc_training_step import nested_state_sha256
from diffusion_policy.trainer.offline_trainer import OfflineTrainer


class DaggerTrainer(OfflineTrainer):
    """Offline BC update over exact expert labels at learner-visited states."""

    def _after_training_data_ready(self, cfg, dataset, normalizer) -> None:
        if not isinstance(dataset, StrictDaggerDataset):
            raise TypeError("DaggerTrainer requires StrictDaggerDataset")
        if cfg.training.resume:
            raise ValueError("strict DAgger uses init_checkpoint, not optimizer resume")
        epochs = int(cfg.training.num_epochs)
        learning_rate = float(cfg.optimizer.learning_rate)
        if not 1 <= epochs <= 4:
            raise ValueError(f"strict DAgger requires 1-4 epochs, got {epochs}")
        if not 1.0e-5 <= learning_rate <= 2.0e-5:
            raise ValueError(
                f"strict DAgger learning rate must be in [1e-5, 2e-5], got {learning_rate}"
            )

        checkpoint = Path(str(cfg.dagger.init_checkpoint)).expanduser().resolve()
        if checkpoint != dataset.manifest.normalizer_checkpoint:
            raise ValueError(
                "init checkpoint and frozen-normalizer checkpoint must be identical: "
                f"{checkpoint} != {dataset.manifest.normalizer_checkpoint}"
            )
        replica = str(cfg.dagger.get("init_replica", dataset.manifest.normalizer_replica))
        payload = load_checkpoint_payload(checkpoint)
        state = checkpoint_replica_state(payload, replica)
        self.agent.load_state_dict(state, strict=True)
        if self.ema_agent is not None:
            self.ema_agent.load_state_dict(state, strict=True)

        expected_hash = nested_state_sha256(normalizer.state_dict())
        loaded_hash = nested_state_sha256(self.agent.normalizer.state_dict())
        if loaded_hash != expected_hash:
            raise RuntimeError("pi0 normalizer changed while loading DAgger initialization")

        # The normalizer stores fitted statistics as Parameters.  It is a
        # fixed pi0 contract, not part of the DAgger optimization problem.
        normalizer_ids = {id(parameter) for parameter in self.agent.normalizer.parameters()}
        for parameter in self.agent.normalizer.parameters():
            parameter.requires_grad_(False)
        for optimizer in self.optimizer.values():
            for group in optimizer.param_groups:
                group["params"] = [
                    parameter for parameter in group["params"]
                    if id(parameter) not in normalizer_ids
                ]
        if any(
            id(parameter) in normalizer_ids
            for optimizer in self.optimizer.values()
            for group in optimizer.param_groups
            for parameter in group["params"]
        ):
            raise RuntimeError("pi0 normalizer remains in the DAgger optimizer")

        audit = {
            "schema": "predactor-strict-dagger-train-v1",
            "init_checkpoint": str(checkpoint),
            "init_replica": replica,
            "normalizer_sha256": loaded_hash,
            "normalizer_frozen": True,
            "manifest": str(dataset.manifest.path),
            "d0_weight": dataset.manifest.d0_weight,
            "dagger_weight": dataset.manifest.dagger_weight,
            "source_counts": dataset.source_counts,
            "learning_rate": learning_rate,
            "num_epochs": epochs,
            "optimizer_state_loaded": False,
        }
        output = Path(self.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "dagger_initialization.json").write_text(
            json.dumps(audit, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"[DaggerTrainer] Initialized weights from pi0 ({replica}); "
            "optimizer/global step start fresh; pi0 normalizer is frozen."
        )
