"""Validated, index-only manifests for strict DAgger training data."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any

import yaml


ALLOWED_ROLES = {"d0", "origin", "recovery", "disturbance", "counterfactual"}
DAGGER_ROLES = ALLOWED_ROLES - {"d0"}


def _resolve_path(value: str, base: Path) -> Path:
    expanded = Path(os.path.expandvars(os.path.expanduser(str(value))))
    return expanded if expanded.is_absolute() else (base / expanded).resolve()


@dataclass(frozen=True)
class DaggerSource:
    name: str
    path: Path
    role: str
    weight: float


@dataclass(frozen=True)
class DaggerManifest:
    path: Path
    normalizer_checkpoint: Path
    normalizer_replica: str
    sources: tuple[DaggerSource, ...]
    sampling_mode: str = "weighted"
    text_embeddings: Path | None = None
    text_commands: tuple[str, ...] = ()

    @classmethod
    def load(cls, path: str | Path, *, require_files: bool = True) -> "DaggerManifest":
        manifest_path = Path(path).expanduser().resolve()
        with manifest_path.open("r", encoding="utf-8") as stream:
            raw: dict[str, Any] = yaml.safe_load(stream) or {}
        if raw.get("schema") != "predactor-strict-dagger-v1":
            raise ValueError("manifest schema must be 'predactor-strict-dagger-v1'")
        sampling_mode = str(raw.get("sampling_mode", "weighted"))
        if sampling_mode not in {"weighted", "aggregate"}:
            raise ValueError("sampling_mode must be 'weighted' or 'aggregate'")

        base = manifest_path.parent
        checkpoint = _resolve_path(raw.get("normalizer_checkpoint", ""), base)
        replica = str(raw.get("normalizer_replica", "agent"))
        if replica not in {"agent", "ema_agent"}:
            raise ValueError("normalizer_replica must be 'agent' or 'ema_agent'")

        sources = []
        seen_names = set()
        for entry in raw.get("sources", []):
            source = DaggerSource(
                name=str(entry["name"]),
                path=_resolve_path(entry["path"], base),
                role=str(entry["role"]),
                weight=float(entry["weight"]),
            )
            if source.name in seen_names:
                raise ValueError(f"duplicate source name: {source.name}")
            if source.role not in ALLOWED_ROLES:
                raise ValueError(
                    f"source {source.name!r} has unsupported role {source.role!r}; "
                    "mixed/beta trajectories are intentionally forbidden"
                )
            if source.weight <= 0:
                raise ValueError(f"source {source.name!r} weight must be positive")
            seen_names.add(source.name)
            sources.append(source)

        if not sources:
            raise ValueError("manifest must contain at least one source")
        d0_sources = [item for item in sources if item.role == "d0"]
        if len(d0_sources) != 1:
            raise ValueError(
                "manifest must contain exactly one immutable D0 source, got "
                f"{len(d0_sources)}"
            )
        d0_weight = sum(item.weight for item in sources if item.role == "d0")
        dagger_weight = sum(item.weight for item in sources if item.role in DAGGER_ROLES)
        if sampling_mode == "weighted":
            if not 0.50 <= d0_weight <= 0.70:
                raise ValueError(f"D0 weight must be in [0.50, 0.70], got {d0_weight:.6f}")
            if not 0.30 <= dagger_weight <= 0.50:
                raise ValueError(
                    f"new DAgger-window weight must be in [0.30, 0.50], got {dagger_weight:.6f}"
                )
        if abs(sum(item.weight for item in sources) - 1.0) > 1.0e-6:
            raise ValueError("source weights must sum to 1.0")

        text_cfg = raw.get("text_conditioning") or {}
        text_path = text_cfg.get("embeddings")
        text_embeddings = _resolve_path(text_path, base) if text_path else None
        text_commands = tuple(str(value) for value in text_cfg.get("commands", ()))

        if require_files:
            missing = [item.path for item in sources if not item.path.exists()]
            if not checkpoint.is_file():
                missing.append(checkpoint)
            if text_embeddings is not None and not text_embeddings.is_file():
                missing.append(text_embeddings)
            if missing:
                raise FileNotFoundError(
                    "manifest references missing paths:\n" +
                    "\n".join(f"  - {item}" for item in missing)
                )
            for source in sources:
                if source.role == "d0":
                    continue
                receipt_path = source.path.with_suffix(".receipt.json")
                if not receipt_path.is_file():
                    raise FileNotFoundError(
                        f"strict DAgger source {source.name!r} lacks build receipt: "
                        f"{receipt_path}"
                    )
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                if receipt.get("schema") != "predactor-strict-dagger-shard-v1":
                    raise ValueError(f"{receipt_path}: invalid strict-shard schema")
                expected_learner_targets = sampling_mode == "aggregate"
                if receipt.get("learner_actions_used_as_targets") is not expected_learner_targets:
                    raise ValueError(
                        f"{receipt_path}: learner-prefix target convention does not "
                        f"match sampling_mode={sampling_mode!r}"
                    )
                recorded_output = Path(str(receipt.get("output", ""))).resolve()
                if recorded_output != source.path.resolve():
                    raise ValueError(
                        f"{receipt_path}: output path does not match manifest source"
                    )
                role_counts = receipt.get("role_counts") or {}
                if set(role_counts) != {source.role}:
                    raise ValueError(
                        f"{receipt_path}: receipt roles {sorted(role_counts)} do not "
                        f"match manifest role {source.role!r}"
                    )

        return cls(
            path=manifest_path,
            normalizer_checkpoint=checkpoint,
            normalizer_replica=replica,
            sources=tuple(sources),
            sampling_mode=sampling_mode,
            text_embeddings=text_embeddings,
            text_commands=text_commands,
        )

    @property
    def d0_weight(self) -> float:
        return sum(item.weight for item in self.sources if item.role == "d0")

    @property
    def dagger_weight(self) -> float:
        return 1.0 - self.d0_weight
