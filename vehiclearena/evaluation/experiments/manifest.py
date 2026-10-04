"""Versioned, hash-checked experiment manifests."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List


SCHEMA = "vehiclearena-experiment-manifest-v0.2"


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), default=str)


def sha256_data(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_fingerprint(root: Path) -> str:
    """Hash executable experiment/simulation sources, excluding outputs."""
    digest = hashlib.sha256()
    patterns = ("*.py", "*.yaml")
    paths = sorted({
        path
        for pattern in patterns
        for path in root.rglob(pattern)
        if "__pycache__" not in path.parts
        and "outputs" not in path.parts
        and "runs" not in path.parts
    })
    for path in paths:
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(path.read_bytes())
    # Browser assets determine the actual images received by the agent, so
    # Python-only hashes cannot safely identify a rendered experiment.
    web_root = root.parent / "web3d"
    if root.name == "vehiclearena" and web_root.is_dir():
        for path in sorted(web_root.iterdir()):
            if path.is_file() and path.suffix in {".js", ".html", ".css", ".py"}:
                digest.update(f"web3d/{path.name}".encode("utf-8"))
                digest.update(path.read_bytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class ExperimentVariant:
    variant_id: str
    base_scenario_id: str
    factors: Dict[str, Any]
    scenario: Dict[str, Any]
    agent_overrides: Dict[str, dict] = field(default_factory=dict)
    requires_llm: bool = False
    repeat_index: int = 0
    notes: str = ""

    @property
    def scenario_hash(self) -> str:
        return sha256_data(self.scenario)

    def to_dict(self) -> dict:
        return {
            "variant_id": self.variant_id,
            "base_scenario_id": self.base_scenario_id,
            "factors": self.factors,
            "scenario_hash": self.scenario_hash,
            "scenario": self.scenario,
            "agent_overrides": self.agent_overrides,
            "requires_llm": self.requires_llm,
            "repeat_index": self.repeat_index,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, value: dict) -> "ExperimentVariant":
        variant = cls(
            variant_id=value["variant_id"],
            base_scenario_id=value["base_scenario_id"],
            factors=dict(value.get("factors", {})),
            scenario=dict(value["scenario"]),
            agent_overrides=dict(value.get("agent_overrides", {})),
            requires_llm=bool(value.get("requires_llm", False)),
            repeat_index=int(value.get("repeat_index", 0)),
            notes=str(value.get("notes", "")),
        )
        expected = value.get("scenario_hash")
        if expected and expected != variant.scenario_hash:
            raise ValueError(
                f"Scenario hash mismatch for {variant.variant_id}")
        return variant


@dataclass
class ExperimentManifest:
    experiment_id: str
    description: str
    variants: List[ExperimentVariant]
    metadata: Dict[str, Any] = field(default_factory=dict)
    schema: str = SCHEMA
    created_at: str = field(default_factory=lambda: datetime.now(
        timezone.utc).isoformat())
    source_hash: str = ""

    def validate(self) -> None:
        if self.schema != SCHEMA:
            raise ValueError(
                f"Unsupported experiment manifest schema: {self.schema}")
        if not self.experiment_id:
            raise ValueError("experiment_id is required")
        if not self.variants:
            raise ValueError("Manifest contains no variants")
        ids = [variant.variant_id for variant in self.variants]
        if len(ids) != len(set(ids)):
            raise ValueError("variant_id values must be unique")
        for variant in self.variants:
            if not variant.base_scenario_id:
                raise ValueError(
                    f"{variant.variant_id} has no base_scenario_id")
            scenario_id = variant.scenario.get("scenario_id")
            if scenario_id != variant.variant_id:
                raise ValueError(
                    f"{variant.variant_id} scenario_id is {scenario_id!r}")
            if "road_network_id" not in variant.scenario:
                raise ValueError(
                    f"{variant.variant_id} has no road_network_id")

    def to_dict(self, include_manifest_hash: bool = True) -> dict:
        payload = {
            "schema": self.schema,
            "experiment_id": self.experiment_id,
            "description": self.description,
            "created_at": self.created_at,
            "source_hash": self.source_hash,
            "metadata": self.metadata,
            "variants": [variant.to_dict() for variant in self.variants],
        }
        if include_manifest_hash:
            payload["manifest_hash"] = sha256_data(payload)
        return payload

    @property
    def manifest_hash(self) -> str:
        return sha256_data(self.to_dict(include_manifest_hash=False))

    def write(self, path: Path) -> Path:
        self.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temp, path)
        return path

    @classmethod
    def from_dict(cls, value: dict) -> "ExperimentManifest":
        expected_hash = value.get("manifest_hash")
        raw = {key: item for key, item in value.items()
               if key != "manifest_hash"}
        if expected_hash and sha256_data(raw) != expected_hash:
            raise ValueError("Manifest hash mismatch")
        manifest = cls(
            schema=value.get("schema", ""),
            experiment_id=value["experiment_id"],
            description=value.get("description", ""),
            created_at=value.get("created_at", ""),
            source_hash=value.get("source_hash", ""),
            metadata=dict(value.get("metadata", {})),
            variants=[ExperimentVariant.from_dict(item)
                      for item in value.get("variants", [])],
        )
        manifest.validate()
        return manifest


def load_manifest(path: Path) -> ExperimentManifest:
    return ExperimentManifest.from_dict(json.loads(
        Path(path).read_text(encoding="utf-8")))


def build_manifest(
    experiment_id: str,
    description: str,
    variants: Iterable[ExperimentVariant],
    *,
    source_root: Path | None = None,
    metadata: Dict[str, Any] | None = None,
) -> ExperimentManifest:
    source_hash = source_fingerprint(source_root) if source_root else ""
    manifest = ExperimentManifest(
        experiment_id=experiment_id,
        description=description,
        variants=list(variants),
        metadata=dict(metadata or {}),
        source_hash=source_hash,
    )
    manifest.validate()
    return manifest
