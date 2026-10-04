"""Prepare the two formal VehicleArena study manifests."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Dict

from evaluation.experiments.manifest import (
    ExperimentVariant, build_manifest, sha256_file, source_fingerprint,
)
from evaluation.experiments.scene_catalog import iter_catalog_scenes


STUDIES = ("Basic", "MultiLLM")


def prepare_scene_manifests(
    output_dir: Path,
    catalog_dir: Path,
    *,
    quick: bool = False,
    source_root: Path | None = None,
) -> Dict[str, Path]:
    """Create one model-agnostic variant per frozen study task.

    Model names and credentials are execution parameters, not physical scene
    factors. Basic has one focal LLM. MultiLLM declares fixed peer entity IDs;
    the run command must hold their model assignment fixed while changing only
    the focal model.
    """
    output_dir = Path(output_dir)
    catalog_dir = Path(catalog_dir)
    source_root = source_root or Path(__file__).resolve().parents[2]
    grouped = {study: [] for study in STUDIES}
    for entry, scenario_path, expected_path in iter_catalog_scenes(catalog_dir):
        study = str(entry["experiment_id"])
        if study not in grouped:
            raise ValueError(f"unsupported formal study {study!r}")
        scenario = json.loads(scenario_path.read_text(encoding="utf-8"))
        expected = json.loads(expected_path.read_text(encoding="utf-8"))
        scenario.setdefault("experiment_scene", {})["setup_assertions"] = (
            copy.deepcopy(expected.get("setup_assertions", [])))
        grouped[study].append((entry, scenario))

    if quick:
        grouped = {study: items[:1] for study, items in grouped.items()}

    descriptions = {
        "Basic": (
            "Single evaluated Driving Agent with frozen SUMO background "
            "vehicles and pedestrians."),
        "MultiLLM": (
            "Marginal system impact when only the focal Driving Agent model "
            "changes and peer LLM assignments remain fixed."),
    }
    code_hash = source_fingerprint(source_root)
    catalog_hash = sha256_file(catalog_dir / "catalog.json")
    paths: Dict[str, Path] = {}

    for study, items in grouped.items():
        # An isolated regression catalog may intentionally contain only one
        # study (for example, Basic-only single-LLM validation).  Do not build
        # an invalid empty manifest for the absent study.
        if not items:
            continue
        variants = []
        for entry, scenario in items:
            scene = scenario["experiment_scene"]
            focal_id = str(scene["focal_vehicle_id"])
            peer_ids = [str(item) for item in scene["fixed_peer_vehicle_ids"]]
            variants.append(ExperimentVariant(
                variant_id=str(scenario["scenario_id"]),
                base_scenario_id=str(scenario["scenario_id"]),
                factors={
                    "study": "basic" if study == "Basic" else "multi_llm",
                    "focal_vehicle_id": focal_id,
                    "fixed_peer_vehicle_ids": peer_ids,
                    "network": scenario["road_network_id"],
                    "interaction_tags": list(entry.get("tags", [])),
                },
                scenario=scenario,
                requires_llm=True,
                notes=(
                    "Change only the focal model; use the identical frozen "
                    "scene and Personal Agent request tape for comparisons."),
            ))

        manifest = build_manifest(
            study,
            descriptions[study],
            variants,
            metadata={
                "catalog_hash": catalog_hash,
                "catalog_dir": str(catalog_dir),
                "task_count": len(variants),
                "random_seed_configurable": False,
                "background_authority": "sumo",
                "focal_replacement_only": True,
                "personal_agent_protocol": "live_before_driver_wake",
                "quick": bool(quick),
            },
        )
        manifest.source_hash = code_hash
        path = output_dir / f"{study}.manifest.json"
        manifest.write(path)
        paths[study] = path
    return paths
