"""Stable compatibility identity for paired SUMO reference trajectories.

``source_hash`` and ``manifest_hash`` remain useful provenance, but they are
too broad to decide whether two physical trajectories may be compared.  This
module defines the smaller, explicit contract that does decide compatibility.

The revision must be bumped whenever a simulation change can alter physical
traffic outcomes without changing the frozen scenario, map, SUMO version, or
fixed-peer execution contract.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional

from evaluation.experiments.manifest import sha256_data, sha256_file


REFERENCE_PROTOCOL_SCHEMA = "vehiclearena-reference-protocol-v1"
REFERENCE_PROTOCOL_REVISION = "sumo-physical-traffic-v1"

_PHYSICS_IDENTITY_FIELDS = (
    "name",
    "sumo_version",
    "step_length_s",
    "vehicle_dynamics_authority",
    "vehicle_pose_authority",
    "vehicle_route_execution_authority",
    "vehicle_collision_authority",
    "vehicle_pedestrian_collision_authority",
    "pedestrian_physics_authority",
)


def expected_reference_policy(experiment_id: str, scenario: Mapping) -> str:
    """Return the reference policy paired with a treatment scenario."""
    if str(experiment_id) == "MultiLLM":
        return "focal_sumo_fixed_peers"
    return "all_sumo"


def _peer_protocol(spec: Any) -> dict:
    """Keep behavioural peer settings while excluding endpoint credentials."""
    fields = (
        "agent_type",
        "model",
        "max_turns",
        "temperature",
        "max_tokens",
        "thinking_mode",
        "reasoning_effort",
        "chat_template_enable_thinking",
        "context_window_tokens",
        "todo_max_ttl_s",
        "heartbeat_interval_s",
    )
    values = {
        key: getattr(spec, key, None)
        for key in fields
    }
    driver_prompt = getattr(spec, "driver_prompt", "") or ""
    values["driver_prompt_hash"] = sha256_data(driver_prompt)
    return values


def build_reference_protocol(
    *,
    experiment_id: str,
    variant: Mapping,
    network_path: Path,
    physics_engine: Mapping,
    resolved_specs: Optional[Mapping[str, Any]] = None,
) -> dict:
    """Build the auditable protocol record embedded in each completed run."""
    scenario = variant.get("scenario", {}) or {}
    scene = scenario.get("experiment_scene", {}) or {}
    fixed_peer_ids = sorted(map(str, scene.get(
        "fixed_peer_vehicle_ids", [])))
    specs = resolved_specs or {}
    missing_peers = [peer_id for peer_id in fixed_peer_ids
                     if peer_id not in specs]
    if missing_peers:
        raise ValueError(
            "reference protocol is missing resolved fixed peers: "
            f"{missing_peers!r}")
    network_path = Path(network_path)
    if not network_path.is_file():
        raise FileNotFoundError(
            f"reference protocol map is missing: {network_path}")
    compiled_manifest_path = Path(str(physics_engine.get(
        "map_manifest", "")))
    if (physics_engine.get("name") == "sumo"
            and not compiled_manifest_path.is_file()):
        raise FileNotFoundError(
            "reference protocol SUMO map manifest is missing: "
            f"{compiled_manifest_path}")
    compiled_map = {}
    if compiled_manifest_path.is_file():
        compiled_manifest = json.loads(compiled_manifest_path.read_text(
            encoding="utf-8"))
        compiled_map = {
            "format": compiled_manifest.get("format"),
            # SUMO map conversion hashes the lane-level bytes together with
            # its compiler revision, so compiler changes invalidate pairing.
            "source_sha256": compiled_manifest.get("source_sha256"),
        }
    return {
        "schema": REFERENCE_PROTOCOL_SCHEMA,
        "revision": REFERENCE_PROTOCOL_REVISION,
        "experiment_id": str(experiment_id),
        "variant_id": str(variant.get("variant_id", "")),
        "scenario_hash": str(variant.get("scenario_hash", "")),
        "reference_policy": expected_reference_policy(
            str(experiment_id), scenario),
        "road_network": {
            "id": str(scenario.get("road_network_id", "")),
            "lane_level_sha256": sha256_file(network_path),
            "sumo_compiled_map": compiled_map,
        },
        "physics": {
            key: physics_engine.get(key)
            for key in _PHYSICS_IDENTITY_FIELDS
        },
        "focal_vehicle_id": str(scene.get("focal_vehicle_id", "")),
        "fixed_peer_vehicle_ids": fixed_peer_ids,
        "fixed_peer_protocols": {
            peer_id: _peer_protocol(specs[peer_id])
            for peer_id in fixed_peer_ids
        },
    }


def protocol_hash(protocol: Mapping) -> str:
    """Hash a complete protocol record using the manifest canonical form."""
    return sha256_data(dict(protocol))


def protocol_identity(payload: Mapping) -> dict:
    """Validate and expose a run's protocol identity without trusting its hash."""
    record = payload.get("reference_protocol")
    actual = payload.get("protocol_hash")
    if record is None and actual is None:
        return {"status": "missing", "protocol_hash": None}
    if not isinstance(record, Mapping) or not actual:
        return {"status": "invalid", "protocol_hash": actual,
                "reason": "protocol_record_or_hash_missing"}
    if record.get("schema") != REFERENCE_PROTOCOL_SCHEMA:
        return {"status": "invalid", "protocol_hash": actual,
                "reason": "protocol_schema_unsupported"}
    calculated = protocol_hash(record)
    if str(actual) != calculated:
        return {"status": "invalid", "protocol_hash": actual,
                "calculated_protocol_hash": calculated,
                "reason": "protocol_hash_invalid"}
    return {"status": "valid", "protocol_hash": calculated}


def reference_compatibility(treatment: Mapping, reference: Mapping) -> dict:
    """Decide pairing by protocol hash, with exact-hash legacy fallback."""
    treatment_identity = protocol_identity(treatment)
    reference_identity = protocol_identity(reference)
    statuses = {
        treatment_identity["status"], reference_identity["status"]}

    if statuses != {"missing"}:
        for label, identity in (
                ("treatment", treatment_identity),
                ("reference", reference_identity)):
            if identity["status"] != "valid":
                return {
                    "applicable": False,
                    "compatibility_mode": "protocol_hash",
                    "reason": f"paired_{label}_protocol_{identity['status']}",
                    "details": identity,
                }
        if (treatment_identity["protocol_hash"]
                != reference_identity["protocol_hash"]):
            return {
                "applicable": False,
                "compatibility_mode": "protocol_hash",
                "reason": "paired_protocol_hash_mismatch",
                "treatment_protocol_hash": treatment_identity[
                    "protocol_hash"],
                "reference_protocol_hash": reference_identity[
                    "protocol_hash"],
            }
        provenance_mismatches = [
            key for key in ("source_hash", "manifest_hash")
            if treatment.get(key) != reference.get(key)
        ]
        return {
            "applicable": True,
            "compatibility_mode": "protocol_hash",
            "protocol_hash": treatment_identity["protocol_hash"],
            "provenance_mismatches": provenance_mismatches,
        }

    # Older artifacts did not persist the stable protocol.  Preserve their
    # former safety contract instead of silently assigning a new identity.
    for key in ("scenario_hash", "source_hash", "manifest_hash"):
        treatment_value = (
            treatment.get("variant", {}).get("scenario_hash")
            if key == "scenario_hash" else treatment.get(key))
        reference_value = (
            reference.get("variant", {}).get("scenario_hash")
            if key == "scenario_hash" else reference.get(key))
        if not treatment_value or not reference_value:
            return {
                "applicable": False,
                "compatibility_mode": "legacy_exact_hashes",
                "reason": f"paired_{key}_missing",
            }
        if treatment_value != reference_value:
            return {
                "applicable": False,
                "compatibility_mode": "legacy_exact_hashes",
                "reason": f"paired_{key}_mismatch",
            }
    return {
        "applicable": True,
        "compatibility_mode": "legacy_exact_hashes",
        "protocol_hash": None,
        "provenance_mismatches": [],
        "warning": "legacy_artifacts_without_protocol_hash",
    }
