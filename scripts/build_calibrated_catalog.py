#!/usr/bin/env python3
"""Build a complete calibrated catalog from SUMO and peer-LLM outcomes."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from evaluation.experiments.time_window_calibration import (
    CALIBRATION_SCHEMA,
    apply_calibration_entry,
    scenario_physical_fingerprint,
)


PROTOCOL = "hybrid-peer-reference-with-selected-all-sumo-margin-20-v1"


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _attempt(outcome: dict) -> dict:
    return {
        "run_duration_s": outcome.get("run_duration_s"),
        "success": bool(outcome.get("success")),
        "completion_time_s": outcome.get("completion_time_s"),
        "collision_count": outcome.get("collision_count"),
        "agent_callback_error_count": outcome.get(
            "agent_callback_error_count"),
        "simulated_until_s": outcome.get("simulated_until_s"),
        "error": outcome.get("error", ""),
    }


def _load_peer_outcomes(root: Path) -> dict[str, dict]:
    outcomes: dict[str, dict] = {}
    paths = sorted(root.glob("endpoint-*/case-outcomes/initial/*.json"))
    paths += sorted(root.glob("endpoint-*/case-outcomes/retry/*.json"))
    for path in paths:
        outcome = _read(path)
        scene_id = str(outcome["scene_id"])
        current = outcomes.get(scene_id)
        if current is None or bool(outcome.get("success")):
            outcomes[scene_id] = outcome
    return outcomes


def _entry(
    *, scene_id: str, scenario: dict, completion_s: float,
    last_event_s: float, margin_s: float, reference_policy: str,
    outcomes: list[dict], actor_configurations: list[str],
    fixed_peer_protocol: dict | None, generated_at: str,
) -> dict:
    time_limit_s = round(max(float(completion_s), float(last_event_s))
                         + float(margin_s), 6)
    return {
        "scene_id": scene_id,
        "protocol": PROTOCOL,
        "physical_fingerprint": scenario_physical_fingerprint(scenario),
        "method": "reference_max_sumo_terminal_or_last_event_plus_margin",
        "reference_policy": reference_policy,
        "fixed_peer_protocol": copy.deepcopy(fixed_peer_protocol),
        "generated_at": generated_at,
        "max_successful_completion_s": round(float(completion_s), 6),
        "last_scheduled_event_s": round(float(last_event_s), 6),
        "environment_schedule_basis_s": None,
        "margin_s": float(margin_s),
        "time_limit_s": time_limit_s,
        "case_count": len(outcomes),
        "successful_case_count": sum(
            bool(item.get("success")) for item in outcomes),
        "failed_case_count": sum(
            not bool(item.get("success")) for item in outcomes),
        "actor_configurations": sorted(set(actor_configurations)),
        "outcomes": outcomes,
    }


def build(args: argparse.Namespace) -> dict:
    source = args.source.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    output.mkdir(parents=True)
    catalog_dir = output / "catalog"
    shutil.copytree(source, catalog_dir)

    catalog_path = catalog_dir / "catalog.json"
    catalog = _read(catalog_path)
    base_registry = _read(args.all_sumo_registry.resolve())
    peer_outcomes = _load_peer_outcomes(args.peer_outcomes.resolve())
    margin20_ids = set(args.sumo_margin20_scene)
    catalog_ids = {str(item["scene_id"]) for item in catalog["entries"]}
    unknown = sorted(margin20_ids - catalog_ids)
    if unknown:
        raise ValueError(f"unknown +20 SUMO scenes: {unknown}")

    now = datetime.now(timezone.utc).isoformat()
    registry_entries: dict[str, dict] = {}
    source_counts = {
        "basic_all_sumo_plus_10": 0,
        "multillm_peer_plus_10": 0,
        "selected_all_sumo_plus_20": 0,
    }
    selected_details = []

    for catalog_entry in catalog["entries"]:
        scene_id = str(catalog_entry["scene_id"])
        experiment_id = str(catalog_entry["experiment_id"])
        scenario_path = catalog_dir / str(catalog_entry["scenario"])
        scenario = _read(scenario_path)
        base = base_registry["entries"].get(scene_id)
        if base is None or base.get("max_successful_completion_s") is None:
            raise ValueError(f"missing successful all-SUMO entry: {scene_id}")
        last_event_s = float(base.get("last_scheduled_event_s", 0.0))

        if experiment_id == "Basic":
            completion_s = float(base["max_successful_completion_s"])
            outcomes = copy.deepcopy(base.get("outcomes", []))
            entry = _entry(
                scene_id=scene_id, scenario=scenario,
                completion_s=completion_s, last_event_s=last_event_s,
                margin_s=10.0, reference_policy="all_sumo",
                outcomes=outcomes,
                actor_configurations=list(base.get(
                    "actor_configurations", ["all_sumo"])),
                fixed_peer_protocol=None, generated_at=now,
            )
            source_counts["basic_all_sumo_plus_10"] += 1
        elif scene_id in margin20_ids:
            completion_s = float(base["max_successful_completion_s"])
            outcomes = copy.deepcopy(base.get("outcomes", []))
            entry = _entry(
                scene_id=scene_id, scenario=scenario,
                completion_s=completion_s, last_event_s=last_event_s,
                margin_s=20.0, reference_policy="all_sumo",
                outcomes=outcomes,
                actor_configurations=list(base.get(
                    "actor_configurations", ["all_sumo"])),
                fixed_peer_protocol=None, generated_at=now,
            )
            source_counts["selected_all_sumo_plus_20"] += 1
            selected_details.append({
                "scene_id": scene_id,
                "all_sumo_completion_s": completion_s,
                "time_limit_s": entry["time_limit_s"],
            })
        else:
            outcome = peer_outcomes.get(scene_id)
            if not outcome or not outcome.get("success") \
                    or outcome.get("completion_time_s") is None:
                raise ValueError(
                    f"missing successful peer-LLM outcome: {scene_id}")
            outcome = copy.deepcopy(outcome)
            outcome.setdefault("attempts", [_attempt(outcome)])
            entry = _entry(
                scene_id=scene_id, scenario=scenario,
                completion_s=float(outcome["completion_time_s"]),
                last_event_s=last_event_s, margin_s=10.0,
                reference_policy="focal_sumo_fixed_peers",
                outcomes=[outcome],
                actor_configurations=[str(outcome["actor_configuration"])],
                fixed_peer_protocol=outcome.get("fixed_peer_protocol"),
                generated_at=now,
            )
            source_counts["multillm_peer_plus_10"] += 1

        time_limit_s = float(entry["time_limit_s"])
        scenario_hardness = scenario.get("experiment_scene", {}).get(
            "hardness_profile")
        if isinstance(scenario_hardness, dict):
            scenario_hardness["calibration_status"] = "calibrated"
            scenario_hardness["time_limit_s"] = time_limit_s
            entry["physical_fingerprint"] = scenario_physical_fingerprint(
                scenario)

        catalog_hardness = catalog_entry.get("hardness_profile")
        if isinstance(catalog_hardness, dict):
            catalog_hardness["calibration_status"] = "calibrated"
            catalog_hardness["time_limit_s"] = time_limit_s
        if "time_window_calibration" in catalog_entry:
            catalog_entry["time_window_calibration"] = copy.deepcopy(entry)

        expected_path = catalog_dir / str(catalog_entry["expected"])
        expected = _read(expected_path)
        expected_hardness = expected.get("hardness_profile")
        if isinstance(expected_hardness, dict):
            expected_hardness["calibration_status"] = "calibrated"
            expected_hardness["time_limit_s"] = time_limit_s
            _write(expected_path, expected)

        registry_entries[scene_id] = entry
        if not apply_calibration_entry(scenario, entry):
            raise RuntimeError(f"fingerprint mismatch: {scene_id}")
        _write(scenario_path, scenario)

    registry = {
        "schema": CALIBRATION_SCHEMA,
        "protocol": PROTOCOL,
        "generated_at": now,
        "method": "hybrid_reference_time_window_calibration",
        "formula": {
            "default": (
                "max(reference SUMO terminal time, last scheduled event) "
                "+ 10 simulation seconds"),
            "selected_all_sumo": (
                "max(all-SUMO completion time, last scheduled event) "
                "+ 20 simulation seconds"),
        },
        "scene_count": len(registry_entries),
        "source_counts": source_counts,
        "selected_all_sumo_margin20_scene_ids": sorted(margin20_ids),
        "entries": registry_entries,
    }
    registry_path = output / "time_window_calibration.json"
    _write(registry_path, registry)

    catalog["time_window_calibration"] = {
        "schema": CALIBRATION_SCHEMA,
        "registry": "../time_window_calibration.json",
        "protocol": PROTOCOL,
        "applied_scene_count": len(registry_entries),
        "status": "calibrated",
        "source_counts": source_counts,
    }
    if isinstance(catalog.get("npc_behavior"), dict):
        catalog["npc_behavior"]["calibration_required"] = False
    if isinstance(catalog.get("new_map_hard_multillm"), dict):
        catalog["new_map_hard_multillm"]["calibration_status"] = "calibrated"
        catalog["new_map_hard_multillm"]["registry"] = (
            "../time_window_calibration.json"
        )
    _write(catalog_path, catalog)

    catalog_readme = catalog_dir / "README.md"
    if catalog_readme.exists():
        text = catalog_readme.read_text(encoding="utf-8")
        text = text.replace(
            "Basic（180 个任务）和 Multi-LLM（20 个任务）",
            "Basic（180 个任务）和 Multi-LLM（40 个任务）",
        )
        catalog_readme.write_text(text, encoding="utf-8")

    limits = [float(item["time_limit_s"])
              for item in registry_entries.values()]
    report = {
        "schema": CALIBRATION_SCHEMA,
        "protocol": PROTOCOL,
        "generated_at": now,
        "catalog": str(catalog_dir),
        "registry": str(registry_path),
        "scene_count": len(registry_entries),
        "source_counts": source_counts,
        "selected_all_sumo_margin20": sorted(
            selected_details, key=lambda item: item["scene_id"]),
        "time_limit_min_s": min(limits),
        "time_limit_max_s": max(limits),
    }
    _write(output / "time-window-calibration-report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--all-sumo-registry", type=Path, required=True)
    parser.add_argument("--peer-outcomes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--sumo-margin20-scene", action="append", default=[],
        help="MultiLLM scene using all-SUMO completion plus 20 seconds",
    )
    args = parser.parse_args()
    report = build(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
