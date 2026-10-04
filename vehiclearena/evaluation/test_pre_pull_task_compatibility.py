"""The restored driver uses all new tasks, without stale reference claims."""
import copy
import json
from pathlib import Path

from evaluation.experiments.manifest import load_manifest
from evaluation.experiments.time_window_calibration import (
    CALIBRATION_PROTOCOL, load_calibration_registry, scenario_physical_fingerprint,
)
from evaluation.multi_agent_runner import _NONVISUAL_WAKE_EVENTS
from simulation.sumo_traffic_manager import SumoTrafficManager
from evaluation.experiments.task_compatibility import adapt_scene


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "evaluation/experiments/scenarios"


def test_all_current_tasks_are_clean_and_have_current_baselines():
    catalog = json.loads((CATALOG / "catalog.json").read_text())
    assert catalog["counts"] == {"Basic": 180, "MultiLLM": 20}
    assert len(catalog["entries"]) == 200
    registry = load_calibration_registry()
    scene_ids = {entry["scene_id"] for entry in catalog["entries"]}
    assert set(registry["entries"]) <= scene_ids
    assert set(registry.get("pending_scene_ids", [])) == (
        scene_ids - set(registry["entries"]))
    for entry in catalog["entries"]:
        scene = json.loads((CATALOG / entry["scenario"]).read_text())
        baseline = registry["entries"].get(entry["scene_id"])
        if baseline is None:
            assert "time_window_calibration" not in scene
        else:
            assert baseline["protocol"] == CALIBRATION_PROTOCOL
            assert baseline["physical_fingerprint"] == scenario_physical_fingerprint(scene)
            assert scene["total_time_s"] == baseline["time_limit_s"]
            assert scene["time_window_calibration"]["time_limit_s"] == baseline["time_limit_s"]
            assert baseline["case_count"] == 1
            if baseline["successful_case_count"]:
                assert baseline["time_limit_s"] == round(
                    max(
                        baseline["max_successful_completion_s"],
                        baseline["last_scheduled_event_s"],
                    ) + 10.0,
                    6,
                )
            else:
                assert baseline["max_successful_completion_s"] is None
        for vehicle in scene["vehicles"]:
            assert "native_lane_change_enabled" not in vehicle.get("initial_physical_state", {})
            if vehicle["agent_config"]["type"] == "sumo":
                assert not {"target_speed_kmh", "desired_speed_kmh"}.intersection(
                    vehicle.get("initial_physical_state", {}))
            else:
                assert vehicle["initial_node"] != vehicle["destination_node"]
        for pedestrian in scene.get("pedestrians", []):
            assert "crossing_trigger" not in pedestrian.get("initial_physical_state", {})
        assert not scene.get("experiment_world_events")


def test_manifests_retain_source_provenance_and_new_task_set():
    for study, count in (("Basic", 180), ("MultiLLM", 40)):
        manifest = load_manifest(ROOT / "evaluation/experiments/manifests" /
                                 f"{study}.manifest.json")
        assert len(manifest.variants) == count
        assert len(manifest.source_hash) == 64


def test_task_migration_reaches_an_idempotent_fixed_point_without_physics():
    catalog = json.loads((CATALOG / "catalog.json").read_text())
    for entry in catalog["entries"]:
        scene = json.loads((CATALOG / entry["scenario"]).read_text())
        expected = json.loads((CATALOG / entry["expected"]).read_text())
        adapt_scene(scene, expected, None)
        migrated = copy.deepcopy((scene, expected))
        adapt_scene(scene, expected, None)
        assert (scene, expected) == migrated, entry["scene_id"]


def test_no_evaluated_reference_driver_and_pa_wake_is_visible():
    assert not hasattr(SumoTrafficManager, "_evaluated_npc_should_hold_for_pedestrian")
    assert not hasattr(SumoTrafficManager, "_evaluated_npc_should_hold_for_shared_corridor")
    assert not hasattr(SumoTrafficManager, "release_triggered_pedestrians")
    assert "personal_agent_update" in _NONVISUAL_WAKE_EVENTS
