"""Regression for the 12 sampled tasks rejected for stale NPC speed hints."""

import copy
import json
from pathlib import Path

import pytest

from evaluation.experiments.manifest import load_manifest
from evaluation.experiments.scene_catalog import normalize_native_npc_inputs
from evaluation.experiments.scene_validator import SceneValidationError, SceneValidator


TASK_IDS = {
    "basic_011_signalized_intersection__shanghai_lujiazui",
    "basic_047_narrow_road__prague_vodickova__forward_peer_mid",
    "basic_126_narrow_road__prague_vodickova__backward_peer_deep",
    "basic_127_narrow_road__prague_vodickova__forward_peer_near_exit",
    "basic_142_obstacle_gap_change__guangzhou_tianhe",
    "basic_150_platoon_pressure__hongkong_central",
    "basic_158_merge_stream__amsterdam_centrum",
    "basic_188_full_network_route__expansion__haikou_longhua",
    "basic_190_full_network_route__expansion__losangeles_downtown",
    "basic_213_unprotected_left_turn__expansion__harbin_zhongyang",
    "basic_228_platoon_pressure__expansion__seoul_gangnam",
    "basic_240_merge_stream__expansion__xiamen_huli",
}
FORBIDDEN = {"target_speed_kmh", "desired_speed_kmh"}


def test_normalization_preserves_llm_controls_and_removes_npc_events():
    scenario = {
        "vehicles": [
            {"vehicle_id": "npc", "agent_config": {"type": "sumo"},
             "initial_physical_state": {
                 "speed_kmh": 24, "target_speed_kmh": 25,
                 "desired_speed_kmh": 30, "lane_id": "lane", "progress": 0.4}},
            {"vehicle_id": "ego", "agent_config": {"type": "llm"},
             "initial_physical_state": {
                 "speed_kmh": 20, "target_speed_kmh": 25, "desired_speed_kmh": 30}},
            {"vehicle_id": "default", "agent_config": {"type": "sumo"}},
        ],
        "experiment_world_events": [{
            "entity_id": "npc", "at_s": 4,
            "action": "set_vehicle_speed", "speed_kmh": 0}],
        "total_time_s": 90,
    }
    expected = copy.deepcopy(scenario)
    for key in FORBIDDEN:
        expected["vehicles"][0]["initial_physical_state"].pop(key)
    expected.pop("experiment_world_events")
    normalize_native_npc_inputs(scenario)
    assert scenario == expected
    normalize_native_npc_inputs(scenario)
    assert scenario == expected  # Idempotent; no implicit actor state insertion.


def test_twelve_cleaned_tasks_match_manifests_and_pass_setup_validation():
    root = Path(__file__).resolve().parents[1]
    catalog = root / "evaluation/experiments/scenarios"
    manifest = load_manifest(root / "evaluation/experiments/manifests/Basic.manifest.json")
    frozen = {v.variant_id: v.scenario for v in manifest.variants}
    entries = [e for e in json.loads((catalog / "catalog.json").read_text())["entries"]
               if e["scene_id"] in TASK_IDS]
    assert len(entries) == 12
    validator = SceneValidator(root / "simulation/road_networks")
    npc_count = 0
    for entry in sorted(entries, key=lambda e: (e["network"], e["scene_id"])):
        scene = json.loads((catalog / entry["scenario"]).read_text())
        expected = json.loads((catalog / entry["expected"]).read_text())
        validator.validate(scene, expected)
        for vehicle in scene["vehicles"]:
            if vehicle["agent_config"]["type"] == "sumo":
                npc_count += 1
                state = vehicle["initial_physical_state"]
                assert "speed_kmh" in state
                assert not FORBIDDEN.intersection(state)
        executable = copy.deepcopy(scene)
        executable["experiment_scene"]["setup_assertions"] = expected["setup_assertions"]
        assert frozen[entry["scene_id"]] == executable
    # The remote baseline contains 45 NPCs across this sample.  Local task
    # variants intentionally add route-relevant traffic, so compatibility
    # requires retaining (rather than replacing) that baseline population.
    assert npc_count >= 45


@pytest.mark.parametrize("mutation", ["stopped", "llm", "world_event"])
def test_native_oncoming_contract_rejects_non_native_control(mutation):
    root = Path(__file__).resolve().parents[1]
    folder = root / "evaluation/experiments/scenarios/Basic" / (
        "basic_188_full_network_route__expansion__haikou_longhua")
    scene = json.loads((folder / "scenario.json").read_text())
    expected = json.loads((folder / "expected.json").read_text())
    peer = next(v for v in scene["vehicles"] if v["vehicle_id"] == "oncoming_01")
    if mutation == "stopped":
        peer["initial_physical_state"]["speed_kmh"] = 0
    elif mutation == "llm":
        peer["agent_config"]["type"] = "llm"
    else:
        scene["experiment_world_events"] = [{
            "at_s": 1, "entity_id": "oncoming_01",
            "action": "set_vehicle_speed", "speed_kmh": 0,
        }]
    with pytest.raises(SceneValidationError, match="oncoming|world events"):
        SceneValidator(root / "simulation/road_networks").validate(scene, expected)
