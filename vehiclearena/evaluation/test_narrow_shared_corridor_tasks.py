"""The narrow-road study uses a real unsignalized shared corridor."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from evaluation.experiments.scene_catalog import (
    NARROW_CORRIDOR_NETWORK,
    NARROW_CORRIDOR_VARIANTS,
    Topology,
    _narrow_road_sequence,
    normalize_native_npc_inputs,
)
from evaluation.experiments.scene_validator import (
    SceneValidationError,
    SceneValidator,
)
from evaluation.experiments.telemetry import ExperimentTrackedEngine
from simulation.multi_sim_engine import MultiScenario


ROOT = Path(__file__).resolve().parents[2]
NETWORK_ROOT = ROOT / "vehiclearena" / "simulation" / "road_networks"
SCENARIO_ROOT = (
    ROOT / "vehiclearena" / "evaluation" / "experiments" / "scenarios")


@pytest.fixture(scope="module")
def topology() -> Topology:
    return Topology(NARROW_CORRIDOR_NETWORK, NETWORK_ROOT)


@pytest.mark.parametrize("variant", NARROW_CORRIDOR_VARIANTS,
                         ids=lambda item: item[0])
def test_variant_is_a_valid_shared_corridor_sequence(
    topology: Topology, variant,
) -> None:
    asset = _narrow_road_sequence(topology, variant)
    normalize_native_npc_inputs(asset.scenario)
    validator = SceneValidator(NETWORK_ROOT)
    validator._runtime[NARROW_CORRIDOR_NETWORK] = topology.runtime
    validator.validate(asset.scenario, asset.expected)

    vehicles = {item["vehicle_id"]: item
                for item in asset.scenario["vehicles"]}
    assert vehicles["ego"]["is_evaluated"] is True
    assert vehicles["oncoming"]["is_evaluated"] is False
    assert any(item["kind"] == "narrow_shared_corridor_sequence"
               for item in asset.expected["setup_assertions"])


def test_moving_oncoming_outside_corridor_is_rejected(
    topology: Topology,
) -> None:
    asset = _narrow_road_sequence(topology, NARROW_CORRIDOR_VARIANTS[0])
    scenario = copy.deepcopy(asset.scenario)
    normalize_native_npc_inputs(scenario)
    scenario["vehicles"][1]["initial_physical_state"]["lane_id"] = (
        scenario["vehicles"][0]["initial_physical_state"]["lane_id"])
    validator = SceneValidator(NETWORK_ROOT)
    validator._runtime[NARROW_CORRIDOR_NETWORK] = topology.runtime
    with pytest.raises(SceneValidationError, match="oncoming initial"):
        validator.validate(scenario, asset.expected)


def test_frozen_catalog_contains_nine_sequence_tasks() -> None:
    catalog = json.loads((SCENARIO_ROOT / "catalog.json").read_text())
    entries = [
        item for item in catalog["entries"]
        if item["experiment_id"] == "Basic"
        and "separated_two_way_meeting" in item["tags"]
    ]
    assert len(entries) == 9
    assert {int(item["scene_id"].split("_")[1]) for item in entries} == {
        45, 46, 47, 48, 125, 126, 127, 128, 194}
    assert {item["network"] for item in entries} == {
        NARROW_CORRIDOR_NETWORK, "helsinki_keskusta"}
    assert [item["environment_group"] for item in entries] == [
        "weather_only", "daynight_only", "combined", "control",
        "weather_only", "daynight_only", "combined", "control",
        "daynight_only",
    ]

    validator = SceneValidator(NETWORK_ROOT)
    for entry in entries:
        scenario = json.loads(
            (SCENARIO_ROOT / entry["scenario"]).read_text())
        expected = json.loads(
            (SCENARIO_ROOT / entry["expected"]).read_text())
        validator.validate(scenario, expected)


def _reference_scenario() -> dict:
    path = (
        SCENARIO_ROOT / "Basic"
        / "basic_045_narrow_road__prague_vodickova__forward_peer_early"
        / "scenario.json")
    scenario = json.loads(path.read_text())
    for vehicle in scenario["vehicles"]:
        vehicle.setdefault("agent_config", {})["type"] = "sumo"
    return scenario


@pytest.mark.parametrize("study,scene_id", [
    ("Basic", "basic_045_narrow_road__prague_vodickova__forward_peer_early"),
    ("MultiLLM", "multi_021_narrow_meeting__prague_vodickova__forward_peer_early"),
])
def test_evaluated_role_does_not_install_a_yielding_driver(study, scene_id):
    raw = json.loads((SCENARIO_ROOT / study / scene_id / "scenario.json").read_text())
    raw.update(total_time_s=8.0, physics_only_mode=True,
               stop_when_all_vehicles_terminal=False)
    raw.pop("weather_keyframes", None)
    raw.pop("daynight_keyframes", None)
    trajectories = []
    for evaluated in (False, True):
        scenario = copy.deepcopy(raw)
        for vehicle in scenario["vehicles"]:
            vehicle["agent_config"] = {"type": "sumo"}
            vehicle["is_evaluated"] = evaluated
        result = ExperimentTrackedEngine(MultiScenario.from_dict(scenario)).run({})
        assert "evaluated_npc_shared_corridor_holds" not in result.physics_engine
        assert "sumo_speed_factor_policy" not in result.physics_engine
        fields = ("vehicle_id", "time_s", "pose_x_m", "pose_y_m", "speed_kmh")
        trajectories.append([tuple(row[key] for key in fields)
                             for row in result._vehicle_trajectory])
    assert trajectories[0] == trajectories[1]


def test_replacement_needs_no_shared_corridor_collision_overlay() -> None:
    scenario = _reference_scenario()
    scenario["total_time_s"] = 8.0
    scenario["stop_when_all_vehicles_terminal"] = False
    for vehicle in scenario["vehicles"]:
        # Scoring roles never change the native driving policy.
        vehicle["is_evaluated"] = False

    result = ExperimentTrackedEngine(
        MultiScenario.from_dict(scenario)).run({})

    assert result._collision_log == []
    assert result.physics_engine["vehicle_collision_authority"] == "sumo"
