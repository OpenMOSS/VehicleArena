"""Native NPC initial conditions are not persistent controller commands."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from evaluation.experiments.scene_catalog import normalize_native_npc_inputs
from evaluation.experiments.scene_validator import SceneValidationError, SceneValidator
from evaluation.experiments.telemetry import ExperimentTrackedEngine
from simulation.multi_sim_engine import MultiScenario
from simulation.sumo_traffic_manager import SumoTrafficManager

ROOT = Path(__file__).parents[1]
SCENE = ROOT / "evaluation/fixtures/pre_pull_catalog/Basic/basic_005_lane_change"


def test_background_actors_receive_no_per_step_actuator_commands():
    manager = object.__new__(SumoTrafficManager)
    manager.vehicles = {"npc": SimpleNamespace(
        is_llm=False, route_failed=False)}
    manager._sumo = SimpleNamespace(vehicle=Mock(), person=Mock())
    manager._sumo.vehicle.getIDList.return_value = ["npc"]
    manager._prepare_default_continuation = Mock(return_value=False)
    manager._ensure_sumo_vehicle = Mock()
    manager._physics_time = 1.0
    manager.pedestrians = {"ped": SimpleNamespace(
        is_llm=False, has_arrived=False)}
    manager._sumo_pedestrian_proxy = {"ped": "proxy"}
    manager._sumo.person.getIDList.return_value = ["proxy"]
    manager._ensure_sumo_pedestrian = Mock()

    manager._submit_controls()

    assert manager._sumo.vehicle.method_calls == [
        ("getIDList", (), {})]
    assert manager._sumo.person.method_calls == [
        ("getIDList", (), {})]


def test_cleanup_preserves_llm_targets_and_removes_npc_world_events():
    scenario = {"vehicles": [
        {"vehicle_id": "ego", "agent_config": {"type": "llm"},
         "initial_physical_state": {"speed_kmh": 20, "target_speed_kmh": 30,
                                    "desired_speed_kmh": 40}},
        {"vehicle_id": "npc", "agent_config": {"type": "sumo"},
         "initial_physical_state": {"speed_kmh": 12, "target_speed_kmh": 12,
                                    "desired_speed_kmh": 30}}],
        "experiment_world_events": [{"at_s": 4, "entity_id": "npc",
                                     "action": "set_vehicle_speed", "speed_kmh": 0}]}
    original = copy.deepcopy(scenario)
    normalize_native_npc_inputs(scenario)
    assert scenario["vehicles"][0] == original["vehicles"][0]
    assert scenario["vehicles"][1]["initial_physical_state"] == {"speed_kmh": 12}
    assert "experiment_world_events" not in scenario
    cleaned = copy.deepcopy(scenario)
    normalize_native_npc_inputs(scenario)
    assert scenario == cleaned


def test_catalog_validator_rejects_inert_npc_targets():
    scenario = json.loads((SCENE / "scenario.json").read_text())
    expected = json.loads((SCENE / "expected.json").read_text())
    validator = SceneValidator(ROOT / "simulation/road_networks")
    assert validator.validate(scenario, expected)["scene_id"] == scenario["scenario_id"]
    scenario["vehicles"][1]["initial_physical_state"]["target_speed_kmh"] = 12
    with pytest.raises(SceneValidationError, match="native SUMO NPC"):
        validator.validate(scenario, expected)


def test_real_sumo_cleanup_preserves_motion_without_speed_override():
    cleaned = json.loads((SCENE / "scenario.json").read_text())
    cleaned.update(total_time_s=10, physics_only_mode=True, enable_driving_evaluation=False)
    legacy = copy.deepcopy(cleaned)
    for vehicle in legacy["vehicles"]:
        if vehicle["agent_config"]["type"] == "sumo":
            state = vehicle["initial_physical_state"]
            state.update(target_speed_kmh=state["speed_kmh"], desired_speed_kmh=30)
    motion = []
    for raw in [legacy, cleaned]:
        engine = ExperimentTrackedEngine(MultiScenario.from_dict(raw))
        engine.run({})
        assert not engine.agent_callback_errors
        rows = engine._vehicle_trajectory
        assert max(r["speed_kmh"] for r in rows if r["vehicle_id"] == "slow_lead") > 30
        fields = ("vehicle_id", "time_s", "pose_x_m", "pose_y_m", "speed_kmh", "current_lane_id")
        motion.append([tuple(row[k] for k in fields) for row in rows])
    assert motion[0] == motion[1]


def test_real_sumo_npc_motion_does_not_follow_mirrored_target_speed():
    scenario = json.loads((SCENE / "scenario.json").read_text())
    scenario.update(
        total_time_s=10,
        physics_only_mode=False,
        enable_driving_evaluation=False,
    )
    engine = ExperimentTrackedEngine(MultiScenario.from_dict(scenario))
    injected = False

    def inject_stale_override(*_args, **_kwargs):
        nonlocal injected
        if not injected:
            npc = engine.traffic_mgr.vehicles["slow_lead"]
            npc.target_speed_kmh = 0.0
            injected = True
        return []

    engine.run({"ego": inject_stale_override})
    assert injected
    rows = [
        row for row in engine._vehicle_trajectory
        if row["vehicle_id"] == "slow_lead"
    ]
    assert max(row["speed_kmh"] for row in rows) > 30
