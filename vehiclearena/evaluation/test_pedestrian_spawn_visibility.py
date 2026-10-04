"""Scheduled pedestrians are not physical hazards before SUMO inserts them."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from evaluation.experiments.telemetry import ExperimentTrackedEngine
from simulation.multi_sim_engine import MultiScenario
from simulation.perception_model import PerceptionModel, VehicleSignalState
from simulation.sumo_traffic_manager import SumoTrafficManager
from simulation.world_state import WorldState


@pytest.fixture
def world():
    manager = object.__new__(SumoTrafficManager)
    pedestrian = SimpleNamespace(
        ped_id="ped", is_spawned=False, has_arrived=False, is_crashed=False,
        is_on_crosswalk=True, active_crosswalk_id="cw", crossing_progress=0.0,
        authored_crosswalk_path_xy=None, authored_crosswalk_id=None,
        authored_crosswalk_progress=0.0, speed=1.4,
        position=SimpleNamespace(at_node="junction", crossing_from=None,
                                 crossing_to=None),
    )
    vehicle = SimpleNamespace(
        vehicle_id="ego", current_speed_kmh=30.0, length_m=4.6,
        pose_x_m=0.0, pose_y_m=0.0, yaw_rad=0.0,
        signal_state=VehicleSignalState(), arrived=False,
    )
    manager.vehicles = {"ego": vehicle}
    manager.pedestrians = {"ped": pedestrian}
    manager._lane_geometry = SimpleNamespace(
        connector_crosswalk_points=lambda _: [
            {"crosswalk_id": "cw", "connector_distance_s_m": 10.0}],
        pedestrian_pose=lambda _: (10.0, 0.0),
        _node_to_junction={}, data={"crosswalks": []},
    )
    manager._distance_to_connector_s = lambda *args: 10.0
    manager._current_weather = "sunny"
    manager._current_wind_speed_mps = 0.0
    manager._is_night = False
    manager._physics_time = 0.0
    manager.perception_log = []
    manager.perception_log_limit = 100
    manager.perception_log_dropped = 0
    manager.perception_model = PerceptionModel(manager)
    road = SimpleNamespace(nodes={"junction": SimpleNamespace(has_crosswalk=True)})
    state = WorldState(manager, road)
    state._observer_distance_to_node = lambda *args: 10.0
    return manager, pedestrian, vehicle, state


@pytest.mark.parametrize("authored", [False, True])
@pytest.mark.parametrize("lifecycle", ["scheduled", "spawned", "arrived", "crashed"])
def test_connector_hazards_require_a_live_spawned_occupant(world, authored, lifecycle):
    manager, pedestrian, vehicle, _ = world
    pedestrian.is_spawned = lifecycle != "scheduled"
    pedestrian.has_arrived = lifecycle == "arrived"
    pedestrian.is_crashed = lifecycle == "crashed"
    if authored:
        pedestrian.is_on_crosswalk = False
        pedestrian.authored_crosswalk_path_xy = [(10.0, -2.0), (10.0, 2.0)]
        pedestrian.authored_crosswalk_id = "cw"
    hazards = manager._connector_pedestrian_hazards(vehicle, "connector")
    if lifecycle == "spawned":
        assert len(hazards) == 1
        assert hazards[0]["pedestrian_id"] == "ped"
        assert hazards[0]["vehicle_ttc_s"] == pytest.approx(1.2)
    else:
        assert hazards == []


@pytest.mark.parametrize("spawned", [False, True])
@pytest.mark.parametrize("claimed_distance", [None, 10.0])
def test_visual_distance_fallback_cannot_reveal_scheduled_pedestrians(
    world, spawned, claimed_distance,
):
    manager, pedestrian, _, state = world
    pedestrian.is_spawned = spawned
    detection = manager.perception_model.detect_entity(
        "ego", "ped", claimed_distance_m=claimed_distance)
    assert (detection is not None) == spawned
    assert [item.entity_id for item in state.look_around("ego")] == (
        ["ped"] if spawned else [])
    if not spawned:
        assert not any(row.get("detected") for row in manager.perception_log)


@pytest.mark.parametrize("spawned", [False, True])
@pytest.mark.parametrize("observer", ["", "ego"])
def test_crosswalk_count_excludes_scheduled_pedestrians(world, spawned, observer):
    _, pedestrian, _, state = world
    pedestrian.is_spawned = spawned
    info = state.scan_crosswalk("junction", observer_id=observer)
    assert info.has_crosswalk
    assert info.pedestrians_crossing == int(spawned)
    assert len(info.pedestrian_details) == int(spawned)


@pytest.mark.parametrize("name,expected_start_s", [
    ("basic_002_crosswalk__hongkong_central", 2.6),
    ("basic_097_crosswalk__guangzhou_tianhe", 0.4),
    ("basic_098_crosswalk__wuhan_hankou", 2.6),
    ("basic_168_crosswalk__expansion__wuhan_hankou__site_08", 0.7),
])
def test_sumo_spawn_gates_awareness_and_scoring(name, expected_start_s):
    path = Path(__file__).parent / "experiments/scenarios/Basic" / name / "scenario.json"
    raw = json.loads(path.read_text())
    starts = {p["ped_id"]: p["start_time"] * 60 for p in raw["pedestrians"]}
    assert all(start == pytest.approx(expected_start_s) for start in starts.values())
    raw.update(physics_only_mode=True, total_time_s=4.0,
               stop_when_all_vehicles_terminal=False, enable_driving_evaluation=True)
    for vehicle in raw["vehicles"]:
        vehicle["agent_config"] = {"type": "sumo"}
    focal_id = raw["experiment_scene"]["focal_vehicle_id"]

    class ObservedEngine(ExperimentTrackedEngine):
        def __init__(self, scenario):
            super().__init__(scenario)
            self.awareness_samples = []

        def _log_per_substep(self, physics_time, tick_index, trigger_events):
            super()._log_per_substep(physics_time, tick_index, trigger_events)
            awareness = self.traffic_mgr.get_driving_awareness(
                focal_id, ground_truth=True)
            self.awareness_samples.append((physics_time, awareness["pedestrian_hazards"]))
            for hazard in awareness["pedestrian_hazards"]:
                assert self.traffic_mgr.pedestrians[hazard["pedestrian_id"]].is_spawned

    engine = ObservedEngine(MultiScenario.from_dict(raw))
    result = engine.run({})
    assert not engine.agent_callback_errors
    before = [hazards for time, hazards in engine.awareness_samples if time < expected_start_s - 1e-6]
    assert before and not any(before)
    assert any(hazards for time, hazards in engine.awareness_samples if time >= expected_start_s)
    for row in result._pedestrian_trajectory:
        if row["time_s"] < starts[row["ped_id"]] - 1e-6:
            assert not row["spawned"]
    report = result.vehicle_results[focal_id].driving_evaluation
    for event in report["events"]:
        if str(event.get("hazard", "")).startswith("pedestrian:"):
            assert event["time_s"] >= expected_start_s - 1e-6
    for episode in report["decision_episodes"]:
        if episode["type"] == "pedestrian_response":
            assert episode["start_time_s"] >= expected_start_s - 1e-6
    for deduction in report["driving_process"]["deductions"]:
        if deduction["type"].startswith("pedestrian_"):
            assert deduction["start_time_s"] >= expected_start_s - 1e-6
