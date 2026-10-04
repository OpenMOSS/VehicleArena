"""Contract tests for the mandatory SUMO physics engine."""

from __future__ import annotations

import json
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from PIL import Image

from simulation.multi_sim_engine import MultiScenario, MultiSimEngine
from simulation.lane_level_map import LaneLevelMapBuilder
from simulation.road_networks import load_road_network
from simulation.sumo_map import SumoMapConverter
from simulation.sumo_traffic_manager import SumoTrafficManager
from evaluation.experiments.telemetry import ExperimentTrackedEngine
from visualization.lane_world_renderer import Viewport
from visualization.sumo_native_renderer import SumoNativeRenderer


ROOT = Path(__file__).resolve().parents[1]
MAP = ROOT / "simulation" / "road_networks" / "beijing_guomao_lane_level.json"
MELBOURNE_MAP = (
    ROOT / "simulation" / "road_networks" / "melbourne_cbd_lane_level.json"
)
OSAKA_MAP = (
    ROOT / "simulation" / "road_networks" / "osaka_namba_lane_level.json"
)
OSAKA_BASE_MAP = ROOT / "simulation" / "road_networks" / "osaka_namba.json"
SCENE = (
    ROOT / "evaluation" / "fixtures" / "pre_pull_catalog"
    / "Basic" / "basic_009_signalized_intersection" / "scenario.json"
)
CONNECTOR_SCENE = (
    ROOT / "evaluation" / "fixtures" / "sumo"
    / "connector_side_contact.json"
)
VEHICLE_PED_SCENE = (
    ROOT / "evaluation" / "fixtures" / "sumo"
    / "vehicle_pedestrian_contact.json"
)
PEDESTRIAN_SCENE = (
    ROOT / "evaluation" / "fixtures" / "pre_pull_catalog"
    / "Basic" / "basic_001_crosswalk" / "scenario.json"
)
FOLLOWING_SCENE = (
    ROOT / "evaluation" / "fixtures" / "pre_pull_catalog"
    / "Basic" / "basic_013_straight_following" / "scenario.json"
)
CAPABILITY_SCENE = (
    ROOT / "evaluation" / "fixtures" / "pre_pull_catalog"
    / "Basic" / "basic_025_capability_probe" / "scenario.json"
)
FREE_PATH_PEDESTRIAN_SCENE = (
    ROOT / "evaluation" / "fixtures" / "pre_pull_catalog"
    / "Basic" / "basic_073_unmarked_jaywalk" / "scenario.json"
)


pytestmark = pytest.mark.skipif(
    shutil.which("sumo") is None or shutil.which("netconvert") is None
    or not MAP.exists(),
    reason="SUMO runtime or offline road-network bundle is unavailable",
)


def test_lane_map_compiles_without_losing_authored_lanes(tmp_path):
    bundle = SumoMapConverter().convert(MAP, cache_root=tmp_path)
    source = json.loads(MAP.read_text(encoding="utf-8"))
    assert bundle.source_path == str(MAP.resolve())
    root = ET.parse(bundle.net_file).getroot()
    authored_edge_ids = set(bundle.lane_by_edge_index)
    external_edges = [
        edge for edge in root.findall("edge")
        if edge.get("id") in authored_edge_ids
    ]
    compiled_lanes = [
        lane for edge in external_edges for lane in edge.findall("lane")
    ]
    assert len(bundle.edge_by_lane) == len(source["lanes"])
    assert len(compiled_lanes) == (
        len(source["lanes"]) + len(authored_edge_ids))
    assert len(bundle.sidewalk_lane_by_edge) == len(authored_edge_ids)
    assert len(bundle.crosswalk_edge_by_id) == len(source["crosswalks"])
    assert len(bundle.connector_link_index) == sum(
        connector.get("signal_controlled", False)
        for connector in source["connectors"]
    )
    assert len(bundle.connector_via_lane) == len(source["connectors"])
    assert bundle.tls_link_count

    # A turning connector must keep its authored interior geometry. Merely
    # preserving from/to lane IDs would still let netconvert invent a second
    # spline through the junction.
    authored = next(
        item for item in source["connectors"]
        if len(item["centerline_xy"]) >= 5
    )
    compiled_lanes = {
        lane.get("id"): lane
        for edge in root.findall("edge") for lane in edge.findall("lane")
    }
    compiled_shape = [
        tuple(float(value) for value in point.split(","))
        for point in compiled_lanes[
            bundle.connector_via_lane[authored["id"]]
        ].get("shape", "").split()
    ]
    authored_middle = tuple(authored["centerline_xy"][len(
        authored["centerline_xy"]) // 2])
    assert min(
        (point[0] - authored_middle[0]) ** 2
        + (point[1] - authored_middle[1]) ** 2
        for point in compiled_shape
    ) <= 0.02 ** 2


def test_route_endpoint_arrival_is_not_reported_as_physical_stop(tmp_path):
    raw = json.loads(SCENE.read_text(encoding="utf-8"))
    raw["sumo_config"] = {
        "cache_root": str(tmp_path),
        "suppress_warnings": True,
    }
    raw.update({"physics_only_mode": True, "total_time_s": 0.0})
    engine = MultiSimEngine(MultiScenario.from_dict(raw))
    engine.run({})
    vehicle = engine.traffic_mgr.vehicles["queue_1"]
    vehicle.current_speed_kmh = 25.0
    vehicle.pending_arrival = True

    events = engine.traffic_mgr._finalize_pending_arrivals(12.3)

    assert len(events) == 1
    assert events[0].details == {
        "completion": "route_endpoint_crossed",
        "terminal_crossing_speed_kmh": 25.0,
        "present_in_physics_world": False,
        "physically_stopped_at_destination": False,
    }
    assert vehicle.arrived
    assert not vehicle.present_in_physics_world
    assert vehicle.terminal_crossing_speed_kmh == 25.0


def test_sumo_background_uses_native_safe_following(tmp_path):
    raw = json.loads(SCENE.read_text(encoding="utf-8"))
    raw["sumo_config"] = {
        "cache_root": str(tmp_path),
        "suppress_warnings": True,
    }
    raw.update({"physics_only_mode": True, "total_time_s": 0.0})
    engine = MultiSimEngine(MultiScenario.from_dict(raw))
    engine.run({})
    background = engine.traffic_mgr.vehicles["queue_2"]
    assert background.control_authority == "sumo"
    assert engine.traffic_mgr._speed_mode(background) == 31

    background.is_llm = True
    assert engine.traffic_mgr._speed_mode(background) == 102


def test_llm_route_guidance_does_not_install_a_complete_sumo_route(tmp_path):
    network = load_road_network("beijing_guomao")
    manager = SumoTrafficManager(network, sumo_config={
        "cache_root": str(tmp_path),
        "suppress_warnings": True,
    })
    try:
        vehicle = manager.register_vehicle(
            "ego", "n33399858", "n35722739", is_llm=True)
        assert len(manager._route_edges(vehicle)) > 1

        result = manager.enable_llm_maneuver_authority("ego")

        assert result["success"] is True
        assert vehicle.route_control_authority == "llm_maneuver"
        assert vehicle.suggested_lane_route_actions
        assert vehicle.lane_route_actions == []
        assert manager._route_edges(vehicle) == [
            manager.sumo_map.edge_by_lane[vehicle.current_lane_id]]
    finally:
        manager.close()


def test_default_llm_moves_and_explicit_straight_can_replace_default(tmp_path):
    network = load_road_network("beijing_guomao")
    manager = SumoTrafficManager(network, sumo_config={
        "cache_root": str(tmp_path),
        "suppress_warnings": True,
    })
    try:
        vehicle = manager.register_vehicle(
            "ego", "n33399858", "n35722739", is_llm=True)
        manager.enable_llm_maneuver_authority("ego")
        initial_lane_id = vehicle.current_lane_id
        manager.set_vehicle_speed("ego", 50.0)

        # Override before entry; at 20 s default following has already
        # reached the connector, where changing steering must be rejected.
        manager.advance_world_to(5.0)

        assert vehicle.arrived is False
        assert vehicle.current_lane_id == initial_lane_id
        assert vehicle.current_speed_kmh == pytest.approx(50.0, abs=0.1)
        assert vehicle.planned_maneuver_source == "default_straight"
        assert not vehicle.active_connector_id
        assert not manager._sumo.vehicle.getStops("ego")
        first_connector = next(
            action for action in vehicle.suggested_lane_route_actions
            if action["type"] == "connector"
            and action["from_lane_id"] == vehicle.current_lane_id)

        accepted = manager.select_vehicle_maneuver(
            "ego", first_connector["turn"])

        assert accepted["success"] is True
        assert vehicle.planned_maneuver_source == "explicit"
        assert vehicle.planned_connector_id == accepted["connector_id"]
        assert len(vehicle.lane_route_actions) == 1
        assert not manager._sumo.vehicle.getStops("ego")
        assert len(manager._route_edges(vehicle)) == 2
        manager.advance_world_to(40.0)
        assert vehicle.current_lane_id != initial_lane_id
        assert vehicle.planned_maneuver_source != "explicit"
    finally:
        manager.close()


@pytest.mark.parametrize("select_maneuver", [True, False])
def test_moving_llm_gets_startup_decision_before_navigation_hold(select_maneuver):
    from evaluation.multi_agent_runner import (
        apply_resolved_agent_authorities, resolve_agent_specs,
    )
    path = ROOT / "evaluation/fixtures/pre_pull_catalog/MultiLLM" / \
        "multi_009_two_lane_merge/scenario.json"
    raw = json.loads(path.read_text())
    # Authored approaches are no longer consumed by netconvert's junction
    # expansion. Allow travel to the hold boundary before asserting a stop;
    # a vehicle 28 m away at 22 km/h need not stop within two seconds.
    raw["total_time_s"] = 8.0
    scenario = MultiScenario.from_dict(raw)
    apply_resolved_agent_authorities(scenario, resolve_agent_specs(scenario))
    engine = MultiSimEngine(scenario)
    observed = {}

    def make_callback(vehicle_id):
        def callback(_world, t, *_args, **_kwargs):
            if vehicle_id not in observed:
                manager = engine.traffic_mgr
                vehicle = manager.vehicles[vehicle_id]
                observed[vehicle_id] = (t, vehicle.current_speed_kmh)
                if select_maneuver:
                    result = manager.select_vehicle_maneuver(vehicle_id, "straight")
                    assert result["success"]
            return []
        return callback

    result = engine.run({v.vehicle_id: make_callback(v.vehicle_id)
                         for v in scenario.vehicles})
    assert observed == {"ego": (0.0, 22.0), "merging_vehicle": (0.0, 24.0)}
    assert not engine.agent_callback_errors
    if not select_maneuver:
        # Both cars continue their lanes with no safety intervention. The
        # authored merge now collides instead of being protected by a gate.
        assert any(v.is_crashed for v in engine.traffic_mgr.vehicles.values())
        for vehicle in engine.traffic_mgr.vehicles.values():
            assert not hasattr(engine.traffic_mgr, "_sumo_navigation_holds")


def test_llm_cannot_extend_route_past_destination_lane(tmp_path):
    network = load_road_network("beijing_guomao")
    manager = SumoTrafficManager(network, sumo_config={
        "cache_root": str(tmp_path),
        "suppress_warnings": True,
    })
    try:
        vehicle = manager.register_vehicle(
            "ego", "n11374620112", "n10676184554", is_llm=True)
        manager.enable_llm_maneuver_authority("ego")
        first_connector = next(
            action for action in vehicle.suggested_lane_route_actions
            if action["type"] == "connector"
            and action["from_lane_id"] == vehicle.current_lane_id)
        assert manager.select_vehicle_maneuver(
            "ego", first_connector["turn"])["success"] is True
        manager.set_vehicle_speed("ego", 50.0)

        manager.advance_world_to(40.0)

        assert manager._current_lane_terminates_at_destination(vehicle)
        rejected = manager.select_vehicle_maneuver("ego", "straight")
        assert rejected == {
            "success": False,
            "reason": "destination_is_end_of_current_lane",
        }

        manager.advance_world_to(60.0)
        assert vehicle.arrived is True
        assert vehicle.present_in_physics_world is False
    finally:
        manager.close()


def test_sumo_background_queue_discharges_without_rear_end_contact(tmp_path):
    raw = json.loads(SCENE.read_text(encoding="utf-8"))
    raw.update({
        "sumo_config": {
            "cache_root": str(tmp_path),
            "suppress_warnings": True,
        },
        "physics_only_mode": True,
        "total_time_s": 12.0,
        "stop_when_all_vehicles_terminal": False,
    })
    engine = MultiSimEngine(MultiScenario.from_dict(raw))
    engine.run({})

    assert not engine.traffic_mgr.collision_log
    assert all(
        not vehicle.is_crashed
        for vehicle in engine.traffic_mgr.vehicles.values())
    # Every logged lamp value must be a real edge. SUMO state used to be
    # copied directly into VehicleState and then overwritten by the generic
    # physical brake-lamp helper, producing the same event every 0.1 s.
    published = {}
    for event in engine.traffic_mgr.signal_event_log:
        vehicle_state = published.setdefault(event["vehicle_id"], {})
        for key, value in event["changed"].items():
            assert vehicle_state.get(key) is not value
            vehicle_state[key] = value


def test_sumo_keeps_its_installed_route_as_public_cursor_advances(tmp_path):
    """A mirrored lane cursor must never trigger a second SUMO setRoute."""
    raw = json.loads(CAPABILITY_SCENE.read_text(encoding="utf-8"))
    raw.update({
        "sumo_config": {
            "cache_root": str(tmp_path),
            "suppress_warnings": True,
        },
        "physics_only_mode": True,
        "enable_driving_evaluation": False,
        "total_time_s": 60.0,
        "stop_when_all_vehicles_terminal": False,
    })
    for vehicle in raw["vehicles"]:
        vehicle["agent_config"] = {"type": "sumo"}
        vehicle["is_evaluated"] = False

    engine = MultiSimEngine(MultiScenario.from_dict(raw))
    engine.run({})

    assert engine.traffic_mgr._physics_time == pytest.approx(60.0)
    assert all(
        vehicle.physical_pose_authority == "sumo"
        for vehicle in engine.traffic_mgr.vehicles.values()
        if not vehicle.arrived
    )


def test_exactly_placed_sumo_vehicle_accumulates_distance_and_progress(tmp_path):
    raw = json.loads(FOLLOWING_SCENE.read_text(encoding="utf-8"))
    raw.update({
        "sumo_config": {
            "cache_root": str(tmp_path),
            "suppress_warnings": True,
        },
        "physics_only_mode": True,
        "total_time_s": 1.0,
        "stop_when_all_vehicles_terminal": False,
    })
    engine = ExperimentTrackedEngine(MultiScenario.from_dict(raw))
    result = engine.run({})
    ego = engine.traffic_mgr.vehicles["ego"]
    report = result.vehicle_results["ego"].driving_evaluation
    assert ego.distance_traveled_m > 5.0
    assert report["metrics"]["distance_traveled_m"] > 5.0
    assert report["metrics"]["route_progress"] > 0.0
    assert report["score_status"] == "provisional"
    ego_rows = [
        row for row in result._vehicle_trajectory
        if row["vehicle_id"] == "ego"]
    assert ego_rows[0]["time_s"] == 0.0
    assert ego_rows[0]["distance_traveled_m"] == 0.0
    assert ego_rows[1]["distance_traveled_m"] < 2.0
    assert abs(
        ego_rows[1]["edge_progress"] - ego_rows[0]["edge_progress"]
    ) < 0.01
    judge_world = engine._passenger_judge_world_snapshot("ego")
    lead = next(item for item in judge_world["nearby_entities"]
                if item["entity_id"] == "lead")
    assert "distance_m" not in lead
    assert lead["center_distance_m"] > lead[
        "same_lane_bumper_clearance_m"]


def test_sumo_signal_links_follow_vehiclearena_phase(tmp_path):
    bundle = SumoMapConverter().convert(MAP, cache_root=tmp_path)
    source = json.loads(MAP.read_text(encoding="utf-8"))
    connector_id = "connector::intersection::n1802595139::14"
    link_index = bundle.connector_link_index[connector_id]
    root = ET.parse(bundle.net_file).getroot()
    logic = next(
        item for item in root.findall("tlLogic")
        if item.get("id") == bundle.connector_tls[connector_id]
    )
    plan = next(
        item for item in source["signal_plans"]
        if item["node_id"] == bundle.connector_tls[connector_id]
    )
    compiled = iter(logic.findall("phase"))
    for authored in plan["phases"]:
        green = next(compiled)
        expected = (
            "g" if connector_id in authored["connector_ids"] else "r")
        assert green.get("state")[link_index] == expected
        if authored["yellow_s"] > 0:
            yellow = next(compiled)
            expected = (
                "y" if connector_id in authored["connector_ids"] else "r")
            assert yellow.get("state")[link_index] == expected
        if authored["all_red_s"] > 0:
            assert next(compiled).get("state")[link_index] == "r"

    # Crossing links follow the authored pedestrian phase and are otherwise
    # red, so native SUMO pedestrians own signal compliance.
    logic_by_tls = {
        item.get("id"): item for item in root.findall("tlLogic")
    }
    for crosswalk_id, crossing_index in (
            bundle.crosswalk_link_index.items()):
        crossing_logic = logic_by_tls[bundle.crosswalk_tls[crosswalk_id]]
        assert {phase.get("state")[crossing_index]
                for phase in crossing_logic.findall("phase")} <= {"G", "r"}
        assert any(
            phase.get("state")[crossing_index] == "r"
            for phase in crossing_logic.findall("phase"))


def test_direct_llm_command_can_be_scored_for_running_red(tmp_path):
    raw = json.loads(SCENE.read_text(encoding="utf-8"))
    raw.update({
        "sumo_config": {
            "cache_root": str(tmp_path),
            "suppress_warnings": True,
        },
        "total_time_s": 8.0,
    })
    raw["vehicles"] = raw["vehicles"][:1]
    raw["vehicles"][0]["agent_config"] = {
        "type": "llm", "heartbeat_interval_s": 1.0,
    }

    def drive(vw, *_args, **_kwargs):
        vw.navigation.navigation_select_maneuver("straight")
        vw.navigation.navigation_set_speed(30)
        return ["select_straight", "set_speed_30"]

    result = MultiSimEngine(MultiScenario.from_dict(raw)).run({
        "queue_1": drive,
    })
    report = result.vehicle_results["queue_1"].driving_evaluation
    assert result.physics_engine["name"] == "sumo"
    assert report["metrics"]["red_light_violations"] == 1
    assert not report["hard_safety_passed"]


def test_initial_connector_vehicles_start_on_authored_paths(tmp_path):
    raw = json.loads(CONNECTOR_SCENE.read_text(encoding="utf-8"))
    initial_progress = {
        item["vehicle_id"]: item["initial_physical_state"]["progress"]
        for item in raw["vehicles"]
    }
    raw.update({
        "sumo_config": {
            "cache_root": str(tmp_path),
            "suppress_warnings": True,
        },
        "total_time_s": 0.1,
        "stop_when_all_vehicles_terminal": False,
    })
    engine = MultiSimEngine(MultiScenario.from_dict(raw))
    engine.run({})

    # A source-edge spawn teleport would produce path progress unrelated to
    # the authored connector state.  This fixture intentionally starts in
    # side contact, so the hidden SUMO bootstrap must publish exactly one
    # authoritative collision at public time zero instead of losing it.
    collisions = engine.traffic_mgr._collision_log
    assert len(collisions) == 1
    assert collisions[0].time_s == pytest.approx(0.0)
    assert collisions[0].physics_source == "sumo"
    for vehicle_id, progress in initial_progress.items():
        vehicle = engine.traffic_mgr.vehicles[vehicle_id]
        assert vehicle.active_connector_id
        assert vehicle.physical_pose_authority == "sumo"
        shared_pose = engine.traffic_mgr._lane_geometry.vehicle_pose(vehicle)
        # netconvert slightly smooths connector geometry; SUMO's live pose is
        # authoritative while its authored-path projection must remain local.
        assert abs(shared_pose[0] - vehicle.pose_x_m) < 1.0
        assert abs(shared_pose[1] - vehicle.pose_y_m) < 1.0
        assert abs(shared_pose[2] - vehicle.yaw_rad) < 0.05
        assert progress <= vehicle.edge_progress <= progress + 0.05
    assert engine.traffic_mgr._physics_time == pytest.approx(0.1)


def test_vehicle_collision_is_reported_only_by_sumo(tmp_path):
    raw = json.loads(CONNECTOR_SCENE.read_text(encoding="utf-8"))
    raw.update({
        "sumo_config": {
            "cache_root": str(tmp_path),
            "suppress_warnings": True,
        },
        "total_time_s": 1.2,
        "stop_when_all_vehicles_terminal": False,
    })
    for vehicle in raw["vehicles"]:
        vehicle["agent_config"] = {
            "type": "llm", "heartbeat_interval_s": 0.5,
        }

    def drive(vw, *_args, **_kwargs):
        vw.navigation.navigation_set_speed(18)
        return ["set_speed_18"]

    engine = MultiSimEngine(MultiScenario.from_dict(raw))
    engine.run({"ego": drive, "cross_traffic": drive})
    collisions = engine.traffic_mgr._collision_log
    assert len(collisions) == 1
    assert {collisions[0].entity_a, collisions[0].entity_b} == {
        "ego", "cross_traffic",
    }
    assert collisions[0].physics_source == "sumo"
    assert all(
        engine.traffic_mgr.vehicles[vehicle_id].is_crashed
        for vehicle_id in ("ego", "cross_traffic"))


@pytest.mark.parametrize("bumper_gap_m, expect_collision", [
    (1.0, False),
    (-0.1, True),
])
def test_collision_requires_physical_contact(bumper_gap_m, expect_collision):
    raw = json.loads(FOLLOWING_SCENE.read_text(encoding="utf-8"))
    lanes = json.loads(MAP.read_text(encoding="utf-8"))["lanes"]
    lane_id = raw["vehicles"][0]["initial_physical_state"]["lane_id"]
    lane_length_m = next(
        float(lane["length_m"]) for lane in lanes if lane["id"] == lane_id)
    raw.update({
        "sumo_config": {"suppress_warnings": True},
        "physics_only_mode": True,
        "total_time_s": 0.1,
        "stop_when_all_vehicles_terminal": False,
    })
    for vehicle in raw["vehicles"]:
        vehicle["agent_config"] = {"type": "sumo"}
        vehicle["initial_physical_state"].update({
            "speed_kmh": 0,
            "target_speed_kmh": 0,
            "desired_speed_kmh": 0,
        })
    raw["vehicles"][0]["initial_physical_state"]["progress"] = 0.544
    # Sedan length is 4.6 m: one case leaves a one-metre gap, while the
    # other overlaps the car bodies by 0.1 m.
    raw["vehicles"][1]["initial_physical_state"]["progress"] = (
        0.544 + (4.6 + bumper_gap_m) / lane_length_m)

    engine = MultiSimEngine(MultiScenario.from_dict(raw))
    engine.run({})

    assert bool(engine.traffic_mgr._collision_log) == expect_collision
    assert all(vehicle.is_crashed == expect_collision for vehicle in
               engine.traffic_mgr.vehicles.values())


def test_vehicle_pedestrian_contact_is_reported_by_sumo(tmp_path):
    raw = json.loads(VEHICLE_PED_SCENE.read_text(encoding="utf-8"))
    raw.update({
        "sumo_config": {
            "cache_root": str(tmp_path),
            "suppress_warnings": True,
        },
        "physics_only_mode": True,
        "total_time_s": 0.1,
        "stop_when_all_vehicles_terminal": False,
    })
    engine = MultiSimEngine(MultiScenario.from_dict(raw))
    engine.run({})
    collisions = engine.traffic_mgr._collision_log
    assert len(collisions) == 1
    assert collisions[0].collision_type == "vehicle_pedestrian"
    assert collisions[0].physics_source == "sumo"
    assert engine.traffic_mgr.vehicles["ego"].physical_pose_authority == "sumo"
    assert engine.traffic_mgr.pedestrians[
        "pedestrian"].physical_pose_authority == "sumo"
    assert engine.traffic_mgr.pedestrians["pedestrian"].is_crashed


def test_mapped_pedestrian_motion_and_progress_come_from_sumo(tmp_path):
    raw = json.loads(PEDESTRIAN_SCENE.read_text(encoding="utf-8"))
    initial_progress = raw["pedestrians"][0][
        "initial_physical_state"]["progress"]
    raw.update({
        "sumo_config": {"cache_root": str(tmp_path)},
        "physics_only_mode": True,
        "total_time_s": 0.5,
        "stop_when_all_vehicles_terminal": False,
    })
    engine = MultiSimEngine(MultiScenario.from_dict(raw))
    engine.run({})
    pedestrian = engine.traffic_mgr.pedestrians["pedestrian"]
    assert pedestrian.control_authority == "sumo"
    assert pedestrian.physical_pose_authority == "sumo"
    assert initial_progress < pedestrian.crossing_progress < 1.0
    assert pedestrian.physical_pose_xy is not None
    assert engine.traffic_mgr.execute_pedestrian_action(
        "pedestrian", "pedestrian_wait") == {
            "success": False,
            "reason": "pedestrian_is_sumo_controlled",
        }


def test_authored_off_network_pedestrian_path_is_rejected(tmp_path):
    raw = json.loads(
        FREE_PATH_PEDESTRIAN_SCENE.read_text(encoding="utf-8"))
    state = raw["pedestrians"][0]["initial_physical_state"]
    state.pop("crosswalk_id", None)
    state["walking_path_xy"] = [[0.0, 0.0], [5.0, 0.0]]
    raw.update({
        "sumo_config": {"cache_root": str(tmp_path)},
        "physics_only_mode": True,
        "total_time_s": 0.5,
        "stop_when_all_vehicles_terminal": False,
    })
    with pytest.raises(ValueError, match="SUMO-only physics"):
        MultiScenario.from_dict(raw)


def test_llm_pedestrian_can_pause_and_resume_a_sumo_leg(tmp_path):
    network = load_road_network("beijing_guomao")
    manager = SumoTrafficManager(
        network, sumo_config={"cache_root": str(tmp_path)})
    try:
        pedestrian = manager.register_pedestrian(
            "walker", "n1819087311",
            ["n1819087311", "n1819087312"],
            is_llm=True,
        )
        crosswalk = manager._lane_geometry._crosswalk_by_id[
            "crosswalk::intersection::n1317216453::5"]
        pedestrian.start_crossing(
            "n1819087311", "n1819087312", crosswalk["length_m"],
            crosswalk_id=crosswalk["id"],
            path_xy=crosswalk["centerline_xy"],
        )
        pedestrian.crossing_progress = 0.5
        manager.advance_world_to(0.3)
        moving_progress = pedestrian.crossing_progress
        assert moving_progress > 0.5

        assert manager.execute_pedestrian_action(
            "walker", "pedestrian_wait")["success"]
        manager.advance_world_to(0.6)
        assert pedestrian.crossing_progress == pytest.approx(
            moving_progress, abs=0.01)

        assert manager.execute_pedestrian_action(
            "walker", "pedestrian_walk", {"speed": 1.4})["success"]
        manager.advance_world_to(0.9)
        assert pedestrian.crossing_progress > moving_progress
    finally:
        manager.close()


def test_complex_junction_crossing_is_native_sumo_topology(
    tmp_path,
):
    network = load_road_network("fuzhou_wuyi")
    manager = SumoTrafficManager(
        network, sumo_config={"cache_root": str(tmp_path)})
    try:
        crosswalk_id = "crosswalk::intersection::n3092398809::0"
        crosswalk = manager._lane_geometry._crosswalk_by_id[crosswalk_id]
        source_lane = next(
            lane for lane in manager._lane_geometry._lane_by_id.values()
            if lane["segment_id"] == crosswalk["road_segment_id"])
        start_node = source_lane["start_node"]
        end_node = source_lane["end_node"]
        pedestrian = manager.register_pedestrian(
            "complex-walker", start_node, [start_node, end_node])
        pedestrian.start_crossing(
            start_node, end_node, crosswalk["length_m"],
            crosswalk_id=crosswalk_id,
            path_xy=crosswalk["centerline_xy"],
        )
        manager.advance_world_to(0.5)
        assert pedestrian.physical_pose_authority == "sumo"
        assert pedestrian.crossing_progress > 0.0
        crossing_edge = manager.sumo_map.crosswalk_edge_by_id[crosswalk_id]
        assert crossing_edge.startswith(":intersection::n3092398809_c")
        compiled = ET.parse(manager.sumo_map.net_file).getroot().find(
            f"edge[@id='{crossing_edge}']")
        assert compiled is not None
        assert compiled.get("function") == "crossing"
        assert manager.physics_metadata[
            "vehicle_pedestrian_collision_authority"] == "sumo"
    finally:
        manager.close()


def test_split_carriageway_has_two_distinct_native_crossings(tmp_path):
    source = json.loads(MELBOURNE_MAP.read_text(encoding="utf-8"))
    approaches = [
        item for item in source["pedestrian_approaches"]
        if item["node_id"] == "intersection::n2190478937"
    ]
    assert {item["id"].rsplit("::", 1)[-1] for item in approaches} == {
        "east_pavement", "central_refuge", "west_pavement",
    }

    bundle = SumoMapConverter().convert(
        MELBOURNE_MAP, cache_root=tmp_path)
    crossing_ids = [
        "crosswalk::intersection::n2190478937::1",
        "crosswalk::intersection::n2190478937::2",
    ]
    compiled = [bundle.crosswalk_edge_by_id[value] for value in crossing_ids]
    assert len(set(compiled)) == 2
    root = ET.parse(bundle.net_file).getroot()
    assert all(
        root.find(f"edge[@id='{edge_id}']").get("function") == "crossing"
        for edge_id in compiled
    )


def test_overlapping_crosswalks_are_one_native_multi_edge_crossing(tmp_path):
    source = json.loads(OSAKA_MAP.read_text(encoding="utf-8"))
    merged_id = "crosswalk::intersection::n1617139425::5"
    removed_ids = {
        "crosswalk::intersection::n1617139425::6",
        "crosswalk::intersection::n1617139425::7",
    }
    crosswalks = {item["id"]: item for item in source["crosswalks"]}
    assert not removed_ids.intersection(crosswalks)
    assert len(crosswalks[merged_id]["crossed_road_segment_ids"]) == 3

    bundle = SumoMapConverter().convert(OSAKA_MAP, cache_root=tmp_path)
    manifest = json.loads(Path(bundle.manifest_file).read_text(
        encoding="utf-8"))
    crossing_edge = bundle.crosswalk_edge_by_id[merged_id]
    compiled = ET.parse(bundle.net_file).getroot().find(
        f"edge[@id='{crossing_edge}']")
    assert compiled is not None
    assert compiled.get("function") == "crossing"
    assert set(compiled.get("crossingEdges", "").split()) == set(
        manifest["crosswalk_input_edges"][merged_id])
    assert len(manifest["crosswalk_input_edges"][merged_id]) == 4


def test_manual_pedestrian_topology_survives_lane_map_regeneration():
    source = json.loads(OSAKA_BASE_MAP.read_text(encoding="utf-8"))
    rebuilt = LaneLevelMapBuilder().build(source, "osaka_namba")
    merged_id = "crosswalk::intersection::n1617139425::5"
    crosswalks = {item["id"]: item for item in rebuilt["crosswalks"]}
    assert "crosswalk::intersection::n1617139425::6" not in crosswalks
    assert "crosswalk::intersection::n1617139425::7" not in crosswalks
    assert len(crosswalks[merged_id]["crossed_road_segment_ids"]) == 3
    assert len(rebuilt["pedestrian_approaches"]) == 10


def test_failed_episode_releases_libsumo_instance(tmp_path):
    raw = json.loads(SCENE.read_text(encoding="utf-8"))
    raw.update({
        "sumo_config": {"cache_root": str(tmp_path)},
        "total_time_s": 0.1,
    })
    raw["vehicles"] = raw["vehicles"][:1]
    raw["vehicles"][0]["initial_physical_state"] = {
        "connector_id": "connector::does-not-exist",
        "progress": 0.5,
    }
    with pytest.raises(ValueError, match="unknown initial connector_id"):
        MultiSimEngine(MultiScenario.from_dict(raw)).run({})
    assert SumoTrafficManager._active_instance is None


def test_road_closure_is_forwarded_to_sumo_lanes(tmp_path):
    raw = json.loads(SCENE.read_text(encoding="utf-8"))
    raw.update({
        "sumo_config": {
            "cache_root": str(tmp_path),
            "suppress_warnings": True,
        },
        "total_time_s": 0.1,
        "stop_when_all_vehicles_terminal": False,
        "road_events": [{
            "edge_id": "n11374620112_n35722739",
            "event_type": "road_closure",
            "start_tick": 0,
            "end_tick": 1,
            "lanes_affected": 3,
        }],
    })
    raw["vehicles"] = raw["vehicles"][:1]
    result = MultiSimEngine(MultiScenario.from_dict(raw)).run({})
    assert result.physics_engine["closed_lanes"] >= 3
    assert result.physics_engine["vehicle_collision_authority"] == "sumo"


@pytest.mark.skipif(
    shutil.which("sumo-gui") is None
    or (not os.environ.get("DISPLAY") and shutil.which("Xvfb") is None),
    reason="sumo-gui requires a display or Xvfb",
)
def test_native_sumo_renderer_captures_png_and_gif(tmp_path):
    network = load_road_network("beijing_guomao")
    manager = SumoTrafficManager(network, sumo_config={
        "cache_root": str(tmp_path / "cache"),
        "gui": True,
        "gui_width": 480,
        "gui_height": 480,
    })
    try:
        manager.register_vehicle(
            "render-car", "n33399858", "n35722739")
        viewport = Viewport.around_node(
            manager._lane_geometry.data, "n33399858", 70.0)
        renderer = SumoNativeRenderer(
            manager, viewport, width=480, height=480)

        png = tmp_path / "native.png"
        png_result = renderer.render_png_at(0.2, png)
        assert png_result.start_time_s == pytest.approx(0.2)
        with Image.open(png) as image:
            assert image.size == (480, 480)

        gif = tmp_path / "native.gif"
        gif_result = renderer.render_gif(0.3, 1.3, 0.5, gif)
        assert gif_result.frame_count == 3
        with Image.open(gif) as image:
            assert image.size == (480, 480)
            assert image.n_frames >= 2
        assert manager.physics_metadata["rendering_authority"] == "sumo-gui"
    finally:
        manager.close()
    assert SumoTrafficManager._active_instance is None
