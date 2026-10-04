"""Default steering contracts; real SUMO tests do not call a model or camera."""
import copy

import pytest

from simulation.road_networks import load_road_network
from simulation.traffic_manager import TrafficCoordinator
from simulation.sumo_traffic_manager import SumoTrafficManager


START = "n280957413_n36704818::lane_1"
AFTER_LEFT = "n11113007817_n1859897266::lane_3"
AFTER_STRAIGHT = "n11113007817_n11113007820::lane_3"


def place(manager, lane_id=START, destination=""):
    lane = manager._lane_geometry._lane_by_id[lane_id]
    v = manager.register_vehicle("ego", lane["start_node"], is_llm=True,
                                 start_lane=int(lane["index"]))
    v.current_lane_id = lane_id
    v.current_lane = int(lane["index"])
    v.current_segment = lane["segment_id"]
    v.current_node = lane["start_node"]
    v.edge_progress = 0.5
    v.destination_node = destination
    manager._initialize_vehicle_pose(v)
    assert manager.enable_llm_maneuver_authority("ego")["success"]
    return v


@pytest.fixture
def manager():
    return TrafficCoordinator(load_road_network("beijing_guomao"))


def test_default_does_not_consult_destination_route_and_can_be_overridden(manager, monkeypatch):
    v = place(manager)
    with monkeypatch.context() as patch:
        patch.setattr(manager, "_replan_lane_route", lambda *a: pytest.fail("route consulted"))
        assert manager._prepare_default_continuation(v)
    assert v.planned_maneuver_source == "default_straight"
    assert v.planned_turn == "straight"
    selected = manager.select_vehicle_maneuver("ego", "left")
    assert selected["success"]
    assert v.planned_maneuver_source == "explicit"
    assert v.planned_turn == "left"
    assert not manager._prepare_default_continuation(v)
    assert not manager.select_vehicle_maneuver("ego", "straight")["success"]


def test_default_allows_lane_change_but_bad_command_does_not_revoke_it(manager):
    v = place(manager, AFTER_LEFT)
    manager._prepare_default_continuation(v)
    original = v.planned_connector_id
    assert not manager.change_lane("ego", -1).success
    assert v.planned_connector_id == original
    # Lane 2 is a same-direction neighbour to lane 3 on this approach.
    assert manager.change_lane("ego", 2).success
    assert not v.planned_connector_id
    assert not manager._prepare_default_continuation(v)


@pytest.mark.parametrize("count", [0, 2])
def test_missing_or_ambiguous_straight_does_not_notify_choose_or_brake(manager, count):
    v = place(manager)
    speed = v.current_speed_kmh
    target = v.target_speed_kmh
    straight = next(c for c in manager._lane_geometry._connectors_from[START]
                    if c["turn"] == "straight")
    manager._lane_geometry._connectors_from[START] = [copy.deepcopy(straight) for _ in range(count)]
    for _ in range(5):
        assert not manager._prepare_default_continuation(v)
    assert not v.planned_connector_id and not v.lane_route_actions
    assert manager._pending_events == []
    assert v.current_speed_kmh == speed and v.target_speed_kmh == target


def test_entered_default_cannot_be_overridden(manager):
    v = place(manager)
    manager._prepare_default_continuation(v)
    v.active_connector_id = v.planned_connector_id
    v.planned_connector_id = ""
    assert not manager.select_vehicle_maneuver("ego", "left")["success"]
    assert not manager.change_lane("ego", 2).success


@pytest.mark.parametrize("turn_once", [False, True])
def test_real_sumo_turn_once_then_follow_without_further_commands(turn_once):
    manager = SumoTrafficManager(load_road_network("beijing_guomao"))
    try:
        goal = manager._lane_geometry._lane_by_id[AFTER_STRAIGHT]["end_node"]
        v = place(manager, START if turn_once else AFTER_LEFT, goal)
        if turn_once:
            manager.advance_world_to(0.0)
            assert v.planned_maneuver_source == "default_straight"
            assert manager.select_vehicle_maneuver("ego", "left")["success"]
        manager.set_vehicle_speed("ego", 25)
        entered = []
        for tick in range(1, 1001):
            for event in manager.advance_world_to(tick / 10):
                if event.type == "connector_entered":
                    entered.append(event.details["turn"])
            if v.arrived:
                break
        assert v.arrived
        assert not v.is_crashed
        assert entered == (["left", "straight"] if turn_once else ["straight"])
        assert v.current_lane_id == AFTER_STRAIGHT
        assert not v.planned_connector_id
        assert not v.active_connector_id
        assert manager._speed_mode(v) == 102
    finally:
        manager.close()


@pytest.mark.parametrize("choose", [False, True])
def test_real_sumo_no_unique_straight_never_brakes_and_choice_can_prevent_failure(choose):
    manager = SumoTrafficManager(load_road_network("beijing_guomao"))
    try:
        v = place(manager)
        # Remove only the straight option from the local actuator choices.
        # Compiled network remains unchanged: SUMO must not choose it itself.
        choices = manager._lane_geometry._connectors_from[START]
        manager._lane_geometry._connectors_from[START] = [c for c in choices if c["turn"] != "straight"]
        manager.set_vehicle_speed("ego", 25)
        manager.advance_world_to(1)
        assert not manager._sumo.vehicle.getStops("ego")
        if choose:
            assert manager.select_vehicle_maneuver("ego", "left")["success"]
        events = manager.advance_world_to(30)
        if choose:
            assert not v.route_failed
            assert v.active_connector_id or v.current_lane_id != START
        else:
            assert v.route_failed and not v.arrived and not v.is_crashed
            assert not v.present_in_physics_world
            assert v.current_speed_kmh == pytest.approx(25, abs=0.1)
            assert sum(e.type == "route_failed" for e in events) == 1
            assert not manager.collision_log
    finally:
        manager.close()


def test_real_lane_change_replaces_default_from_old_lane():
    manager = SumoTrafficManager(load_road_network("beijing_guomao"))
    try:
        v = place(manager, AFTER_LEFT)
        manager.set_vehicle_speed("ego", 15)
        manager.advance_world_to(0.1)
        old_connector = v.planned_connector_id
        assert old_connector
        assert manager.change_lane("ego", 2).success
        manager.advance_world_to(5)
        assert not v.is_changing_lane
        assert v.current_lane == 2
        assert v.planned_maneuver_source == "default_straight"
        assert v.planned_connector_id != old_connector
        assert v.planned_from_lane_id == v.current_lane_id
        assert not manager._sumo.vehicle.getStops("ego")
    finally:
        manager.close()
