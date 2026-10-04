"""Destination distance must not be confused with a physical route end."""
import pytest

from evaluation.driving_evaluator import DrivingEvaluator
from simulation.road_networks import load_road_network
from simulation.traffic_manager import TrafficCoordinator


START = "n10203791835_n9141198422::lane_0"
DEAD_END = "n10203791833_n10203791834::lane_0"
GOAL = "n7141497649"


def place(manager, vehicle, lane_id, progress=0.0):
    lane = manager._lane_geometry._lane_by_id[lane_id]
    vehicle.current_lane_id = lane_id
    vehicle.current_lane = lane["index"]
    vehicle.current_segment = lane["segment_id"]
    vehicle.current_node = lane["start_node"]
    vehicle.edge_progress = progress
    manager._initialize_vehicle_pose(vehicle)


@pytest.fixture
def navigation():
    manager = TrafficCoordinator(load_road_network("beijing_guomao"))
    lane = manager._lane_geometry._lane_by_id[START]
    vehicle = manager.register_vehicle("ego", lane["start_node"], is_llm=True)
    place(manager, vehicle, START)
    vehicle.destination_node = GOAL
    assert manager.enable_llm_maneuver_authority("ego")["success"]
    evaluator = DrivingEvaluator(manager, manager.road_network, ["ego"])
    return manager, vehicle, evaluator


def test_original_dead_end_has_unknown_distance_and_progress_then_recovers(navigation):
    manager, vehicle, evaluator = navigation
    evaluator.observe(0)
    vehicle.edge_progress = 0.5
    evaluator.observe(1)
    best = evaluator.finalize()["ego"]["metrics"]["route_progress"]
    assert 0 < best < 0.999

    place(manager, vehicle, DEAD_END, 1.0)
    assert not manager._replan_lane_route(vehicle, 2)
    assert manager.navigation_route_preview("ego") is None
    assert manager.get_navigation_status("ego") == {
        "status": "blocked", "remaining_distance_m": None, "next_maneuver": None,
    }
    evaluator.observe(2)
    metrics = evaluator.finalize()["ego"]["metrics"]
    assert metrics["route_progress"] is None
    assert metrics["best_route_progress"] == best
    assert evaluator._vehicles["ego"].last_remaining_distance_m is None

    # A local steering decision may clear this flag, but missing destination
    # guidance must still not turn the dead end into an arrival instruction.
    vehicle.lane_route_blocked = False
    assert manager.get_navigation_status("ego")["status"] == "blocked"

    # Reposition only in this test to exercise recovery to the same goal.
    place(manager, vehicle, START, 0.75)
    assert manager._replan_lane_route(vehicle, 3)
    assert manager.get_navigation_status("ego")["status"] == "active"
    evaluator.observe(3)
    assert evaluator.finalize()["ego"]["metrics"]["route_progress"] > best


@pytest.mark.parametrize("authority", ["llm_maneuver", "sumo"])
def test_blocked_status_does_not_report_stale_actions_as_arrival(navigation, authority):
    manager, vehicle, _ = navigation
    vehicle.route_control_authority = authority
    vehicle.lane_route_blocked = True
    vehicle.edge_progress = 1.0
    vehicle.lane_route_actions = [{"type": "lane_change", "target_lane_index": 1}]
    status = manager.get_navigation_status("ego")
    assert status["remaining_distance_m"] is None
    assert status["next_maneuver"] is None


@pytest.mark.parametrize("status,remaining", [
    ("blocked", 0.0), ("route_failed", 0.0), ("inactive", 0.0),
    ("unavailable", 0.0), ("active", None), ("active", float("nan")),
    ("active", float("inf")), ("active", -1.0),
])
def test_invalid_samples_neither_add_progress_nor_use_physical_action_fallback(
        navigation, status, remaining):
    _, vehicle, evaluator = navigation
    acc = evaluator._vehicles["ego"]
    update = lambda state, distance: evaluator._observe_navigation(
        acc, vehicle, {"status": state, "remaining_distance_m": distance})
    update("active", 100.0)
    update("active", 60.0)
    vehicle.lane_route_actions = [{"type": "connector"}]
    vehicle.lane_route_action_index = 1
    vehicle.edge_progress = 1.0
    update(status, remaining)
    assert acc.last_remaining_distance_m is None
    assert not acc.route_distance_available
    assert evaluator._route_progress(acc, vehicle) == pytest.approx(0.4)
    metrics = evaluator.finalize()["ego"]["metrics"]
    assert metrics["route_progress"] is None
    assert metrics["best_route_progress"] == 0.4

    vehicle.destination_node = "different_goal"
    update(status, remaining)
    assert acc.initial_remaining_distance_m is None
    assert evaluator._route_progress(acc, vehicle) == 0.0


def test_route_failure_is_unknown_but_real_arrival_is_complete(navigation):
    manager, vehicle, evaluator = navigation
    evaluator.observe(0)
    vehicle.route_failed = True
    vehicle.route_failure_reason = "unresolved_route_endpoint"
    status = manager.get_navigation_status("ego")
    assert status["status"] == "route_failed"
    assert status["remaining_distance_m"] is None
    assert status["next_maneuver"] is None
    evaluator.observe(1)
    assert evaluator.finalize()["ego"]["metrics"]["route_progress"] is None

    vehicle.route_failed = False
    vehicle.arrived = True
    assert manager.get_navigation_status("ego")["remaining_distance_m"] == 0.0
    assert evaluator.finalize()["ego"]["metrics"]["route_progress"] == 1.0


def test_initially_blocked_route_has_no_progress_baseline(navigation):
    manager, vehicle, evaluator = navigation
    place(manager, vehicle, DEAD_END, 1.0)
    assert not manager._replan_lane_route(vehicle, 0)
    evaluator.observe(0)
    acc = evaluator._vehicles["ego"]
    assert acc.initial_remaining_distance_m is None
    metrics = evaluator.finalize()["ego"]["metrics"]
    assert metrics["route_progress"] is None
    assert metrics["best_route_progress"] == 0.0


def test_future_lane_change_does_not_make_adjacent_car_current_leader(
        navigation):
    """A future route action is not evidence of a current rear-end hazard."""
    manager, ego, _ = navigation
    lane_groups = [
        lanes for lanes in manager._lane_geometry._lanes.values()
        if len(lanes) >= 2 and all(lane["length_m"] > 80 for lane in lanes[:2])
    ]
    source_lane, target_lane = lane_groups[0][:2]
    peer = manager.register_vehicle("adjacent_peer", target_lane["start_node"])
    place(manager, ego, source_lane["id"], progress=0.50)
    place(manager, peer, target_lane["id"], progress=0.54)
    ego.is_changing_lane = False
    ego.target_lane = -1
    ego.lane_route_actions = [{
        "type": "lane_change",
        "from_lane_id": source_lane["id"],
        "to_lane_id": target_lane["id"],
        "target_lane_index": target_lane["index"],
    }]
    ego.lane_route_action_index = 0

    leader, gap_m = manager._route_path_leader(ego)

    assert leader is None
    assert gap_m == float("inf")
