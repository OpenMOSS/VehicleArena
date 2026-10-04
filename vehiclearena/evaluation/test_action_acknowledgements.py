"""Action acknowledgements are not observations of completed physical motion."""
from dataclasses import asdict

import pytest

from evaluation.context_runtime import build_vehicle_self_now
from evaluation.driving_evaluator import DrivingEvaluator
from simulation.multi_sim_engine import MultiSimEngine
from simulation.road_networks import load_road_network
from simulation.sumo_traffic_manager import SumoTrafficManager
from vehiclearena import VehicleWorld


@pytest.fixture
def manager():
    manager = SumoTrafficManager(load_road_network("beijing_guomao"))
    lane = manager._lane_geometry._lane_by_id["n11113007817_n1859897266::lane_3"]
    v = manager.register_vehicle("ego", lane["start_node"], is_llm=True, start_lane=3)
    v.current_lane_id = lane["id"]
    v.current_lane = 3
    v.current_segment = lane["segment_id"]
    v.current_node = lane["start_node"]
    v.edge_progress = 0.5
    manager._initialize_vehicle_pose(v)
    assert manager.enable_llm_maneuver_authority("ego")["success"]
    manager.set_vehicle_speed("ego", 0)
    manager.advance_world_to(0.1)
    try:
        yield manager
    finally:
        manager.close()


def intercepts(manager, queued=True):
    engine = object.__new__(MultiSimEngine)
    engine.traffic_mgr = manager
    engine.road_network = manager.road_network
    engine._vw = {"ego": VehicleWorld()}
    engine._sim_time = 0.1
    engine._collect_agent_commands = queued
    engine._next_agent_command_id = 1
    engine._agent_command_queue = []
    engine._pending_command_receipts = []
    engine.agent_callback_errors = []
    engine._install_driving_intercepts("ego")
    return engine, engine._vw["ego"].navigation


@pytest.mark.parametrize("queued", [True, False])
def test_lane_ack_contains_only_request_target_not_completion(manager, queued):
    engine, nav = intercepts(manager, queued)
    result = nav.navigation_change_lane("right")
    assert result["success"] and result["target_lane"] == 2
    assert result["status"] == ("queued" if queued else "accepted")
    for key in ("lane", "new_lane", "is_changing_lane", "lane_change_duration_s",
                "lane_change_progress", "completed"):
        assert key not in result
    assert manager.vehicles["ego"].current_lane == 3
    assert manager.vehicles["ego"].lane_change_completions == 0
    if queued:
        assert manager.vehicles["ego"].lane_change_attempts == 0
        engine._commit_agent_commands(0.1)
    assert manager.vehicles["ego"].lane_change_attempts == 1
    rejected = nav.navigation_change_lane("right")
    assert not rejected["success"]
    assert manager.vehicles["ego"].lane_change_attempts == 1


@pytest.mark.parametrize("duration,completed", [(2.5, True), (3.5, True), (6.0, True), (3.5, False)])
def test_native_lane_change_counts_physical_completion_not_expiry(manager, duration, completed):
    v = manager.vehicles["ego"]
    v.lane_change_duration_s = duration
    native = manager._sumo.vehicle
    if not completed:
        # Fault injection only: reproduce the historical stopped-on-lane
        # state. Production code never installs this navigation stop.
        native.setStop("ego", native.getRoadID("ego"),
                       pos=native.getLanePosition("ego"),
                       laneIndex=native.getLaneIndex("ego"), duration=100)
        manager.advance_world_to(0.2)
        assert native.isStopped("ego")
    assert manager.change_lane("ego", 2).success
    assert not manager.change_lane("ego", 2).success
    manager.advance_world_to(7)
    assert v.current_lane == (2 if completed else 3)
    assert not v.is_changing_lane
    assert "ego" not in manager._sumo_lane_changes
    assert v.lateral_offset_m == pytest.approx(native.getLateralLanePosition("ego"))
    if completed:
        width = manager._sumo.lane.getWidth(native.getLaneID("ego"))
        assert abs(v.lateral_offset_m) + v.width_m / 2 <= width / 2 + 0.05
    assert v.lane_change_attempts == 1
    assert v.lane_change_completions == int(completed)
    assert v.lane_change_uncompleted == int(not completed)
    before = (v.lane_change_completions, v.lane_change_uncompleted)
    manager.advance_world_to(9)
    assert (v.lane_change_completions, v.lane_change_uncompleted) == before
    evaluator = DrivingEvaluator(manager, manager.road_network, ["ego"])
    evaluator.observe(9)
    metrics = evaluator.finalize()["ego"]["metrics"]
    assert metrics["lane_changes"] == int(completed)
    assert metrics["lane_change_attempts"] == 1
    assert metrics["lane_change_uncompleted"] == int(not completed)
    assert metrics["lane_change_pending"] == 0


def test_lane_index_crossing_does_not_count_as_complete(manager):
    v = manager.vehicles["ego"]
    assert manager.change_lane("ego", 2).success
    manager.advance_world_to(2.7)
    assert v.current_lane == 2
    assert abs(v.lateral_speed_mps) > 0.1
    assert v.lane_change_completions == 0
    assert v.lane_change_request_outcome == "pending"


def test_stop_intent_is_not_reported_as_observed_standstill(manager):
    from simulation.world_state import WorldState
    v = manager.vehicles["ego"]
    v.current_speed_kmh = 20
    assert manager.emergency_stop_vehicle("ego")["success"]
    world = WorldState(manager, manager.road_network)
    assert world.get_entity_location("ego")["is_stopped"] is False


def test_terminal_interrupt_counts_once_and_authored_state_is_not_an_attempt(manager):
    v = manager.vehicles["ego"]
    v.is_changing_lane = True  # authored state, no accepted request
    v.is_crashed = True
    manager._observe_lane_change_outcomes()
    assert v.lane_change_uncompleted == 0
    v.is_crashed = False
    assert manager.change_lane("ego", 2).success
    v.is_crashed = True
    manager._observe_lane_change_outcomes()
    manager._observe_lane_change_outcomes()
    assert v.lane_change_attempts == v.lane_change_uncompleted == 1
    assert v.lane_change_completions == 0


def test_stopped_native_vehicle_with_stale_lateral_speed_does_not_remain_pending(manager):
    v = manager.vehicles["ego"]
    native = manager._sumo.vehicle
    native.setStop("ego", native.getRoadID("ego"),
                   pos=native.getLanePosition("ego"),
                   laneIndex=native.getLaneIndex("ego"), duration=100)
    manager.advance_world_to(0.2)
    assert manager.change_lane("ego", 2).success
    manager.advance_world_to(0.3)
    assert native.isStopped("ego")
    # SUMO can retain the pre-stop lateral speed. Reproduce that observation
    # at the request deadline without altering the vehicle's native pose.
    v.lateral_speed_mps = 0.9
    manager._physics_time = 4.0
    manager._observe_lane_change_outcomes()
    assert v.lane_change_uncompleted == 1
    assert v.lane_change_request_outcome == "uncompleted"


def test_self_observation_never_exposes_timer_or_internal_outcome(manager):
    from simulation.world_state import FrozenStateView
    v = manager.vehicles["ego"]
    v.lane_change_progress = 1.0
    v.lane_change_request_outcome = "completed"
    result = build_vehicle_self_now(asdict(v))
    assert result["current_lane"] == 3
    for key in ("is_changing_lane", "lane_change_progress", "lane_change_request_outcome",
                "lane_change_attempts", "lane_change_completions", "lane_change_uncompleted"):
        assert key not in result
    frozen = FrozenStateView.from_state(v).as_dict()
    assert "lane_change_progress" not in frozen
    assert "lane_change_request_outcome" not in frozen
    assert "lane_change_completions" not in frozen


@pytest.mark.parametrize("method,arguments", [
    ("navigation_set_speed", {"speed_kmh": 20}),
    ("navigation_emergency_stop", {}),
    ("navigation_select_maneuver", {"direction": "straight"}),
    ("navigation_u_turn", {}),
])
def test_other_motion_tools_explicitly_acknowledge_queue_only(manager, method, arguments):
    _, nav = intercepts(manager)
    result = getattr(nav, method)(**arguments)
    assert result["success"] and result["status"] == "queued"
    assert result["queued_for_world_commit"]
    assert manager.vehicles["ego"].current_speed_kmh == 0
    assert "completed" not in result and "is_stopped" not in result


@pytest.mark.parametrize("method,arguments", [
    ("navigation_set_speed", {"speed_kmh": 20}),
    ("navigation_emergency_stop", {}),
    ("navigation_change_lane", {"direction": "right"}),
    ("navigation_select_maneuver", {"direction": "straight"}),
    ("navigation_u_turn", {}),
])
def test_standalone_motion_tools_cannot_report_success_without_backend(method, arguments):
    result = getattr(VehicleWorld().navigation, method)(**arguments)
    assert result == {"success": False, "reason": "driving_backend_unavailable"}


def test_display_navigation_tools_use_real_routes_and_exit_clears_preview(manager):
    _, nav = intercepts(manager)
    lane = manager._lane_geometry._lane_by_id[manager.vehicles["ego"].current_lane_id]
    target = lane["end_node"]
    assert nav.navigation_route_plan(target)["success"]
    assert nav.navigation_reroute()["success"]
    assert nav._lane_route_preview_provider(target)
    original = nav.current_route.destination
    bad = nav.navigation_destination_change("not a real mapped destination")
    assert not bad["success"] and nav.current_route.destination == original
    assert not nav.navigation_midWay_add([target])["success"]
    assert not nav.navigation_midWay_delete(number=0)["success"]
    assert not nav.waypoints
    assert nav.navigation_exit()["success"]
    assert nav._lane_route_preview is None
    assert nav._lane_route_preview_provider(target) is None
    assert not nav.navigation_reroute()["success"]


def test_failed_reroute_does_not_claim_a_map_route(manager):
    _, nav = intercepts(manager)
    assert nav.navigation_route_plan("n7141497649")["success"]
    v = manager.vehicles["ego"]
    lane = manager._lane_geometry._lane_by_id["n10203791833_n10203791834::lane_0"]
    v.current_lane_id = lane["id"]
    v.current_node = lane["start_node"]
    result = nav.navigation_reroute()
    assert not result["success"]
    assert nav._lane_route_preview is None


def test_assigned_name_takes_precedence_over_other_nodes_full_name(manager, monkeypatch):
    v = manager.vehicles["ego"]
    lane = manager._lane_geometry._lane_by_id[v.current_lane_id]
    v.destination_node = lane["end_node"]
    other = next(node for node_id, node in manager.road_network.nodes.items()
                 if node_id != v.destination_node)
    monkeypatch.setattr(other, "name", "Shared destination label")
    v.destination_name = other.full_name
    _, nav = intercepts(manager)
    assert nav.navigation_route_plan(v.destination_name)["success"]
    assert nav._preview_destination_node == v.destination_node
    assert not nav.navigation_route_plan("not a real mapped destination")["success"]


def test_runtime_map_queries_never_return_demo_pois_or_zero_road_data(manager):
    engine, _ = intercepts(manager)
    module = engine._vw["ego"].map
    assert module.map_query_road("a", "b") == {
        "success": False, "reason": "road_query_backend_unavailable"}
    for result in (module.map_search_poi("gas station"), module.map_get_nearby()):
        assert result == {"success": False, "reason": "poi_data_unavailable"}
