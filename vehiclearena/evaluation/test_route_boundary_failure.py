"""Route ends are per-car failures, never an automatic brake or fake impact."""
import copy
import json
import math
from pathlib import Path

import pytest

from evaluation.experiments.telemetry import ExperimentTrackedEngine
from evaluation.personal_agent import _terminal_event
from simulation.multi_sim_engine import MultiScenario
from simulation.sumo_traffic_manager import SumoTrafficManager
from simulation.road_networks import load_road_network


CATALOG = Path(__file__).parent / "fixtures/pre_pull_catalog"
SHORT_LANE = "n9231274164_n9231274165::lane_0"


@pytest.fixture
def no_camera(monkeypatch):
    import visualization.web3d_camera

    class NoCamera:
        def render(self, *args, **kwargs):
            return None

        def close(self):
            pass

    monkeypatch.setattr(visualization.web3d_camera, "Web3DCameraRenderer", NoCamera)


def scene(with_peer=False, physics_only=False):
    raw = json.loads((CATALOG / "MultiLLM" /
        "multi_014_unsignalized_four_way__hongkong_central/scenario.json").read_text())
    bad = copy.deepcopy(raw["vehicles"][-1])
    bad["initial_node"] = "n9231274165"
    bad["destination_node"] = ""  # isolate an unresolved end, no impossible startup route
    bad["initial_physical_state"] = {
        "lane_id": SHORT_LANE, "progress": 0.05,
        "speed_kmh": 18, "target_speed_kmh": 18,
    }
    bad["agent_config"]["heartbeat_interval_s"] = 0.1
    peer = copy.deepcopy(raw["vehicles"][1])
    peer["initial_physical_state"].update(speed_kmh=0, target_speed_kmh=0)
    peer["agent_config"]["heartbeat_interval_s"] = 0.1
    raw["vehicles"] = [bad, peer] if with_peer else [bad]
    raw.update(total_time_s=2, physics_only_mode=physics_only)
    return MultiScenario.from_dict(raw)


@pytest.mark.parametrize("physics_only", [False, True])
@pytest.mark.parametrize("with_peer", [False, True])
def test_short_lane_failure_is_terminal_only_for_that_car(no_camera, physics_only, with_peer):
    engine = ExperimentTrackedEngine(scene(with_peer, physics_only))
    calls, closed = [], []

    class Driver:
        def __init__(self, vid):
            self.vid = vid

        def __call__(self, vw, t, *args, **kwargs):
            calls.append((self.vid, t))
            return []

        def finalize_passenger(self, vw, t, *args, **kwargs):
            closed.append((self.vid, t, kwargs["_wake_events"]))

    callbacks = {v.vehicle_id: Driver(v.vehicle_id) for v in engine.scenario.vehicles}
    result = engine.run(callbacks)
    assert all(event.event_type != "navigation_decision_required"
               for event in engine.wake_broker.log)
    v = engine.traffic_mgr.vehicles["approach_4"]
    assert v.route_failed and not v.arrived and not v.is_crashed
    assert not v.present_in_physics_world
    assert 0 < v.route_failure_time_s < 2
    assert v.current_speed_kmh == pytest.approx(18)  # no invented zero-speed sample
    assert v.terminal_crossing_speed_kmh == pytest.approx(18)
    assert not engine.traffic_mgr.collision_log
    assert not engine.agent_callback_errors
    report = result.to_dict()["vehicles"]["approach_4"]
    assert report["route_failed"] and not report["arrived"]
    assert report["route_failure_reason"] == "unresolved_route_endpoint"
    metrics = report["driving_evaluation"]["metrics"]
    assert metrics["route_failed"]
    assert metrics["observed_time_s"] == pytest.approx(v.route_failure_time_s)
    rows = [r for r in engine._vehicle_trajectory if r["vehicle_id"] == "approach_4"]
    assert rows[-1]["route_failed"]
    assert rows[-1]["time_s"] == v.route_failure_time_s
    assert sum(r["route_failed"] for r in rows) == 1
    assert all(t < v.route_failure_time_s for vid, t in calls if vid == "approach_4")
    if not physics_only:
        terminal = [r for r in closed if r[0] == "approach_4"]
        assert len(terminal) == 1
        assert _terminal_event({"current_events": terminal[0][2]})
    if with_peer:
        peer = engine.traffic_mgr.vehicles["approach_2"]
        assert not peer.route_failed and peer.present_in_physics_world
        assert engine._sim_time == 2
        if not physics_only:
            assert any(vid == "approach_2" and t > v.route_failure_time_s
                       for vid, t in calls)
    else:
        assert engine._sim_time == v.route_failure_time_s
    # Removed bodies cannot remain in spatial queries or accept new controls.
    assert "approach_4" not in engine.road_network._vehicle_positions
    manager = engine.traffic_mgr
    for command in (
        lambda: manager.set_vehicle_speed("approach_4", 30),
        lambda: manager.emergency_stop_vehicle("approach_4"),
        lambda: manager.select_vehicle_maneuver("approach_4", "left"),
        lambda: manager.set_vehicle_destination("approach_4", "n526956446"),
    ):
        assert command()["reason"] == "vehicle_route_failed"
    assert manager.change_lane("approach_4", 1).reason == "vehicle_route_failed"


@pytest.mark.parametrize("study,name", [
    ("Basic", "basic_005_lane_change"),
    ("Basic", "basic_033_continuous_turns"),
    ("MultiLLM", "multi_009_two_lane_merge"),
    ("MultiLLM", "multi_014_unsignalized_four_way__hongkong_central"),
])
def test_assigned_display_names_round_trip_through_navigation(no_camera, study, name):
    raw = json.loads((CATALOG / study / name / "scenario.json").read_text())
    raw.update(total_time_s=0.1)
    engine = ExperimentTrackedEngine(MultiScenario.from_dict(raw))
    planned = {}

    def driver(vw, t, *args, **kwargs):
        for event in kwargs.get("_wake_events", []):
            if event["event_type"] != "driving_task_assigned":
                continue
            detail = event["details"]
            vid = event["entity_id"]
            nav = vw.navigation
            receipt = nav.navigation_route_plan(detail["destination_name"])
            planned[vid] = (detail, receipt, nav._preview_destination_node)
        return []

    callbacks = {v.vehicle_id: driver for v in engine.scenario.vehicles
                 if v.agent_type == "llm"}
    engine.run(callbacks)
    assert not engine.agent_callback_errors
    assert set(planned) == set(callbacks)
    for vid, (detail, receipt, preview_node) in planned.items():
        assert receipt["success"], (vid, detail, receipt)
        assert preview_node == detail["destination_node"]
        assert engine.traffic_mgr.vehicles[vid].destination_name == detail["destination_name"]


def test_collision_at_route_end_takes_priority():
    manager = SumoTrafficManager(load_road_network("hongkong_central"))
    try:
        v = manager.register_vehicle("ego", "n9231274165", is_llm=True)
        v.pending_route_failure = True
        v.is_crashed = True  # native contact has already been committed
        assert manager._finalize_route_failures(1.2) == []
        assert v.is_crashed and not v.route_failed and not v.arrived
        assert not v.present_in_physics_world
    finally:
        manager.close()


@pytest.mark.parametrize("speed_kmh", [0.0, 0.36])
def test_original_dead_end_allows_lateral_motion_without_navigation_stop(speed_kmh):
    # basic_033 used to install a 1e9-second setStop on this four-lane
    # dead end. SUMO skips lateral updates for stopped-on-lane vehicles,
    # even with laneChangeMode=0. Ordinary zero speed must not create a stop.
    manager = SumoTrafficManager(load_road_network("beijing_guomao"))
    try:
        lane_id = "n10203791833_n10203791834::lane_0"
        lane = manager._lane_geometry._lane_by_id[lane_id]
        v = manager.register_vehicle("ego", lane["start_node"], is_llm=True,
                                     start_lane=0)
        v.current_lane_id = lane_id
        v.current_lane = 0
        v.current_segment = lane["segment_id"]
        v.current_node = lane["start_node"]
        v.edge_progress = 0.05
        v.destination_node = ""
        manager._initialize_vehicle_pose(v)
        assert manager.enable_llm_maneuver_authority("ego")["success"]
        assert manager.set_vehicle_speed("ego", speed_kmh)["success"]
        manager.advance_world_to(0.1)
        assert manager._has_unresolved_route_endpoint(v)
        native = manager._sumo.vehicle
        original_position = native.getPosition("ego")
        original_index = native.getLaneIndex("ego")
        target_id = "n10203791833_n10203791834::lane_1"
        target_index = manager.sumo_map.lane_index_by_lane[target_id]
        assert target_index != original_index
        assert manager.change_lane("ego", 1).success
        for tick in range(2, 61):
            manager.advance_world_to(tick / 10)
            assert not v.route_failed and not v.is_crashed
            assert not native.getStops("ego")
            assert not native.isStopped("ego")
            assert native.getSpeed("ego") == pytest.approx(speed_kmh / 3.6)
        # Check native pose and lane, not the adapter's timer-based progress.
        # Entering the target lane does not imply it has reached lane centre.
        assert native.getLaneIndex("ego") == target_index
        assert math.dist(native.getPosition("ego"), original_position) > 2.0
        assert v.current_lane_id == target_id
        assert not manager.collision_log

        # Resuming forward motion must end in a route failure, not an
        # indefinitely parked car or a fabricated collision/arrival.
        assert manager.set_vehicle_speed("ego", 18)["success"]
        manager.advance_world_to(10)
        assert v.route_failed and not v.arrived and not v.is_crashed
        assert not v.present_in_physics_world
        assert not manager.collision_log
    finally:
        manager.close()


def test_replay_original_short_lane_crash_without_model_calls(no_camera):
    # Accepted motion commands from the 2026-09-08 Qwen failure. Text reasons
    # do not alter physics; all times, speeds and acceleration limits retained.
    commands = [
        (0, "navigation_set_speed", dict(speed_kmh=30, acceleration_mps2=2, deceleration_mps2=2.5)),
        (4.3, "navigation_set_speed", dict(speed_kmh=0, deceleration_mps2=6)),
        (6, "navigation_set_speed", dict(speed_kmh=40, acceleration_mps2=2.5)),
        (9, "navigation_select_maneuver", dict(direction="right")),
        (9, "navigation_set_speed", dict(speed_kmh=25)),
        (10.4, "navigation_set_speed", dict(speed_kmh=15)),
        (10.8, "navigation_set_speed", dict(speed_kmh=18, acceleration_mps2=1.5, deceleration_mps2=2.5)),
    ]
    raw = json.loads((CATALOG / "MultiLLM" /
        "multi_014_unsignalized_four_way__hongkong_central/scenario.json").read_text())
    raw["vehicles"] = [raw["vehicles"][-1]]
    raw["vehicles"][0]["agent_config"]["heartbeat_interval_s"] = 0.1
    raw.update(total_time_s=20, max_parallel_model_calls=1)
    engine = ExperimentTrackedEngine(MultiScenario.from_dict(raw))
    applied = []

    def driver(vw, t, *args, **kwargs):
        while commands and commands[0][0] <= t + 1e-9:
            when, tool, arguments = commands.pop(0)
            assert t == pytest.approx(when)
            assert getattr(vw.navigation, tool)(**arguments)["success"]
            applied.append(when)
        return []

    result = engine.run({"approach_4": driver})
    assert len(applied) == 7 and not engine.agent_callback_errors
    v = engine.traffic_mgr.vehicles["approach_4"]
    assert v.route_failed and not v.arrived and not v.is_crashed
    assert 17.8 < v.route_failure_time_s < 20
    assert v.current_lane_id == SHORT_LANE
    assert v.current_speed_kmh == pytest.approx(18)
    assert result.to_dict()["vehicles"]["approach_4"]["route_failed"]
    assert not engine.traffic_mgr.collision_log
    row = next(r for r in engine._vehicle_trajectory if r["time_s"] == 17.8)
    assert row["current_lane_id"] == SHORT_LANE
    assert row["speed_kmh"] == pytest.approx(18)
    assert row["present_in_physics_world"]
