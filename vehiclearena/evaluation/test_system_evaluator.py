"""Regression tests for physical system evaluation."""

from types import SimpleNamespace

from evaluation.experiments.actor_scope import (
    map_trip_vehicle_ids,
    persistent_obstacle_vehicle_ids,
    required_trip_vehicle_ids,
)
from evaluation.experiments.system_evaluator import (
    candidate_deadlocks,
    confirmed_deadlocks,
    evaluate_system,
)


def _vehicle(*, arrived=True, arrival_time_s=5.0):
    return SimpleNamespace(
        arrived=arrived,
        arrival_time_s=arrival_time_s,
        driving_evaluation=None,
    )


def _row(vehicle_id, time_s, x, *, distance=0.0, speed=0.0,
         arrived=False, crashed=False, waiting_red=False):
    return {
        "vehicle_id": vehicle_id,
        "time_s": float(time_s),
        "pose_x_m": float(x),
        "pose_y_m": 0.0,
        "yaw_rad": 0.0,
        "length_m": 4.6,
        "current_segment": "road",
        "current_lane_id": "road::lane_0",
        "active_connector_id": "",
        "distance_traveled_m": float(distance),
        "speed_kmh": float(speed),
        "arrived": arrived,
        "crashed": crashed,
        "waiting_red_light": waiting_red,
        "present_in_physics_world": True,
        "last_control_command": {},
    }


def test_actor_scope_excludes_only_initial_persistent_obstacles():
    scenario = {
        "vehicles": [
            {"vehicle_id": "ego"},
            {"vehicle_id": "wreck",
             "initial_physical_state": {"crashed": True}},
            {"vehicle_id": "barrier"},
        ],
        "experiment_scene": {"focal_vehicle_id": "ego"},
    }
    assert persistent_obstacle_vehicle_ids(scenario) == {"wreck"}
    assert required_trip_vehicle_ids(scenario) == {"ego", "barrier"}
    assert map_trip_vehicle_ids(scenario) == {"barrier"}


def test_system_metrics_exclude_obstacle_but_keep_it_as_queue_blocker():
    scenario = {
        "vehicles": [
            {"vehicle_id": "ego"},
            {"vehicle_id": "npc"},
            {"vehicle_id": "obstacle",
             "initial_physical_state": {"crashed": True}},
        ],
        "experiment_scene": {"focal_vehicle_id": "ego"},
    }
    result = SimpleNamespace(
        vehicle_results={
            "ego": _vehicle(arrival_time_s=2.0),
            "npc": _vehicle(arrival_time_s=2.0),
            "obstacle": _vehicle(arrived=False, arrival_time_s=None),
        },
        _collision_log=[],
    )
    trajectory = [
        _row("ego", 0.0, 0.0),
        _row("npc", 0.0, 0.0),
        _row("obstacle", 0.0, 7.0, crashed=True),
        _row("ego", 1.0, 0.0),
        _row("npc", 1.0, 0.0),
        _row("obstacle", 1.0, 7.0, crashed=True),
        _row("ego", 2.0, 1.0, arrived=True),
        _row("npc", 2.0, 1.0, arrived=True),
        _row("obstacle", 2.0, 7.0, crashed=True),
    ]

    metrics = evaluate_system(
        result, trajectory, duration_s=3.0, scenario=scenario)

    assert metrics["required_trip_vehicle_ids"] == ["ego", "npc"]
    assert metrics["map_trip_vehicle_ids"] == ["npc"]
    assert metrics["persistent_obstacle_vehicle_ids"] == ["obstacle"]
    assert metrics["arrival_rate"] == 1.0
    assert set(metrics["trip_vehicle_metrics"]) == {"ego", "npc"}
    assert metrics["trip_vehicle_metrics"]["npc"]["queue_wait_s"] == 2.0


def test_crashed_trip_vehicle_is_blocker_but_not_queue_member():
    scenario = {
        "vehicles": [
            {"vehicle_id": "ego"},
            {"vehicle_id": "npc"},
            {"vehicle_id": "crashed_npc"},
        ],
        "experiment_scene": {"focal_vehicle_id": "ego"},
    }
    result = SimpleNamespace(
        vehicle_results={
            "ego": _vehicle(arrived=False, arrival_time_s=None),
            "npc": _vehicle(arrived=False, arrival_time_s=None),
            "crashed_npc": _vehicle(arrived=False, arrival_time_s=None),
        },
        _collision_log=[],
    )
    trajectory = []
    for time_s in (0.0, 1.0, 2.0):
        trajectory.extend([
            _row("ego", time_s, 0.0),
            _row("npc", time_s, 4.6),
            _row("crashed_npc", time_s, 9.2, crashed=True),
        ])

    metrics = evaluate_system(
        result, trajectory, duration_s=3.0, scenario=scenario)

    assert metrics["max_queue_length"] == 2
    assert metrics["queue_vehicle_seconds_total"] == 2.0
    assert metrics["trip_vehicle_metrics"]["npc"]["queue_wait_s"] == 2.0
    assert metrics["trip_vehicle_metrics"]["crashed_npc"][
        "queue_wait_s"] == 0.0
    assert "crashed_npc" in metrics["max_queue_detail"]["components"][0]
    assert "crashed_npc" in metrics["max_queue_detail"][
        "physical_blocker_vehicle_ids"]
    assert "crashed_npc" not in metrics["max_queue_detail"][
        "queue_components"][0]


def test_crashed_focal_remains_blocker_but_is_not_queue_length_member():
    scenario = {
        "vehicles": [
            {"vehicle_id": "ego"},
            {"vehicle_id": "npc_1"},
            {"vehicle_id": "npc_2"},
        ],
        "experiment_scene": {"focal_vehicle_id": "ego"},
    }
    result = SimpleNamespace(
        vehicle_results={
            "ego": _vehicle(arrived=False, arrival_time_s=None),
            "npc_1": _vehicle(arrived=False, arrival_time_s=None),
            "npc_2": _vehicle(arrived=False, arrival_time_s=None),
        },
        _collision_log=[],
    )
    trajectory = []
    for time_s in (0.0, 1.0, 2.0):
        trajectory.extend([
            _row("ego", time_s, 0.0, crashed=True),
            _row("npc_1", time_s, 4.6),
            _row("npc_2", time_s, 9.2),
        ])

    metrics = evaluate_system(
        result, trajectory, duration_s=3.0, scenario=scenario)

    assert metrics["max_queue_length"] == 2
    assert metrics["queue_vehicle_seconds_total"] == 4.0
    assert metrics["trip_vehicle_metrics"]["ego"]["queue_wait_s"] == 0.0
    assert metrics["trip_vehicle_metrics"]["npc_1"]["queue_wait_s"] == 2.0
    assert metrics["trip_vehicle_metrics"]["npc_2"]["queue_wait_s"] == 2.0
    assert "ego" in metrics["max_queue_detail"]["components"][0]
    assert "ego" in metrics["max_queue_detail"][
        "physical_blocker_vehicle_ids"]


def test_unrelated_stalled_vehicles_are_candidates_but_not_confirmed_deadlock():
    trajectory = []
    for time_s in range(21):
        trajectory.extend([
            _row("a", time_s, 0.0),
            _row("b", time_s, 1000.0),
        ])

    assert len(candidate_deadlocks(
        trajectory, eligible_vehicle_ids={"a", "b"})) == 1
    assert confirmed_deadlocks(trajectory, {"a", "b"}) == []


def test_persistent_connected_stall_is_confirmed_deadlock():
    trajectory = []
    for time_s in range(21):
        trajectory.extend([
            _row("a", time_s, 0.0),
            _row("b", time_s, 7.0),
        ])

    confirmed = confirmed_deadlocks(trajectory, {"a", "b"})
    assert len(confirmed) == 1
    assert confirmed[0]["entity_ids"] == ["a", "b"]
