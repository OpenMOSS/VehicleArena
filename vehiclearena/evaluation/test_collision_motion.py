"""Collision terminal flags must not fabricate a physical stop."""

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from evaluation.experiments.telemetry import ExperimentTrackedEngine
from simulation.multi_sim_engine import MultiScenario
from simulation.sumo_traffic_manager import SumoTrafficManager
from simulation.traffic_manager import TrafficCoordinator


@pytest.mark.parametrize("pair", ["vehicles", "vehicle_pedestrian", "pedestrian_vehicle"])
@pytest.mark.parametrize("is_llm", [False, True])
def test_contact_preserves_measured_vehicle_motion(pair, is_llm):
    manager = TrafficCoordinator.__new__(TrafficCoordinator)
    manager._pending_events = []
    manager.vehicles = {
        vid: SimpleNamespace(is_llm=is_llm, current_speed_kmh=speed,
                             acceleration_mps2=accel, target_speed_kmh=50,
                             is_crashed=False, pending_arrival=True)
        for vid, speed, accel in [("a", 50.0, 2.0), ("b", 30.0, -4.0)]
    }
    manager.pedestrians = {"p": SimpleNamespace(
        is_crashed=False, speed=1.0, pending_arrival=True)}
    if pair == "vehicles":
        manager._apply_collision("a", "b", "junction", "sumo_junction", 75, 0,
                                 time_s=7.5)
        affected = ["a", "b"]
    else:
        entities = ("a", "vehicle", "p", "pedestrian")
        if pair == "pedestrian_vehicle":
            entities = ("p", "pedestrian", "a", "vehicle")
        manager._apply_entity_collision(*entities, "junction", "vehicle_pedestrian",
                                        75, 0, time_s=7.5)
        affected = ["a"]
        assert manager.pedestrians["p"].is_crashed
        assert manager.pedestrians["p"].speed == 0
    for vid in affected:
        v = manager.vehicles[vid]
        assert v.is_crashed and not v.pending_arrival
        assert v.target_speed_kmh == 0
        assert (v.current_speed_kmh, v.acceleration_mps2) == (
            (50.0, 2.0) if vid == "a" else (30.0, -4.0))
    assert len(manager._pending_events) == 2


@pytest.mark.skipif(
    shutil.which("sumo") is None or shutil.which("netconvert") is None,
    reason="SUMO/netconvert unavailable",
)
def test_four_way_collision_replay_matches_native_motion(monkeypatch):
    """Replay only the original driving commands, with no model or browser."""
    class NoCamera:
        def render(self, *args, **kwargs):
            return None

        def close(self):
            pass

    monkeypatch.setattr("visualization.web3d_camera.Web3DCameraRenderer", NoCamera)
    catalog = Path(__file__).parent / "experiments/scenarios/MultiLLM"
    raw = json.loads((catalog / (
        "multi_033_synchronized_four_way__beijing_guomao/scenario.json"
    )).read_text())
    # The local benchmark replaced the remote task IDs and synchronized
    # approach distances. Keep its road/connector geometry, but replay the
    # original collision-regression initial positions inside this test only.
    reference_progress = {
        "ego": 0.8484,
        "approach_2": 0.760474,
        "approach_3": 0.739954,
        "approach_4": 0.786805,
    }
    # The catalog scenario gained map_flow/map_route/map_cross traffic in the
    # task-config refresh; this regression replays the original four-vehicle
    # collision geometry only, so drop vehicles that were not part of it.
    raw["vehicles"] = [
        vehicle for vehicle in raw["vehicles"]
        if vehicle["vehicle_id"] in reference_progress
    ]
    for vehicle in raw["vehicles"]:
        vehicle["initial_physical_state"]["progress"] = reference_progress[
            vehicle["vehicle_id"]]
    experiment_scene = raw.get("experiment_scene") or {}
    if isinstance(experiment_scene.get("fixed_peer_vehicle_ids"), list):
        experiment_scene["fixed_peer_vehicle_ids"] = [
            vid for vid in experiment_scene["fixed_peer_vehicle_ids"]
            if vid in reference_progress
        ]
    raw.pop("weather_keyframes", None)
    raw.pop("daynight_keyframes", None)
    raw.update(total_time_s=12, tick_interval_s=0.1,
               stop_when_all_vehicles_terminal=False)
    for v in raw["vehicles"]:
        if v["agent_config"]["type"] == "llm":
            v["agent_config"]["heartbeat_interval_s"] = 0.1
    schedules = {
        "ego": [(0, {"speed_kmh": 50})],
        "approach_2": [(0, {"speed_kmh": 50}), (1.7, {"speed_kmh": 25}),
                       (2.7, {"speed_kmh": 15}), (3.2, {"speed_kmh": 10}),
                       (4.2, {"speed_kmh": 30})],
        "approach_3": [(0, {"speed_kmh": 45}),
                       (2.8, {"speed_kmh": 20, "deceleration_mps2": 4}),
                       (4.6, {"speed_kmh": 50})],
        "approach_4": [(0, {"speed_kmh": 50, "acceleration_mps2": 2})],
    }
    wakes = {vid: [] for vid in schedules}

    def driver(vid):
        remaining = list(schedules[vid])

        def callback(vw, t, *args, **kwargs):
            wakes[vid].append(t)
            while remaining and t + 1e-6 >= remaining[0][0]:
                _, command = remaining.pop(0)
                assert vw.navigation.navigation_set_speed(**command)["success"]
            return []
        return callback

    # Read SUMO immediately after each contact commit, including impact frames.
    native = {}
    original_commit = SumoTrafficManager._commit_sumo_collisions

    def commit(manager, t):
        original_commit(manager, t)
        active = set(manager._sumo.vehicle.getIDList())
        for vid in schedules.keys() & active:
            v = manager.vehicles[vid]
            speed = manager._sumo.vehicle.getSpeed(vid) * 3.6
            accel = manager._sumo.vehicle.getAcceleration(vid)
            assert v.current_speed_kmh == pytest.approx(speed)
            assert v.acceleration_mps2 == pytest.approx(accel)
            native[vid, round(t, 6)] = (speed, accel)

    monkeypatch.setattr(SumoTrafficManager, "_commit_sumo_collisions", commit)
    engine = ExperimentTrackedEngine(MultiScenario.from_dict(raw))
    engine.run({vid: driver(vid) for vid in schedules})
    assert not engine.agent_callback_errors
    contacts = engine.traffic_mgr.collision_log
    assert [(c.time_s, {c.entity_a, c.entity_b}) for c in contacts] == [
        (7.5, {"approach_3", "approach_4"}),
        (7.8, {"approach_2", "approach_4"})]
    for vid, impact in [("approach_2", 7.8), ("approach_3", 7.5), ("approach_4", 7.5)]:
        rows = [x for x in engine._vehicle_trajectory
                if x["vehicle_id"] == vid and x["time_s"] >= impact]
        assert rows and rows[0]["speed_kmh"] > 0
        assert rows[-1]["speed_kmh"] == 0
        assert all(x["crashed"] and x["target_speed_kmh"] == 0 for x in rows)
        for row in rows:
            speed, accel = native[vid, row["time_s"]]
            assert row["speed_kmh"] == pytest.approx(speed, abs=0.0001)
            assert row["acceleration_mps2"] == pytest.approx(accel, abs=0.0001)
        assert all(b["speed_kmh"] <= a["speed_kmh"] + 0.0001
                   for a, b in zip(rows, rows[1:]))
        assert all(t < impact for t in wakes[vid])
        assert not engine.traffic_mgr.set_vehicle_speed(vid, 50)["success"]
    assert any(t > 7.8 for t in wakes["ego"])
