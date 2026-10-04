"""Tests for the read-only SUMO-to-Web3D frame boundary."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from simulation.perception_model import VehicleSignalState
from simulation.multi_sim_engine import MultiScenario, MultiSimEngine
from visualization import web3d_live


class _LaneGeometry:
    def signal_state(self, connector_id, time_s):
        assert connector_id == "connector-a"
        return SimpleNamespace(signal="green")

    @staticmethod
    def pedestrian_pose(pedestrian):
        return pedestrian.physical_pose_xy


def test_live_frame_contains_only_authoritative_dynamic_state():
    vehicle = SimpleNamespace(
        is_llm=True,
        control_authority="llm",
        present_in_physics_world=True,
        arrived=False,
        pose_x_m=12.5,
        pose_y_m=18.0,
        yaw_rad=0.25,
        width_m=1.9,
        length_m=4.6,
        current_speed_kmh=36.0,
        acceleration_mps2=1.2,
        z_level=0,
        signal_state=VehicleSignalState(left_indicator=True),
        is_crashed=False,
    )
    pedestrian = SimpleNamespace(
        is_spawned=True,
        has_arrived=False,
        physical_pose_xy=[14.0, 19.0],
        physical_yaw_rad=-0.5,
        control_authority="sumo",
        collision_radius_m=0.4,
        speed=1.3,
        is_crashed=False,
    )
    manager = SimpleNamespace(
        vehicles={"ego": vehicle},
        pedestrians={"walker": pedestrian},
        _lane_geometry=SimpleNamespace(
            data={
                "intersections": [{
                    "id": "junction-a", "center_xy": [10.0, 20.0]}],
                "lanes": [{"id": "lane-a"}],
                "connectors": [{
                    "id": "connector-a", "from_lane": "lane-a",
                    "turn": "straight",
                }],
                "stop_lines": [{
                    "lane_id": "lane-a",
                    "node_id": "junction-a",
                    "line_xy": [[10.0, 20.0], [12.0, 20.0]],
                }],
            },
            signal_state=_LaneGeometry().signal_state,
            pedestrian_pose=_LaneGeometry().pedestrian_pose,
        ),
        _current_weather="heavy_rain",
        _current_wind_speed_mps=7.0,
        _daylight_level=25,
        _is_night=True,
    )
    engine = SimpleNamespace(
        traffic_mgr=manager,
        scenario=SimpleNamespace(
            road_network_id="test_map", physics_step_s=0.1),
    )

    frame = web3d_live.Web3DFrameEncoder(
        focus_entity_id="ego", radius_m=150,
    ).encode(engine, 1.2, 12)

    assert frame["physics_source"] == "sumo"
    assert frame["sim_time_s"] == 1.2
    assert frame["signals"] == {"connector-a": "green"}
    assert frame["environment"]["weather"] == "heavy_rain"
    ego = next(item for item in frame["actors"] if item["id"] == "ego")
    assert ego["position_world_xy"] == pytest.approx([
        12.5 - 2.3 * math.cos(0.25), 18.0 - 2.3 * math.sin(0.25)], abs=1e-4)
    assert ego["position_reference"] == "body_center"
    assert (vehicle.pose_x_m, vehicle.pose_y_m) == (12.5, 18.0)
    assert ego["speed_mps"] == 10.0
    assert ego["signals"]["left_indicator"] is True
    walker = next(
        item for item in frame["actors"] if item["kind"] == "pedestrian")
    assert walker["yaw_rad"] == -0.5
    assert walker["position_world_xy"] == [14.0, 19.0]


@pytest.mark.parametrize("yaw", [0, math.pi / 2, math.pi, -math.pi / 2, 0.63])
@pytest.mark.parametrize("length", [4.6, 8.0])
@pytest.mark.parametrize("vehicle_id", ["ego", "background"])
@pytest.mark.parametrize("z_level", [-1, 0, 1])
def test_vehicle_render_origin_preserves_sumo_front_bumper(yaw, length, vehicle_id, z_level):
    vehicle = SimpleNamespace(
        control_authority="llm" if vehicle_id == "ego" else "sumo",
        pose_x_m=12.5, pose_y_m=-18.0, yaw_rad=yaw,
        width_m=1.9, length_m=length, current_speed_kmh=36,
        acceleration_mps2=0, z_level=z_level, signal_state={}, is_crashed=False)
    before = vars(vehicle).copy()
    first = web3d_live.Web3DFrameEncoder._vehicle_actor(vehicle_id, vehicle, "ego")
    second = web3d_live.Web3DFrameEncoder._vehicle_actor(vehicle_id, vehicle, "ego")
    assert vars(vehicle) == before
    assert first == second  # Re-encoding cannot accumulate a pose offset.
    assert first["is_focus"] == (vehicle_id == "ego")
    assert first["z_level"] == z_level  # Preserve authoritative topology.
    assert first["elevation_m"] == 0.0  # Do not invent altitude from it.
    cx, cy = first["position_world_xy"]
    # The visible front bumper and rear now coincide with SUMO's body.
    assert [cx + length / 2 * math.cos(yaw), cy + length / 2 * math.sin(yaw)] == pytest.approx(
        [vehicle.pose_x_m, vehicle.pose_y_m], abs=1e-4)
    assert [cx - length / 2 * math.cos(yaw), cy - length / 2 * math.sin(yaw)] == pytest.approx(
        [vehicle.pose_x_m - length * math.cos(yaw),
         vehicle.pose_y_m - length * math.sin(yaw)], abs=1e-4)


def test_live_stream_url_replaces_existing_session_parameter():
    value = web3d_live.Web3DFramePublisher._with_session(
        "http://127.0.0.1:8765/api/live/frame?session_id=old&x=1",
        "new-session",
    )
    assert value.endswith("x=1&session_id=new-session")


def test_world_observer_failure_is_isolated_from_physics():
    class BrokenObserver:
        def __init__(self):
            self.closed = False

        def on_world_step(self, *args):
            raise RuntimeError("display failed")

        def close(self):
            self.closed = True

    engine = MultiSimEngine(MultiScenario.from_dict({
        "scenario_id": "observer-isolation",
        "road_network_id": "beijing_guomao",
        "vehicles": [],
    }))
    observer = BrokenObserver()
    engine.add_world_observer(observer)
    engine._notify_world_observers(1.0, 10, [])

    assert engine._world_observers == []
    assert observer.closed is True
    assert engine.world_observer_errors[0]["time_s"] == 1.0
