"""Regression tests for individual optical/acoustic sensing and cues."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from simulation.perception_model import (
    HornEmission, PerceptionModel, VehicleSignalState)
from tool_utils import (
    dispatch_lazy_discovery, generate_lazy_discovery_tools,
    get_brief_module_api)
from vehiclearena import VehicleWorld
from simulation.multi_sim_engine import MultiScenario, MultiSimEngine
from visualization.web3d_live import Web3DFrameEncoder


def _vehicle(
    vehicle_id: str, x: float, *, yaw: float = 0.0,
    profile: str = "human_driver_standard", speed_kmh: float = 0.0,
) -> SimpleNamespace:
    return SimpleNamespace(
        vehicle_id=vehicle_id,
        pose_x_m=x,
        pose_y_m=0.0,
        yaw_rad=yaw,
        perception_profile_name=profile,
        perception_overrides={},
        signal_state=VehicleSignalState(),
        cabin_open_fraction=0.0,
        current_speed_kmh=speed_kmh,
        length_m=4.6,
        width_m=1.9,
        z_level=0,
        arrived=False,
    )


def _perception_world() -> SimpleNamespace:
    vehicles = {
        "ego": _vehicle("ego", 0.0),
        "other": _vehicle("other", 50.0, yaw=3.141592653589793),
    }
    return SimpleNamespace(
        vehicles=vehicles,
        pedestrians={},
        horn_events=[],
        perception_log=[],
        perception_log_limit=10_000,
        perception_log_dropped=0,
        _current_weather="sunny",
        _current_wind_speed_mps=0.0,
        _is_night=False,
        _physics_time=0.0,
    )


def test_ego_lighting_and_weather_change_visual_envelope_without_hidden_external_bonus():
    tm = _perception_world()
    model = PerceptionModel(tm)
    tm._is_night = True
    dark_range = model.visual_envelope("ego")["effective_range_m"]
    tm.vehicles["ego"].signal_state.low_beam = True
    low_beam_range = model.visual_envelope("ego")["effective_range_m"]
    assert low_beam_range > dark_range

    tm._current_weather = "fog"
    fog_low_range = model.visual_envelope("ego")["effective_range_m"]
    tm.vehicles["ego"].signal_state.low_beam = False
    tm.vehicles["ego"].signal_state.high_beam = True
    fog_high_range = model.visual_envelope("ego")["effective_range_m"]
    assert fog_high_range < fog_low_range

    tm._current_weather = "sunny"
    tm.vehicles["ego"].signal_state.high_beam = False
    without_external_light = model.visual_envelope(
        "ego")["effective_range_m"]
    tm.vehicles["other"].signal_state.high_beam = True
    tm._perception_environment_version = 1
    with_external_light = model.visual_envelope(
        "ego")["effective_range_m"]
    assert with_external_light == without_external_light


def test_visible_vehicle_signals_are_part_of_detection():
    tm = _perception_world()
    model = PerceptionModel(tm)
    tm.vehicles["other"].yaw_rad = 0.0
    tm.vehicles["other"].signal_state.left_indicator = True
    tm.vehicles["other"].signal_state.brake_light = True
    detection = model.detect_entity("ego", "other")
    assert detection is not None
    assert detection.observed_signals["left_indicator"] is True
    assert detection.observed_signals["brake_light"] is True


def test_vehicle_lamps_are_filtered_by_observer_side():
    tm = _perception_world()
    model = PerceptionModel(tm)
    other = tm.vehicles["other"]
    other.yaw_rad = 0.0
    other.signal_state.brake_light = True
    other.signal_state.high_beam = True

    # Ego is behind the target and sees rear lamps, not its headlamps.
    detection = model.detect_entity("ego", "other")
    assert detection is not None
    assert detection.observed_signals["brake_light"] is True
    assert "high_beam" not in detection.observed_signals

    # Move ego in front of the target: the visible lamp set reverses.
    tm.vehicles["ego"].pose_x_m = 85.0
    detection = model.detect_entity("ego", "other")
    assert detection is not None
    assert detection.observed_signals["high_beam"] is True
    assert "brake_light" not in detection.observed_signals


def test_window_and_weather_change_horn_reception():
    tm = _perception_world()
    tm.vehicles["other"].pose_x_m = 100.0
    tm.horn_events = [HornEmission(
        event_id="h1", source_id="other", start_time_s=1.0,
        duration_s=0.2, intensity="normal", source_db=105.0,
        pose_x_m=100.0, pose_y_m=0.0)]
    model = PerceptionModel(tm)
    assert model.heard_horns("ego", 1.1)

    tm.vehicles["ego"].perception_profile_name = "human_driver_limited"
    assert not model.heard_horns("ego", 1.1)
    closed_range = model.acoustic_envelope(
        "ego")["effective_range_m"]
    tm.vehicles["ego"].cabin_open_fraction = 1.0
    assert model.heard_horns("ego", 1.1)
    open_range = model.acoustic_envelope(
        "ego")["effective_range_m"]
    assert open_range > closed_range

    tm._current_weather = "heavy_rain"
    tm._current_wind_speed_mps = 15.0
    storm_range = model.acoustic_envelope(
        "ego")["effective_range_m"]
    assert storm_range < open_range


def test_front_and_rear_radar_measure_anonymous_closing_tracks():
    tm = _perception_world()
    tm.vehicles["ego"].current_speed_kmh = 36.0
    tm.vehicles["other"].pose_x_m = 30.0
    tm.vehicles["other"].yaw_rad = 0.0
    tm.vehicles["other"].current_speed_kmh = 18.0
    tm.vehicles["behind"] = _vehicle(
        "behind", -20.0, speed_kmh=54.0)
    model = PerceptionModel(tm)

    front = model.radar_scan("ego", "frontRadar")
    rear = model.radar_scan("ego", "rearRadar")

    assert front["success"] is True
    assert rear["success"] is True
    assert len(front["tracks"]) == 1
    assert len(rear["tracks"]) == 1
    assert front["tracks"][0]["direction"] == "front"
    assert rear["tracks"][0]["direction"] == "rear"
    assert front["tracks"][0]["closing_speed_mps"] > 4.5
    assert rear["tracks"][0]["closing_speed_mps"] > 4.5
    assert front["tracks"][0]["ttc_s"] is not None
    assert "_warning_corridor_relevant" not in front["tracks"][0]
    assert "vehicle_id" not in front["tracks"][0]
    assert front["tracks"][0]["track_id"].startswith("trk_")


def test_radar_keeps_adjacent_track_but_warning_corridor_rejects_it():
    tm = _perception_world()
    ego = tm.vehicles["ego"]
    other = tm.vehicles["other"]
    ego.current_speed_kmh = 36.0
    ego.current_segment = "road"
    ego.current_lane_id = "road::lane_0"
    ego.active_connector_id = ""
    ego.is_changing_lane = False
    ego.target_lane = -1
    ego.lateral_speed_mps = 0.0
    other.pose_x_m = 30.0
    other.pose_y_m = 3.5
    other.yaw_rad = 0.0
    other.current_speed_kmh = 0.0
    other.current_segment = "road"
    other.current_lane_id = "road::lane_1"
    other.active_connector_id = ""
    other.is_changing_lane = False
    other.target_lane = -1
    other.lateral_speed_mps = 0.0
    model = PerceptionModel(tm)

    adjacent = model.radar_scan(
        "ego", "frontRadar", include_monitor_metadata=True)["tracks"][0]
    assert adjacent["ttc_s"] is not None
    assert adjacent["_warning_corridor_relevant"] is False
    assert adjacent["_warning_corridor_relation"] == \
        "separate_stable_lane"

    # Once the adjacent target actually moves into ego's corridor, the same
    # radar observation becomes warning-relevant before body overlap.
    other.is_changing_lane = True
    other.lateral_speed_mps = -1.0
    cutting_in = model.radar_scan(
        "ego", "frontRadar", include_monitor_metadata=True)["tracks"][0]
    assert cutting_in["_warning_corridor_relevant"] is True
    assert cutting_in["_warning_corridor_relation"] == \
        "predicted_corridor_entry"


def test_radar_is_installed_equipment_and_uses_strict_overrides():
    tm = _perception_world()
    model = PerceptionModel(tm)
    vw = VehicleWorld(equipment_profile="full")
    vw.frontRadar.configure({"range_m": 90.0})
    vw.frontRadar.bind_scan_provider(
        lambda name, overrides: model.radar_scan("ego", name, overrides))
    result = vw.frontRadar.scan()
    assert result["success"] is True
    assert result["spec"]["range_m"] == 90.0
    tools = generate_lazy_discovery_tools()
    catalog = dispatch_lazy_discovery(
        vw, "get_module_api", {"module": "frontRadar"}, tools)
    assert "frontRadar__scan" in catalog
    loaded = dispatch_lazy_discovery(
        vw, "load_tools", {"tools": ["frontRadar__scan"]}, tools)
    assert loaded["success"] is True
    assert loaded["loaded"] == ["frontRadar__scan"]

    without_rear = VehicleWorld(
        equipment_profile="full", disable_modules=["rearRadar"])
    assert without_rear.has_module("frontRadar")
    assert not without_rear.has_module("rearRadar")
    with pytest.raises(ValueError, match="unknown frontRadar override"):
        MultiScenario.from_dict({
            "scenario_id": "bad_radar_override",
            "road_network_id": "beijing_guomao",
            "vehicles": [{
                "vehicle_id": "ego", "initial_node": "n33399858",
                "sensor_overrides": {"frontRadar": {"magic": 1}},
            }],
        })


def test_engine_binds_configured_radar_to_each_vehicle_world():
    scenario = MultiScenario.from_dict({
        "scenario_id": "radar_engine_binding",
        "road_network_id": "beijing_guomao",
        "total_time_s": 0.3,
        "vehicles": [{
            "vehicle_id": "ego",
            "initial_node": "n33399858",
            "agent_config": {"type": "llm"},
            "sensor_overrides": {
                "frontRadar": {"range_m": 77.0},
            },
        }],
    })
    scans = []

    def callback(vw, *args, **kwargs):
        scans.append(vw.frontRadar.scan())
        return []

    engine = MultiSimEngine(scenario)
    result = engine.run({"ego": callback})
    assert scans and scans[0]["success"] is True
    assert scans[0]["spec"]["range_m"] == 77.0
    assert scans[0]["source"] == "continuous_0.1s_radar_cache"
    assert all(
        not any(key.startswith("_warning_") for key in track)
        for track in scans[0]["tracks"])
    capability = result.to_dict()["vehicles"]["ego"]["capabilities"]
    assert capability["sensor_specs"]["frontRadar"]["range_m"] == 77.0
    assert engine._radar_frames["ego"]["frontRadar"][
        "sample_time_s"] == pytest.approx(0.3)

    # The public tool reads the already sampled frame; it cannot privately
    # recompute against a different world state inside the same wake.
    engine._radar_frames["ego"]["frontRadar"]["tracks"] = [{
        "track_id": "trk_cached", "distance_m": 12.0,
    }]
    cached = engine._vw["ego"].frontRadar.scan()
    assert cached["tracks"][0]["track_id"] == "trk_cached"

    # Warnings are transition based: enter, suppress unchanged frames, then
    # report clear. They tell the LLM which radar tool to inspect.
    engine._radar_frames["ego"]["frontRadar"]["tracks"] = [{
        "track_id": "trk_adjacent", "distance_m": 1.0, "ttc_s": 1.0,
        "_warning_corridor_relevant": False,
    }]
    assert engine._radar_warning_events(0.3, {"ego": callback}) == []
    engine._radar_frames["ego"]["frontRadar"]["tracks"] = [{
        "track_id": "trk_risk", "distance_m": 1.0, "ttc_s": 1.0,
        "_warning_corridor_relevant": True,
    }]
    entered = engine._radar_warning_events(0.4, {"ego": callback})
    repeated = engine._radar_warning_events(0.5, {"ego": callback})
    assert [event.event_type for event in entered] == [
        "front_collision_warning"]
    assert entered[0].state == "entered"
    assert entered[0].details["observation_tool"] == "frontRadar__scan"
    assert repeated == []
    engine._radar_frames["ego"]["frontRadar"]["tracks"] = []
    cleared = engine._radar_warning_events(0.6, {"ego": callback})
    assert [event.state for event in cleared] == ["cleared"]
    engine._radar_frames["ego"]["frontRadar"]["tracks"] = [{
        "track_id": "trk_caution", "distance_m": 100.0, "ttc_s": 3.0,
        "_warning_corridor_relevant": True,
    }]
    assert engine._radar_warning_events(0.7, {"ego": callback}) == []
    reentered = engine._radar_warning_events(1.6, {"ego": callback})
    assert [event.state for event in reentered] == ["entered"]
    assert reentered[0].details["level"] == "caution"


def test_road_perception_tools_are_lazy_loaded():
    vw = VehicleWorld()
    tools = generate_lazy_discovery_tools()
    initial_names = {item["function"]["name"] for item in tools}
    assert "road_perception__look_ahead" not in initial_names
    catalog = get_brief_module_api(
        "road_perception", vw.available_module_names())
    assert "road_perception__look_ahead" in catalog
    rejected = dispatch_lazy_discovery(
        vw, "load_tools",
        {"tools": ["road_perception__look_ahead", "not_a_real_tool"]},
        tools)
    assert rejected["success"] is False
    assert "road_perception__look_ahead" not in {
        item["function"]["name"] for item in tools}
    result = dispatch_lazy_discovery(
        vw, "load_tools",
        {"tools": ["road_perception__look_ahead"]}, tools)
    assert result["success"] is True
    assert "road_perception__look_ahead" in {
        item["function"]["name"] for item in tools}

    bare_tools = generate_lazy_discovery_tools()
    bare = dispatch_lazy_discovery(
        vw, "load_tools", {"tools": ["look_ahead"]}, bare_tools)
    assert bare["success"] is False


def test_scenario_validates_entity_perception_profiles():
    scenario = MultiScenario.from_dict({
        "scenario_id": "perception_profiles",
        "road_network_id": "beijing_guomao",
        "vehicles": [{
            "vehicle_id": "ego", "initial_node": "n33399858",
            "perception_profile": "enhanced_vision",
            "perception_overrides": {"forward_visual_range_m": 210.0},
        }],
        "pedestrians": [{
            "ped_id": "p", "initial_node": "n33399858",
            "perception_profile": "pedestrian_standard",
        }],
    })
    assert scenario.vehicles[0].perception_profile == "enhanced_vision"
    assert scenario.vehicles[0].perception_overrides[
        "forward_visual_range_m"] == 210.0
    assert scenario.pedestrians[0].perception_profile == \
        "pedestrian_standard"
    with pytest.raises(ValueError, match="unknown perception profile"):
        MultiScenario.from_dict({
            "scenario_id": "bad_perception_profile",
            "road_network_id": "beijing_guomao",
            "vehicles": [{
                "vehicle_id": "ego", "initial_node": "n33399858",
                "perception_profile": "magic_sensor",
            }],
        })


def test_web3d_signal_frame_matches_connector_enforcement_state():
    connector_id = "connector::intersection::n495771183::9"
    scenario = MultiScenario.from_dict({
        "scenario_id": "movement_signal_consistency",
        "road_network_id": "shanghai_lujiazui",
        "total_time_s": 0.0,
        "tick_interval_s": 2.0,
        "vehicles": [{
            "vehicle_id": "ego",
            "initial_node": "n602392045",
            "destination_node": "n478379222",
            "initial_lane": 0,
            "is_evaluated": False,
            "agent_config": {"type": "sumo"},
            "initial_physical_state": {
                "progress": 0.253546,
                "speed_kmh": 0.0,
                "lane_id": "n602392045_n84489700::lane_0",
            },
        }],
    })
    engine = MultiSimEngine(scenario)
    engine.run({})
    frame = Web3DFrameEncoder(
        focus_entity_id="ego",
        junction_id="intersection::n495771183",
    ).encode(engine, 0.0, 0)
    authoritative = engine.traffic_mgr._lane_geometry.signal_state(
        connector_id, 0.0)
    assert authoritative is not None
    assert frame["signals"][connector_id] == authoritative.signal


def test_explicit_maneuver_selection_rebuilds_signal_connector():
    connector_id = "connector::intersection::n1317216453::4"
    scenario = MultiScenario.from_dict({
        "scenario_id": "exact_placement_signal_cache",
        "road_network_id": "beijing_guomao",
        "total_time_s": 0.1,
        "tick_interval_s": 10.0,
        "vehicles": [{
            "vehicle_id": "ego",
            "initial_node": "n345589058",
            "destination_node": "n10676277830",
            "initial_lane": 3,
            "agent_config": {"type": "llm"},
            "initial_physical_state": {
                "lane_id": "n1317216454_n345589058::lane_3",
                "progress": 0.95,
                "speed_kmh": 0.0,
            },
        }],
    })
    observed = {}

    def callback(vw, _t, _messages, _memory, _tick_index, **kwargs):
        if "selection" not in observed:
            observed["selection"] = (
                vw.navigation.navigation_select_maneuver("straight"))
        return []

    engine = MultiSimEngine(scenario)
    engine.run({"ego": callback})
    vehicle = engine.traffic_mgr.vehicles["ego"]
    awareness = engine.traffic_mgr.get_driving_awareness("ego")
    authoritative = engine.traffic_mgr._lane_geometry.signal_state(
        connector_id, 0.0)
    assert authoritative is not None
    assert observed["selection"]["success"] is True
    assert awareness["connector_id"] == connector_id
    assert vehicle.planned_connector_id == connector_id
    assert awareness["traffic_light"] == authoritative.signal
    assert awareness["route_lane_change_required"] is False
    assert "does not mean the lane drops" in awareness[
        "lane_end_semantics"]
