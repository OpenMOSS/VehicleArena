"""Regression checks for processed LiDAR BEV and route mini-maps."""

from __future__ import annotations

import copy
import io

import pytest

from PIL import Image, ImageDraw

from evaluation.context_runtime import estimate_request_tokens
from simulation.road_networks import load_road_network
from simulation.traffic_manager import TrafficCoordinator
from visualization.agent_visual_renderer import (
    AgentVisualRenderer,
    multimodal_image_message,
)


def _manager():
    manager = TrafficCoordinator(load_road_network("beijing_guomao"))
    manager.register_vehicle(
        "ego", "n33399858", "n35722739", is_llm=True)
    return manager


def test_lidar_bev_is_deterministic_multimodal_png():
    renderer = AgentVisualRenderer(_manager(), width=512, height=512)
    first = renderer.render_lidar_bev("ego", 0.0)
    second = renderer.render_lidar_bev("ego", 0.0)
    assert first.kind == "lidar_bev"
    assert first.view == "ego_top_down"
    assert first.png_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    assert first.sha256 == second.sha256
    message = multimodal_image_message(first, label="LidarBEV")
    assert message["content"][1]["image_url"]["url"].startswith(
        "data:image/png;base64,")
    estimate = estimate_request_tokens([message], [])
    assert 1800 <= estimate < 5000


def test_lidar_bev_defaults_to_1024_square():
    renderer = AgentVisualRenderer(_manager())
    assert (renderer.width, renderer.height) == (1024, 1024)


def test_lidar_bev_ignores_signal_phase_day_weather_and_vehicle_lamps():
    manager = _manager()
    ego = manager.vehicles["ego"]
    other = manager.register_vehicle(
        "other", "n33399858", "n35722739",
        start_lane=ego.current_lane)
    other.current_node = ego.current_node
    other.current_segment = ego.current_segment
    other.current_lane = ego.current_lane
    other.current_lane_id = ego.current_lane_id
    other.edge_progress = min(0.95, ego.edge_progress + 0.04)
    manager._initialize_vehicle_pose(other)
    manager._update_road_network_position(other)
    renderer = AgentVisualRenderer(manager, width=512, height=512)

    clear = renderer.render_lidar_bev("ego", 2.0)
    manager._is_night = True
    manager._daylight_level = 0
    manager._current_weather = "heavy_fog"
    ego.signal_state.high_beam = True
    other.signal_state.high_beam = True
    other.signal_state.left_indicator = True
    # A different simulation time also advances the map's signal phase.
    changed_optics = renderer.render_lidar_bev("ego", 17.0)
    assert clear.sha256 == changed_optics.sha256


def test_lidar_bev_contains_no_traffic_light_colours():
    manager = _manager()
    renderer = AgentVisualRenderer(manager, width=512, height=512)
    image = Image.open(io.BytesIO(
        renderer.render_lidar_bev("ego", 0.0).png_bytes)).convert("RGB")
    colours = {
        colour for _, colour in image.getcolors(
            maxcolors=image.width * image.height)}
    traffic_light_colours = {
        (238, 61, 66), (255, 200, 48), (38, 220, 120)}
    assert not traffic_light_colours & colours


def test_lidar_bev_uses_distinct_ego_and_other_body_colours():
    manager = _manager()
    ego = manager.vehicles["ego"]
    other = manager.register_vehicle(
        "other", "n33399858", "n35722739",
        start_lane=ego.current_lane)
    other.current_node = ego.current_node
    other.current_segment = ego.current_segment
    other.current_lane = ego.current_lane
    other.current_lane_id = ego.current_lane_id
    other.edge_progress = min(0.95, ego.edge_progress + 0.04)
    manager._initialize_vehicle_pose(other)
    manager._update_road_network_position(other)
    renderer = AgentVisualRenderer(manager, width=512, height=512)
    image = Image.open(io.BytesIO(
        renderer.render_lidar_bev("ego", 0.0).png_bytes)).convert("RGB")
    colours = {
        colour for _, colour in image.getcolors(
            maxcolors=image.width * image.height)}
    assert renderer._EGO_VEHICLE in colours
    assert renderer._OTHER_VEHICLE in colours


def test_vehicle_direction_arrow_points_from_rear_to_front():
    image = Image.new("RGB", (64, 64), (0, 0, 0))
    draw = ImageDraw.Draw(image)
    AgentVisualRenderer._draw_direction_arrow(
        draw, [(27, 10), (37, 10), (37, 50), (27, 50)],
        fill=(255, 255, 255))
    lit = [
        (x, y) for y in range(64) for x in range(64)
        if image.getpixel((x, y)) == (255, 255, 255)]
    assert len({x for x, y in lit if y < 27}) > len(
        {x for x, y in lit if y > 33})


def test_lidar_local_transform_keeps_physical_right_on_screen_right():
    transform = AgentVisualRenderer._local_transform((0.0, 0.0, 0.0))
    assert transform((0.0, -1.0)) == (1.0, 0.0)
    assert transform((0.0, 1.0)) == (-1.0, 0.0)


def test_complex_junction_has_filled_surface_envelopes():
    renderer = AgentVisualRenderer(_manager(), width=512, height=512)
    assert renderer._junction_surfaces
    assert all(
        len(surface["polygon_xy"]) >= 3
        for surface in renderer._junction_surfaces)


def test_minimap_highlights_route_without_live_traffic():
    manager = _manager()
    vehicle = manager.vehicles["ego"]
    assert vehicle.lane_route_actions
    renderer = AgentVisualRenderer(manager, width=512, height=512)
    rendered = renderer.render_minimap("ego", 0.0, scope="route")
    local = renderer.render_minimap("ego", 0.0, scope="local")
    assert rendered.route_available is True
    assert rendered.kind == "navigation_minimap"
    assert rendered.view == "ego_heading_up_route"
    assert rendered.png_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    assert rendered.sha256 != local.sha256
    image = Image.open(io.BytesIO(rendered.png_bytes)).convert("RGB")
    colours = {
        colour for _, colour in image.getcolors(
            maxcolors=image.width * image.height)
    }
    assert renderer._MINIMAP_ROAD in colours
    assert renderer._ROUTE in colours
    # The selected route is now one continuous colour. The former yellow
    # connector segment made a legal turn look like a warning or obstruction.
    assert (255, 198, 58) not in colours


@pytest.mark.parametrize("scope", ["route", "local"])
def test_failed_preview_removes_route_but_keeps_basemap_and_ego(scope):
    manager = _manager()
    renderer = AgentVisualRenderer(manager, width=512, height=512)
    vehicle = manager.vehicles["ego"]
    valid = manager.navigation_route_preview("ego")
    assert valid and vehicle.lane_route_actions
    before = renderer.render_minimap("ego", 0, scope=scope, route_preview=valid)
    failed = renderer.render_minimap("ego", 0, scope=scope, route_preview=None)
    assert failed.route_available is False
    assert renderer._remaining_route_lines(vehicle, None, (0, 0, 0)) == ([], "")
    assert before.sha256 != failed.sha256
    image = Image.open(io.BytesIO(failed.png_bytes)).convert("RGB")
    colours = set(image.getdata())
    assert renderer._MINIMAP_ROAD in colours
    assert renderer._EGO_VEHICLE in colours
    assert renderer._MINIMAP_DESTINATION not in colours
    # Cyan remains on the ego marker's halo only, never on road geometry.
    cyan = [(x, y) for y in range(512) for x in range(512)
            if image.getpixel((x, y)) == renderer._ROUTE]
    # Antialiasing may leave no exact cyan pixels on the thin halo.
    assert all(abs(x - 256) < 22 and 0.7 * 512 < y < 0.9 * 512
               for x, y in cyan)
    recovered = renderer.render_minimap("ego", 0, scope=scope, route_preview=valid)
    assert recovered.route_available is True and recovered.sha256 == before.sha256


def test_saved_minimap_route_updates_pose_without_replanning(monkeypatch):
    manager = _manager()
    vehicle = manager.vehicles["ego"]
    saved = manager.navigation_route_preview("ego")
    assert saved
    original = copy.deepcopy(saved)
    renderer = AgentVisualRenderer(manager, width=512, height=512)

    def unexpected_replan(*args, **kwargs):
        pytest.fail("Displaying a saved route must not invoke the planner")

    monkeypatch.setattr(manager, "navigation_route_preview", unexpected_replan)
    before = renderer.render_minimap("ego", 0, route_preview=saved)
    vehicle.edge_progress = 0.75
    manager._initialize_vehicle_pose(vehicle)
    manager._update_road_network_position(vehicle)
    # Internal physical/scoring route updates must not replace saved geometry.
    manager._replan_lane_route(vehicle, tick=1)
    after = renderer.render_minimap("ego", 1, route_preview=saved)
    assert saved == original
    assert before.route_available and after.route_available
    assert before.sha256 != after.sha256


def test_dead_end_from_continuous_turns_has_no_default_route_highlight():
    manager = _manager()
    v = manager.vehicles["ego"]
    lane = manager._lane_geometry._lane_by_id["n10203791833_n10203791834::lane_0"]
    v.current_lane_id = lane["id"]
    v.current_lane = lane["index"]
    v.current_segment = lane["segment_id"]
    v.current_node = lane["start_node"]
    v.destination_node = "n7141497649"
    v.edge_progress = 1
    manager._initialize_vehicle_pose(v)
    assert manager.navigation_route_preview("ego") is None
    renderer = AgentVisualRenderer(manager, width=512, height=512)
    # Even legacy/direct callers cannot fall back to stale physical actions.
    rendered = renderer.render_minimap("ego", 75.1)
    assert rendered.route_available is False


def test_minimap_keeps_ego_low_and_centred_in_heading_up_view():
    rendered = AgentVisualRenderer(
        _manager(), width=512, height=512).render_minimap(
            "ego", 0.0, scope="route")
    image = Image.open(io.BytesIO(rendered.png_bytes)).convert("RGB")
    marker_pixels = [
        (x, y)
        for y in range(image.height)
        for x in range(image.width)
        if image.getpixel((x, y)) == AgentVisualRenderer._EGO_VEHICLE
    ]
    assert marker_pixels
    centre_x = sum(point[0] for point in marker_pixels) / len(marker_pixels)
    centre_y = sum(point[1] for point in marker_pixels) / len(marker_pixels)
    assert abs(centre_x - image.width / 2) <= 2
    assert image.height * 0.70 < centre_y < image.height * 0.90


def test_minimap_marks_visible_route_end_and_draws_scale():
    manager = _manager()
    vehicle = manager.vehicles["ego"]
    vehicle.edge_progress = 0.75
    manager._initialize_vehicle_pose(vehicle)
    manager._update_road_network_position(vehicle)
    renderer = AgentVisualRenderer(manager, width=512, height=512)
    rendered = renderer.render_minimap(
        "ego",
        0.0,
        scope="route",
        route_preview={
            "start_lane_id": vehicle.current_lane_id,
            "goal_lane_id": vehicle.current_lane_id,
            "actions": [],
        },
    )
    image = Image.open(io.BytesIO(rendered.png_bytes)).convert("RGB")
    colours = {
        colour for _, colour in image.getcolors(
            maxcolors=image.width * image.height)}

    assert renderer._MINIMAP_DESTINATION in colours
    assert renderer._MINIMAP_SCALE in colours
