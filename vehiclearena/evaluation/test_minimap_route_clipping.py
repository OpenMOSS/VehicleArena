"""The saved navigation plan is immutable; only its displayed prefix advances."""

import copy
import io
from types import SimpleNamespace

import pytest
from PIL import Image

from simulation.road_networks import load_road_network
from simulation.traffic_manager import TrafficCoordinator
from visualization.agent_visual_renderer import AgentVisualRenderer


def _fixture():
    renderer = AgentVisualRenderer.__new__(AgentVisualRenderer)
    renderer._minimap_progress = {}
    renderer._lane_by_id = {
        "in": {"centerline_xy": [(0, 0), (10, 0), (20, 0)]},
        "adjacent": {"centerline_xy": [(0, 3), (20, 3)]},
        "out": {"centerline_xy": [(20, 10), (10, 10), (-10, 10)]},
    }
    renderer._connector_by_id = {
        "turn": {"from_lane": "in", "to_lane": "out",
                 "centerline_xy": [(20, 0), (25, 5), (20, 10)]},
        "wrong": {"from_lane": "in", "to_lane": "elsewhere",
                  "centerline_xy": [(20, 0), (40, 0)]},
    }
    preview = {"start_lane_id": "in", "goal_lane_id": "out", "actions": [
        {"type": "connector", "connector_id": "turn",
         "from_lane_id": "in", "to_lane_id": "out"}]}
    vehicle = SimpleNamespace(vehicle_id="ego", current_lane_id="in",
                              active_connector_id="")
    return renderer, vehicle, preview


def test_each_view_clips_current_lane_without_changing_saved_plan():
    renderer, vehicle, preview = _fixture()
    original = copy.deepcopy(preview)
    lines, goal = renderer._remaining_route_lines(vehicle, preview, (4, 0))
    assert lines[0] == [(4, 0), (10, 0), (20, 0)]
    assert goal == "out"
    # This future return lane is behind the ego in heading-up coordinates.
    assert lines[-1][-1] == (-10, 10)
    later, _ = renderer._remaining_route_lines(vehicle, preview, (13, 0))
    assert later[0] == [(13, 0), (20, 0)]
    assert preview == original
    # Tiny pose regressions must not restore a traversed tail.
    again, _ = renderer._remaining_route_lines(vehicle, preview, (12.9, 0))
    assert again == later


def test_connector_then_outgoing_lane_discard_consumed_parts():
    renderer, vehicle, preview = _fixture()
    vehicle.active_connector_id = "turn"
    # SUMO still reports the incoming lane while inside a connector.
    lines, _ = renderer._remaining_route_lines(vehicle, preview, (22.5, 2.5))
    assert len(lines) == 2
    assert lines[0] == [(22.5, 2.5), (25, 5), (20, 10)]
    vehicle.active_connector_id = ""
    vehicle.current_lane_id = "out"
    lines, _ = renderer._remaining_route_lines(vehicle, preview, (15, 10))
    assert lines == [[(15, 10), (10, 10), (-10, 10)]]


def test_lane_change_target_tail_is_clipped_and_stays_clipped_off_route():
    renderer, vehicle, _ = _fixture()
    preview = {"start_lane_id": "in", "goal_lane_id": "adjacent", "actions": [
        {"type": "lane_change", "from_lane_id": "in", "to_lane_id": "adjacent"}]}
    lines, _ = renderer._remaining_route_lines(vehicle, preview, (12, 1))
    assert lines == [[(12, 0), (20, 0)], [(12, 3), (20, 3)]]
    vehicle.current_lane_id = "off_route"
    off_route, _ = renderer._remaining_route_lines(vehicle, preview, (19, 3))
    assert off_route == lines  # No global-nearest guess on a different lane.
    vehicle.current_lane_id = "adjacent"
    lines, _ = renderer._remaining_route_lines(vehicle, preview, (16, 3))
    assert lines == [[(16, 3), (20, 3)]]


def test_wrong_turn_keeps_original_future_route_not_consumed_approach():
    renderer, vehicle, preview = _fixture()
    renderer._remaining_route_lines(vehicle, preview, (18, 0))
    vehicle.active_connector_id = "wrong"
    lines, _ = renderer._remaining_route_lines(vehicle, preview, (30, 0))
    assert lines[0] == renderer._connector_by_id["turn"]["centerline_xy"]
    assert len(lines) == 2
    vehicle.current_lane_id = "elsewhere"
    vehicle.active_connector_id = ""
    assert renderer._remaining_route_lines(vehicle, preview, (40, 0))[0] == lines


def test_progress_is_per_vehicle_and_resets_on_explicit_plan_replacement():
    renderer, vehicle, preview = _fixture()
    renderer._remaining_route_lines(vehicle, preview, (18, 0))
    peer = SimpleNamespace(vehicle_id="peer", current_lane_id="in", active_connector_id="")
    assert renderer._remaining_route_lines(peer, preview, (2, 0))[0][0][0] == (2, 0)
    replaced = copy.deepcopy(preview)
    assert renderer._remaining_route_lines(vehicle, replaced, (3, 0))[0][0][0] == (3, 0)
    assert renderer._remaining_route_lines(vehicle, None, (4, 0)) == ([], "")
    assert "ego" not in renderer._minimap_progress
    assert renderer._remaining_route_lines(vehicle, replaced, (1, 0))[0][0][0] == (1, 0)


def test_future_repeated_lane_is_not_removed_by_global_deduplication():
    renderer, vehicle, preview = _fixture()
    renderer._connector_by_id["back"] = {
        "from_lane": "out", "to_lane": "in",
        "centerline_xy": [(-10, 10), (0, 0)]}
    preview["actions"].append({"type": "connector", "connector_id": "back",
                               "from_lane_id": "out", "to_lane_id": "in"})
    preview["goal_lane_id"] = "in"
    lines, _ = renderer._remaining_route_lines(vehicle, preview, (15, 0))
    assert len(lines) == 5
    assert lines[0][0] == (15, 0)
    assert lines[-1][0] == (0, 0)  # Later visit, not an already-traveled prefix.
    vehicle.current_lane_id = "out"
    renderer._remaining_route_lines(vehicle, preview, (15, 10))
    vehicle.current_lane_id = "in"
    assert renderer._remaining_route_lines(vehicle, preview, (2, 0))[0] == [
        [(2, 0), (10, 0), (20, 0)]]


@pytest.mark.parametrize("scope", ["local", "route"])
def test_rendered_png_has_no_highlight_tail_and_refreshes_without_planning(scope, monkeypatch):
    manager = TrafficCoordinator(load_road_network("beijing_guomao"))
    vehicle = manager.register_vehicle("ego", "n33399858", "n35722739", is_llm=True)
    renderer = AgentVisualRenderer(manager, width=512, height=512)
    preview = {"start_lane_id": vehicle.current_lane_id,
               "goal_lane_id": vehicle.current_lane_id, "actions": []}
    monkeypatch.setattr(manager, "navigation_route_preview",
                        lambda *a, **k: pytest.fail("view invoked route planner"))
    images = []
    for t, progress in enumerate((0.45, 0.75)):
        vehicle.edge_progress = progress
        manager._initialize_vehicle_pose(vehicle)
        rendered = renderer.render_minimap("ego", t, scope=scope, route_preview=preview)
        images.append(rendered)
        assert rendered.sim_time_s == t and rendered.route_available
        image = Image.open(io.BytesIO(rendered.png_bytes)).convert("RGB")
        marker_y = max(y for y in range(512) for x in range(512)
                       if image.getpixel((x, y)) == renderer._EGO_VEHICLE)
        # Beyond the marker halo there must be no cyan current-lane tail.
        assert not any(image.getpixel((x, y)) == renderer._ROUTE
                       for y in range(marker_y + 20, 512) for x in range(512))
    assert images[0].sha256 != images[1].sha256
