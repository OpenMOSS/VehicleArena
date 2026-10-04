"""Regression tests for the browser-3D HD-map scene exporter."""

from __future__ import annotations

import math

import pytest

from visualization.web3d_scene import (
    _clip_marking_outside_junctions,
    build_web3d_scene,
)


def test_web3d_scene_uses_authoritative_lane_level_geometry():
    scene = build_web3d_scene(
        "beijing_tiananmen",
        junction_id="n31194143",
        radius_m=145,
    )

    assert scene["schema"] == "vehiclearena-web3d-v0.1"
    assert scene["map"]["junction_id"] == "n31194143"
    assert len(scene["static"]["roads"]) >= 20
    assert len(scene["static"]["stop_lines"]) >= 4
    assert len(scene["static"]["crosswalks"]) == 4
    assert len(scene["static"]["signals"]) >= 4
    assert any(
        head.get("connector_id")
        for mast in scene["static"]["signals"]
        for head in mast["heads"]
    )
    signal_masts = scene["static"]["signals"]
    signal_heads = [
        head for mast in signal_masts for head in mast["heads"]]
    assert len(signal_heads) > len(signal_masts)
    assert len({mast["approach_id"] for mast in signal_masts}) == len(
        signal_masts)
    for mast in signal_masts:
        assert mast["placement"] == "far_side_exit_overhead"
        assert mast["stop_line_to_mast_m"] > 10.0
        assert mast["roadside_clearance_m"] >= 0.8
        assert mast["heads"]
        assert all(
            head["distance_from_stop_line_m"] > 10.0
            for head in mast["heads"]
        )
        # In the exported X/Z plane the approach's physical right vector is
        # (-sin(heading), cos(heading)). The shared pole must lie beyond every
        # lane head on that side, not within an incoming lane.
        right = (
            -math.sin(mast["heading_rad"]),
            math.cos(mast["heading_rad"]),
        )
        pole = mast["pole_position_xz"]
        for head in mast["heads"]:
            offset = (
                pole[0] - head["position_xz"][0],
                pole[1] - head["position_xz"][1],
            )
            assert offset[0] * right[0] + offset[1] * right[1] > 0.8
    assert all(
        len(road["polygon_xz"]) >= 3
        for road in scene["static"]["roads"]
    )
    junction_guides = [
        marking for marking in scene["static"]["markings"]
        if marking["kind"] == "junction_guide"
    ]
    # Straight guides from crossing approaches previously formed a grid.
    assert not junction_guides
    connector_roads = [
        road for road in scene["static"]["roads"]
        if road["kind"] == "connector"]
    turn_guides = [
        marking for marking in scene["static"]["markings"]
        if marking["kind"] == "junction_turn_guide"
    ]
    assert turn_guides
    assert all(guide["dashed"] for guide in turn_guides)
    assert all(guide["turn"] in {"left", "right"}
               for guide in turn_guides)
    assert len({
        (guide["approach_id"], guide["turn"])
        for guide in turn_guides
    }) == len(turn_guides)
    assert len(turn_guides) < len(connector_roads)
    ground_arrows = scene["static"]["ground_arrows"]
    assert ground_arrows
    assert len({arrow["lane_id"] for arrow in ground_arrows}) == len(
        ground_arrows)
    assert all(arrow["distance_before_stop_line_m"] == 8.0
               for arrow in ground_arrows)
    stop_lines_by_id = {
        stop_line["id"]: stop_line
        for stop_line in scene["static"]["stop_lines"]
    }
    assert all(arrow["stop_line_id"] in stop_lines_by_id
               for arrow in ground_arrows)
    assert all(
        stop_lines_by_id[arrow["stop_line_id"]]["lane_id"]
        == arrow["lane_id"]
        for arrow in ground_arrows
    )
    for arrow in ground_arrows:
        stop_points = stop_lines_by_id[
            arrow["stop_line_id"]]["points_xz"]
        stop_midpoint = tuple(
            sum(point[axis] for point in stop_points) / len(stop_points)
            for axis in (0, 1)
        )
        # These test-map approaches are straight: the projected longitudinal
        # offset should therefore equal the configured eight metres exactly.
        assert math.dist(arrow["position_xz"], stop_midpoint) == pytest.approx(
            8.0, abs=0.02)
    assert all(arrow["movements"] for arrow in ground_arrows)
    assert all(
        set(arrow["movements"]) <= {"left", "straight", "right", "uturn"}
        for arrow in ground_arrows
    )
    assert all(
        arrow["movements"] == arrow["legal_movements"]
        for arrow in ground_arrows
    )
    assert any(len(arrow["movements"]) > 1 for arrow in ground_arrows)

    actors = scene["snapshot"]["actors"]
    ego = next(actor for actor in actors if actor["id"] == "ego")
    assert ego["control"] == "llm"
    assert len(ego["path_xz"]) >= 3
    assert any(actor.get("control") == "sumo" for actor in actors)
    assert any(actor["kind"] == "pedestrian" for actor in actors)


def test_shanghai_shared_lane_arrow_keeps_straight_left_and_uturn():
    """Basic110 must not depict its legal straight route as a turn-only lane."""
    scene = build_web3d_scene(
        "shanghai_lujiazui",
        junction_id="intersection::n554691049",
        radius_m=145,
        include_demo_actors=False,
    )
    arrow = next(
        item for item in scene["static"]["ground_arrows"]
        if item["lane_id"] == "n554691043_n554691049::lane_1"
    )
    assert arrow["legal_movements"] == ["left", "straight", "uturn"]
    assert arrow["movements"] == arrow["legal_movements"]
    # The same contract applies to all approaches, not only the focal lane.
    for item in scene["static"]["ground_arrows"]:
        assert item["movements"] == item["legal_movements"]


def test_web3d_scene_selects_a_valid_junction_for_another_map():
    scene = build_web3d_scene(
        "beijing_guomao",
        junction_id=None,
        radius_m=90,
    )
    assert scene["map"]["id"] == "beijing_guomao"
    assert scene["map"]["junction_id"]
    assert scene["static"]["roads"]


def test_web3d_scene_can_center_a_live_view_without_demo_actors():
    scene = build_web3d_scene(
        "beijing_guomao",
        center_world_xy=[20.0, -30.0],
        radius_m=100,
        include_demo_actors=False,
    )
    assert scene["map"]["center_world_xy"] == [20.0, -30.0]
    assert scene["snapshot"]["actors"] == []


@pytest.mark.parametrize("radius_m", [20, 501])
def test_web3d_scene_rejects_unbounded_exports(radius_m):
    with pytest.raises(ValueError, match="radius_m"):
        build_web3d_scene(radius_m=radius_m)


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("points,expected", [
    ([[-2, 0], [2, 0]], [[[-2, 0], [-1, 0]], [[1, 0], [2, 0]]]),
    ([[-2, 0], [0, 0], [2, 0]], [[[-2, 0], [-1, 0]], [[1, 0], [2, 0]]]),
    ([[0, 0], [.5, .5]], []),
    ([[-2, 2], [0, 2], [2, 2]], [[[-2, 2], [0, 2], [2, 2]]]),
    ([[-2, 1], [2, 1]], [[[-2, 1], [2, 1]]]),  # Along the boundary.
    ([[-2, 0], [0, 2]], [[[-2, 0], [0, 2]]]),  # Touches one corner.
    ([[-2, 0], [-2, 0], [0, 0]], [[[-2, 0], [-1, 0]]]),
])
def test_road_paint_clips_junction_interiors_without_bridging_gaps(reverse, points, expected):
    polygon = [[-1, -1], [1, -1], [1, 1], [-1, 1]]
    if reverse:
        polygon.reverse()
    assert _clip_marking_outside_junctions(points, [polygon]) == expected


def test_road_paint_clips_multiple_overlapping_junctions():
    polygons = [
        [[-2, -1], [1, -1], [1, 1], [-2, 1]],
        [[0, -1], [2, -1], [2, 1], [0, 1]],
    ]
    assert _clip_marking_outside_junctions([[-3, 0], [3, 0]], polygons) == [
        [[-3, 0], [-2, 0]], [[2, 0], [3, 0]]]


@pytest.mark.parametrize("map_id,center", [
    ("hangzhou_binjiang", [-825, -165]),
    ("beijing_tiananmen", None),
    ("shanghai_lujiazui", None),
])
def test_ordinary_road_paint_stays_outside_junctions(map_id, center):
    scene = build_web3d_scene(map_id, center_world_xy=center, radius_m=220,
                             include_demo_actors=False)
    static = scene["static"]
    polygons = [s["polygon_xz"] for s in static["junction_surfaces"]]
    assert static["markings"] and static["ground_arrows"]
    assert static["stop_lines"] and static["crosswalks"]
    for marking in static["markings"]:
        assert marking["kind"] != "junction_guide"
        if marking["kind"] not in {"road_edge", "lane_divider"}:
            continue
        points = marking["points_xz"]
        assert len(points) >= 2
        parts = _clip_marking_outside_junctions(points, polygons)
        original_length = sum(math.dist(a, b) for a, b in zip(points, points[1:]))
        reclipped_length = sum(math.dist(a, b) for part in parts
                               for a, b in zip(part, part[1:]))
        assert reclipped_length == pytest.approx(original_length, abs=1e-6)
