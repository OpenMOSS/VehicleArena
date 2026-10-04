"""Build compact browser-3D geometry from an authoritative HD map.

Static map geometry and dynamic SUMO state deliberately use separate schemas.
The map is loaded once; live frames only update actors, signals and weather.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from simulation.road_networks import load_road_network
from simulation.traffic_manager import TrafficCoordinator
from visualization.signal_layout import signal_stop_lines


Point = Tuple[float, float]


def _distance(left: Sequence[float], right: Sequence[float]) -> float:
    return math.hypot(
        float(left[0]) - float(right[0]),
        float(left[1]) - float(right[1]),
    )


def _polyline_length(points: Sequence[Sequence[float]]) -> float:
    return sum(_distance(a, b) for a, b in zip(points, points[1:]))


def _point_and_tangent_before_reference(
    points: Sequence[Sequence[float]],
    reference: Sequence[float],
    distance_m: float,
) -> Optional[Tuple[Point, Point]]:
    """Sample a lane centre a fixed distance before a projected reference."""
    if len(points) < 2:
        return None
    lengths = [_distance(first, second)
               for first, second in zip(points, points[1:])]
    best_distance_sq = math.inf
    reference_station = 0.0
    walked = 0.0
    reference_x, reference_y = map(float, reference[:2])
    for (first, second), length in zip(zip(points, points[1:]), lengths):
        if length <= 1e-9:
            continue
        first_x, first_y = map(float, first[:2])
        dx = float(second[0]) - first_x
        dy = float(second[1]) - first_y
        ratio = max(0.0, min(1.0, (
            (reference_x - first_x) * dx
            + (reference_y - first_y) * dy
        ) / (length * length)))
        projected_x = first_x + dx * ratio
        projected_y = first_y + dy * ratio
        distance_sq = (
            (reference_x - projected_x) ** 2
            + (reference_y - projected_y) ** 2
        )
        if distance_sq < best_distance_sq:
            best_distance_sq = distance_sq
            reference_station = walked + length * ratio
        walked += length

    target = max(
        0.0, reference_station - max(0.0, float(distance_m)))
    walked = 0.0
    for index, ((first, second), length) in enumerate(zip(
            zip(points, points[1:]), lengths)):
        if walked + length < target and index < len(lengths) - 1:
            walked += length
            continue
        if length <= 1e-9:
            continue
        ratio = max(0.0, min(1.0, (target - walked) / length))
        dx = float(second[0]) - float(first[0])
        dy = float(second[1]) - float(first[1])
        return (
            (float(first[0]) + dx * ratio,
             float(first[1]) + dy * ratio),
            (dx / length, dy / length),
        )
    return None


def _local_point(point: Sequence[float], center: Point) -> List[float]:
    # Three.js uses X/Z for the horizontal plane. Negating source Y keeps the
    # browser camera orientation consistent with the 2-D driving renderer.
    return [
        round(float(point[0]) - center[0], 3),
        round(center[1] - float(point[1]), 3),
    ]


def _local_polyline(
    points: Iterable[Sequence[float]], center: Point,
) -> List[List[float]]:
    result = [_local_point(point, center) for point in points]
    return [
        point for index, point in enumerate(result)
        if index == 0 or point != result[index - 1]
    ]


def _convex_hull(points: Iterable[Sequence[float]]) -> List[List[float]]:
    values = sorted({
        (round(float(point[0]), 4), round(float(point[1]), 4))
        for point in points
    })
    if len(values) <= 2:
        return [list(point) for point in values]

    def cross(origin: Point, a: Point, b: Point) -> float:
        return ((a[0] - origin[0]) * (b[1] - origin[1])
                - (a[1] - origin[1]) * (b[0] - origin[0]))

    lower: List[Point] = []
    for point in values:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: List[Point] = []
    for point in reversed(values):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return [list(point) for point in lower[:-1] + upper[:-1]]


def _point_segment_distance(point: Point, a: Point, b: Point) -> float:
    dx, dy = b[0] - a[0], b[1] - a[1]
    length_squared = dx * dx + dy * dy
    if length_squared <= 1e-12:
        return _distance(point, a)
    ratio = max(0.0, min(
        1.0,
        ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy)
        / length_squared,
    ))
    projection = a[0] + ratio * dx, a[1] + ratio * dy
    return _distance(point, projection)


def _clip_marking_outside_junctions(
    points: Sequence[Sequence[float]],
    polygons: Sequence[Sequence[Sequence[float]]],
) -> List[List[List[float]]]:
    """Subtract convex junction interiors from road paint, not road geometry.

    Split segments at the actual hull boundary, including segments with both
    endpoints outside. Boundary-following paint is retained. The hulls come
    from _junction_polygon; no geometry dependency or map rewrite is needed.
    """
    parts = [[list(point) for point in points]] if len(points) >= 2 else []
    for polygon in polygons:
        if len(polygon) < 3:
            continue
        edges = list(zip(polygon, [*polygon[1:], polygon[0]]))
        area = sum(a[0] * b[1] - a[1] * b[0] for a, b in edges)
        if abs(area) < 1e-9:
            continue
        orientation = 1 if area > 0 else -1
        min_x, max_x = min(p[0] for p in polygon), max(p[0] for p in polygon)
        min_y, max_y = min(p[1] for p in polygon), max(p[1] for p in polygon)
        clipped = []
        for part in parts:
            if (max(p[0] for p in part) <= min_x
                    or min(p[0] for p in part) >= max_x
                    or max(p[1] for p in part) <= min_y
                    or min(p[1] for p in part) >= max_y):
                clipped.append(part)
                continue
            current = []
            for start, end in zip(part, part[1:]):
                dx, dy = end[0] - start[0], end[1] - start[1]
                if math.hypot(dx, dy) < 1e-9:
                    continue
                enter, leave = 0.0, 1.0
                for a, b in edges:
                    ex, ey = b[0] - a[0], b[1] - a[1]
                    origin = orientation * (ex * (start[1] - a[1])
                                            - ey * (start[0] - a[0]))
                    delta = orientation * (ex * dy - ey * dx)
                    if abs(delta) < 1e-9:
                        if origin <= 1e-9:  # Outside or along the boundary.
                            enter, leave = 1.0, 0.0
                            break
                    elif delta > 0:
                        enter = max(enter, -origin / delta)
                    else:
                        leave = min(leave, -origin / delta)
                intervals = [(0.0, 1.0)]
                if leave - enter > 1e-9:
                    intervals = [(0.0, enter), (leave, 1.0)]
                for low, high in intervals:
                    if high - low <= 1e-9:
                        continue
                    first = [start[0] + low * dx, start[1] + low * dy]
                    last = [start[0] + high * dx, start[1] + high * dy]
                    if current and _distance(current[-1], first) > 1e-7:
                        clipped.append(current)
                        current = []
                    if not current:
                        current.append(first)
                    current.append(last)
            if current:
                clipped.append(current)
        parts = clipped
    return parts


def _point_polyline_distance(
    point: Point, polyline: Sequence[Sequence[float]],
) -> float:
    return min(
        (_point_segment_distance(
            point,
            (float(a[0]), float(a[1])),
            (float(b[0]), float(b[1])),
        ) for a, b in zip(polyline, polyline[1:])),
        default=float("inf"),
    )


def _choose_junction(data: dict, requested: Optional[str]) -> dict:
    intersections = {
        item["id"]: item for item in data["intersections"]}
    if requested:
        try:
            return intersections[requested]
        except KeyError as exc:
            raise ValueError(f"unknown junction {requested!r}") from exc
    for plan in data.get("signal_plans", []):
        intersection = intersections.get(plan["node_id"])
        if intersection is not None:
            return intersection
    return max(
        intersections.values(),
        key=lambda item: float(item.get("extent_m", 0.0)),
    )


def _junction_polygon(
    data: dict, junction: dict, center: Point,
) -> List[List[float]]:
    junction_id = junction["id"]
    points: List[Sequence[float]] = []
    for connector in data["connectors"]:
        if connector["node_id"] != junction_id:
            continue
        points.extend(connector.get("left_boundary_xy", []))
        points.extend(connector.get("right_boundary_xy", []))
    for lane in data["lanes"]:
        if lane.get("start_junction") == junction_id:
            points.extend(lane.get("left_boundary_xy", [])[:1])
            points.extend(lane.get("right_boundary_xy", [])[:1])
        if lane.get("end_junction") == junction_id:
            points.extend(lane.get("left_boundary_xy", [])[-1:])
            points.extend(lane.get("right_boundary_xy", [])[-1:])
    if len(points) < 3:
        x, y = junction["center_xy"]
        extent = max(4.0, float(junction.get("extent_m", 8.0)))
        points = [
            (x - extent, y - extent), (x + extent, y - extent),
            (x + extent, y + extent), (x - extent, y + extent),
        ]
    return _convex_hull(_local_polyline(points, center))


def _path_for_lane(
    lane: dict, connector: Optional[dict], lanes_by_id: Dict[str, dict],
    center: Point,
) -> List[List[float]]:
    points: List[Sequence[float]] = list(lane["centerline_xy"])
    if connector is not None:
        points.extend(connector["centerline_xy"][1:])
        target = lanes_by_id.get(connector["to_lane"])
        if target is not None:
            points.extend(target["centerline_xy"][1:])
    return _local_polyline(points, center)


def _demo_actors(
    data: dict, junction_id: str, center: Point,
) -> List[dict]:
    lanes_by_id = {lane["id"]: lane for lane in data["lanes"]}
    connectors_from: Dict[str, List[dict]] = {}
    for connector in data["connectors"]:
        if connector["node_id"] == junction_id:
            connectors_from.setdefault(
                connector["from_lane"], []).append(connector)
    approaches = [
        lane for lane in data["lanes"]
        if lane.get("end_junction") == junction_id
        and lane["id"] in connectors_from
    ]
    approaches.sort(key=lambda lane: (
        str(lane["segment_id"]), int(lane["index"])))
    if not approaches:
        return []

    selected = []
    used_segments = set()
    for lane in approaches:
        if lane["segment_id"] in used_segments:
            continue
        selected.append(lane)
        used_segments.add(lane["segment_id"])
    for lane in approaches:
        if len(selected) >= 7:
            break
        if lane not in selected:
            selected.append(lane)

    actors = []
    palette = [
        "#31b6e7", "#f59e42", "#65c466", "#d86bde",
        "#f2c14e", "#e66565", "#8b9cf6",
    ]
    for index, lane in enumerate(selected[:7]):
        connectors = connectors_from[lane["id"]]
        connector = next(
            (item for item in connectors if item.get("turn") == "straight"),
            connectors[0],
        )
        path = _path_for_lane(lane, connector, lanes_by_id, center)
        lane_length = _polyline_length(lane["centerline_xy"])
        clearance = 28.0 if index == 0 else 12.0 + index * 5.0
        actors.append({
            "id": "ego" if index == 0 else f"npc_{index}",
            "kind": "vehicle",
            "control": "llm" if index == 0 else "sumo",
            "color": palette[index % len(palette)],
            "dimensions_m": [1.9, 1.55, 4.6],
            "path_xz": path,
            "initial_s_m": round(max(0.0, lane_length - clearance), 3),
            "speed_mps": 5.5 if index == 0 else 3.8 + (index % 3),
            "phase_group": index % 2,
        })

    crosswalks = [
        item for item in data.get("crosswalks", [])
        if item.get("node_id") == junction_id
        and len(item.get("centerline_xy", [])) >= 2
    ]
    for index, crosswalk in enumerate(crosswalks[:4]):
        path = _local_polyline(crosswalk["centerline_xy"], center)
        actors.append({
            "id": f"pedestrian_{index + 1}",
            "kind": "pedestrian",
            "color": "#ffd36a",
            "dimensions_m": [0.55, 1.72, 0.55],
            "path_xz": path,
            "initial_s_m": float(index) * 1.8,
            "speed_mps": 1.1 + 0.1 * index,
            "phase_group": 2,
        })
    return actors


def _decorative_buildings(
    centerlines: Sequence[Sequence[Sequence[float]]], radius_m: float,
) -> List[dict]:
    buildings = []
    step = 34
    limit = int(radius_m // step)
    for grid_x in range(-limit, limit + 1):
        for grid_z in range(-limit, limit + 1):
            x, z = grid_x * step, grid_z * step
            if math.hypot(x, z) < 38.0 or math.hypot(x, z) > radius_m:
                continue
            if min(
                (_point_polyline_distance((x, z), line)
                 for line in centerlines),
                default=float("inf"),
            ) < 15.0:
                continue
            value = abs(grid_x * 37 + grid_z * 53)
            buildings.append({
                "id": f"building_{grid_x}_{grid_z}",
                "position_xz": [x, z],
                "dimensions_m": [
                    18 + value % 9,
                    12 + value % 31,
                    18 + (value * 3) % 11,
                ],
                "color": [
                    "#43515a", "#55636b", "#4d5963", "#657078",
                ][value % 4],
            })
    return buildings[:44]


def build_web3d_scene(
    map_id: str = "beijing_tiananmen",
    *,
    junction_id: Optional[str] = "n31194143",
    radius_m: float = 145.0,
    center_world_xy: Optional[Sequence[float]] = None,
    include_demo_actors: bool = True,
) -> dict:
    """Return one compact, JSON-serialisable browser scene.

    The current lane geometry is XY-only: z_level expresses topological
    separation, not metric altitude. Render a flat projection until a complete
    elevation/ramp/terrain model exists, including for OSM integer layer tags.
    Multiplying levels by 5 invents underground roads without modelling the
    surrounding terrain or ramps. This projection never modifies the map.
    """
    if radius_m < 50.0 or radius_m > 500.0:
        raise ValueError("radius_m must be between 50 and 500")
    road_network = load_road_network(map_id)
    manager = TrafficCoordinator(road_network)
    data = manager._lane_geometry.data
    if junction_id and not any(
            item["id"] == junction_id for item in data["intersections"]):
        # The default junction only belongs to the default map. Other maps
        # automatically choose their first signalised intersection.
        if map_id != "beijing_tiananmen":
            junction_id = None
    if center_world_xy is not None:
        if len(center_world_xy) != 2:
            raise ValueError("center_world_xy must contain exactly two values")
        center = tuple(map(float, center_world_xy))
        junction = min(
            data["intersections"],
            key=lambda item: _distance(item["center_xy"], center),
        )
    else:
        junction = _choose_junction(data, junction_id)
        center = tuple(map(float, junction["center_xy"]))

    def nearby(points: Sequence[Sequence[float]]) -> bool:
        return bool(points) and min(
            _distance(point, center) for point in points) <= radius_m

    lanes = [
        lane for lane in data["lanes"]
        if nearby(lane.get("centerline_xy", []))
    ]
    lane_ids = {lane["id"] for lane in lanes}
    connectors = [
        item for item in data["connectors"]
        if item.get("from_lane") in lane_ids
        and nearby(item.get("centerline_xy", []))
    ]
    junctions = [
        item for item in data["intersections"]
        if _distance(item["center_xy"], center) <= radius_m
    ]

    roads = []
    centerlines = []
    lane_groups: Dict[Tuple[str, str], List[dict]] = {}
    for lane in lanes:
        polygon = list(lane.get("left_boundary_xy", [])) + list(reversed(
            lane.get("right_boundary_xy", [])))
        if len(polygon) < 3:
            continue
        centerline = _local_polyline(lane["centerline_xy"], center)
        centerlines.append(centerline)
        roads.append({
            "id": lane["id"],
            "kind": "lane",
            "polygon_xz": _local_polyline(polygon, center),
            "elevation_m": 0.0,
        })
        lane_groups.setdefault(
            (lane["segment_id"], lane["direction"]), []).append(lane)
    for connector in connectors:
        polygon = list(connector.get("left_boundary_xy", [])) + list(
            reversed(connector.get("right_boundary_xy", [])))
        if len(polygon) >= 3:
            roads.append({
                "id": connector["id"],
                "kind": "connector",
                "polygon_xz": _local_polyline(polygon, center),
                "elevation_m": 0.0,
            })

    surfaces = [{
        "id": item["id"],
        "polygon_xz": _junction_polygon(data, item, center),
        "elevation_m": 0.0,
    } for item in junctions]

    markings = []
    junction_polygons = [surface["polygon_xz"] for surface in surfaces]

    def add_road_marking(lane_id: str, points: List[List[float]], outer: bool):
        for part in _clip_marking_outside_junctions(points, junction_polygons):
            markings.append({
                "kind": "road_edge" if outer else "lane_divider",
                "lane_id": lane_id,
                "elevation_m": 0.0,
                "points_xz": part,
                "color": "#e7edf0" if outer else "#e8e4d3",
                "dashed": not outer,
            })

    for group in lane_groups.values():
        group.sort(key=lambda lane: int(lane.get("directional_index", 0)))
        for index, lane in enumerate(group):
            right = _local_polyline(lane["right_boundary_xy"], center)
            left = _local_polyline(lane["left_boundary_xy"], center)
            if index == 0:
                add_road_marking(lane["id"], right, True)
            is_outer = index == len(group) - 1
            add_road_marking(lane["id"], left, is_outer)

    # Crossing approaches must not continue their straight lane dividers into
    # a junction-wide grid. Keep only the sparse curved turn guides below.
    # Connector roads and every legal movement arrow remain authoritative.
    connectors_by_source: Dict[str, List[dict]] = {}
    for connector in connectors:
        connectors_by_source.setdefault(
            str(connector.get("from_lane", "")), []).append(connector)
    # Add one readable curved guide for each legal turn on an approach.  We
    # deliberately do not expose every lane-to-lane connector: large junctions
    # can contain dozens of equivalent connector curves and drawing all of
    # them turns the driver's view into a mesh.  The representative uses the
    # inside boundary of the natural turn lane so it reads as a lane guide,
    # not as a route centreline.
    visible_lanes_by_id = {str(lane["id"]): lane for lane in lanes}
    turn_guide_radius_m = min(float(radius_m), 70.0)
    turn_guide_junction_ids = {
        str(item["id"])
        for item in junctions
        if _distance(item["center_xy"], center) <= turn_guide_radius_m
    }
    turn_guide_candidates: Dict[Tuple[str, str, str, str], List[dict]] = {}
    for connector in connectors:
        turn = str(connector.get("turn", ""))
        # U-turns share the left-turn guide. Rendering their tight loop as an
        # additional marking is visually noisy and is not normal road paint.
        if turn not in {"left", "right"}:
            continue
        if str(connector.get("node_id", "")) not in turn_guide_junction_ids:
            continue
        source_lane = visible_lanes_by_id.get(
            str(connector.get("from_lane", "")))
        if source_lane is None:
            continue
        key = (
            str(connector.get("node_id", "")),
            str(source_lane.get("segment_id", "")),
            str(source_lane.get("direction", "")),
            turn,
        )
        turn_guide_candidates.setdefault(key, []).append(connector)

    for key, candidates in sorted(turn_guide_candidates.items()):
        turn = key[-1]
        choose_leftmost = turn == "left"
        connector = max(
            candidates,
            key=lambda item: int(visible_lanes_by_id[
                str(item["from_lane"])].get("directional_index", 0)),
        ) if choose_leftmost else min(
            candidates,
            key=lambda item: int(visible_lanes_by_id[
                str(item["from_lane"])].get("directional_index", 0)),
        )
        boundary_key = (
            "right_boundary_xy" if choose_leftmost
            else "left_boundary_xy"
        )
        guide = _local_polyline(connector.get(boundary_key, []), center)
        if len(guide) < 2:
            continue
        approach_id = "::".join(key[:-1])
        markings.append({
            "kind": "junction_turn_guide",
            "connector_id": str(connector["id"]),
            "source_lane_id": str(connector["from_lane"]),
            "elevation_m": 0.0,
            "approach_id": approach_id,
            "turn": turn,
            "points_xz": guide,
            "color": "#d8d4c8",
            "dashed": True,
            "dash_size_m": 1.8,
            "gap_size_m": 2.6,
            "opacity": 0.64,
        })

    movement_order = ("left", "straight", "right", "uturn")
    nearby_junction_ids = {item["id"] for item in junctions}
    stop_line_by_lane = {
        str(item.get("lane_id", "")): item
        for item in data.get("stop_lines", [])
        if item.get("node_id") in nearby_junction_ids
        and item.get("lane_id") in lane_ids
        and len(item.get("line_xy", [])) >= 2
    }
    lane_positions = {
        str(lane["id"]): (index, len(group))
        for group in lane_groups.values()
        for index, lane in enumerate(group)
    }
    ground_arrows = []
    for lane in lanes:
        stop_line = stop_line_by_lane.get(str(lane["id"]))
        if stop_line is None:
            # Direction arrows belong to an incoming approach. A connector
            # alone is not sufficient: outbound lanes must remain unmarked.
            continue
        lane_connectors = connectors_by_source.get(str(lane["id"]), [])
        legal_movements = {
            str(connector.get("turn", ""))
            for connector in lane_connectors
        }
        legal_movements = [
            movement for movement in movement_order
            if movement in legal_movements
        ]
        lane_index, lane_count = lane_positions[str(lane["id"])]
        is_leftmost_lane = lane_index == lane_count - 1
        # Road paint must expose every legal movement, independently of the
        # selected route. Do not infer turn restrictions from lane position
        # or truncate compound arrows for visual simplicity.
        movements = list(legal_movements)
        stop_points = stop_line["line_xy"]
        stop_midpoint = (
            sum(float(point[0]) for point in stop_points) / len(stop_points),
            sum(float(point[1]) for point in stop_points) / len(stop_points),
        )
        sampled = _point_and_tangent_before_reference(
            lane.get("centerline_xy", []), stop_midpoint, 8.0)
        if not movements or sampled is None:
            continue
        position, tangent = sampled
        # Source world XY becomes browser X/-Y on the ground plane.
        rotation_y = math.atan2(tangent[0], -tangent[1])
        ground_arrows.append({
            "id": f"movement_arrow::{lane['id']}",
            "lane_id": str(lane["id"]),
            "position_xz": _local_point(position, center),
            "rotation_y_rad": round(rotation_y, 6),
            "z_level": int(lane.get("z_level", 0)),
            "elevation_m": 0.0,
            "stop_line_id": str(stop_line["id"]),
            "distance_before_stop_line_m": 8.0,
            "is_leftmost_lane": is_leftmost_lane,
            "legal_movements": legal_movements,
            "movements": movements,
        })

    stop_lines = []
    signals = []
    lanes_by_id = {lane["id"]: lane for lane in data["lanes"]}
    signalised_junction_ids = {
        str(plan.get("node_id", ""))
        for plan in data.get("signal_plans", [])
    }
    # The primary signal for an incoming approach belongs on the far-side
    # exit boundary of the junction, not immediately above its stop line.
    # Connector geometry spans exactly that junction interior, so its outer
    # points give us a map-derived exit plane for every approach direction.
    junction_geometry_points: Dict[str, List[Point]] = {}
    for connector in data.get("connectors", []):
        node_id = str(connector.get("node_id", ""))
        if node_id not in nearby_junction_ids:
            continue
        bucket = junction_geometry_points.setdefault(node_id, [])
        for field in (
                "left_boundary_xy", "right_boundary_xy", "centerline_xy"):
            bucket.extend(
                (float(point[0]), float(point[1]))
                for point in connector.get(field, [])
                if len(point) >= 2
            )
    signal_approaches: Dict[Tuple[str, str, str, int], dict] = {}
    for item in signal_stop_lines(data):
        if item.get("node_id") not in nearby_junction_ids:
            continue
        points = item.get("line_xy", [])
        if len(points) < 2:
            continue
        local_line = _local_polyline(points, center)
        stop_lines.append({
            "id": item["id"],
            "lane_id": str(item.get("lane_id", "")),
            "node_id": str(item.get("node_id", "")),
            "points_xz": local_line,
            "elevation_m": 0.0,
        })
        lane = lanes_by_id.get(item.get("lane_id"))
        if (lane is None
                or str(item.get("node_id", ""))
                not in signalised_junction_ids
                or len(lane.get("centerline_xy", [])) < 2):
            continue
        signal_groups = manager._lane_geometry.signal_groups_for_lane(lane["id"])
        if not signal_groups:
            continue
        a, b = lane["centerline_xy"][-2:]
        heading = math.atan2(
            -(float(b[1]) - float(a[1])),
            float(b[0]) - float(a[0]),
        )
        dx = float(b[0]) - float(a[0])
        dy = float(b[1]) - float(a[1])
        length = math.hypot(dx, dy)
        if length <= 1e-9:
            continue
        travel = (dx / length, dy / length)
        # Right-hand traffic: one shared mast stands beyond the right edge of
        # the incoming carriageway. Its arm crosses the approach and carries
        # independently controlled movement heads over their incoming lanes.
        road_right = (travel[1], -travel[0])
        key = (
            str(item.get("node_id", "")),
            str(lane.get("segment_id", "")),
            str(lane.get("direction", "")),
            int(lane.get("z_level", 0)),
        )
        approach = signal_approaches.setdefault(key, {
            "heading_rad": heading,
            "travel": travel,
            "road_right": road_right,
            "edge_candidates": [],
            "junction_points": junction_geometry_points.get(
                str(item.get("node_id", "")), []),
            "heads": [],
        })
        approach["edge_candidates"].extend([
            (float(point[0]), float(point[1])) for point in points])
        midpoint_world = (
            (float(points[0][0]) + float(points[-1][0])) / 2.0,
            (float(points[0][1]) + float(points[-1][1])) / 2.0,
        )
        # Keep all movement heads inside their own lane's projected width.
        # Order is driver's left-to-right: U-turn, left, straight, right.
        spacing = min(1.25, float(lane.get("width_m", 3.5)) / len(signal_groups))
        for index, signal_group in enumerate(signal_groups):
            offset = (index - (len(signal_groups) - 1) / 2) * spacing
            approach["heads"].append({
                "id": f"signal_head::{item['id']}::{signal_group['connector_id']}",
                "lane_id": str(lane["id"]),
                "directional_index": int(lane.get(
                    "directional_index", lane.get("index", 0))),
                "position_world_xy": (
                    midpoint_world[0] + road_right[0] * offset,
                    midpoint_world[1] + road_right[1] * offset),
                "width_m": min(1.05, spacing * 0.82),
                **signal_group,
            })

    roadside_clearance_m = 0.9
    exit_side_clearance_m = 0.35
    for key, approach in sorted(signal_approaches.items()):
        road_right = approach["road_right"]
        travel = approach["travel"]
        junction_points = approach["junction_points"]
        if not junction_points:
            # A signalised approach without connector geometry is not a valid
            # lane-level junction and cannot be placed truthfully in 3-D.
            continue
        exit_projection = max(
            point[0] * travel[0] + point[1] * travel[1]
            for point in junction_points
        ) + exit_side_clearance_m
        right_edge = max(
            approach["edge_candidates"],
            key=lambda point: (
                point[0] * road_right[0] + point[1] * road_right[1]),
        )
        right_edge_projection = (
            right_edge[0] * travel[0] + right_edge[1] * travel[1])
        distance_to_exit_m = exit_projection - right_edge_projection
        if distance_to_exit_m <= 0.0:
            continue
        pole_world = (
            right_edge[0] + road_right[0] * roadside_clearance_m
            + travel[0] * distance_to_exit_m,
            right_edge[1] + road_right[1] * roadside_clearance_m
            + travel[1] * distance_to_exit_m,
        )
        heads = sorted(
            approach["heads"],
            key=lambda head: head["directional_index"],
        )
        for head in heads:
            head_projection = (
                head["position_world_xy"][0] * travel[0]
                + head["position_world_xy"][1] * travel[1]
            )
            head_distance_to_exit_m = exit_projection - head_projection
            head["mount_position_world_xy"] = (
                head["position_world_xy"][0]
                + travel[0] * head_distance_to_exit_m,
                head["position_world_xy"][1]
                + travel[1] * head_distance_to_exit_m,
            )
            head["distance_from_stop_line_m"] = head_distance_to_exit_m
        farthest_head = max(
            heads,
            key=lambda head: _distance(
                pole_world, head["mount_position_world_xy"]),
        )
        node_id, segment_id, direction, z_level = key
        signals.append({
            "id": f"signal_mast::{node_id}::{segment_id}::{direction}",
            "node_id": node_id,
            "approach_id": f"{segment_id}::{direction}",
            "placement": "far_side_exit_overhead",
            "pole_position_xz": _local_point(pole_world, center),
            "arm_end_xz": _local_point(
                farthest_head["mount_position_world_xy"], center),
            "heading_rad": round(float(approach["heading_rad"]), 6),
            "z_level": z_level,
            "elevation_m": 0.0,
            "roadside_clearance_m": roadside_clearance_m,
            "stop_line_to_mast_m": round(distance_to_exit_m, 3),
            "heads": [{
                "id": head["id"],
                "lane_id": head["lane_id"],
                "position_xz": _local_point(
                    head["mount_position_world_xy"], center),
                "connector_id": head["connector_id"],
                "connector_ids": head["connector_ids"],
                "turn": head["turn"],
                "width_m": head["width_m"],
                "distance_from_stop_line_m": round(
                    head["distance_from_stop_line_m"], 3),
            } for head in heads],
        })

    crosswalks = []
    for item in data.get("crosswalks", []):
        if item.get("node_id") not in nearby_junction_ids:
            continue
        stripes = [
            _local_polyline(polygon, center)
            for polygon in item.get("stripe_polygons_xy", [])
            if len(polygon) >= 3
        ]
        crosswalks.append({
            "id": item["id"],
            "stripes_xz": stripes,
        })

    actors = (
        _demo_actors(data, junction["id"], center)
        if include_demo_actors else [])
    return {
        "schema": "vehiclearena-web3d-v0.1",
        "map": {
            "id": map_id,
            "junction_id": junction["id"],
            "title": getattr(road_network, "name", "") or map_id,
            "center_world_xy": list(center),
            "radius_m": float(radius_m),
            "source": data.get("source_map", map_id),
            "elevation_mode": "flat_2d",
        },
        "static": {
            "roads": roads,
            "junction_surfaces": surfaces,
            "markings": markings,
            "ground_arrows": ground_arrows,
            "stop_lines": stop_lines,
            "crosswalks": crosswalks,
            "signals": signals,
            "buildings": _decorative_buildings(centerlines, radius_m),
        },
        "snapshot": {
            "sim_time_s": 0.0,
            "weather": "clear",
            "daylight": "day",
            "actors": actors,
        },
        "notice": (
            "Static HD-map geometry is authoritative. Dynamic actors are "
            "included only for an explicitly requested static preview. "
            "Elevation is a flat 2D projection; topology layers do not "
            "represent measured road height or reconstructed overpasses."
        ),
    }
