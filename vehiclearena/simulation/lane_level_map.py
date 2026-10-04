"""Generate VehicleArena lane-level maps from an OSM topology source.

The generated map contains:

* lane centre lines and boundaries in a local metric coordinate system;
* directed incoming/outgoing lane ends at intersections;
* cubic-Bezier intersection connectors;
* inferred stop lines and directional signal-junction crosswalks;
* connector conflict pairs computed from sampled geometry.

It intentionally uses only the Python standard library so generated maps are
deterministic and independently inspectable.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

Point = Tuple[float, float]


def _distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _unit(a: Point, b: Point) -> Point:
    length = _distance(a, b)
    if length < 1e-9:
        return 1.0, 0.0
    return (b[0] - a[0]) / length, (b[1] - a[1]) / length


def _offset_polyline(points: Sequence[Point], offset: float) -> List[Point]:
    """Offset a polyline using averaged normals at interior vertices."""
    if len(points) < 2:
        return list(points)
    normals: List[Point] = []
    for a, b in zip(points, points[1:]):
        ux, uy = _unit(a, b)
        normals.append((-uy, ux))
    result = []
    for idx, point in enumerate(points):
        if idx == 0:
            nx, ny = normals[0]
        elif idx == len(points) - 1:
            nx, ny = normals[-1]
        else:
            nx = normals[idx - 1][0] + normals[idx][0]
            ny = normals[idx - 1][1] + normals[idx][1]
            norm = math.hypot(nx, ny)
            if norm > 1e-9:
                nx, ny = nx / norm, ny / norm
            else:
                nx, ny = normals[idx]
        result.append((point[0] + nx * offset, point[1] + ny * offset))
    return result


def _trim_polyline(points: Sequence[Point], start_m: float,
                   end_m: float) -> Tuple[List[Point], float, float]:
    """Trim both ends and return geometry plus the applied trim distances."""
    if len(points) < 2:
        return list(points), 0.0, 0.0

    def trim_start(polyline: Sequence[Point], distance_m: float) -> List[Point]:
        remaining = max(0.0, distance_m)
        result = list(polyline)
        while len(result) >= 2:
            length = _distance(result[0], result[1])
            if length < 1e-9:
                result.pop(0)
                continue
            if remaining < length:
                ratio = remaining / length
                result[0] = (
                    result[0][0] + (result[1][0] - result[0][0]) * ratio,
                    result[0][1] + (result[1][1] - result[0][1]) * ratio,
                )
                return result
            remaining -= length
            result.pop(0)
        return list(polyline[-2:])

    total = sum(_distance(a, b) for a, b in zip(points, points[1:]))
    # A one-metre remnant is not a usable road lane and used to create
    # thousands of near-zero physical edges.  Preserve at least 40% of the
    # source geometry and never less than 3m for a normal road link.
    minimum_remainder = min(total, max(3.0, total * 0.4))
    budget = max(0.0, total - minimum_remainder)
    requested = start_m + end_m
    if requested > budget and requested > 0:
        factor = budget / requested
        start_m *= factor
        end_m *= factor
    result = trim_start(points, start_m)
    result = list(reversed(trim_start(list(reversed(result)), end_m)))
    return result, start_m, end_m


def _bezier(p0: Point, p1: Point, p2: Point, p3: Point,
            samples: int = 12) -> List[Point]:
    result = []
    for i in range(samples + 1):
        t = i / samples
        u = 1.0 - t
        result.append((
            u ** 3 * p0[0] + 3 * u * u * t * p1[0]
            + 3 * u * t * t * p2[0] + t ** 3 * p3[0],
            u ** 3 * p0[1] + 3 * u * u * t * p1[1]
            + 3 * u * t * t * p2[1] + t ** 3 * p3[1],
        ))
    return result


def _orientation(a: Point, b: Point, c: Point) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (
        b[1] - a[1]) * (c[0] - a[0])


def _segments_cross(a: Point, b: Point, c: Point, d: Point) -> bool:
    o1, o2 = _orientation(a, b, c), _orientation(a, b, d)
    o3, o4 = _orientation(c, d, a), _orientation(c, d, b)
    return o1 * o2 < -1e-7 and o3 * o4 < -1e-7


def _polylines_cross(a: Sequence[Point], b: Sequence[Point]) -> bool:
    return any(
        _segments_cross(a0, a1, b0, b1)
        for a0, a1 in zip(a, a[1:])
        for b0, b1 in zip(b, b[1:])
    )


def _polyline_hits_corridor(
    line: Sequence[Point], centerline: Sequence[Point], half_width: float,
) -> bool:
    """Whether a polyline enters a crosswalk's finite-width corridor."""
    if _polylines_cross(line, centerline):
        return True
    return any(
        _point_polyline_distance(point, centerline) <= half_width
        for point in line
    ) or any(
        _point_polyline_distance(point, line) <= half_width
        for point in centerline
    )


def _rectangle(center: Point, along: Point, length: float,
               width: float) -> List[Point]:
    """Rectangle whose long axis is ``along``."""
    ax, ay = along
    norm = math.hypot(ax, ay)
    ax, ay = ((ax / norm, ay / norm) if norm > 1e-9 else (1.0, 0.0))
    nx, ny = -ay, ax
    hl, hw = length / 2.0, width / 2.0
    return [
        (center[0] + ax * sl * hl + nx * sw * hw,
         center[1] + ay * sl * hl + ny * sw * hw)
        for sl, sw in ((1, 1), (1, -1), (-1, -1), (-1, 1))
    ]


def _point_segment_distance(point: Point, a: Point, b: Point) -> float:
    dx, dy = b[0] - a[0], b[1] - a[1]
    length_sq = dx * dx + dy * dy
    if length_sq < 1e-12:
        return _distance(point, a)
    t = max(0.0, min(
        1.0, ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy)
        / length_sq))
    projection = a[0] + t * dx, a[1] + t * dy
    return _distance(point, projection)


def _point_polyline_distance(point: Point, line: Sequence[Point]) -> float:
    return min(
        (_point_segment_distance(point, a, b)
         for a, b in zip(line, line[1:])),
        default=float("inf"),
    )


def _turn_type(in_heading: Point, out_heading: Point) -> str:
    cross = in_heading[0] * out_heading[1] - in_heading[1] * out_heading[0]
    dot = in_heading[0] * out_heading[0] + in_heading[1] * out_heading[1]
    angle = math.degrees(math.atan2(cross, dot))
    if abs(angle) < 35:
        return "straight"
    if abs(angle) > 145:
        return "uturn"
    return "left" if angle > 0 else "right"


@dataclass
class _LaneEnd:
    lane_id: str
    node_id: str
    point: Point
    heading: Point
    incoming: bool
    lane_index: int
    lane_count: int


class LaneLevelMapBuilder:
    """Convert a topology-source road network into lane-level data."""

    def __init__(self, lane_width_m: float = 3.5,
                 junction_cutback_m: float = 8.0):
        self.lane_width_m = lane_width_m
        self.junction_cutback_m = junction_cutback_m

    def build(self, source: dict, source_name: str = "") -> dict:
        nodes = {n["id"]: n for n in source["nodes"]}
        origin_lat = sum(n["lat"] for n in nodes.values()) / len(nodes)
        origin_lng = sum(n["lng"] for n in nodes.values()) / len(nodes)
        cos_lat = math.cos(math.radians(origin_lat))

        def project(lat: float, lng: float) -> Point:
            return (
                (lng - origin_lng) * 111_320.0 * cos_lat,
                (lat - origin_lat) * 110_540.0,
            )

        node_xy = {
            nid: project(n["lat"], n["lng"]) for nid, n in nodes.items()
        }

        # OSM commonly represents a divided-road intersection as four nearby
        # signal nodes joined by short internal road segments. Collapse those
        # nodes into one physical IntersectionArea before generating lanes.
        parent = {nid: nid for nid in nodes}

        def find(node_id: str) -> str:
            while parent[node_id] != node_id:
                parent[node_id] = parent[parent[node_id]]
                node_id = parent[node_id]
            return node_id

        def union(a: str, b: str) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)

        for segment in source["segments"]:
            a, b = segment["from_node"], segment["to_node"]
            source_length = float(segment.get(
                "distance_meters",
                _distance(node_xy[a], node_xy[b])))
            signal_micro_link = (
                nodes[a].get("signal")
                and nodes[b].get("signal")
                and _distance(node_xy[a], node_xy[b]) <= 60.0)
            topology_micro_link = source_length < 3.0
            if signal_micro_link or topology_micro_link:
                union(a, b)

        raw_groups: Dict[str, List[str]] = {}
        for node_id in nodes:
            raw_groups.setdefault(find(node_id), []).append(node_id)
        junction_groups: Dict[str, List[str]] = {}
        node_to_junction: Dict[str, str] = {}
        for members in raw_groups.values():
            if len(members) > 1:
                junction_id = f"intersection::{min(members)}"
            else:
                junction_id = members[0]
            junction_groups[junction_id] = sorted(members)
            for member in members:
                node_to_junction[member] = junction_id

        junction_centers = {}
        junction_extent = {}
        for junction_id, members in junction_groups.items():
            center = (
                sum(node_xy[n][0] for n in members) / len(members),
                sum(node_xy[n][1] for n in members) / len(members),
            )
            junction_centers[junction_id] = center
            junction_extent[junction_id] = max(
                [_distance(center, node_xy[n]) for n in members] + [0.0])

        incident_widths: Dict[str, List[float]] = {nid: [] for nid in nodes}
        incident_counts: Dict[str, int] = {nid: 0 for nid in nodes}
        for segment in source["segments"]:
            lane_width = max(
                2.2, float(segment.get(
                    "lane_width_meters", self.lane_width_m)))
            width = max(1, int(segment.get("lanes", 1))) * lane_width
            for node_id in (segment["from_node"], segment["to_node"]):
                incident_widths[node_id].append(width)
                incident_counts[node_id] += 1
        junction_radii = {}
        for node_id, node in nodes.items():
            width = max(incident_widths[node_id] or [self.lane_width_m])
            is_junction = incident_counts[node_id] >= 3
            if node.get("signal"):
                radius = max(14.0, width * 0.72)
            elif is_junction:
                radius = max(9.0, width * 0.58)
            else:
                radius = 2.5
            junction_radii[node_id] = radius

        lanes: List[dict] = []
        lane_ends: Dict[str, List[_LaneEnd]] = {
            junction_id: [] for junction_id in junction_groups}

        for segment in source["segments"]:
            from_junction = node_to_junction[segment["from_node"]]
            to_junction = node_to_junction[segment["to_node"]]
            # This segment lies inside a clustered physical intersection.
            # It is replaced by generated lane connectors.
            if from_junction == to_junction \
                    and len(junction_groups[from_junction]) > 1:
                continue
            raw_geometry = segment.get("geometry") or [
                [nodes[segment["from_node"]]["lat"],
                 nodes[segment["from_node"]]["lng"]],
                [nodes[segment["to_node"]]["lat"],
                 nodes[segment["to_node"]]["lng"]],
            ]
            centre = [project(lat, lng) for lat, lng in raw_geometry]
            if len(centre) < 2 or _distance(centre[0], centre[-1]) < 0.1:
                continue
            # OSM geometry is not guaranteed to follow from_node → to_node.
            # Orient it explicitly before creating directed lane geometry.
            from_xy = node_xy[segment["from_node"]]
            if _distance(centre[-1], from_xy) < _distance(centre[0], from_xy):
                centre.reverse()
            lane_count = max(1, int(segment.get("lanes", 1)))
            lane_width = max(
                2.2, float(segment.get(
                    "lane_width_meters", self.lane_width_m)))
            shared_bidirectional = False
            if segment.get("oneway", False):
                directions = ["forward"] * lane_count
            elif (
                lane_count == 1
                and segment.get("lanes_forward") is None
                and segment.get("lanes_backward") is None
            ):
                # One physical lane shared by traffic in both directions.
                directions = ["backward", "forward"]
                shared_bidirectional = True
            else:
                forward = segment.get("lanes_forward")
                backward = segment.get("lanes_backward")
                if forward is None:
                    forward = (lane_count + 1) // 2
                if backward is None:
                    backward = max(1, lane_count - int(forward))
                forward = max(1, int(forward))
                backward = max(1, int(backward))
                directions = (
                    ["backward"] * backward
                    + ["forward"] * forward)

            for index, direction in enumerate(directions):
                # Positive offsets are left of the source geometry.
                # Within either travel direction, directional rank 0 is the
                # rightmost lane and increasing ranks move physically left.
                # Forward traffic occupies the right half of a two-way road.
                if segment.get("oneway", False):
                    offset = (
                        index - (len(directions) - 1) / 2) * lane_width
                elif shared_bidirectional:
                    offset = 0.0
                elif direction == "forward":
                    forward_indices = [
                        i for i, d in enumerate(directions) if d == "forward"]
                    rank = forward_indices.index(index)
                    offset = -(
                        len(forward_indices) - rank - 0.5) * lane_width
                else:
                    backward_indices = [
                        i for i, d in enumerate(directions) if d == "backward"]
                    rank = backward_indices.index(index)
                    offset = (
                        len(backward_indices) - rank - 0.5) * lane_width

                line = _offset_polyline(centre, offset)
                if direction == "backward":
                    line.reverse()
                source_length = sum(
                    _distance(a, b) for a, b in zip(line, line[1:]))
                directional_indices = [
                    i for i, item_direction in enumerate(directions)
                    if item_direction == direction]
                directional_rank = directional_indices.index(index)
                directional_count = len(directional_indices)
                start_node = (segment["from_node"] if direction == "forward"
                              else segment["to_node"])
                end_node = (segment["to_node"] if direction == "forward"
                            else segment["from_node"])
                line, applied_start_trim, applied_end_trim = _trim_polyline(
                    line, junction_radii[start_node],
                    junction_radii[end_node])
                physical_index = 0 if shared_bidirectional else index
                lane_id = (
                    f"{segment['id']}::lane_shared_{direction}"
                    if shared_bidirectional
                    else f"{segment['id']}::lane_{index}")
                left = _offset_polyline(line, lane_width / 2)
                right = _offset_polyline(line, -lane_width / 2)
                start_junction = node_to_junction[start_node]
                end_junction = node_to_junction[end_node]
                geometry_length = sum(
                    _distance(a, b) for a, b in zip(line, line[1:]))
                explicit_level = int(segment.get("layer", 0) or 0)
                if explicit_level == 0 and segment.get("bridge"):
                    explicit_level = 1
                if explicit_level == 0 and segment.get("tunnel"):
                    explicit_level = -1
                lane = {
                    "id": lane_id,
                    "segment_id": segment["id"],
                    "index": physical_index,
                    "directional_index": directional_rank,
                    "direction": direction,
                    "shared_bidirectional": shared_bidirectional,
                    "width_m": lane_width,
                    "speed_limit_kmh": segment.get("speed_limit", 50),
                    "length_m": round(geometry_length, 3),
                    "source_length_m": round(source_length, 3),
                    "drivable_length_ratio": round(
                        geometry_length / source_length, 6)
                        if source_length > 0 else 1.0,
                    "start_node": start_node,
                    "end_node": end_node,
                    "start_junction": start_junction,
                    "end_junction": end_junction,
                    "centerline_xy": _rounded(line),
                    "left_boundary_xy": _rounded(left),
                    "right_boundary_xy": _rounded(right),
                    "junction_trim_start_m": round(
                        applied_start_trim, 3),
                    "junction_trim_end_m": round(
                        applied_end_trim, 3),
                    "z_level": explicit_level,
                    "grade_separation_source": (
                        "osm_layer"
                        if segment.get("layer") not in (None, 0)
                        else "osm_bridge"
                        if segment.get("bridge")
                        else "osm_tunnel"
                        if segment.get("tunnel")
                        else "at_grade_topology"),
                    "turn_lanes": (
                        segment.get("turn_lanes_forward", [])
                        if direction == "forward"
                        else segment.get("turn_lanes_backward", [])),
                    "source": "osm+inferred_offset",
                }
                lanes.append(lane)
                start_heading = _unit(line[0], line[1])
                end_heading = _unit(line[-2], line[-1])
                lane_ends[start_junction].append(_LaneEnd(
                    lane_id, start_junction, line[0], start_heading, False,
                    directional_rank, directional_count))
                lane_ends[end_junction].append(_LaneEnd(
                    lane_id, end_junction, line[-1], end_heading, True,
                    directional_rank, directional_count))

        # Infer grade-separated carriageways that geometrically pass through
        # a clustered intersection but have no topological endpoint in it.
        # This recovers a common OSM pattern (mainline over/under local road)
        # even when the topology source omitted bridge/tunnel/layer tags.
        clustered = {
            junction_id: members
            for junction_id, members in junction_groups.items()
            if len(members) > 1
        }
        for lane in lanes:
            if lane.get("z_level", 0) != 0:
                continue
            line = [tuple(point) for point in lane["centerline_xy"]]
            for junction_id in clustered:
                if junction_id in (
                        lane["start_junction"], lane["end_junction"]):
                    continue
                threshold = junction_extent[junction_id] + 3.0
                if _point_polyline_distance(
                        junction_centers[junction_id], line) <= threshold:
                    lane["z_level"] = -1
                    lane["grade_separation_source"] = (
                        "inferred_geometric_crossing_without_topology")
                    break

        lane_levels = {lane["id"]: lane["z_level"] for lane in lanes}
        lane_widths = {lane["id"]: lane["width_m"] for lane in lanes}
        connectors: List[dict] = []
        connector_by_node: Dict[str, List[dict]] = {}
        for node_id, ends in lane_ends.items():
            incoming = [end for end in ends if end.incoming]
            outgoing = [end for end in ends if not end.incoming]
            local = []
            for src in incoming:
                candidates_by_turn: Dict[str, List[_LaneEnd]] = {
                    "straight": [], "left": [], "right": [], "uturn": []}
                eligible: List[Tuple[str, _LaneEnd]] = []
                for dst in outgoing:
                    if lane_levels[src.lane_id] != lane_levels[dst.lane_id]:
                        continue
                    turn = _turn_type(src.heading, dst.heading)
                    same_segment = (
                        src.lane_id.split("::", 1)[0]
                        == dst.lane_id.split("::", 1)[0])
                    # A near-180° connection between two different OSM
                    # segments can be a hairpin/topology bend. Reserve the
                    # U-turn semantic for returning onto the opposite
                    # directed lane of the same physical road segment.
                    if turn == "uturn" and not same_segment:
                        cross = (
                            src.heading[0] * dst.heading[1]
                            - src.heading[1] * dst.heading[0])
                        turn = "left" if cross >= 0 else "right"
                    if same_segment and turn != "uturn":
                        continue
                    candidates_by_turn[turn].append(dst)
                    eligible.append((turn, dst))

                selected: List[Tuple[str, _LaneEnd, str]] = []
                src_norm = (
                    src.lane_index / (src.lane_count - 1)
                    if src.lane_count > 1 else 0.5)

                def candidate_cost(
                        dst: _LaneEnd, prefer_straight: bool = False) -> float:
                    dst_norm = (
                        dst.lane_index / (dst.lane_count - 1)
                        if dst.lane_count > 1 else 0.5)
                    dx = dst.point[0] - src.point[0]
                    dy = dst.point[1] - src.point[1]
                    lateral = abs(
                        src.heading[0] * dy - src.heading[1] * dx)
                    rank_cost = abs(src_norm - dst_norm) * 8.0
                    dot = max(-1.0, min(
                        1.0, src.heading[0] * dst.heading[0]
                        + src.heading[1] * dst.heading[1]))
                    angle = math.acos(dot)
                    if prefer_straight:
                        # At bends, merges and lane-count transitions the
                        # geometric classifier may call the only continuation
                        # a left/right turn. Prefer the smallest heading change
                        # while preserving normalized lane order.
                        return (angle * 30.0 + lateral * 1.5 + rank_cost
                                + math.hypot(dx, dy))
                    return lateral * 3.0 + rank_cost + math.hypot(dx, dy)

                for turn, candidates in candidates_by_turn.items():
                    if not candidates:
                        continue
                    # Basic lane discipline. Lane rank 0 is rightmost in the
                    # travel direction; the highest rank is leftmost.
                    if turn == "left" and src.lane_index != src.lane_count - 1:
                        continue
                    if turn == "right" and src.lane_index != 0:
                        continue
                    if turn == "uturn" and src.lane_index != src.lane_count - 1:
                        continue

                    selected.append((
                        turn, min(candidates, key=candidate_cost), "lane_rule"))

                if not selected and eligible:
                    # Do not leave a physically traversable incoming lane
                    # stranded merely because it is a middle lane at a curved
                    # or non-standard junction.
                    turn, dst = min(
                        eligible,
                        key=lambda item: candidate_cost(
                            item[1], prefer_straight=True))
                    selected.append((turn, dst, "continuity_fallback"))

                for turn, dst, selection in selected:
                    distance = max(
                        4.0, min(18.0, _distance(src.point, dst.point) * 0.38))
                    p1 = (src.point[0] + src.heading[0] * distance,
                          src.point[1] + src.heading[1] * distance)
                    p2 = (dst.point[0] - dst.heading[0] * distance,
                          dst.point[1] - dst.heading[1] * distance)
                    if turn == "uturn":
                        nx, ny = -src.heading[1], src.heading[0]
                        lateral = max(
                            3.5,
                            self.lane_width_m * 1.25)
                        p1 = (p1[0] + nx * lateral,
                              p1[1] + ny * lateral)
                        p2 = (p2[0] + nx * lateral,
                              p2[1] + ny * lateral)
                    curve = _bezier(src.point, p1, p2, dst.point)
                    connector_width = min(
                        lane_widths.get(src.lane_id, self.lane_width_m),
                        lane_widths.get(dst.lane_id, self.lane_width_m))
                    connector = {
                        "id": f"connector::{node_id}::{len(local)}",
                        "node_id": node_id,
                        "from_lane": src.lane_id,
                        "to_lane": dst.lane_id,
                        "turn": turn,
                        "centerline_xy": _rounded(curve),
                        "left_boundary_xy": _rounded(
                            _offset_polyline(curve, connector_width / 2)),
                        "right_boundary_xy": _rounded(
                            _offset_polyline(curve, -connector_width / 2)),
                        "width_m": connector_width,
                        "length_m": round(sum(
                            _distance(a, b)
                            for a, b in zip(curve, curve[1:])), 3),
                        "signal_controlled": any(
                            nodes[n].get("signal")
                            for n in junction_groups[node_id]),
                        "z_level": lane_levels[src.lane_id],
                        "source": "inferred_connector",
                        "selection": selection,
                    }
                    connectors.append(connector)
                    local.append(connector)
            connector_by_node[node_id] = local

        conflicts = []
        for node_id, local in connector_by_node.items():
            for i, first in enumerate(local):
                for second in local[i + 1:]:
                    if first["from_lane"] == second["from_lane"]:
                        continue
                    if first.get("z_level", 0) != second.get("z_level", 0):
                        continue
                    merge_conflict = (
                        first["to_lane"] == second["to_lane"])
                    swept_corridor_conflict = _polyline_hits_corridor(
                        first["centerline_xy"],
                        second["centerline_xy"],
                        max(1.0, min(
                            first.get("width_m", self.lane_width_m),
                            second.get("width_m", self.lane_width_m),
                        ) / 2.0),
                    )
                    if merge_conflict or swept_corridor_conflict:
                        conflicts.append({
                            "node_id": node_id,
                            "connector_a": first["id"],
                            "connector_b": second["id"],
                        })

        stop_lines = []
        crosswalks = []
        for node_id, ends in lane_ends.items():
            incoming = [end for end in ends if end.incoming]
            outgoing = [end for end in ends if not end.incoming]
            is_signal = any(
                nodes[n].get("signal") for n in junction_groups[node_id])
            if is_signal:
                for end in incoming:
                    # A signal-tagged topology endpoint with no reachable
                    # outgoing lane is not a traversable junction. Drawing a
                    # stop line there would invent road connectivity.
                    if not any(
                            lane_levels[end.lane_id]
                            == lane_levels[candidate.lane_id]
                            and end.lane_id.split("::", 1)[0]
                            != candidate.lane_id.split("::", 1)[0]
                            for candidate in outgoing):
                        continue
                    hx, hy = end.heading
                    # Incoming lane ends are already cut back to the junction
                    # boundary; the stop line lies one metre before that edge.
                    centre = (end.point[0] - hx,
                              end.point[1] - hy)
                    nx, ny = -hy, hx
                    lane_width = lane_widths.get(
                        end.lane_id, self.lane_width_m)
                    half = lane_width / 2
                    stop_lines.append({
                        "id": f"stop::{end.lane_id}",
                        "node_id": node_id,
                        "lane_id": end.lane_id,
                        "line_xy": _rounded([
                            (centre[0] - nx * half, centre[1] - ny * half),
                            (centre[0] + nx * half, centre[1] + ny * half),
                        ]),
                        "source": "inferred_from_signal",
                    })
                # Build one directional crosswalk for each physical road
                # mouth. Lane endpoints already lie at the junction boundary,
                # so their lateral span gives the carriageway width.
                mouths: Dict[str, List[_LaneEnd]] = {}
                for end in ends:
                    segment_id = end.lane_id.split("::", 1)[0]
                    mouths.setdefault(segment_id, []).append(end)
                for mouth_index, (segment_id, mouth) in enumerate(
                        sorted(mouths.items())):
                    if not mouth:
                        continue
                    center = (
                        sum(item.point[0] for item in mouth) / len(mouth),
                        sum(item.point[1] for item in mouth) / len(mouth),
                    )
                    # Heading points along the road; the crossing runs across
                    # it. Align opposite-direction lane headings before
                    # averaging.
                    reference = mouth[0].heading
                    aligned = []
                    for item in mouth:
                        hx, hy = item.heading
                        if hx * reference[0] + hy * reference[1] < 0:
                            hx, hy = -hx, -hy
                        aligned.append((hx, hy))
                    hx = sum(item[0] for item in aligned)
                    hy = sum(item[1] for item in aligned)
                    norm = math.hypot(hx, hy)
                    hx, hy = (
                        (hx / norm, hy / norm) if norm > 1e-9
                        else reference)
                    crossing_axis = (-hy, hx)
                    projections = [
                        (item.point[0] - center[0]) * crossing_axis[0]
                        + (item.point[1] - center[1]) * crossing_axis[1]
                        for item in mouth]
                    road_span = (
                        max(projections) - min(projections)
                        + self.lane_width_m)
                    length = max(self.lane_width_m * 2.0, road_span + 2.0)
                    width = 4.0
                    start = (
                        center[0] - crossing_axis[0] * length / 2,
                        center[1] - crossing_axis[1] * length / 2)
                    end = (
                        center[0] + crossing_axis[0] * length / 2,
                        center[1] + crossing_axis[1] * length / 2)
                    crosswalk_id = (
                        f"crosswalk::{node_id}::{mouth_index}")
                    conflicting = [
                        connector["id"]
                        for connector in connector_by_node.get(node_id, [])
                        if _polyline_hits_corridor(
                            [tuple(point) for point
                             in connector["centerline_xy"]],
                            [start, end], width / 2 + 1.0)
                    ]
                    # A road mouth with no vehicle connector is a topology
                    # endpoint, not a traversable intersection crossing.
                    if not conflicting:
                        continue
                    stripe_count = max(3, int(length // 1.2))
                    stripe_length = min(0.6, length / stripe_count * 0.55)
                    stripes = []
                    for stripe_index in range(stripe_count):
                        offset = (
                            -length / 2
                            + (stripe_index + 0.5)
                            * length / stripe_count)
                        stripe_center = (
                            center[0] + crossing_axis[0] * offset,
                            center[1] + crossing_axis[1] * offset)
                        stripes.append(_rounded(_rectangle(
                            stripe_center, crossing_axis,
                            stripe_length, width)))
                    crosswalks.append({
                        "id": crosswalk_id,
                        "node_id": node_id,
                        "road_segment_id": segment_id,
                        "center_xy": _round_point(center),
                        "centerline_xy": _rounded([start, end]),
                        "polygon_xy": _rounded(_rectangle(
                            center, crossing_axis, length, width)),
                        "stripe_polygons_xy": stripes,
                        "length_m": round(length, 3),
                        "width_m": width,
                        "from_wait_area": {
                            "id": f"{crosswalk_id}::wait_a",
                            "xy": _round_point((
                                start[0] - crossing_axis[0] * 2.5,
                                start[1] - crossing_axis[1] * 2.5))},
                        "to_wait_area": {
                            "id": f"{crosswalk_id}::wait_b",
                            "xy": _round_point((
                                end[0] + crossing_axis[0] * 2.5,
                                end[1] + crossing_axis[1] * 2.5))},
                        "conflicting_connectors": conflicting,
                        "source": "inferred_from_lane_mouth_geometry",
                        "confidence": 0.65,
                    })

        # Build lane-level signal plans. Every vehicle phase is a set of
        # non-conflicting connectors; yellow and all-red clearance are
        # explicit parts of the cycle. With no surveyed controller data this
        # is a conservative inferred plan, but its safety relationships are
        # exact with respect to the generated connector graph.
        lane_by_id = {lane["id"]: lane for lane in lanes}
        geometric_conflicts = {
            frozenset((item["connector_a"], item["connector_b"]))
            for item in conflicts}
        signal_plans = []
        for node_id, local in connector_by_node.items():
            if not local or not any(
                    nodes[n].get("signal")
                    for n in junction_groups[node_id]):
                continue
            categories = {
                "east_west_through": [],
                "north_south_through": [],
                "east_west_protected_left": [],
                "north_south_protected_left": [],
            }
            for connector in local:
                source_lane = lane_by_id[connector["from_lane"]]
                line = source_lane["centerline_xy"]
                dx = line[-1][0] - line[-2][0]
                dy = line[-1][1] - line[-2][1]
                axis = "east_west" if abs(dx) >= abs(dy) else "north_south"
                movement = (
                    "protected_left"
                    if connector["turn"] in ("left", "uturn")
                    else "through")
                categories[f"{axis}_{movement}"].append(connector)

            phases = []
            for label, group in categories.items():
                # Greedily split any non-cardinal or merge conflicts that
                # remain within a nominal approach phase.
                buckets: List[List[dict]] = []
                for connector in sorted(group, key=lambda item: item["id"]):
                    placed = False
                    for bucket in buckets:
                        unsafe = any(
                            frozenset((connector["id"], other["id"]))
                            in geometric_conflicts
                            or (connector["to_lane"] == other["to_lane"]
                                and connector["from_lane"]
                                != other["from_lane"])
                            for other in bucket)
                        if not unsafe:
                            bucket.append(connector)
                            placed = True
                            break
                    if not placed:
                        buckets.append([connector])
                # ``25 s through`` / ``10 s protected left`` is the budget
                # for one directional movement family, not for every bucket
                # created by inferred micro-conflicts. Repeating the full
                # budget per bucket produced 180--230 s urban cycles and made
                # a pedestrian wait several minutes. Preserve conflict-free
                # buckets while sharing the family budget among them.
                family_green_s = (
                    25.0 if label.endswith("through") else 10.0)
                bucket_green_s = max(
                    5.0, family_green_s / max(1, len(buckets)))
                for index, bucket in enumerate(buckets):
                    phases.append({
                        "id": f"{node_id}::{label}::{index}",
                        "label": label,
                        "green_s": round(bucket_green_s, 3),
                        "yellow_s": 3.0,
                        "all_red_s": 1.5,
                        "connector_ids": [
                            connector["id"] for connector in bucket],
                        "pedestrian_green": False,
                    })
            phases.append({
                "id": f"{node_id}::pedestrian",
                "label": "pedestrian_all_cross",
                "green_s": 10.0,
                "yellow_s": 0.0,
                "all_red_s": 1.5,
                "connector_ids": [],
                "pedestrian_green": True,
            })
            signal_plans.append({
                "id": f"signal_plan::{node_id}",
                "node_id": node_id,
                "phases": phases,
                "cycle_s": round(sum(
                    phase["green_s"] + phase["yellow_s"]
                    + phase["all_red_s"] for phase in phases), 3),
                "source": "inferred_lane_connector_conflicts",
            })

        # Complex pedestrian topology is authored in the base map instead of
        # guessed by the runtime SUMO compiler. This keeps manual corrections
        # reproducible when lane-level companions are regenerated.
        pedestrian_topology = source.get(
            "lane_level_pedestrian_topology", {})
        if not isinstance(pedestrian_topology, dict):
            raise ValueError(
                "lane_level_pedestrian_topology must be an object")
        crosswalk_by_id = {item["id"]: item for item in crosswalks}
        remove_crosswalk_ids = set(pedestrian_topology.get(
            "remove_crosswalk_ids", []))
        unknown_removals = remove_crosswalk_ids - set(crosswalk_by_id)
        if unknown_removals:
            raise ValueError(
                "pedestrian topology removes unknown crosswalks: "
                f"{sorted(unknown_removals)}")
        crosswalks = [
            item for item in crosswalks
            if item["id"] not in remove_crosswalk_ids
        ]
        crosswalk_by_id = {item["id"]: item for item in crosswalks}
        for update in pedestrian_topology.get("crosswalk_updates", []):
            crosswalk_id = str(update.get("id", ""))
            if crosswalk_id not in crosswalk_by_id:
                raise ValueError(
                    "pedestrian topology updates unknown crosswalk "
                    f"{crosswalk_id!r}")
            crosswalk_by_id[crosswalk_id].update({
                key: value for key, value in update.items()
                if key != "id"
            })
        pedestrian_approaches = sorted(
            pedestrian_topology.get("pedestrian_approaches", []),
            key=lambda item: item["id"],
        )

        return {
            "schema": "vehiclearena-lane-level-v0.3",
            "lane_index_convention": (
                "directional_rightmost_zero_left_increasing"),
            "source_map": source_name,
            "projection": {
                "type": "local_equirectangular",
                "origin_lat": origin_lat,
                "origin_lng": origin_lng,
                "units": "meters",
            },
            "defaults": {"lane_width_m": self.lane_width_m},
            "nodes_xy": [
                {"id": nid, "xy": _round_point(xy),
                 "signal": bool(nodes[nid].get("signal"))}
                for nid, xy in node_xy.items()
            ],
            "intersections": [
                {
                    "id": junction_id,
                    "member_nodes": members,
                    "center_xy": _round_point(junction_centers[junction_id]),
                    "extent_m": round(
                        junction_extent[junction_id]
                        + max(junction_radii[node] for node in members), 3),
                    "kind": (
                        "clustered_physical_intersection"
                        if len(members) > 1 else "topology_junction"),
                    "signal": any(nodes[n].get("signal") for n in members),
                }
                for junction_id, members in junction_groups.items()
            ],
            "lanes": lanes,
            "connectors": connectors,
            "connector_conflicts": conflicts,
            "stop_lines": stop_lines,
            "crosswalks": crosswalks,
            "pedestrian_approaches": pedestrian_approaches,
            "signal_plans": signal_plans,
            "quality": {
                "geometry_is_inferred": True,
                "crosswalks_are_inferred": True,
                "signal_plans_are_inferred": True,
                "survey_grade": False,
                "suitable_for": "lane_level_simulation",
                "not_suitable_for": "survey_grade_claims",
                "counts": {
                    "lanes": len(lanes),
                    "connectors": len(connectors),
                    "connector_conflicts": len(conflicts),
                    "stop_lines": len(stop_lines),
                    "crosswalks": len(crosswalks),
                    "pedestrian_approaches": len(pedestrian_approaches),
                    "signal_plans": len(signal_plans),
                },
            },
        }

def _round_point(point: Point) -> List[float]:
    return [round(point[0], 3), round(point[1], 3)]


def _rounded(points: Iterable[Point]) -> List[List[float]]:
    return [_round_point(point) for point in points]


def build_file(source_path: str, output_path: str) -> dict:
    source = json.loads(Path(source_path).read_text(encoding="utf-8"))
    result = LaneLevelMapBuilder().build(source, Path(source_path).stem)
    Path(output_path).write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
