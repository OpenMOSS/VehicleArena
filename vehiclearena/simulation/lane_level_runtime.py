"""Lane geometry, routing and signal semantics for VehicleArena."""

from __future__ import annotations

import math
import heapq
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

Point = Tuple[float, float]


def _distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _polyline_length(points: Sequence[Point]) -> float:
    return sum(_distance(a, b) for a, b in zip(points, points[1:]))


def _segment_intersection(
    a: Point, b: Point, c: Point, d: Point,
) -> Optional[Tuple[Point, float, float]]:
    """Return the intersection and segment ratios for two finite segments."""
    abx, aby = b[0] - a[0], b[1] - a[1]
    cdx, cdy = d[0] - c[0], d[1] - c[1]
    denominator = abx * cdy - aby * cdx
    if abs(denominator) < 1e-9:
        return None
    acx, acy = c[0] - a[0], c[1] - a[1]
    first_ratio = (acx * cdy - acy * cdx) / denominator
    second_ratio = (acx * aby - acy * abx) / denominator
    if not (-1e-7 <= first_ratio <= 1.0 + 1e-7
            and -1e-7 <= second_ratio <= 1.0 + 1e-7):
        return None
    return (
        (a[0] + abx * first_ratio, a[1] + aby * first_ratio),
        max(0.0, min(1.0, first_ratio)),
        max(0.0, min(1.0, second_ratio)),
    )


def _polyline_intersection_distances(
    first: Sequence[Point], second: Sequence[Point],
) -> Optional[Tuple[Point, float, float]]:
    """Return the earliest shared point and distance along both polylines."""
    first_base = 0.0
    candidates = []
    for a, b in zip(first, first[1:]):
        first_length = _distance(a, b)
        second_base = 0.0
        for c, d in zip(second, second[1:]):
            second_length = _distance(c, d)
            hit = _segment_intersection(a, b, c, d)
            if hit:
                point, first_ratio, second_ratio = hit
                candidates.append((
                    first_base + first_length * first_ratio,
                    second_base + second_length * second_ratio,
                    point,
                ))
            second_base += second_length
        first_base += first_length
    if not candidates:
        return None
    first_s, second_s, point = min(
        candidates, key=lambda item: (item[0], item[1]))
    return point, first_s, second_s


def _point_segment_projection(
    point: Point, a: Point, b: Point,
) -> Tuple[float, float]:
    """Return (distance, ratio) for a point projected onto a segment."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    length_sq = dx * dx + dy * dy
    if length_sq < 1e-12:
        return _distance(point, a), 0.0
    ratio = max(0.0, min(
        1.0,
        ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy)
        / length_sq,
    ))
    projection = a[0] + dx * ratio, a[1] + dy * ratio
    return _distance(point, projection), ratio


def _nearest_polyline_distance(
    path: Sequence[Point], target: Sequence[Point],
) -> float:
    """Approximate distance along *path* nearest to the target polyline."""
    best = (float("inf"), 0.0)
    path_base = 0.0
    for a, b in zip(path, path[1:]):
        segment_length = _distance(a, b)
        for point in target:
            distance, ratio = _point_segment_projection(point, a, b)
            candidate = path_base + segment_length * ratio
            if (distance, candidate) < best:
                best = distance, candidate
        path_base += segment_length
    return best[1]


def _polyline_nearest_distances(
    first: Sequence[Point], second: Sequence[Point],
) -> Optional[Tuple[Point, float, float]]:
    """Return the nearest sampled corridor point and distance along both paths."""
    if len(first) < 2 or len(second) < 2:
        return None
    best = (float("inf"), 0.0, 0.0, first[0], second[0])

    first_vertex_s = 0.0
    for index, point in enumerate(first):
        second_base = 0.0
        for a, b in zip(second, second[1:]):
            distance, ratio = _point_segment_projection(point, a, b)
            projected = (
                a[0] + (b[0] - a[0]) * ratio,
                a[1] + (b[1] - a[1]) * ratio,
            )
            candidate = (
                distance, first_vertex_s,
                second_base + _distance(a, b) * ratio,
                point, projected)
            if candidate[:3] < best[:3]:
                best = candidate
            second_base += _distance(a, b)
        if index < len(first) - 1:
            first_vertex_s += _distance(first[index], first[index + 1])

    second_vertex_s = 0.0
    for index, point in enumerate(second):
        first_base = 0.0
        for a, b in zip(first, first[1:]):
            distance, ratio = _point_segment_projection(point, a, b)
            projected = (
                a[0] + (b[0] - a[0]) * ratio,
                a[1] + (b[1] - a[1]) * ratio,
            )
            candidate = (
                distance,
                first_base + _distance(a, b) * ratio,
                second_vertex_s, projected, point)
            if candidate[:3] < best[:3]:
                best = candidate
            first_base += _distance(a, b)
        if index < len(second) - 1:
            second_vertex_s += _distance(
                second[index], second[index + 1])

    _, first_s, second_s, first_point, second_point = best
    midpoint = (
        (first_point[0] + second_point[0]) / 2.0,
        (first_point[1] + second_point[1]) / 2.0,
    )
    return midpoint, first_s, second_s


def _point_polyline_distance(
    point: Point, path: Sequence[Point],
) -> float:
    return min(
        (_point_segment_projection(point, a, b)[0]
         for a, b in zip(path, path[1:])),
        default=float("inf"),
    )


def _earliest_corridor_distance(
    path: Sequence[Point], corridor_centerline: Sequence[Point],
    half_width_m: float, sample_step_m: float = 0.25,
) -> Optional[float]:
    """Earliest path distance entering a finite-width centerline corridor."""
    path_base = 0.0
    for a, b in zip(path, path[1:]):
        length = _distance(a, b)
        samples = max(1, math.ceil(length / sample_step_m))
        for index in range(samples + 1):
            ratio = index / samples
            point = (
                a[0] + (b[0] - a[0]) * ratio,
                a[1] + (b[1] - a[1]) * ratio,
            )
            if (_point_polyline_distance(
                    point, corridor_centerline)
                    <= half_width_m):
                return path_base + length * ratio
        path_base += length
    return None


def pose_at(points: Sequence[Point], distance_s: float) -> Tuple[float, float, float]:
    """Return x, y, heading at metric distance along a polyline."""
    remaining = max(0.0, distance_s)
    for a, b in zip(points, points[1:]):
        length = _distance(a, b)
        if length < 1e-9:
            continue
        if remaining <= length:
            ratio = remaining / length
            x = a[0] + (b[0] - a[0]) * ratio
            y = a[1] + (b[1] - a[1]) * ratio
            return x, y, math.atan2(b[1] - a[1], b[0] - a[0])
        remaining -= length
    a, b = points[-2], points[-1]
    return b[0], b[1], math.atan2(b[1] - a[1], b[0] - a[0])


@dataclass(frozen=True)
class VehicleFootprint:
    length_m: float = 4.6
    width_m: float = 1.9

    def corners(self, pose: Tuple[float, float, float]) -> List[Point]:
        x, y, heading = pose
        forward = math.cos(heading), math.sin(heading)
        left = -forward[1], forward[0]
        half_l, half_w = self.length_m / 2, self.width_m / 2
        return [
            (x + forward[0] * sl * half_l + left[0] * sw * half_w,
             y + forward[1] * sl * half_l + left[1] * sw * half_w)
            for sl, sw in ((1, 1), (1, -1), (-1, -1), (-1, 1))
        ]


@dataclass(frozen=True)
class LaneSignalState:
    signal: str
    remaining_seconds: float
    phase_id: str
    pedestrian_green: bool = False


class LaneGeometryRuntime:
    """Authoritative lane geometry and route runtime used by TrafficCoordinator."""

    def __init__(self, data: dict):
        self._validate_map_contract(data)
        self.data = data
        self.projection = data["projection"]
        raw_nodes = data["nodes_xy"]
        self.nodes_xy = (
            {item["id"]: tuple(item["xy"]) for item in raw_nodes}
            if isinstance(raw_nodes, list)
            else {key: tuple(value) for key, value in raw_nodes.items()})
        self._lanes: Dict[Tuple[str, str], List[dict]] = {}
        for lane in data["lanes"]:
            self._lanes.setdefault(
                (lane["segment_id"], lane["direction"]), []).append(lane)
        for lanes in self._lanes.values():
            lanes.sort(key=lambda lane: lane["index"])
        self._connectors = {
            (item["from_lane"], item["to_lane"]): [
                tuple(point) for point in item["centerline_xy"]]
            for item in data["connectors"]}
        self._connector_records = {
            (item["from_lane"], item["to_lane"]): item
            for item in data["connectors"]}
        self._connector_by_id = {
            item["id"]: item for item in data["connectors"]}
        self._connector_lengths = {
            item["id"]: _polyline_length([
                tuple(point) for point in item["centerline_xy"]])
            for item in data["connectors"]}
        self._connectors_from: Dict[str, List[dict]] = {}
        for item in data["connectors"]:
            self._connectors_from.setdefault(
                item["from_lane"], []).append(item)
        self._lane_by_id = {lane["id"]: lane for lane in data["lanes"]}
        self._lane_lengths = {
            lane["id"]: _polyline_length([
                tuple(point) for point in lane["centerline_xy"]])
            for lane in data["lanes"]}
        self._node_to_junction = {}
        for intersection in data.get("intersections", []):
            for node_id in intersection.get("member_nodes", []):
                self._node_to_junction[node_id] = intersection["id"]
        self._conflict_degree: Dict[str, int] = {}
        self._connector_conflicts: Dict[str, set] = {}
        self._connector_conflict_points: Dict[str, List[dict]] = {}
        for conflict in data.get("connector_conflicts", []):
            first, second = (
                conflict["connector_a"], conflict["connector_b"])
            self._connector_conflicts.setdefault(first, set()).add(second)
            self._connector_conflicts.setdefault(second, set()).add(first)
            for connector_id in (first, second):
                self._conflict_degree[connector_id] = (
                    self._conflict_degree.get(connector_id, 0) + 1)
            first_record = self._connector_by_id.get(first)
            second_record = self._connector_by_id.get(second)
            if not first_record or not second_record:
                continue
            hit = _polyline_intersection_distances(
                [tuple(point)
                 for point in first_record["centerline_xy"]],
                [tuple(point)
                 for point in second_record["centerline_xy"]],
            )
            if not hit:
                hit = _polyline_nearest_distances(
                    [tuple(point)
                     for point in first_record["centerline_xy"]],
                    [tuple(point)
                     for point in second_record["centerline_xy"]],
                )
            if not hit:
                continue
            point, first_s, second_s = hit
            self._connector_conflict_points.setdefault(
                first, []).append({
                    "other_connector_id": second,
                    "self_distance_s_m": first_s,
                    "other_distance_s_m": second_s,
                    "point_xy": point,
                })
            self._connector_conflict_points.setdefault(
                second, []).append({
                    "other_connector_id": first,
                    "self_distance_s_m": second_s,
                    "other_distance_s_m": first_s,
                    "point_xy": point,
                })
        self._signal_plans = {
            item["node_id"]: item for item in data.get("signal_plans", [])}
        self._signal_plan_by_connector = {}
        for plan in self._signal_plans.values():
            for phase in plan["phases"]:
                for connector_id in phase["connector_ids"]:
                    self._signal_plan_by_connector[connector_id] = plan
        self._stop_lane_ids = {
            item["lane_id"] for item in data.get("stop_lines", [])}
        self._crosswalk_by_id = {
            item["id"]: item for item in data.get("crosswalks", [])}
        self._crosswalks_by_junction: Dict[str, List[dict]] = {}
        self._crosswalk_conflicts_by_connector: Dict[
            str, List[dict]] = {}
        for item in data.get("crosswalks", []):
            self._crosswalks_by_junction.setdefault(
                item["node_id"], []).append(item)
            crosswalk_line = [
                tuple(point)
                for point in item.get("centerline_xy", [])]
            for connector_id in item.get(
                    "conflicting_connectors", []):
                connector = self._connector_by_id.get(connector_id)
                if not connector or len(crosswalk_line) < 2:
                    continue
                connector_line = [
                    tuple(point)
                    for point in connector["centerline_xy"]]
                corridor_entry_s = _earliest_corridor_distance(
                    connector_line, crosswalk_line,
                    max(0.1, float(item.get("width_m", 4.0)) / 2.0))
                hit = _polyline_intersection_distances(
                    connector_line, crosswalk_line)
                conflict_s = (
                    corridor_entry_s
                    if corridor_entry_s is not None
                    else hit[1] if hit
                    else _nearest_polyline_distance(
                        connector_line, crosswalk_line))
                self._crosswalk_conflicts_by_connector.setdefault(
                    connector_id, []).append({
                        "crosswalk_id": item["id"],
                        "connector_distance_s_m": conflict_s,
                    })

    @classmethod
    def load(cls, path: str) -> "LaneGeometryRuntime":
        import json
        with open(path, "r", encoding="utf-8") as fh:
            return cls(json.load(fh))

    @staticmethod
    def _validate_map_contract(data: dict) -> None:
        expected_schema = "vehiclearena-lane-level-v0.3"
        if data.get("schema") != expected_schema:
            raise ValueError(
                f"unsupported lane-level schema {data.get('schema')!r}; "
                f"expected {expected_schema!r}. "
                "Regenerate the companion map.")
        if data.get("lane_index_convention") != \
                "directional_rightmost_zero_left_increasing":
            raise ValueError(
                "lane-level map does not use the required lane convention "
                "(directional lane 0 is rightmost and indexes increase left); "
                "regenerate the companion map")
        required = {
            "projection", "nodes_xy", "intersections", "lanes",
            "connectors", "connector_conflicts", "stop_lines",
            "crosswalks", "pedestrian_approaches", "signal_plans", "quality",
        }
        missing = sorted(required - set(data))
        if missing:
            raise ValueError(
                f"lane-level map is missing required fields: {missing}")
        lane_ids = [item.get("id") for item in data["lanes"]]
        connector_ids = [item.get("id") for item in data["connectors"]]
        crosswalk_ids = [item.get("id") for item in data["crosswalks"]]
        approach_ids = [
            item.get("id")
            for item in data["pedestrian_approaches"]
        ]
        if len(lane_ids) != len(set(lane_ids)):
            raise ValueError("lane-level map contains duplicate lane IDs")
        if len(connector_ids) != len(set(connector_ids)):
            raise ValueError(
                "lane-level map contains duplicate connector IDs")
        if len(crosswalk_ids) != len(set(crosswalk_ids)):
            raise ValueError(
                "lane-level map contains duplicate crosswalk IDs")
        if len(approach_ids) != len(set(approach_ids)):
            raise ValueError(
                "lane-level map contains duplicate pedestrian approach IDs")
        if data.get("quality", {}).get("survey_grade") is not False:
            raise ValueError(
                "generated simulation maps must explicitly declare "
                "survey_grade=false")
        known_junctions = {
            str(lane[side])
            for lane in data["lanes"]
            for side in ("start_junction", "end_junction")
        }
        segment_junctions: Dict[str, set[str]] = {}
        for lane in data["lanes"]:
            segment_junctions.setdefault(
                str(lane["segment_id"]), set()).update({
                    str(lane["start_junction"]),
                    str(lane["end_junction"]),
                })
        for index, crosswalk in enumerate(data["crosswalks"]):
            missing_crosswalk = {
                "id", "node_id", "road_segment_id", "centerline_xy",
                "polygon_xy",
                "stripe_polygons_xy", "width_m",
            } - set(crosswalk)
            if missing_crosswalk:
                raise ValueError(
                    f"crosswalk[{index}] is missing required fields: "
                    f"{sorted(missing_crosswalk)}")
            segment_ids = crosswalk.get(
                "crossed_road_segment_ids",
                [crosswalk.get("road_segment_id")],
            )
            if (not isinstance(segment_ids, list) or not segment_ids
                    or any(not isinstance(value, str) or not value
                           for value in segment_ids)):
                raise ValueError(
                    f"crosswalk[{index}] has invalid crossed road segments")
            non_incident = [
                segment_id for segment_id in segment_ids
                if crosswalk["node_id"] not in segment_junctions.get(
                    segment_id, set())
            ]
            if non_incident:
                raise ValueError(
                    f"crosswalk[{index}] references non-incident road "
                    f"segments: {non_incident}")
        for index, approach in enumerate(
                data["pedestrian_approaches"]):
            missing_approach = {
                "id", "node_id", "centerline_xy", "width_m",
            } - set(approach)
            if missing_approach:
                raise ValueError(
                    f"pedestrian_approach[{index}] is missing required "
                    f"fields: {sorted(missing_approach)}")
            if approach["node_id"] not in known_junctions:
                raise ValueError(
                    f"pedestrian_approach[{index}] references unknown "
                    f"junction {approach['node_id']}")
            if len(approach["centerline_xy"]) < 2:
                raise ValueError(
                    f"pedestrian_approach[{index}] has no usable "
                    "centerline")
            if float(approach["width_m"]) <= 0:
                raise ValueError(
                    f"pedestrian_approach[{index}] has non-positive width")
            supported = approach.get("supports_crosswalk_id")
            if supported is not None and supported not in set(crosswalk_ids):
                raise ValueError(
                    f"pedestrian_approach[{index}] references unknown "
                    f"crosswalk {supported}")

    def connector_record(self, connector_id: str) -> Optional[dict]:
        """Return a connector record by its stable ID."""
        return self._connector_by_id.get(connector_id)

    def connector_length(self, connector_id: str) -> float:
        """Return physical connector path length in metres."""
        return self._connector_lengths.get(connector_id, 0.0)

    def lane_length(self, lane_id: str, fallback: float = 0.0) -> float:
        """Return authoritative metric length for a directed lane."""
        return self._lane_lengths.get(lane_id, fallback)

    def uturn_plan(self, current_lane_id: str) -> Optional[dict]:
        """Plan adjacent lane changes plus a same-road U-turn connector."""
        current = self._lane_by_id.get(current_lane_id)
        if not current:
            return None
        same_direction = self._lanes.get(
            (current["segment_id"], current["direction"]), [])
        candidates = []
        for source in same_direction:
            for connector in self._connectors_from.get(source["id"], []):
                target = self._lane_by_id.get(connector["to_lane"])
                if (
                    connector.get("turn") != "uturn"
                    or not target
                    or target["segment_id"] != current["segment_id"]
                    or target["direction"] == current["direction"]
                ):
                    continue
                candidates.append((
                    abs(source["index"] - current["index"]),
                    source, target, connector))
        if not candidates:
            return None
        _, source, target, connector = min(
            candidates,
            key=lambda item: (
                item[0], item[1]["index"], item[3]["id"]))

        by_index = {lane["index"]: lane for lane in same_direction}
        actions = []
        cursor = current
        while cursor["id"] != source["id"]:
            step = 1 if source["index"] > cursor["index"] else -1
            next_lane = by_index.get(cursor["index"] + step)
            if not next_lane:
                return None
            actions.append({
                "type": "lane_change",
                "from_lane_id": cursor["id"],
                "to_lane_id": next_lane["id"],
                "target_lane_index": next_lane["index"],
            })
            cursor = next_lane
        actions.append({
            "type": "connector",
            "from_lane_id": source["id"],
            "to_lane_id": target["id"],
            "connector_id": connector["id"],
            "turn": "uturn",
        })
        return {
            "actions": actions,
            "connector_id": connector["id"],
            "target_lane_id": target["id"],
        }

    def connector_conflict_points(
        self, connector_id: str,
    ) -> List[dict]:
        """Return path-relative conflict points for a connector."""
        return list(self._connector_conflict_points.get(
            connector_id, ()))

    def connector_crosswalk_points(
        self, connector_id: str,
    ) -> List[dict]:
        """Return path-relative crosswalk conflicts for a connector."""
        return list(self._crosswalk_conflicts_by_connector.get(
            connector_id, ()))

    def initial_vehicle_pose(self, vehicle: Any, road_network: Any):
        """Project an authored spawn state onto immutable map geometry.

        This helper is used only before SUMO materialises an actor. It does
        not advance time, velocity, lane changes or any other physical state.
        """
        if vehicle.active_connector_id:
            record = self._connector_by_id.get(vehicle.active_connector_id)
            if record:
                line = [tuple(point) for point in record["centerline_xy"]]
                x, y, heading = pose_at(
                    line, max(0.0, min(1.0, vehicle.edge_progress))
                    * _polyline_length(line))
                source = self._lane_by_id.get(record["from_lane"], {})
                return (
                    x, y, heading, source.get("z_level", 0),
                    f"connector::{record['id']}")
        if not vehicle.current_segment:
            point = self.nodes_xy.get(vehicle.current_node)
            return (*point, 0.0, 0, "") if point else None
        segment = road_network.get_segment(vehicle.current_segment)
        if not segment:
            return None
        direction = self._travel_direction(vehicle, segment)
        lanes = self._lanes.get((vehicle.current_segment, direction), [])
        if not lanes:
            return None
        def lane_for_index(index: int):
            return next(
                (item for item in lanes if item["index"] == index),
                lanes[min(max(0, index), len(lanes) - 1)])

        lane = lane_for_index(vehicle.current_lane)
        vehicle.current_lane_id = lane["id"]
        line = [tuple(point) for point in lane["centerline_xy"]]
        x, y, heading = pose_at(
            line, max(0.0, min(1.0, vehicle.edge_progress))
            * _polyline_length(line))
        return x, y, heading, lane.get("z_level", 0), lane["id"]

    def vehicle_pose(self, vehicle: Any):
        """Return the latest stored pose without deriving a trajectory."""
        if vehicle.is_crashed and getattr(vehicle, "crash_pose", None):
            return tuple(vehicle.crash_pose)
        path_id = (
            f"connector::{vehicle.active_connector_id}"
            if vehicle.active_connector_id else vehicle.current_lane_id
        )
        return (
            float(vehicle.pose_x_m), float(vehicle.pose_y_m),
            float(vehicle.yaw_rad), int(vehicle.z_level), path_id,
        )

    def plan_lane_route(
        self, start_node: str, destination_node: str,
        current_lane_id: str = "",
        blocked_segments: Optional[Sequence[str]] = None,
    ) -> Optional[dict]:
        """Plan end-to-end on the lane graph using connectors and lane changes."""
        start_junction = self._node_to_junction.get(
            start_node, start_node)
        destination_junction = self._node_to_junction.get(
            destination_node, destination_node)
        if current_lane_id:
            starts = [current_lane_id] if current_lane_id in self._lane_by_id else []
        else:
            starts = [
                lane["id"] for lane in self._lane_by_id.values()
                if (lane["start_node"] == start_node
                    or lane["start_junction"] == start_junction)]
        goals = {
            lane["id"] for lane in self._lane_by_id.values()
            if (lane["end_node"] == destination_node
                or lane["end_junction"] == destination_junction)}
        if not starts or not goals:
            return None
        blocked = set(blocked_segments or ())

        lane_change_weight = 22.0
        turn_weights = {
            "straight": 0.0, "right": 5.0, "left": 9.0, "uturn": 35.0}
        risk_weight = 0.35
        adjacency: Dict[str, List[Tuple[str, float, dict]]] = {}
        for source_id, connectors in self._connectors_from.items():
            if (source_id not in starts
                    and self._lane_by_id[source_id]["segment_id"] in blocked):
                continue
            for connector in connectors:
                # U-turns are explicit driver maneuvers, not ordinary
                # shortest-path edges. They are inserted only by uturn_plan.
                if connector.get("turn") == "uturn":
                    continue
                target_id = connector["to_lane"]
                if self._lane_by_id[target_id]["segment_id"] in blocked:
                    continue
                connector_length = _polyline_length([
                    tuple(point) for point in connector["centerline_xy"]])
                cost = (
                    connector_length + self._lane_lengths[target_id]
                    + turn_weights.get(connector["turn"], 20.0)
                    + risk_weight * self._conflict_degree.get(
                        connector["id"], 0))
                adjacency.setdefault(source_id, []).append((
                    target_id, cost, {
                        "type": "connector",
                        "from_lane_id": source_id,
                        "to_lane_id": target_id,
                        "connector_id": connector["id"],
                        "turn": connector["turn"],
                    }))
        for lanes in self._lanes.values():
            for first, second in zip(lanes, lanes[1:]):
                for source, target in ((first, second), (second, first)):
                    if target["segment_id"] in blocked:
                        continue
                    adjacency.setdefault(source["id"], []).append((
                        target["id"], lane_change_weight, {
                            "type": "lane_change",
                            "from_lane_id": source["id"],
                            "to_lane_id": target["id"],
                            "target_lane_index": target["index"],
                        }))

        distances = {}
        previous = {}
        queue = []
        for lane_id in starts:
            initial = self._lane_lengths[lane_id]
            if initial < distances.get(lane_id, float("inf")):
                distances[lane_id] = initial
                heapq.heappush(queue, (initial, lane_id))
        goal = None
        while queue:
            distance, lane_id = heapq.heappop(queue)
            if distance != distances.get(lane_id):
                continue
            if lane_id in goals:
                goal = lane_id
                break
            for target_id, edge_cost, action in adjacency.get(lane_id, []):
                candidate = distance + edge_cost
                if candidate >= distances.get(target_id, float("inf")):
                    continue
                distances[target_id] = candidate
                previous[target_id] = (lane_id, action)
                heapq.heappush(queue, (candidate, target_id))
        if goal is None:
            return None
        actions = []
        cursor = goal
        while cursor in previous:
            predecessor, action = previous[cursor]
            actions.append(action)
            cursor = predecessor
        actions.reverse()
        return {
            "start_lane_id": cursor,
            "goal_lane_id": goal,
            "actions": actions,
            "cost": round(distances[goal], 3),
        }

    def signal_groups_for_lane(self, lane_id: str) -> List[dict]:
        """Group only identical turn/phase schedules into a display head.

        Parallel connectors may share a head, but a through green must never
        stand in for a protected-left red. Unsignalized movements get no lamp.
        This is a read-only presentation query, not a maneuver selection.
        """
        groups = {}
        order = {"uturn": 0, "left": 1, "straight": 2, "right": 3}
        for connector in self._connectors_from.get(lane_id, []):
            cid = connector["id"]
            plan = self._signal_plan_by_connector.get(cid)
            if plan is None:
                continue
            turn = connector["turn"]
            signature = tuple(cid in phase["connector_ids"]
                              for phase in plan["phases"])
            key = (turn, plan["node_id"], signature)
            groups.setdefault(key, []).append(cid)
        return sorted([
            {"turn": key[0], "connector_id": min(ids),
             "connector_ids": sorted(ids)}
            for key, ids in groups.items()
        ], key=lambda group: (order.get(group["turn"], 4), group["connector_id"]))

    def signal_state(
        self, connector_id: str, time_s: float,
    ) -> Optional[LaneSignalState]:
        plan = self._signal_plan_by_connector.get(connector_id)
        if not plan:
            return None
        cycle = plan["cycle_s"]
        cycle_time = time_s % cycle
        cursor = 0.0
        current_phase = None
        phase_elapsed = 0.0
        for phase in plan["phases"]:
            duration = (
                phase["green_s"] + phase["yellow_s"]
                + phase["all_red_s"])
            if cycle_time < cursor + duration:
                current_phase = phase
                phase_elapsed = cycle_time - cursor
                break
            cursor += duration
        if current_phase is None:
            current_phase = plan["phases"][-1]
            phase_elapsed = 0.0
        allowed = connector_id in current_phase["connector_ids"]
        green_end = current_phase["green_s"]
        yellow_end = green_end + current_phase["yellow_s"]
        if allowed and phase_elapsed < green_end:
            signal = "green"
            remaining = green_end - phase_elapsed
        elif allowed and phase_elapsed < yellow_end:
            signal = "yellow"
            remaining = yellow_end - phase_elapsed
        else:
            signal = "red"
            # Find the start of this connector's next green phase.
            starts = []
            phase_cursor = 0.0
            for phase in plan["phases"]:
                if connector_id in phase["connector_ids"]:
                    delta = (phase_cursor - cycle_time) % cycle
                    starts.append(cycle if delta <= 1e-9 else delta)
                phase_cursor += (
                    phase["green_s"] + phase["yellow_s"]
                    + phase["all_red_s"])
            remaining = min(starts) if starts else 0.0
        ped_green = (
            current_phase.get("pedestrian_green", False)
            and phase_elapsed < current_phase["green_s"])
        return LaneSignalState(
            signal=signal,
            remaining_seconds=round(remaining, 3),
            phase_id=current_phase["id"],
            pedestrian_green=ped_green)

    def pedestrian_signal_state(
        self, node_id: str, time_s: float,
    ) -> Optional[LaneSignalState]:
        plan_node_id = self._node_to_junction.get(node_id, node_id)
        plan = self._signal_plans.get(plan_node_id)
        if not plan:
            return None
        pedestrian_phase = next(
            (phase for phase in plan["phases"]
             if phase.get("pedestrian_green")), None)
        if not pedestrian_phase:
            return None
        # Any connector uses the same cycle locator. Reproduce the phase
        # cursor here because pedestrian phases intentionally have no connector.
        cycle_time = time_s % plan["cycle_s"]
        cursor = 0.0
        for phase in plan["phases"]:
            duration = (
                phase["green_s"] + phase["yellow_s"]
                + phase["all_red_s"])
            if cycle_time < cursor + duration:
                elapsed = cycle_time - cursor
                green = (
                    phase.get("pedestrian_green", False)
                    and elapsed < phase["green_s"])
                return LaneSignalState(
                    "green" if green else "red",
                    round(
                        (phase["green_s"] - elapsed) if green
                        else duration - elapsed, 3),
                    phase["id"], green)
            cursor += duration
        return None

    def stop_progress(self, lane_id: str) -> Optional[float]:
        if lane_id not in self._stop_lane_ids:
            return None
        length = self._lane_lengths.get(lane_id, 0.0)
        return max(0.0, (length - 1.0) / length) if length > 0 else None

    def _travel_direction(self, vehicle: Any, segment: Any) -> str:
        lane = self._lane_by_id.get(vehicle.current_lane_id)
        if lane is not None:
            return str(lane["direction"])
        return (
            "forward" if segment.from_node == vehicle.current_node
            else "backward")

    def pedestrian_pose(self, pedestrian: Any):
        """Return the latest stored SUMO or authored spawn pose."""
        if pedestrian.is_crashed and pedestrian.crash_position_xy:
            return tuple(pedestrian.crash_position_xy)
        if getattr(pedestrian, "physical_pose_xy", None):
            return tuple(pedestrian.physical_pose_xy)
        pos = pedestrian.position
        node_id = pos.at_node or pos.crossing_to
        return self.nodes_xy.get(node_id)
