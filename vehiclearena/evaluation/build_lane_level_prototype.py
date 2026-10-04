"""Build and validate the Beijing Guomao lane-level map prototype."""

from __future__ import annotations

import json
import importlib.util
import math
import os
import sys

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
MODULE_PATH = os.path.join(ROOT, "simulation", "lane_level_map.py")
SPEC = importlib.util.spec_from_file_location("lane_level_map", MODULE_PATH)
LANE_LEVEL_MAP = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = LANE_LEVEL_MAP
SPEC.loader.exec_module(LANE_LEVEL_MAP)
build_file = LANE_LEVEL_MAP.build_file

SOURCE = os.path.join(ROOT, "simulation", "road_networks",
                      "beijing_guomao.json")
OUTPUT = os.path.join(ROOT, "simulation", "road_networks",
                      "beijing_guomao_lane_level.json")


def polyline_length(points):
    return sum(math.hypot(b[0] - a[0], b[1] - a[1])
               for a, b in zip(points, points[1:]))


def validate(data):
    errors = []
    if data.get("schema") != "vehiclearena-lane-level-v0.3":
        errors.append(f"bad schema: {data.get('schema')!r}")
    if data.get("lane_index_convention") != \
            "directional_rightmost_zero_left_increasing":
        errors.append("bad or missing lane_index_convention")
    quality = data.get("quality", {})
    if quality.get("survey_grade") is not False:
        errors.append("quality must explicitly declare survey_grade=false")
    if quality.get("suitable_for") != "lane_level_simulation":
        errors.append("quality.suitable_for must be lane_level_simulation")

    raw_lane_ids = [lane["id"] for lane in data["lanes"]]
    raw_connector_ids = [item["id"] for item in data["connectors"]]
    raw_intersection_ids = [
        item["id"] for item in data.get("intersections", [])]
    if len(raw_lane_ids) != len(set(raw_lane_ids)):
        errors.append("duplicate lane IDs")
    if len(raw_connector_ids) != len(set(raw_connector_ids)):
        errors.append("duplicate connector IDs")
    if len(raw_intersection_ids) != len(set(raw_intersection_ids)):
        errors.append("duplicate intersection IDs")

    lane_ids = set(raw_lane_ids)
    lanes_by_id = {lane["id"]: lane for lane in data["lanes"]}
    connector_ids = set(raw_connector_ids)
    intersections = {
        item["id"]: item for item in data.get("intersections", [])}
    outgoing_by_junction = {}
    incoming_by_junction = {}
    lane_slots = set()
    shared_by_segment = {}
    directional_groups = {}
    for lane in data["lanes"]:
        outgoing_by_junction.setdefault(
            lane["start_junction"], []).append(lane)
        incoming_by_junction.setdefault(
            lane["end_junction"], []).append(lane)
        slot = (lane["segment_id"], lane["direction"], lane["index"])
        if slot in lane_slots:
            errors.append(f"duplicate directed lane slot: {slot}")
        lane_slots.add(slot)
        if lane["start_junction"] not in intersections:
            errors.append(
                f"unknown lane start junction: {lane['id']}")
        if lane["end_junction"] not in intersections:
            errors.append(
                f"unknown lane end junction: {lane['id']}")
        if lane.get("shared_bidirectional"):
            shared_by_segment.setdefault(
                lane["segment_id"], []).append(lane)
        directional_groups.setdefault(
            (lane["segment_id"], lane["direction"]), []).append(lane)

    # Geometry must implement the public control convention: rank 0 is the
    # rightmost lane and every larger directional rank lies to its left in
    # the direction of travel.
    for group_key, group in directional_groups.items():
        ordered = sorted(group, key=lambda item: item["directional_index"])
        ranks = [item["directional_index"] for item in ordered]
        if ranks != list(range(len(ordered))):
            errors.append(f"non-contiguous directional ranks: {group_key}")
            continue
        for right_lane, left_lane in zip(ordered, ordered[1:]):
            right_line = right_lane.get("centerline_xy", [])
            left_line = left_lane.get("centerline_xy", [])
            if len(right_line) < 2 or not left_line:
                continue
            heading_end = next((
                point for point in right_line[1:]
                if math.dist(right_line[0], point) >= 0.1
            ), None)
            if heading_end is None:
                continue
            heading = (
                heading_end[0] - right_line[0][0],
                heading_end[1] - right_line[0][1],
            )
            displacement = (
                left_line[0][0] - right_line[0][0],
                left_line[0][1] - right_line[0][1],
            )
            cross = (heading[0] * displacement[1]
                     - heading[1] * displacement[0])
            if cross <= 1e-6:
                errors.append(
                    "directional rank does not move physically left: "
                    f"{right_lane['id']} -> {left_lane['id']}")
    connected_from = {}
    connector_turns = set()
    for lane in data["lanes"]:
        centerline = lane["centerline_xy"]
        if len(centerline) < 2:
            errors.append(f"lane without geometry: {lane['id']}")
            continue
        length = polyline_length(centerline)
        tolerance = max(0.05, length * 0.002)
        if length < 2.95:
            errors.append(f"short physical lane {length:.3f}: {lane['id']}")
        if abs(length - lane.get("length_m", 0.0)) > tolerance:
            errors.append(f"bad lane length {length:.3f}: {lane['id']}")
        source_length = lane.get("source_length_m", 0.0)
        ratio = length / source_length if source_length > 0 else 1.0
        if ratio < 0.395:
            errors.append(
                f"over-trimmed lane ratio={ratio:.3f}: {lane['id']}")
        if abs(ratio - lane.get("drivable_length_ratio", 0.0)) > 0.005:
            errors.append(f"bad lane length ratio: {lane['id']}")
        recovered_source = (
            length + lane.get("junction_trim_start_m", 0.0)
            + lane.get("junction_trim_end_m", 0.0))
        if abs(recovered_source - source_length) > max(
                0.1, source_length * 0.005):
            errors.append(f"inconsistent lane trims: {lane['id']}")
        left = lane.get("left_boundary_xy", [])
        right = lane.get("right_boundary_xy", [])
        if len(left) != len(centerline) or len(right) != len(centerline):
            errors.append(f"bad lane boundary cardinality: {lane['id']}")
        elif left and right:
            for point_index in (0, -1):
                width = math.dist(left[point_index], right[point_index])
                if abs(width - lane["width_m"]) > 0.15:
                    errors.append(
                        f"bad lane boundary width {width:.3f}: "
                        f"{lane['id']}")
                    break

    for segment_id, shared in shared_by_segment.items():
        directions = {lane["direction"] for lane in shared}
        if len(shared) != 2 or directions != {"forward", "backward"}:
            errors.append(
                f"bad shared bidirectional lane pair: {segment_id}")
            continue
        first, second = shared
        first_line = first["centerline_xy"]
        second_line = list(reversed(second["centerline_xy"]))
        if len(first_line) != len(second_line) or any(
                math.dist(a, b) > 0.02
                for a, b in zip(first_line, second_line)):
            errors.append(
                f"shared lane geometries do not coincide: {segment_id}")

    for connector in data["connectors"]:
        connected_from.setdefault(
            (connector["node_id"], connector["from_lane"]), 0)
        connected_from[
            (connector["node_id"], connector["from_lane"])] += 1
        turn_key = (connector["node_id"], connector["from_lane"],
                    connector["turn"])
        if turn_key in connector_turns:
            errors.append(f"duplicate lane turn: {turn_key}")
        connector_turns.add(turn_key)
        if connector["from_lane"] not in lane_ids:
            errors.append(f"unknown from_lane: {connector['id']}")
        if connector["to_lane"] not in lane_ids:
            errors.append(f"unknown to_lane: {connector['id']}")
        connector_length = polyline_length(connector["centerline_xy"])
        if connector_length < 0.1:
            errors.append(f"degenerate connector: {connector['id']}")
        if abs(connector_length - connector.get(
                "length_m", 0.0)) > max(0.05, connector_length * 0.002):
            errors.append(f"bad connector length: {connector['id']}")
        left = connector.get("left_boundary_xy", [])
        right = connector.get("right_boundary_xy", [])
        centerline = connector.get("centerline_xy", [])
        if len(left) != len(centerline) or len(right) != len(centerline):
            errors.append(
                f"bad connector boundary cardinality: {connector['id']}")
        elif left and right:
            for point_index in (0, -1):
                width = math.dist(left[point_index], right[point_index])
                if abs(width - connector["width_m"]) > 0.2:
                    errors.append(
                        f"bad connector boundary width {width:.3f}: "
                        f"{connector['id']}")
                    break
        if connector["from_lane"] in lanes_by_id:
            gap = math.dist(
                lanes_by_id[connector["from_lane"]]["centerline_xy"][-1],
                connector["centerline_xy"][0])
            if gap > 0.02:
                errors.append(f"connector start gap {gap:.3f}: "
                              f"{connector['id']}")
        if connector["to_lane"] in lanes_by_id:
            gap = math.dist(
                lanes_by_id[connector["to_lane"]]["centerline_xy"][0],
                connector["centerline_xy"][-1])
            if gap > 0.02:
                errors.append(f"connector end gap {gap:.3f}: "
                              f"{connector['id']}")
        if connector["from_lane"] in lanes_by_id \
                and connector["to_lane"] in lanes_by_id:
            from_level = lanes_by_id[connector["from_lane"]].get("z_level", 0)
            to_level = lanes_by_id[connector["to_lane"]].get("z_level", 0)
            if from_level != to_level or connector.get(
                    "z_level", 0) != from_level:
                errors.append(f"cross-level connector: {connector['id']}")
        intersection = intersections.get(connector["node_id"])
        if not intersection:
            errors.append(
                f"connector has unknown intersection: {connector['id']}")
        else:
            farthest = max(
                math.dist(intersection["center_xy"], point)
                for point in connector["centerline_xy"])
            if farthest > intersection["extent_m"] + 25.0:
                errors.append(f"connector outside intersection: "
                              f"{connector['id']}")
        if (connector.get("turn") == "uturn"
                and connector["from_lane"] in lanes_by_id
                and connector["to_lane"] in lanes_by_id):
            source = lanes_by_id[connector["from_lane"]]
            target = lanes_by_id[connector["to_lane"]]
            if (source["segment_id"] != target["segment_id"]
                    or source["direction"] == target["direction"]):
                errors.append(
                    f"invalid U-turn connector: {connector['id']}")
    for junction_id, incoming in incoming_by_junction.items():
        outgoing = outgoing_by_junction.get(junction_id, [])
        if not outgoing:
            continue
        for lane in incoming:
            compatible = [
                candidate for candidate in outgoing
                if candidate.get("z_level", 0) == lane.get("z_level", 0)
                and candidate["segment_id"] != lane["segment_id"]]
            if compatible and not connected_from.get(
                    (junction_id, lane["id"])):
                errors.append(
                    f"unconnected incoming lane: {junction_id} {lane['id']}")
    conflict_keys = set()
    connector_by_id = {
        item["id"]: item for item in data["connectors"]}
    for conflict in data["connector_conflicts"]:
        if conflict["connector_a"] not in connector_ids \
                or conflict["connector_b"] not in connector_ids:
            errors.append(f"unknown connector conflict: {conflict}")
            continue
        key = frozenset((
            conflict["connector_a"], conflict["connector_b"]))
        if len(key) != 2:
            errors.append(f"self connector conflict: {conflict}")
        if key in conflict_keys:
            errors.append(f"duplicate connector conflict: {conflict}")
        conflict_keys.add(key)
        first = connector_by_id[conflict["connector_a"]]
        second = connector_by_id[conflict["connector_b"]]
        if first["node_id"] != second["node_id"]:
            errors.append(f"cross-junction connector conflict: {conflict}")
        if first.get("z_level", 0) != second.get("z_level", 0):
            errors.append(f"cross-level connector conflict: {conflict}")
        swept_conflict = LANE_LEVEL_MAP._polyline_hits_corridor(
            first["centerline_xy"], second["centerline_xy"],
            max(1.0, min(
                first.get("width_m", data["defaults"]["lane_width_m"]),
                second.get("width_m", data["defaults"]["lane_width_m"]),
            ) / 2.0),
        )
        if (first["to_lane"] != second["to_lane"]
                and not swept_conflict):
            errors.append(
                f"spurious connector conflict: {conflict}")
    for stop in data["stop_lines"]:
        width = math.dist(*stop["line_xy"])
        lane = lanes_by_id.get(stop["lane_id"])
        if not lane:
            errors.append(f"stop line has unknown lane: {stop['id']}")
            continue
        if stop["node_id"] not in intersections:
            errors.append(f"stop line has unknown junction: {stop['id']}")
        if abs(width - lane["width_m"]) > 0.05:
            errors.append(f"bad stop-line width {width:.3f}: {stop['id']}")
        if not connected_from.get((stop["node_id"], stop["lane_id"])):
            errors.append(f"stopped lane without connector: {stop['id']}")
    crosswalk_ids = set()
    segment_junctions = {}
    for lane in data["lanes"]:
        segment_junctions.setdefault(lane["segment_id"], set()).update({
            lane["start_junction"], lane["end_junction"],
        })
    for crosswalk in data.get("crosswalks", []):
        if crosswalk["id"] in crosswalk_ids:
            errors.append(f"duplicate crosswalk: {crosswalk['id']}")
        crosswalk_ids.add(crosswalk["id"])
        line = crosswalk.get("centerline_xy", [])
        polygon = crosswalk.get("polygon_xy", [])
        if len(line) < 2 or polyline_length(line) < 3.0:
            errors.append(f"degenerate crosswalk line: {crosswalk['id']}")
        if len(polygon) != 4:
            errors.append(f"bad crosswalk polygon: {crosswalk['id']}")
        if abs(polyline_length(line) - crosswalk.get(
                "length_m", 0.0)) > 0.05:
            errors.append(f"bad crosswalk length: {crosswalk['id']}")
        unknown = set(crosswalk.get(
            "conflicting_connectors", [])) - connector_ids
        if unknown:
            errors.append(
                f"unknown crosswalk conflicts: {crosswalk['id']}")
        if not crosswalk.get("conflicting_connectors"):
            errors.append(
                f"crosswalk without vehicle conflict: {crosswalk['id']}")
        if crosswalk["node_id"] not in intersections:
            errors.append(
                f"crosswalk has unknown junction: {crosswalk['id']}")
        crossed_segments = crosswalk.get(
            "crossed_road_segment_ids", [crosswalk["road_segment_id"]])
        if not crossed_segments or any(
                crosswalk["node_id"] not in segment_junctions.get(
                    segment_id, set())
                for segment_id in crossed_segments):
            errors.append(
                f"crosswalk has non-incident road segment: "
                f"{crosswalk['id']}")
        if line:
            wait_a = crosswalk.get("from_wait_area", {}).get("xy")
            wait_b = crosswalk.get("to_wait_area", {}).get("xy")
            # Waiting domains belong on the sidewalk beyond the painted
            # crossing.  They must not coincide with a lane/connector mouth.
            wait_a_clearance = (
                math.dist(wait_a, line[0]) if wait_a else 0.0)
            wait_b_clearance = (
                math.dist(wait_b, line[-1]) if wait_b else 0.0)
            if (not wait_a
                    or not 2.0 <= wait_a_clearance <= 4.0):
                errors.append(
                    f"bad crosswalk entry wait area: {crosswalk['id']}")
            if (not wait_b
                    or not 2.0 <= wait_b_clearance <= 4.0):
                errors.append(
                    f"bad crosswalk exit wait area: {crosswalk['id']}")
        if not crosswalk.get("stripe_polygons_xy"):
            errors.append(f"crosswalk without stripes: {crosswalk['id']}")
    approach_ids = set()
    for approach in data.get("pedestrian_approaches", []):
        approach_id = approach.get("id")
        if not approach_id or approach_id in approach_ids:
            errors.append(
                f"duplicate or missing pedestrian approach: {approach_id}")
        approach_ids.add(approach_id)
        if approach.get("node_id") not in intersections:
            errors.append(
                f"pedestrian approach has unknown junction: {approach_id}")
        if (len(approach.get("centerline_xy", [])) < 2
                or polyline_length(approach["centerline_xy"]) < 1.0):
            errors.append(
                f"degenerate pedestrian approach: {approach_id}")
        if float(approach.get("width_m", 0.0)) <= 0:
            errors.append(
                f"bad pedestrian approach width: {approach_id}")
        supported = approach.get("supports_crosswalk_id")
        if supported is not None and supported not in crosswalk_ids:
            errors.append(
                f"pedestrian approach has unknown crosswalk: {approach_id}")
    conflict_pairs = {
        frozenset((item["connector_a"], item["connector_b"]))
        for item in data["connector_conflicts"]}
    controlled_seen = set()
    for plan in data.get("signal_plans", []):
        if plan["node_id"] not in intersections:
            errors.append(f"signal plan has unknown junction: {plan['id']}")
        computed_cycle = 0.0
        for phase in plan["phases"]:
            computed_cycle += (
                phase["green_s"] + phase["yellow_s"]
                + phase["all_red_s"])
            phase_connectors = phase["connector_ids"]
            controlled_seen.update(phase_connectors)
            for index, first_id in enumerate(phase_connectors):
                if first_id not in connector_by_id:
                    errors.append(f"unknown signal connector: {first_id}")
                    continue
                for second_id in phase_connectors[index + 1:]:
                    first = connector_by_id[first_id]
                    second = connector_by_id.get(second_id)
                    if not second:
                        continue
                    if (frozenset((first_id, second_id)) in conflict_pairs
                            or (first["to_lane"] == second["to_lane"]
                                and first["from_lane"]
                                != second["from_lane"])):
                        errors.append(
                            f"conflicting green connectors: "
                            f"{first_id} {second_id}")
        if abs(computed_cycle - plan["cycle_s"]) > 0.01:
            errors.append(f"bad signal cycle: {plan['id']}")
    expected_controlled = {
        item["id"] for item in data["connectors"]
        if item.get("signal_controlled")}
    if controlled_seen != expected_controlled:
        errors.append(
            f"signal connector coverage mismatch: "
            f"missing={len(expected_controlled - controlled_seen)} "
            f"extra={len(controlled_seen - expected_controlled)}")
    expected_counts = {
        "lanes": len(data["lanes"]),
        "connectors": len(data["connectors"]),
        "connector_conflicts": len(data["connector_conflicts"]),
        "stop_lines": len(data["stop_lines"]),
        "crosswalks": len(data.get("crosswalks", [])),
        "pedestrian_approaches": len(data.get(
            "pedestrian_approaches", [])),
        "signal_plans": len(data.get("signal_plans", [])),
    }
    if quality.get("counts") != expected_counts:
        errors.append(
            f"quality count mismatch: expected={expected_counts} "
            f"actual={quality.get('counts')}")
    return errors


def main():
    data = build_file(SOURCE, OUTPUT)
    errors = validate(data)
    turns = {}
    for connector in data["connectors"]:
        turns[connector["turn"]] = turns.get(connector["turn"], 0) + 1
    print(json.dumps({
        "output": OUTPUT,
        "lanes": len(data["lanes"]),
        "connectors": len(data["connectors"]),
        "turns": turns,
        "conflicts": len(data["connector_conflicts"]),
        "stop_lines": len(data["stop_lines"]),
        "crosswalks": len(data["crosswalks"]),
        "signal_plans": len(data.get("signal_plans", [])),
        "validation_errors": errors[:20],
        "validation_error_count": len(errors),
    }, ensure_ascii=False, indent=2))
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
