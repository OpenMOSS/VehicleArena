"""Smoke tests for the Guomao lane-level map and route metadata."""

from __future__ import annotations

import importlib.util
import json
import os
import sys


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(ROOT, relative_path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runtime = load_module("lane_level_runtime",
                      "simulation/lane_level_runtime.py")
MAP_PATH = os.path.join(
    ROOT, "simulation", "road_networks",
    "beijing_guomao_lane_level.json")


def _load_map_data():
    with open(MAP_PATH, encoding="utf-8") as handle:
        return json.load(handle)


def test_map():
    data = _load_map_data()
    assert data["lane_index_convention"] == \
        "directional_rightmost_zero_left_increasing"
    assert data["lanes"]
    assert data["connectors"]
    assert data["connector_conflicts"]
    assert data["stop_lines"]


def test_directional_lane_indexes_move_physically_left():
    data = _load_map_data()
    groups = {}
    for lane in data["lanes"]:
        groups.setdefault(
            (lane["segment_id"], lane["direction"]), []).append(lane)
    checked = 0
    for lanes in groups.values():
        ordered = sorted(lanes, key=lambda lane: lane["directional_index"])
        for right_lane, left_lane in zip(ordered, ordered[1:]):
            right = right_lane["centerline_xy"]
            left = left_lane["centerline_xy"]
            heading_end = next((
                point for point in right[1:]
                if ((point[0] - right[0][0]) ** 2
                    + (point[1] - right[0][1]) ** 2) >= 0.01
            ), None)
            if heading_end is None:
                continue
            heading = (
                heading_end[0] - right[0][0],
                heading_end[1] - right[0][1],
            )
            displacement = (
                left[0][0] - right[0][0],
                left[0][1] - right[0][1],
            )
            cross = (heading[0] * displacement[1]
                     - heading[1] * displacement[0])
            assert cross > 1e-6, (right_lane["id"], left_lane["id"])
            checked += 1
    assert checked > 0


def main():
    data = _load_map_data()
    test_map()
    print(json.dumps({
        "map": "beijing_guomao_lane_level",
        "lane_count": len(data["lanes"]),
        "connector_count": len(data["connectors"]),
        "connector_conflict_count": len(data["connector_conflicts"]),
        "status": "PASS",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
