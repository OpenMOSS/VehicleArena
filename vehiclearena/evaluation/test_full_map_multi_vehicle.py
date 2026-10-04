"""Full-map SUMO-native background traffic integration test."""

from __future__ import annotations

import json
import math
import os
from collections import Counter, defaultdict
from itertools import islice, permutations

from simulation.road_networks import load_road_network
from simulation.sumo_traffic_manager import SumoTrafficManager


VEHICLE_COUNT = 40
DURATION_S = 90.0
DT_S = 0.1
OUTPUT = os.path.join(
    os.path.dirname(__file__), "outputs", "full_map_multi_vehicle_test.json")


def main():
    network = load_road_network("beijing_guomao")
    manager = SumoTrafficManager(network)
    runtime = manager._lane_geometry
    nodes = sorted({
        lane[key] for lane in runtime._lane_by_id.values()
        for key in ("start_node", "end_node")
    })
    candidates = []
    for start, destination in islice(permutations(nodes, 2), 5000):
        a, b = runtime.nodes_xy.get(start), runtime.nodes_xy.get(destination)
        if a and b and math.dist(a, b) >= 350.0:
            candidates.append((math.dist(a, b), start, destination))
    candidates.sort(reverse=True)

    used_lanes = set()
    for _, start, destination in candidates:
        if len(manager.vehicles) >= VEHICLE_COUNT:
            break
        plan = runtime.plan_lane_route(start, destination)
        if not plan or len(plan["actions"]) < 2:
            continue
        if plan["start_lane_id"] in used_lanes:
            continue
        vehicle_id = f"fullmap-{len(manager.vehicles) + 1:03d}"
        vehicle = manager.register_vehicle(
            vehicle_id, start, destination)
        used_lanes.add(vehicle.current_lane_id)
        vehicle.edge_progress = 0.05
    assert len(manager.vehicles) == VEHICLE_COUNT

    visited_lanes = defaultdict(set)
    connector_entries = Counter()
    try:
        for step in range(round(DURATION_S / DT_S)):
            now = step * DT_S
            manager.recalculate_speeds(now)
            for event in manager.advance_world_to(now + DT_S):
                if event.type == "connector_entered":
                    connector_entries[event.vehicle_id] += 1
            for vehicle_id, vehicle in manager.vehicles.items():
                assert vehicle.control_authority == "sumo"
                if vehicle.current_lane_id:
                    visited_lanes[vehicle_id].add(vehicle.current_lane_id)

        result = {
            "status": "PASS" if not manager.collision_log else "SAFETY_FAIL",
            "duration_s": DURATION_S,
            "vehicles": len(manager.vehicles),
            "arrived": sum(v.arrived for v in manager.vehicles.values()),
            "crashed": sum(v.is_crashed for v in manager.vehicles.values()),
            "collision_events": len(manager.collision_log),
            "connector_entries": sum(connector_entries.values()),
            "visited_lane_count": len(set().union(*visited_lanes.values())),
            "control_authority": "sumo",
        }
        os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
        with open(OUTPUT, "w", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        manager.close()


if __name__ == "__main__":
    main()
