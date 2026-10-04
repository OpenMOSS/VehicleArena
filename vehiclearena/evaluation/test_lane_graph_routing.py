"""End-to-end lane graph planning and execution test."""

from __future__ import annotations

import json
import os
import sys
import types

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)
simulation_package = types.ModuleType("simulation")
simulation_package.__path__ = [os.path.join(ROOT, "simulation")]
sys.modules["simulation"] = simulation_package

from simulation.road_networks import get_key_node, load_road_network
from simulation.sumo_traffic_manager import SumoTrafficManager


def main():
    network = load_road_network("beijing_guomao")
    manager = SumoTrafficManager(network)
    start = get_key_node("beijing_guomao", "guanghua_road")
    destination = get_key_node("beijing_guomao", "chaoyang_road")

    plan = manager._lane_geometry.plan_lane_route(start, destination)
    assert plan and plan["actions"]
    assert any(action["type"] == "connector" for action in plan["actions"])

    blocked_segment = next(
        manager._lane_geometry._lane_by_id[
            action["to_lane_id"]]["segment_id"]
        for action in plan["actions"]
        if action["type"] == "connector")
    alternate = manager._lane_geometry.plan_lane_route(
        start, destination, blocked_segments=[blocked_segment])
    assert alternate
    assert all(
        manager._lane_geometry._lane_by_id[
            action["to_lane_id"]]["segment_id"] != blocked_segment
        for action in alternate["actions"]
        if action["type"] == "connector")

    vehicle = manager.register_vehicle(
        "route_vehicle", start, destination)
    assert vehicle.current_lane_id
    assert vehicle.lane_route_actions
    for step in range(1800):
        now = step * 0.1
        manager.recalculate_speeds(now)
        manager.advance_world_to(now + 0.1)
        if vehicle.arrived or vehicle.is_crashed:
            break
    assert vehicle.arrived and not vehicle.is_crashed
    assert vehicle.lane_route_action_index == len(
        vehicle.lane_route_actions)

    report = {
        "status": "PASS",
        "start": start,
        "destination": destination,
        "route_cost": plan["cost"],
        "arrival_time_s": round(manager._physics_time, 1),
        "distance_m": round(vehicle.distance_traveled_m, 2),
        "actions_executed": vehicle.lane_route_action_index,
        "blocked_segment_avoided": blocked_segment,
        "alternate_actions": len(alternate["actions"]),
    }
    manager.close()
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
