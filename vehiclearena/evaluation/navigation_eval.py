"""Navigation constraint-based evaluation.

Instead of exact GT matching, navigation is evaluated via constraint checks:
- Safety: obey traffic lights, yield to pedestrians
- Efficiency: don't idle, make progress, reach destination
- Compliance: don't exceed speed limits

Usage:
    from evaluation.navigation_eval import evaluate_navigation

    nav_metrics = evaluate_navigation(
        vehicle_id, vehicle_state, traffic_mgr, road_network,
        tick_log, total_time_s,
    )
    # nav_metrics is a NavigationMetrics dataclass
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class NavigationMetrics:
    """Constraint-based navigation evaluation metrics."""
    vehicle_id: str

    # ── Progress ──
    arrived: bool = False
    distance_traveled_m: float = 0.0
    route_completion: float = 0.0       # fraction of route nodes visited (0-1)
    destination_distance_m: float = 0.0  # remaining distance to destination

    # ── Efficiency ──
    total_time_s: float = 0.0
    moving_time_s: float = 0.0          # time with speed > 0
    idle_time_s: float = 0.0            # time stopped without safety reason
    avg_speed_kmh: float = 0.0
    efficiency_ratio: float = 0.0       # moving_time / total_time

    # ── Compliance ──
    overspeed_count: int = 0            # times speed > segment limit
    overspeed_total_s: float = 0.0      # total seconds over speed limit
    red_light_violations: int = 0       # passed through red light

    # ── Safety ──
    collisions: int = 0
    near_misses: int = 0                # came within 5m of another entity at speed

    # ── Scores (0-1, higher is better) ──
    @property
    def progress_score(self) -> float:
        """1.0 if arrived, else route_completion."""
        if self.arrived:
            return 1.0
        return self.route_completion

    @property
    def efficiency_score(self) -> float:
        """Ratio of moving time to total time. 1.0 = never idle."""
        if self.total_time_s <= 0:
            return 1.0
        return self.efficiency_ratio

    @property
    def compliance_score(self) -> float:
        """1.0 if no violations, decreases with violations."""
        violations = self.overspeed_count + self.red_light_violations
        if violations == 0:
            return 1.0
        return max(0.0, 1.0 - violations * 0.1)

    @property
    def safety_score(self) -> float:
        """1.0 if no collisions/near-misses."""
        if self.collisions > 0:
            return 0.0
        if self.near_misses > 0:
            return max(0.0, 1.0 - self.near_misses * 0.2)
        return 1.0

    @property
    def overall_nav_score(self) -> float:
        """Weighted average of all dimension scores."""
        return (
            0.3 * self.progress_score +
            0.2 * self.efficiency_score +
            0.2 * self.compliance_score +
            0.3 * self.safety_score
        )

    def summary_dict(self) -> dict:
        return {
            "arrived": self.arrived,
            "distance_m": round(self.distance_traveled_m, 1),
            "route_completion": round(self.route_completion, 3),
            "avg_speed_kmh": round(self.avg_speed_kmh, 1),
            "efficiency": round(self.efficiency_ratio, 3),
            "idle_time_s": round(self.idle_time_s, 1),
            "overspeed_count": self.overspeed_count,
            "overspeed_s": round(self.overspeed_total_s, 1),
            "red_light_violations": self.red_light_violations,
            "collisions": self.collisions,
            "scores": {
                "progress": round(self.progress_score, 3),
                "efficiency": round(self.efficiency_score, 3),
                "compliance": round(self.compliance_score, 3),
                "safety": round(self.safety_score, 3),
                "overall": round(self.overall_nav_score, 3),
            },
        }


def evaluate_navigation(
    vehicle_id: str,
    vehicle_state,
    traffic_mgr,
    road_network,
    tick_log: List[dict],
    total_time_s: float,
) -> NavigationMetrics:
    """Compute navigation metrics from tick-by-tick driving log.

    Args:
        vehicle_id: Vehicle identifier.
        vehicle_state: Final VehicleState after simulation.
        traffic_mgr: TrafficCoordinator instance (for collision log).
        road_network: RoadNetwork instance (for speed limits, distances).
        tick_log: List of per-tick snapshots, each with keys:
            time_s, speed_kmh, segment_id, node, is_stopped,
            at_red_light, pedestrian_ahead
        total_time_s: Total simulation duration in seconds.

    Returns:
        NavigationMetrics with all dimensions computed.
    """
    metrics = NavigationMetrics(vehicle_id=vehicle_id)
    metrics.arrived = vehicle_state.arrived
    metrics.distance_traveled_m = vehicle_state.distance_traveled_m
    metrics.total_time_s = total_time_s

    # ── Route completion ──
    if vehicle_state.route:
        route = vehicle_state.route
        # Count how many route nodes were visited
        visited = set()
        for entry in tick_log:
            node = entry.get("node", "")
            if node and node in route:
                visited.add(node)
        metrics.route_completion = len(visited) / len(route) if route else 0.0

        # Remaining distance to destination
        if not vehicle_state.arrived and len(route) > 1:
            dest = route[-1]
            remaining = road_network.plan_route(
                vehicle_state.current_node, dest, 0)
            if remaining:
                dist = 0.0
                for i in range(len(remaining) - 1):
                    seg_id = road_network.make_edge_id(remaining[i], remaining[i + 1])
                    seg = road_network.get_segment(seg_id)
                    if seg:
                        dist += seg.distance_meters
                metrics.destination_distance_m = dist

    # ── Efficiency: moving vs idle time ──
    moving_time = 0.0
    idle_time = 0.0
    prev_time = 0.0
    speed_sum = 0.0
    speed_count = 0

    for entry in tick_log:
        t = entry.get("time_s", 0.0)
        dt = t - prev_time if prev_time > 0 else 0.0
        prev_time = t

        speed = entry.get("speed_kmh", 0.0)
        speed_sum += speed
        speed_count += 1

        if speed > 0.5:
            moving_time += dt
        else:
            # Is there a safety reason to be stopped?
            at_red = entry.get("at_red_light", False)
            ped_ahead = entry.get("pedestrian_ahead", False)
            if not at_red and not ped_ahead:
                idle_time += dt

    metrics.moving_time_s = moving_time
    metrics.idle_time_s = idle_time
    metrics.avg_speed_kmh = speed_sum / speed_count if speed_count > 0 else 0.0
    metrics.efficiency_ratio = (
        moving_time / metrics.total_time_s if metrics.total_time_s > 0 else 1.0
    )

    # ── Compliance: speed limit violations ──
    for entry in tick_log:
        seg_id = entry.get("segment_id", "")
        speed = entry.get("speed_kmh", 0.0)
        if seg_id and speed > 0:
            seg = road_network.get_segment(seg_id)
            if seg and speed > seg.speed_limit + 5:  # 5km/h tolerance
                metrics.overspeed_count += 1
                dt = entry.get("dt_s", 1.0)
                metrics.overspeed_total_s += dt

    # ── Safety: collisions ──
    for collision in traffic_mgr._collision_log:
        if collision.entity_a == vehicle_id or collision.entity_b == vehicle_id:
            metrics.collisions += 1

    return metrics
