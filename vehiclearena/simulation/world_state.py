"""
WorldState — Queryable world state interface for active perception.

This module provides a unified query API that serves as the backend for
perception tools. Agents (vehicles, pedestrians, etc.) query the world
through this interface rather than receiving pre-built observations.

The WorldState wraps TrafficCoordinator and RoadNetwork internals into a
clean, entity-type-agnostic query layer.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from simulation.traffic_manager import TrafficCoordinator
    from simulation.road_network import RoadNetwork


@dataclass
class EntityInfo:
    """Generic info about an entity visible in the world."""
    entity_id: str
    entity_type: str            # "vehicle" | "pedestrian"
    distance_m: float
    speed_kmh: float = 0.0
    is_stopped: bool = False
    is_crossing: bool = False   # pedestrian on crosswalk
    heading: float = 0.0
    bearing_deg: float = 0.0
    confidence: float = 1.0
    effective_range_m: float = 0.0
    modalities: List[str] = field(default_factory=lambda: ["visual"])
    observed_signals: Dict[str, bool] = field(default_factory=dict)


@dataclass
class SignalInfo:
    """Traffic signal state at a node."""
    has_signal: bool = False
    signal: str = ""            # "red" | "yellow" | "green" | "ped_green"
    remaining_seconds: int = 0
    is_ped_green: bool = False
    direction_info: str = ""    # which direction this signal applies to


@dataclass
class RoadInfo:
    """Road segment status."""
    segment_id: str = ""
    speed_limit_kmh: float = 60.0
    is_blocked: bool = False
    observed_flow_speed_kmh: float = 60.0
    lanes: int = 2
    distance_m: float = 0.0


@dataclass
class CrosswalkInfo:
    """Crosswalk status at a node."""
    has_crosswalk: bool = False
    pedestrians_crossing: int = 0
    pedestrian_details: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class WeatherInfo:
    """Current weather status."""
    condition: str = "sunny"


class WorldState:
    """Queryable world state — backend for all perception tools.

    Design principles:
    - Entity-type agnostic: does not know/care who is querying
    - Read-only: queries never mutate state
    - Delegates to TrafficCoordinator/RoadNetwork for actual computation
    """

    def __init__(
        self,
        traffic_mgr: 'TrafficCoordinator',
        road_network: 'RoadNetwork',
        weather: Any = None,
        daynight: Any = None,
        tick: int = 0,
        time_s: float = 0.0,
    ):
        self._tm = traffic_mgr
        self._rn = road_network
        self._weather = weather
        self._daynight = daynight
        self._tick = tick
        self._time_s = time_s

    # ══════════════════════════════════════════════
    # Signal Queries
    # ══════════════════════════════════════════════

    def signal_at(
        self, node_id: str, perspective: str = "vehicle",
        observer_id: str = "",
    ) -> SignalInfo:
        """Query traffic signal at a node.

        Args:
            node_id: The intersection node ID.
            perspective: "vehicle" for vehicle signal, "pedestrian" for ped signal.

        Returns:
            SignalInfo with current signal state.
        """
        if observer_id:
            distance = self._observer_distance_to_node(
                observer_id, node_id)
            envelope = self._tm.perception_model.visual_envelope(
                observer_id, bearing_deg=0.0)
            if (distance is not None
                    and distance > envelope["effective_range_m"]):
                return SignalInfo(
                    has_signal=False,
                    direction_info="not_detected_optically")
        if perspective == "pedestrian":
            lane_runtime = getattr(self._tm, "_lane_geometry", None)
            lane_light = (
                lane_runtime.pedestrian_signal_state(node_id, self._time_s)
                if lane_runtime is not None else None)
            if lane_light is not None:
                return SignalInfo(
                    has_signal=True,
                    signal=("ped_green" if lane_light.pedestrian_green
                            else "ped_red"),
                    remaining_seconds=int(lane_light.remaining_seconds),
                    is_ped_green=lane_light.pedestrian_green,
                )
        if perspective == "vehicle" and observer_id:
            movement = self._tm.get_vehicle_movement_signal(
                observer_id, node_id=node_id, time_s=self._time_s)
            if movement is not None:
                return SignalInfo(
                    has_signal=True,
                    signal=movement["signal"],
                    remaining_seconds=int(math.ceil(
                        movement["remaining_seconds"])),
                    is_ped_green=movement["is_ped_green"],
                    direction_info=(
                        f"{movement['turn'] or 'movement'} via "
                        f"{movement['connector_id']}"),
                )
        light = self._rn.get_traffic_light(node_id, self._tick)
        if not light:
            return SignalInfo(has_signal=False)

        if perspective == "pedestrian":
            # Pedestrian cares about ped_green phase
            signal_str = "ped_green" if light.is_crosswalk_phase else "ped_red"
        else:
            signal_str = light.signal

        return SignalInfo(
            has_signal=True,
            signal=signal_str,
            remaining_seconds=light.remaining_seconds,
            is_ped_green=light.is_crosswalk_phase,
        )

    def _observer_distance_to_node(
        self, observer_id: str, node_id: str,
    ) -> Optional[float]:
        vehicle = self._tm.vehicles.get(observer_id)
        if vehicle is not None:
            return self._vehicle_distance_to_node(vehicle, node_id)
        pedestrian = self._tm.pedestrians.get(observer_id)
        if pedestrian is not None:
            if self._ped_is_at_node(pedestrian, node_id):
                return 0.0
        return None

    def _observed_entity(
        self, observer_id: str, candidate: EntityInfo,
    ) -> Optional[EntityInfo]:
        detection = self._tm.perception_model.detect_entity(
            observer_id, candidate.entity_id,
            claimed_distance_m=candidate.distance_m)
        if detection is None:
            return None
        candidate.distance_m = detection.distance_m
        candidate.bearing_deg = detection.bearing_deg
        candidate.confidence = detection.confidence
        candidate.effective_range_m = detection.effective_range_m
        candidate.modalities = list(detection.modalities)
        candidate.observed_signals = dict(detection.observed_signals)
        return candidate

    # ══════════════════════════════════════════════
    # Entity Queries
    # ══════════════════════════════════════════════

    def look_ahead(self, entity_id: str, distance_m: float = 100.0) -> List[EntityInfo]:
        """Query entities ahead of the given entity within distance.

        Works for both vehicles and pedestrians. For vehicles, "ahead" means
        along the current travel direction on the segment. For pedestrians,
        "ahead" means along their route.

        Args:
            entity_id: The querying entity's ID.
            distance_m: Maximum look-ahead distance in meters.

        Returns:
            List of EntityInfo sorted by distance (nearest first).
        """
        results: List[EntityInfo] = []

        # Check if entity is a vehicle
        vs = self._tm.vehicles.get(entity_id)
        if vs:
            results.extend(
                item for item in (
                    self._observed_entity(entity_id, candidate)
                    for candidate in self._look_ahead_vehicle(vs, distance_m))
                if item is not None)
            return sorted(results, key=lambda e: e.distance_m)

        # Check if entity is a pedestrian
        ps = self._tm.pedestrians.get(entity_id)
        if ps:
            results.extend(self.vehicles_heading_toward(
                ps.position.at_node or ps.position.crossing_to or "",
                distance_m, observer_id=entity_id))
            return sorted(results, key=lambda e: e.distance_m)

        return results

    def _look_ahead_vehicle(self, vs, distance_m: float) -> List[EntityInfo]:
        """Find entities ahead on the authoritative multi-leg lane route."""
        results: List[EntityInfo] = []
        try:
            context = self._tm._build_driver_context(
                vs, self._tm._build_env_view(vs, self._tick))
        except Exception:
            context = None
        if (context is not None and context.leader is not None
                and context.leader_gap_m <= distance_m):
            leader = context.leader
            results.append(EntityInfo(
                entity_id=leader.vehicle_id,
                entity_type="vehicle",
                distance_m=round(context.leader_gap_m, 1),
                speed_kmh=round(leader.current_speed_kmh, 1),
                is_stopped=(
                    leader.current_speed_kmh <= 0.1
                    or leader.is_crashed),
            ))
        if context is not None:
            for hazard in context.pedestrian_hazards:
                if (hazard.distance_to_crosswalk_m < 0
                        or hazard.distance_to_crosswalk_m > distance_m):
                    continue
                pedestrian = self._tm.pedestrians.get(
                    hazard.pedestrian_id)
                results.append(EntityInfo(
                    entity_id=hazard.pedestrian_id,
                    entity_type="pedestrian",
                    distance_m=round(
                        hazard.distance_to_crosswalk_m, 1),
                    speed_kmh=round(
                        (pedestrian.speed * 3.6
                         if pedestrian else 0.0), 1),
                    is_crossing=True,
                ))

        return results

    def _look_ahead_pedestrian(self, ps, distance_m: float) -> List[EntityInfo]:
        """Find entities relevant to a pedestrian (approaching vehicles)."""
        results: List[EntityInfo] = []
        node_id = ps.position.at_node or ps.position.crossing_to
        if not node_id:
            return results
        return self.vehicles_heading_toward(
            node_id, distance_m, observer_id=ps.ped_id)

    def _get_vehicle_end_node(self, vs) -> Optional[str]:
        """Get the endpoint of the vehicle's current physical path."""
        lane_runtime = getattr(self._tm, "_lane_geometry", None)
        if lane_runtime is not None:
            if vs.active_connector_to_lane_id:
                target = lane_runtime._lane_by_id.get(
                    vs.active_connector_to_lane_id)
                if target:
                    return target["start_node"]
            lane = lane_runtime._lane_by_id.get(vs.current_lane_id)
            if lane:
                return lane["end_node"]
        return None

    def _ped_is_at_node(self, ps, node_id: str) -> bool:
        """Check if pedestrian is at or crossing through a specific node."""
        if ps.position.at_node == node_id:
            return True
        if ps.position.crossing_from == node_id or ps.position.crossing_to == node_id:
            return True
        return False

    def look_around(self, entity_id: str, radius_m: float = 50.0) -> List[EntityInfo]:
        """Scan all entities within radius of the given entity.

        Returns all nearby entities regardless of direction.

        Args:
            entity_id: The querying entity's ID.
            radius_m: Search radius in meters.

        Returns:
            List of EntityInfo sorted by distance.
        """
        results: List[EntityInfo] = []
        if self._tm.perception_model.entity_pose(entity_id) is None:
            return results

        # Scan vehicles
        for vid, vs in self._tm.vehicles.items():
            if vid == entity_id:
                continue
            if vs.arrived:
                continue
            geometry = self._tm.perception_model.relative_geometry(
                entity_id, vid)
            if geometry is None:
                continue
            dist, _ = geometry
            if dist <= radius_m:
                candidate = EntityInfo(
                    entity_id=vid,
                    entity_type="vehicle",
                    distance_m=round(dist, 1),
                    speed_kmh=round(vs.current_speed_kmh, 1),
                    is_stopped=vs.current_speed_kmh <= 0.1 or vs.is_crashed)
                observed = self._observed_entity(entity_id, candidate)
                if observed is not None:
                    results.append(observed)

        # Scan pedestrians
        if hasattr(self._tm, 'pedestrians'):
            for pid, ps in self._tm.pedestrians.items():
                if pid == entity_id:
                    continue
                if ps.has_arrived:
                    continue
                geometry = self._tm.perception_model.relative_geometry(
                    entity_id, pid)
                if geometry is None:
                    continue
                dist, _ = geometry
                if dist <= radius_m:
                    candidate = EntityInfo(
                        entity_id=pid,
                        entity_type="pedestrian",
                        distance_m=round(dist, 1),
                        speed_kmh=round(ps.speed * 3.6, 1),
                        is_crossing=ps.is_on_crosswalk)
                    observed = self._observed_entity(entity_id, candidate)
                    if observed is not None:
                        results.append(observed)

        return sorted(results, key=lambda e: e.distance_m)

    def vehicles_heading_toward(
        self, node_id: str, max_dist_m: float = 50.0,
        observer_id: str = "",
    ) -> List[EntityInfo]:
        """Query vehicles heading toward a specific node.

        Useful for pedestrians to check if vehicles are approaching
        their crosswalk.

        Args:
            node_id: The target node (e.g., crosswalk location).
            max_dist_m: Maximum distance to consider.

        Returns:
            List of approaching vehicles sorted by distance.
        """
        results: List[EntityInfo] = []
        for vid, vs in self._tm.vehicles.items():
            if vs.arrived or vs.is_crashed:
                continue
            # Check if vehicle's route passes through this node
            if not self._vehicle_heading_toward_node(vs, node_id):
                continue
            dist = self._vehicle_distance_to_node(vs, node_id)
            if dist is not None and dist <= max_dist_m:
                candidate = EntityInfo(
                    entity_id=vid,
                    entity_type="vehicle",
                    distance_m=round(dist, 1),
                    speed_kmh=round(vs.current_speed_kmh, 1),
                    is_stopped=vs.current_speed_kmh <= 0.1)
                observed = (
                    self._observed_entity(observer_id, candidate)
                    if observer_id else candidate)
                if observed is not None:
                    results.append(observed)

        return sorted(results, key=lambda e: e.distance_m)

    def vehicles_approaching_path(
        self, path_xy: List[List[float]], max_dist_m: float = 50.0,
        observer_id: str = "",
    ) -> List[EntityInfo]:
        """Return vehicles approaching a physical pedestrian path.

        Unlike :meth:`vehicles_heading_toward`, this query follows physical
        geometry. It measures the remaining distance along the
        vehicle's current lane and immediate junction connector to the actual
        geometric intersection with ``path_xy``.
        """
        runtime = getattr(self._tm, "_lane_geometry", None)
        pedestrian_path = [tuple(point) for point in path_xy]
        if runtime is None or len(pedestrian_path) < 2:
            return []

        results: List[EntityInfo] = []
        for vehicle_id, vehicle in self._tm.vehicles.items():
            if vehicle.arrived or vehicle.is_crashed:
                continue
            distance = self._tm._vehicle_distance_to_path(
                vehicle, pedestrian_path)
            if distance is None:
                continue
            if distance > max_dist_m:
                continue
            candidate = EntityInfo(
                entity_id=vehicle_id,
                entity_type="vehicle",
                distance_m=round(distance, 1),
                speed_kmh=round(vehicle.current_speed_kmh, 1),
                is_stopped=(
                    vehicle.current_speed_kmh
                    <= self._tm.config.stopped_speed_threshold_kmh))
            observed = (
                self._observed_entity(observer_id, candidate)
                if observer_id else candidate)
            if observed is not None:
                results.append(observed)
        return sorted(results, key=lambda item: item.distance_m)

    def _vehicle_heading_toward_node(self, vs, node_id: str) -> bool:
        """Check if vehicle is heading toward a specific node."""
        lane_runtime = getattr(self._tm, "_lane_geometry", None)
        if lane_runtime is not None and vs.current_lane_id:
            lane = lane_runtime._lane_by_id.get(vs.current_lane_id)
            target_junction = lane_runtime._node_to_junction.get(
                node_id, node_id)
            if lane and lane.get("end_node") == target_junction:
                return True
            if vs.planned_connector_id:
                plan = lane_runtime._signal_plan_by_connector.get(
                    vs.planned_connector_id)
                if plan and plan.get("node_id") == target_junction:
                    return True
        # Check current segment endpoints
        seg_id = vs.current_segment
        if not seg_id:
            return False
        # If the segment leads to this node
        if '_' in seg_id:
            parts = seg_id.split('_')
            if node_id in parts:
                # Check direction: is the vehicle moving toward this node?
                end_node = self._get_vehicle_end_node(vs)
                return end_node == node_id
        return False

    def _vehicle_distance_to_node(self, vs, node_id: str) -> Optional[float]:
        """Compute distance from vehicle to a specific node (meters)."""
        lane_runtime = getattr(self._tm, "_lane_geometry", None)
        if lane_runtime is not None and vs.current_lane_id:
            lane = lane_runtime._lane_by_id.get(vs.current_lane_id)
            target_junction = lane_runtime._node_to_junction.get(
                node_id, node_id)
            if lane and lane.get("end_junction") == target_junction:
                length = lane_runtime._lane_lengths.get(
                    vs.current_lane_id, 0.0)
                return max(0.0, (1.0 - vs.edge_progress) * length)
        seg_id = vs.current_segment
        if not seg_id:
            return None
        seg = self._rn.get_segment(seg_id)
        if not seg:
            return None

        end_node = self._get_vehicle_end_node(vs)
        if end_node == node_id:
            # Node is at the end of current segment
            return (
                (1.0 - vs.edge_progress)
                * self._tm._vehicle_path_length(vs))

        return None

    # ══════════════════════════════════════════════
    # Crosswalk Queries
    # ══════════════════════════════════════════════

    def scan_crosswalk(
        self, node_id: str, observer_id: str = "",
    ) -> CrosswalkInfo:
        """Query crosswalk status at a node.

        Args:
            node_id: The intersection node ID.

        Returns:
            CrosswalkInfo with pedestrian count and details.
        """
        node = self._rn.nodes.get(node_id)
        lane_runtime = getattr(self._tm, "_lane_geometry", None)
        junction_id = (
            lane_runtime._node_to_junction.get(node_id, node_id)
            if lane_runtime is not None else node_id)
        lane_crosswalk = (
            next((item for item in lane_runtime.data.get("crosswalks", [])
                  if item["node_id"] == junction_id), None)
            if lane_runtime is not None else None)
        has_cw = bool(
            (node and getattr(node, 'has_crosswalk', False))
            or lane_crosswalk)
        if not has_cw:
            return CrosswalkInfo(has_crosswalk=False)
        if observer_id:
            distance = self._observer_distance_to_node(
                observer_id, node_id)
            envelope = self._tm.perception_model.visual_envelope(
                observer_id, bearing_deg=0.0)
            if (distance is not None
                    and distance > envelope["effective_range_m"]):
                return CrosswalkInfo(has_crosswalk=False)

        # Count pedestrians on this crosswalk
        peds = []
        if hasattr(self._tm, 'pedestrians'):
            for pid, ps in self._tm.pedestrians.items():
                if not ps.is_spawned or ps.has_arrived:
                    continue
                if (self._ped_is_at_node(ps, node_id)
                        and ps.is_on_crosswalk
                        and (not observer_id or self._tm.perception_model
                             .detect_entity(observer_id, pid) is not None)):
                    peds.append({
                        "ped_id": pid,
                        "progress": round(ps.crossing_progress, 2),
                    })

        return CrosswalkInfo(
            has_crosswalk=True,
            pedestrians_crossing=len(peds),
            pedestrian_details=peds,
        )

    # ══════════════════════════════════════════════
    # Weather / Environment Queries
    # ══════════════════════════════════════════════

    def check_weather(self) -> WeatherInfo:
        """Query current weather condition."""
        if self._weather is None:
            return WeatherInfo(condition="sunny")
        condition = getattr(self._weather, 'condition', None)
        if condition is None:
            return WeatherInfo(condition="sunny")
        cond_val = condition.value if hasattr(condition, 'value') else str(condition)
        return WeatherInfo(condition=cond_val)

    def check_daynight(self) -> dict:
        """Query current day/night period."""
        if self._daynight is None:
            return {"period": "day", "daylight_level": 100}
        period = getattr(self._daynight, 'period', 'day')
        period_val = period.value if hasattr(period, 'value') else str(period)
        level = getattr(self._daynight, 'daylight_level', 100)
        return {"period": period_val, "daylight_level": level}

    def check_road(self, segment_id: str) -> RoadInfo:
        """Query road segment status.

        Args:
            segment_id: The segment ID to query.

        Returns:
            RoadInfo with speed limit, blockage and SUMO-observed flow.
        """
        seg = self._rn.get_segment(segment_id)
        if not seg:
            return RoadInfo(segment_id=segment_id)

        return RoadInfo(
            segment_id=segment_id,
            speed_limit_kmh=float(seg.speed_limit),
            is_blocked=self._rn.has_road_closure(segment_id, self._tick),
            observed_flow_speed_kmh=(
                self._rn.get_segment_mean_speed_kmh(segment_id)),
            lanes=seg.lanes,
            distance_m=seg.distance_meters,
        )

    # ══════════════════════════════════════════════
    # Position Helpers
    # ══════════════════════════════════════════════

    def get_entity_location(self, entity_id: str) -> dict:
        """Get the current location of any entity.

        Returns:
            Dict with position info (node, segment, progress, etc.)
        """
        vs = self._tm.vehicles.get(entity_id)
        if vs:
            return {
                "type": "vehicle",
                "current_node": vs.current_node,
                "current_segment": vs.current_segment,
                "current_lane_index": vs.current_lane,
                "current_lane_id": vs.current_lane_id,
                "active_connector_id": vs.active_connector_id,
                "planned_connector_id": vs.planned_connector_id,
                "planned_turn": vs.planned_turn,
                "route_required_lane_id": vs.planned_from_lane_id,
                "route_required_lane_index": vs.planned_from_lane_index,
                "edge_progress": round(vs.edge_progress, 3),
                "speed_kmh": round(vs.current_speed_kmh, 1),
                "target_speed_kmh": round(vs.target_speed_kmh, 1),
                "acceleration_mps2": round(vs.acceleration_mps2, 2),
                "heading": round(vs.heading, 1),
                "pose_xy_m": [
                    round(vs.pose_x_m, 3), round(vs.pose_y_m, 3)],
                "yaw_rad": round(vs.yaw_rad, 6),
                "z_level": vs.z_level,
                "is_stopped": vs.current_speed_kmh <= 0.1,
                "is_changing_lane": vs.is_changing_lane,
                "target_lane_index": vs.target_lane,
                "is_crashed": vs.is_crashed,
                "collision_domain": {
                    "shape": "oriented_rectangle",
                    "length_m": vs.length_m,
                    "width_m": vs.width_m,
                },
                "llm_control_command": dict(vs.llm_control_command),
                "active_control_commands": dict(
                    vs.active_control_commands),
                "driving_awareness":
                    self._tm.get_driving_awareness(entity_id),
            }

        if hasattr(self._tm, 'pedestrians'):
            ps = self._tm.pedestrians.get(entity_id)
            if ps:
                pose = self._tm._lane_geometry.pedestrian_pose(ps)
                return {
                    "type": "pedestrian",
                    "at_node": ps.position.at_node,
                    "crossing_from": ps.position.crossing_from,
                    "crossing_to": ps.position.crossing_to,
                    "crossing_progress": round(ps.crossing_progress, 2),
                    "walking_progress": round(ps.walking_progress, 3),
                    "pose_xy_m": (
                        [round(pose[0], 3), round(pose[1], 3)]
                        if pose is not None else None),
                    "is_on_crosswalk": ps.is_on_crosswalk,
                    "active_crosswalk_id": ps.active_crosswalk_id,
                    "is_waiting": ps.is_waiting,
                    "is_crashed": ps.is_crashed,
                    "is_spawned": ps.is_spawned,
                    "speed_mps": round(ps.speed, 2),
                    "collision_domain": {
                        "shape": "circle",
                        "radius_m": ps.collision_radius_m,
                    },
                }

        return {"type": "unknown"}

    def _get_entity_position_m(self, entity_id: str) -> Optional[float]:
        """Get approximate 1D position in meters (for distance calculation).

        This uses a simplified linear model along segments for distance
        estimation between entities on the same or adjacent segments.
        """
        vs = self._tm.vehicles.get(entity_id)
        if vs:
            seg = self._rn.get_segment(vs.current_segment) if vs.current_segment else None
            if seg:
                return self._tm._vehicle_path_position(vs)
            return 0.0

        if hasattr(self._tm, 'pedestrians'):
            ps = self._tm.pedestrians.get(entity_id)
            if ps:
                if ps.is_on_crosswalk:
                    return ps.crossing_progress * 10.0  # approximate crosswalk length
                return 0.0

        return None

    # ══════════════════════════════════════════════
    # Tool Dispatch
    # ══════════════════════════════════════════════

    def dispatch_tool(self, entity_id: str, tool_name: str, params: dict) -> dict:
        """Dispatch a perception tool call to the appropriate query method.

        This is the main entry point called by the agent callback when
        the LLM invokes a perception tool.

        Args:
            entity_id: The entity making the query.
            tool_name: Name of the perception tool.
            params: Tool parameters from LLM.

        Returns:
            Dict with query results (serializable to JSON for LLM).
        """
        from simulation.perception_tools import (
            vehicle_perception_method_name)
        tool_name = vehicle_perception_method_name(tool_name)
        dispatch_map = {
            "look_ahead": self._dispatch_look_ahead,
            "check_signal": self._dispatch_check_signal,
            "scan_crosswalk": self._dispatch_scan_crosswalk,
            "check_weather": self._dispatch_check_weather,
            "check_daynight": self._dispatch_check_daynight,
            "check_road": self._dispatch_check_road,
            "look_around": self._dispatch_look_around,
            "look_for_vehicles": self._dispatch_look_for_vehicles,
            "check_crosswalk": self._dispatch_check_crosswalk,
            "get_location": self._dispatch_get_location,
        }

        handler = dispatch_map.get(tool_name)
        if handler is None:
            return {"error": f"Unknown perception tool: {tool_name}"}
        return handler(entity_id, params)

    def _dispatch_look_ahead(self, entity_id: str, params: dict) -> dict:
        distance = params.get("distance_m", 100.0)
        entities = self.look_ahead(entity_id, distance)
        envelope = self._tm.perception_model.visual_envelope(
            entity_id, bearing_deg=0.0)
        return {
            "requested_range_m": float(distance),
            "perception_envelope": envelope,
            "entities": [vars(e) for e in entities],
        }

    def _dispatch_check_signal(self, entity_id: str, params: dict) -> dict:
        node_id = params.get("node_id")
        if not node_id:
            # Default: entity's next node
            vs = self._tm.vehicles.get(entity_id)
            if vs:
                node_id = self._get_vehicle_end_node(vs)
            elif hasattr(self._tm, 'pedestrians'):
                ps = self._tm.pedestrians.get(entity_id)
                if ps:
                    node_id = ps.position.at_node or ps.position.crossing_to

        if not node_id:
            return {"error": "Cannot determine node. Provide node_id."}

        # Determine perspective from entity type
        perspective = "vehicle"
        if hasattr(self._tm, 'pedestrians') and entity_id in self._tm.pedestrians:
            perspective = "pedestrian"

        info = self.signal_at(
            node_id, perspective, observer_id=entity_id)
        return {
            **vars(info),
            "perception_envelope": self._tm.perception_model.visual_envelope(
                entity_id, bearing_deg=0.0),
        }

    def _dispatch_scan_crosswalk(self, entity_id: str, params: dict) -> dict:
        node_id = params.get("node_id")
        if not node_id:
            vs = self._tm.vehicles.get(entity_id)
            if vs:
                node_id = self._get_vehicle_end_node(vs)
        if not node_id:
            return {"error": "Cannot determine node. Provide node_id."}
        info = self.scan_crosswalk(node_id, observer_id=entity_id)
        lane_runtime = getattr(self._tm, "_lane_geometry", None)
        junction_id = (
            lane_runtime._node_to_junction.get(node_id, node_id)
            if lane_runtime is not None else node_id)
        crosswalks = []
        if lane_runtime is not None:
            crosswalks = [
                {
                    "crosswalk_id": item["id"],
                    "length_m": item.get("length_m"),
                    "width_m": item.get("width_m"),
                    "conflicting_connector_ids": item.get(
                        "conflicting_connectors", []),
                }
                for item in lane_runtime.data.get("crosswalks", [])
                if item["node_id"] == junction_id
            ]
        return {
            "has_crosswalk": info.has_crosswalk,
            "pedestrians_crossing": info.pedestrians_crossing,
            "pedestrian_details": info.pedestrian_details,
            "directional_crosswalks": (
                crosswalks if info.has_crosswalk else []),
        }

    def _dispatch_check_weather(self, entity_id: str, params: dict) -> dict:
        info = self.check_weather()
        return vars(info)

    def _dispatch_check_daynight(self, entity_id: str, params: dict) -> dict:
        return self.check_daynight()

    def _dispatch_check_road(self, entity_id: str, params: dict) -> dict:
        segment_id = params.get("segment_id")
        if not segment_id:
            vs = self._tm.vehicles.get(entity_id)
            if vs:
                segment_id = vs.current_segment
        if not segment_id:
            return {"error": "Cannot determine segment. Provide segment_id."}
        info = self.check_road(segment_id)
        return vars(info)

    def _dispatch_look_around(self, entity_id: str, params: dict) -> dict:
        radius = params.get("radius_m", 50.0)
        entities = self.look_around(entity_id, radius)
        return {
            "requested_radius_m": float(radius),
            "perception_envelope": self._tm.perception_model.visual_envelope(
                entity_id, bearing_deg=90.0),
            "entities": [vars(e) for e in entities],
            "heard_horns": self._tm.perception_model.heard_horns(
                entity_id, self._time_s),
        }

    def _dispatch_look_for_vehicles(self, entity_id: str, params: dict) -> dict:
        """Pedestrian-specific: look for vehicles approaching current node."""
        max_dist = params.get("max_distance_m", 50.0)
        node_id = None
        if hasattr(self._tm, 'pedestrians'):
            ps = self._tm.pedestrians.get(entity_id)
            if ps:
                node_id = ps.position.at_node or ps.position.crossing_to
        if not node_id:
            return {"error": "Cannot determine current node."}
        entities = self.vehicles_heading_toward(
            node_id, max_dist, observer_id=entity_id)
        return {
            "requested_range_m": float(max_dist),
            "perception_envelope": self._tm.perception_model.visual_envelope(
                entity_id, bearing_deg=0.0),
            "approaching_vehicles": [vars(e) for e in entities],
            "heard_horns": self._tm.perception_model.heard_horns(
                entity_id, self._time_s),
        }

    def _dispatch_check_crosswalk(self, entity_id: str, params: dict) -> dict:
        """Pedestrian-specific: check if crosswalk at current node is clear."""
        node_id = None
        if hasattr(self._tm, 'pedestrians'):
            ps = self._tm.pedestrians.get(entity_id)
            if ps:
                node_id = ps.position.at_node or ps.position.crossing_to
        if not node_id:
            return {"error": "Cannot determine current node."}
        info = self.scan_crosswalk(node_id, observer_id=entity_id)
        signal = self.signal_at(
            node_id, "pedestrian", observer_id=entity_id)
        approaching = self.vehicles_heading_toward(
            node_id, 60.0, observer_id=entity_id)
        risky = [
            entity for entity in approaching
            if entity.distance_m <= max(
                8.0, entity.speed_kmh / 3.6 * 3.0)
        ]
        lane_runtime = getattr(self._tm, "_lane_geometry", None)
        junction_id = (
            lane_runtime._node_to_junction.get(node_id, node_id)
            if lane_runtime is not None else node_id)
        crosswalks = (
            [
                {
                    "crosswalk_id": item["id"],
                    "length_m": item.get("length_m"),
                    "width_m": item.get("width_m"),
                    "conflicting_connector_ids": item.get(
                        "conflicting_connectors", []),
                }
                for item in lane_runtime.data.get("crosswalks", [])
                if item["node_id"] == junction_id
            ]
            if lane_runtime is not None else [])
        pedestrian = self._tm.pedestrians.get(entity_id)
        return {
            "has_crosswalk": info.has_crosswalk,
            "is_clear": not risky,
            "signal": signal.signal,
            "signal_remaining_seconds": signal.remaining_seconds,
            "active_crosswalk_id": (
                pedestrian.active_crosswalk_id if pedestrian else None),
            "is_on_crosswalk": bool(
                pedestrian and pedestrian.is_on_crosswalk),
            "directional_crosswalks": (
                crosswalks if info.has_crosswalk else []),
            "approaching_vehicles": [
                vars(entity) for entity in approaching],
            "risk_vehicle_ids": [
                entity.entity_id for entity in risky],
            "perception_envelope": self._tm.perception_model.visual_envelope(
                entity_id, bearing_deg=0.0),
            "heard_horns": self._tm.perception_model.heard_horns(
                entity_id, self._time_s),
        }

    def _dispatch_get_location(self, entity_id: str, params: dict) -> dict:
        result = self.get_entity_location(entity_id)
        result["perception_profile"] = (
            self._tm.perception_model.profile_for(entity_id).as_dict())
        result["visual_envelope"] = (
            self._tm.perception_model.visual_envelope(
                entity_id, bearing_deg=0.0))
        result["acoustic_envelope"] = (
            self._tm.perception_model.acoustic_envelope(entity_id))
        result["heard_horns"] = (
            self._tm.perception_model.heard_horns(
                entity_id, self._time_s))
        vehicle = self._tm.vehicles.get(entity_id)
        if vehicle is not None:
            result["own_signal_state"] = vehicle.signal_state.as_dict()
        return result


def _freeze_observation(value: Any) -> Any:
    """Detach and recursively freeze values exposed to agent callbacks."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return MappingProxyType({
            str(key): _freeze_observation(item)
            for key, item in value.items()
        })
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_observation(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_observation(item) for item in value)
    if hasattr(value, "__dict__"):
        return FrozenStateView.from_state(value)
    try:
        return copy.deepcopy(value)
    except Exception:
        return repr(value)


class FrozenStateView:
    """Read-only, detached entity state supplied to agent callbacks.

    It intentionally excludes private back-references and manager handles.
    Mutating it cannot alter physical truth.
    """

    __slots__ = ("__values",)
    _EXCLUDED = {
        "lane_change_progress", "lane_change_attempts", "lane_change_completions",
        "lane_change_uncompleted", "lane_change_request_outcome",
    }

    def __init__(self, values: Dict[str, Any]):
        object.__setattr__(
            self, "_FrozenStateView__values",
            MappingProxyType(dict(values)))

    @classmethod
    def from_state(cls, state: Any) -> "FrozenStateView":
        values = {}
        for name, value in vars(state).items():
            if name.startswith("_") or name in cls._EXCLUDED:
                continue
            values[name] = _freeze_observation(value)
        # Preserve useful computed pedestrian properties without exposing the
        # live object that implements them.
        for name in ("current_node", "next_node", "destination"):
            if name in values:
                continue
            try:
                values[name] = _freeze_observation(getattr(state, name))
            except Exception:
                pass
        return cls(values)

    def __getattr__(self, name: str) -> Any:
        values = object.__getattribute__(
            self, "_FrozenStateView__values")
        if name in values:
            return values[name]
        raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("agent state observations are read-only")

    def as_dict(self) -> Dict[str, Any]:
        values = object.__getattribute__(
            self, "_FrozenStateView__values")
        return dict(values)


class AgentWorldState:
    """Narrow agent-facing facade over :class:`WorldState`.

    Perception stays read-only. Actions are submitted through an engine-owned
    command boundary, so callbacks never receive ``TrafficCoordinator`` itself.
    """

    __slots__ = ("__world", "__submit_action", "entity_id")

    def __init__(
        self,
        world: WorldState,
        entity_id: str,
        submit_action,
    ):
        object.__setattr__(self, "_AgentWorldState__world", world)
        object.__setattr__(
            self, "_AgentWorldState__submit_action", submit_action)
        object.__setattr__(self, "entity_id", entity_id)

    @property
    def time_s(self) -> float:
        world = object.__getattribute__(
            self, "_AgentWorldState__world")
        return world._time_s

    def dispatch_tool(
        self, entity_id: str, tool_name: str, params: Optional[dict] = None,
    ) -> dict:
        if entity_id != self.entity_id:
            return {"error": "cross-entity perception is not allowed"}
        world = object.__getattribute__(
            self, "_AgentWorldState__world")
        return world.dispatch_tool(entity_id, tool_name, params or {})

    def execute_action(
        self, entity_id: str, action: str, params: Optional[dict] = None,
    ) -> dict:
        if entity_id != self.entity_id:
            return {"success": False,
                    "error": "cross-entity control is not allowed"}
        submit = object.__getattribute__(
            self, "_AgentWorldState__submit_action")
        return submit(entity_id, action, params or {})
