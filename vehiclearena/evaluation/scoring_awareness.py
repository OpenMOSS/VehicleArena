"""Read-only scoring observations; never install a reference driving policy."""
from __future__ import annotations

import math
from typing import Any, Optional


class ScoringTrafficView:
    """Expose the new scorer's observations without modifying traffic control.

    Only evaluators receive this view. The driver and SUMO retain their old
    manager. Shared-corridor occupancy is evidence, never a braking command.
    """

    def __init__(self, manager):
        self._manager = manager

    def __getattr__(self, name):
        return getattr(self._manager, name)

    def get_driving_awareness(self, vehicle_id, *, ground_truth=False):
        awareness = dict(self._manager.get_driving_awareness(
            vehicle_id, ground_truth=ground_truth))
        vehicle = self._manager.get_state(vehicle_id)
        if vehicle is None:
            return awareness
        env = self._manager._build_env_view(vehicle, round(self._physics_time))
        context = self._manager._build_driver_context(
            vehicle, env, apply_perception=not ground_truth)
        oncoming = context.oncoming_vehicle
        gap = context.oncoming_gap_m
        awareness["oncoming"] = ({
            "vehicle_id": oncoming.vehicle_id,
            "gap_m": round(gap, 3) if math.isfinite(gap) else None,
            "speed_kmh": round(oncoming.current_speed_kmh, 2),
        } if oncoming is not None else None)
        awareness["shared_corridor_hazard"] = self._shared_corridor_occupant_ahead(
            vehicle, apply_perception=not ground_truth)
        awareness["connector_path_blocker"] = (
            self._same_direction_connector_blocker_ahead(
                vehicle, apply_perception=not ground_truth))
        return awareness

    def _same_direction_connector_blocker_ahead(
        self, vehicle: Any, *, apply_perception: bool,
        max_gap_m: float = 15.0,
    ) -> Optional[dict]:
        """Find a same-direction body blocking ego's connector corridor.

        Parallel and merging movements can use different connector IDs even
        where their physical paths converge.  Exact-ID route projection then
        misses the leading body.  Keep this evaluator-only observation local:
        both vehicles must be inside the same junction, point in nearly the
        same direction, and overlap the same narrow body corridor.
        """
        connector_id = str(
            getattr(vehicle, "active_connector_id", "") or "")
        ego_connector = self._lane_geometry._connector_by_id.get(
            connector_id)
        if (not ego_connector
                or vehicle.pose_x_m is None
                or vehicle.pose_y_m is None):
            return None

        forward_x = math.cos(float(vehicle.yaw_rad))
        forward_y = math.sin(float(vehicle.yaw_rad))
        left_x, left_y = -forward_y, forward_x
        junction_id = str(ego_connector.get("node_id", ""))
        candidates = []
        for other in self.vehicles.values():
            other_connector_id = str(
                getattr(other, "active_connector_id", "") or "")
            if (other.vehicle_id == vehicle.vehicle_id
                    or other.arrived
                    or not other_connector_id
                    or other.pose_x_m is None
                    or other.pose_y_m is None
                    or getattr(other, "z_level", 0)
                    != getattr(vehicle, "z_level", 0)):
                continue
            other_connector = self._lane_geometry._connector_by_id.get(
                other_connector_id)
            if (not other_connector
                    or str(other_connector.get("node_id", ""))
                    != junction_id):
                continue
            heading_alignment = math.cos(
                float(other.yaw_rad) - float(vehicle.yaw_rad))
            if heading_alignment < math.cos(math.radians(45.0)):
                continue
            dx = float(other.pose_x_m) - float(vehicle.pose_x_m)
            dy = float(other.pose_y_m) - float(vehicle.pose_y_m)
            longitudinal = dx * forward_x + dy * forward_y
            if longitudinal <= 0.0:
                continue
            lateral = abs(dx * left_x + dy * left_y)
            corridor_half_width = (
                (float(vehicle.width_m) + float(other.width_m)) / 2.0
                + 0.4)
            if lateral > corridor_half_width:
                continue
            gap = max(
                0.0,
                longitudinal
                - (float(vehicle.length_m) + float(other.length_m)) / 2.0,
            )
            if gap > max_gap_m:
                continue
            if (apply_perception
                    and self.perception_model.detect_entity(
                        vehicle.vehicle_id, other.vehicle_id,
                        claimed_distance_m=gap) is None):
                continue
            candidates.append((gap, other, other_connector_id))
        if not candidates:
            return None
        gap, other, other_connector_id = min(
            candidates, key=lambda item: (item[0], item[1].vehicle_id))
        return {
            "vehicle_id": other.vehicle_id,
            "gap_m": round(gap, 3),
            "other_connector_id": other_connector_id,
            "relationship": "same_direction_connector_ahead",
        }

    def _upcoming_shared_corridor_lanes(
        self, vehicle: Any,
    ) -> list[dict]:
        """Return the directed shared lanes ahead of one approach."""
        current_lane = self._lane_geometry._lane_by_id.get(
            vehicle.current_lane_id)
        if not current_lane or current_lane.get("shared_bidirectional"):
            return []
        cursor = vehicle.current_lane_id
        shared_lanes: list[dict] = []
        entered_corridor = False
        for action in vehicle.lane_route_actions[
                vehicle.lane_route_action_index:]:
            if action.get("type") != "connector":
                continue
            connector = self._lane_geometry._connector_by_id.get(
                action.get("connector_id"))
            if not connector or connector["from_lane"] != cursor:
                break
            target = self._lane_geometry._lane_by_id[connector["to_lane"]]
            if target.get("shared_bidirectional"):
                entered_corridor = True
                shared_lanes.append(target)
            elif entered_corridor:
                break
            cursor = target["id"]
        return shared_lanes

    def _shared_corridor_occupant_ahead(
        self, vehicle: Any, *, apply_perception: bool,
    ) -> Optional[dict]:
        """Describe a body already committed to the next shared corridor."""
        shared_lanes = self._upcoming_shared_corridor_lanes(vehicle)
        shared_segments = {
            str(lane["segment_id"]) for lane in shared_lanes
        }
        if not shared_segments:
            return None
        same_direction_lane_ids = {
            str(lane["id"]) for lane in shared_lanes
        }
        candidates = []
        for other in self.vehicles.values():
            if (other.vehicle_id == vehicle.vehicle_id
                    or other.arrived or other.is_crashed):
                continue
            lane = self._lane_geometry._lane_by_id.get(
                other.current_lane_id)
            occupancy = None
            if (lane and lane.get("shared_bidirectional")
                    and str(lane["segment_id"]) in shared_segments):
                if str(lane["id"]) in same_direction_lane_ids:
                    continue
                occupancy = "inside_shared_corridor"
            elif other.active_connector_id:
                connector = self._lane_geometry._connector_by_id.get(
                    other.active_connector_id)
                source = (
                    self._lane_geometry._lane_by_id.get(
                        connector["from_lane"])
                    if connector else None)
                if (source and source.get("shared_bidirectional")
                        and str(source["segment_id"]) in shared_segments):
                    if str(source["id"]) in same_direction_lane_ids:
                        continue
                    occupancy = "clearing_exit_connector"
            if occupancy is None:
                continue
            distance = float("inf")
            if (vehicle.pose_x_m is not None
                    and vehicle.pose_y_m is not None
                    and other.pose_x_m is not None
                    and other.pose_y_m is not None):
                distance = max(0.0, math.hypot(
                    float(other.pose_x_m) - float(vehicle.pose_x_m),
                    float(other.pose_y_m) - float(vehicle.pose_y_m),
                ) - (float(vehicle.length_m) + float(other.length_m)) / 2.0)
            if (apply_perception
                    and self.perception_model.detect_entity(
                        vehicle.vehicle_id,
                        other.vehicle_id,
                        claimed_distance_m=distance,
                    ) is None):
                continue
            candidates.append((distance, other, occupancy))
        if not candidates:
            return None
        distance, other, occupancy = min(
            candidates, key=lambda item: (item[0], item[1].vehicle_id))
        return {
            "vehicle_id": other.vehicle_id,
            "gap_m": round(distance, 3) if math.isfinite(distance) else None,
            "occupancy": occupancy,
            "corridor_segment_ids": sorted(shared_segments),
        }
