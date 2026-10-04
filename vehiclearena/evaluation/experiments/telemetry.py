"""Lossless-enough experiment telemetry from the authoritative physics world."""

from __future__ import annotations

import dataclasses
import enum
import math
from typing import Any, Dict, List

from evaluation.multi_agent_runner import TrackedMultiSimEngine


def jsonable(value: Any) -> Any:
    """Convert simulation records to deterministic JSON-compatible values."""
    if dataclasses.is_dataclass(value):
        return {field.name: jsonable(getattr(value, field.name))
                for field in dataclasses.fields(value)}
    if isinstance(value, enum.Enum):
        return value.name.lower()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "__dict__"):
        return {
            str(key): jsonable(item)
            for key, item in vars(value).items()
            if not str(key).startswith("_")
        }
    return str(value)


def build_event_audit(events: List[Any], delivery_log: List[dict]) -> dict:
    """Partition internal events from events successfully shown to agents."""
    event_records = {
        str(record["event_id"]): record
        for record in (jsonable(event.as_dict()) for event in events)
    }
    deliveries = jsonable(delivery_log)
    delivery_by_event: Dict[str, List[dict]] = {}
    for batch in deliveries:
        reference = {
            "entity_id": batch.get("entity_id"),
            "delivered_at_s": batch.get("delivered_at_s"),
            "status": batch.get("status"),
        }
        for event_id in batch.get("event_ids", []):
            delivery_by_event.setdefault(str(event_id), []).append(
                reference)

    agent_visible = []
    evaluator_only = []
    for event_id, record in event_records.items():
        matched = delivery_by_event.get(event_id, [])
        successful = [
            item for item in matched if item.get("status") == "delivered"]
        evaluator_internal = (
            record.get("event_type") == "passenger_judge_due")
        annotated = {
            **record,
            "visibility": (
                "agent_visible"
                if successful and not evaluator_internal
                else "evaluator_only"),
            "delivered_at_s": (
                min(float(item["delivered_at_s"]) for item in successful)
                if successful else None),
            "delivery_records": matched,
        }
        if successful and not evaluator_internal:
            agent_visible.append(annotated)
        else:
            evaluator_only.append(annotated)
    return {
        "summary": {
            "agent_visible_event_count": len(agent_visible),
            "evaluator_only_event_count": len(evaluator_only),
            "delivery_batch_count": len(deliveries),
        },
        "agent_visible_events": agent_visible,
        "evaluator_only_events": evaluator_only,
        "delivery_batches": deliveries,
    }


def _decision(value: Any) -> Any:
    if value is None:
        return None
    return jsonable(value)


def snapshot_world(engine, time_s: float, tick_index: int) -> dict:
    """Capture vehicle and pedestrian state after one fixed physics step."""
    manager = engine.traffic_mgr
    vehicles = []
    for vehicle_id in sorted(manager.vehicles):
        vehicle = manager.vehicles[vehicle_id]
        vehicles.append({
            "vehicle_id": vehicle_id,
            "time_s": round(float(time_s), 6),
            "tick_index": int(tick_index),
            "control_authority": vehicle.control_authority,
            "chassis_profile": vehicle.chassis_profile,
            "pose_x_m": round(float(vehicle.pose_x_m), 4),
            "pose_y_m": round(float(vehicle.pose_y_m), 4),
            "yaw_rad": round(float(vehicle.yaw_rad), 6),
            "length_m": float(vehicle.length_m),
            "width_m": float(vehicle.width_m),
            "current_node": vehicle.current_node,
            "current_segment": vehicle.current_segment,
            "current_lane": int(vehicle.current_lane),
            "current_lane_id": vehicle.current_lane_id,
            "active_connector_id": vehicle.active_connector_id,
            "edge_progress": round(float(vehicle.edge_progress), 6),
            "lane_route_action_index": int(
                vehicle.lane_route_action_index),
            "speed_kmh": round(float(vehicle.current_speed_kmh), 4),
            "target_speed_kmh": round(float(vehicle.target_speed_kmh), 4),
            "acceleration_mps2": round(float(vehicle.acceleration_mps2), 4),
            "distance_traveled_m": round(
                float(vehicle.distance_traveled_m), 4),
            "is_changing_lane": bool(vehicle.is_changing_lane),
            "target_lane": int(vehicle.target_lane),
            "lane_change_attempts": vehicle.lane_change_attempts,
            "lane_change_completions": vehicle.lane_change_completions,
            "lane_change_uncompleted": vehicle.lane_change_uncompleted,
            "lane_change_request_outcome": vehicle.lane_change_request_outcome,
            "lateral_offset_m": round(float(vehicle.lateral_offset_m), 4),
            "maneuver_state": (
                "LANE_CHANGE" if vehicle.is_changing_lane
                else "CONNECTOR" if vehicle.active_connector_id
                else "LANE_KEEP"),
            "waiting_red_light": bool(vehicle.waiting_red_light),
            "arrived": bool(vehicle.arrived),
            "route_failed": bool(vehicle.route_failed),
            "route_failure_reason": vehicle.route_failure_reason,
            "route_failure_time_s": vehicle.route_failure_time_s,
            "present_in_physics_world": bool(
                vehicle.present_in_physics_world),
            "terminal_crossing_speed_kmh": (
                round(float(vehicle.terminal_crossing_speed_kmh), 4)
                if vehicle.terminal_crossing_speed_kmh is not None
                else None),
            "crashed": bool(vehicle.is_crashed),
            "last_control_command": _decision(
                vehicle.llm_control_command if vehicle.is_llm else None),
            "signal_state": jsonable(vehicle.signal_state),
            "perception_profile": vehicle.perception_profile_name,
            "perception_overrides": jsonable(
                vehicle.perception_overrides),
            "cabin_open_fraction": round(
                float(vehicle.cabin_open_fraction), 4),
        })

    pedestrians = []
    for pedestrian_id in sorted(manager.pedestrians):
        pedestrian = manager.pedestrians[pedestrian_id]
        pose = manager._lane_geometry.pedestrian_pose(pedestrian)
        x_m, y_m = (pose if pose is not None else (None, None))
        pedestrians.append({
            "ped_id": pedestrian_id,
            "time_s": round(float(time_s), 6),
            "tick_index": int(tick_index),
            "control_authority": pedestrian.control_authority,
            "pose_x_m": round(float(x_m), 4) if x_m is not None else None,
            "pose_y_m": round(float(y_m), 4) if y_m is not None else None,
            "collision_radius_m": float(pedestrian.collision_radius_m),
            "speed_mps": round(float(pedestrian.speed), 4),
            "current_node": pedestrian.current_node,
            "route_index": int(pedestrian.route_index),
            "spawned": bool(pedestrian.is_spawned),
            "waiting": bool(pedestrian.is_waiting),
            "walking": bool(pedestrian.is_walking),
            "on_crosswalk": bool(pedestrian.is_on_crosswalk),
            "active_crosswalk_id": pedestrian.active_crosswalk_id,
            "crossing_progress": round(
                float(pedestrian.crossing_progress), 6),
            "walking_progress": round(float(pedestrian.walking_progress), 6),
            "arrived": bool(pedestrian.has_arrived),
            "crashed": bool(pedestrian.is_crashed),
            "perception_profile": pedestrian.perception_profile_name,
            "perception_overrides": jsonable(
                pedestrian.perception_overrides),
        })
    return {"vehicles": vehicles, "pedestrians": pedestrians}


class ExperimentTrackedEngine(TrackedMultiSimEngine):
    """Tracked engine that adds complete 10 Hz entity trajectories."""

    def __init__(self, scenario):
        super().__init__(scenario)
        self._vehicle_trajectory: List[Dict[str, Any]] = []
        self._pedestrian_trajectory: List[Dict[str, Any]] = []
        self._physics_events: List[Dict[str, Any]] = []
        self._arrived_vehicle_trajectory_closed: set[str] = set()
        self._arrived_pedestrian_trajectory_closed: set[str] = set()

    def _log_per_substep(self, physics_time, tick_index, trigger_events):
        super()._log_per_substep(physics_time, tick_index, trigger_events)
        snapshot = snapshot_world(self, physics_time, tick_index)
        for row in snapshot["vehicles"]:
            entity_id = row["vehicle_id"]
            if entity_id in self._arrived_vehicle_trajectory_closed:
                continue
            self._vehicle_trajectory.append(row)
            if row["arrived"] or row["route_failed"]:
                self._arrived_vehicle_trajectory_closed.add(entity_id)
        for row in snapshot["pedestrians"]:
            entity_id = row["ped_id"]
            if entity_id in self._arrived_pedestrian_trajectory_closed:
                continue
            self._pedestrian_trajectory.append(row)
            if row["arrived"]:
                self._arrived_pedestrian_trajectory_closed.add(entity_id)
        self._physics_events.extend({
            "time_s": round(float(physics_time), 6),
            "tick_index": int(tick_index),
            **jsonable(event),
        } for event in trigger_events)

    def _finalize_result(self, result):
        super()._finalize_result(result)
        result._vehicle_trajectory = self._vehicle_trajectory
        result._pedestrian_trajectory = self._pedestrian_trajectory
        result._physics_events = self._physics_events
        result._collision_log = jsonable(self.traffic_mgr.collision_log)
        result._event_audit = build_event_audit(
            self.wake_broker.log, self.wake_broker.delivery_log)
        result._agent_callback_errors = jsonable(self.agent_callback_errors)

    def failure_snapshot(self):
        """Read collected evidence after cleanup, without scoring an aborted run.

        Do not query libsumo or call finalization here: the physics connection
        may already be closed, or initialization may not have completed.
        """
        broker = getattr(self, "wake_broker", None)
        manager = getattr(self, "traffic_mgr", None)
        return {
            "last_public_time_s": getattr(self, "_sim_time", None),
            "trajectories": {"vehicles": jsonable(self._vehicle_trajectory),
                             "pedestrians": jsonable(self._pedestrian_trajectory)},
            "physics_events": jsonable(self._physics_events),
            "collision_log": jsonable(getattr(manager, "collision_log", [])),
            "agent_callback_errors": jsonable(getattr(self, "agent_callback_errors", [])),
            "event_audit": (build_event_audit(broker.log, broker.delivery_log)
                            if broker is not None else {}),
        }


def callback_tool_logs(callbacks: Dict[str, Any]) -> Dict[str, list]:
    return {
        entity_id: jsonable(getattr(callback, "_state", {}).get(
            "tool_call_log", []))
        for entity_id, callback in callbacks.items()
    }
