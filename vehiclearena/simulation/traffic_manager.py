"""Simulator-independent traffic decisions, state and perception.

This module deliberately contains no vehicle motion integrator. SUMO/LLM
actions are coordinated here; :mod:`simulation.sumo_traffic_manager` is the
only component allowed to advance physical poses.
"""

from __future__ import annotations

import copy
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from simulation.road_network import RoadNetwork, TrafficLightState, VehicleRef
from simulation.vehicle_state import VehicleState
from simulation.perception_model import (
    HornEmission, PerceptionModel, VehicleSignalState,
)

logger = logging.getLogger(__name__)


# Brake lamps are driven by physical braking demand.  Separate engage/release
# thresholds and a short minimum-on time keep sensor-sized acceleration noise
# from becoming a stream of fake optical events.
_BRAKE_LIGHT_ON_MPS2 = -0.35
_BRAKE_LIGHT_OFF_MPS2 = -0.10
_BRAKE_LIGHT_MIN_ON_S = 0.4


@dataclass
class EnvironmentView:
    """Read-only physical observations used to build driver contexts."""

    segment: Any = None
    segment_id: str = ""
    travel_direction: str = "forward"
    speed_limit_kmh: float = 60.0
    segment_distance_m: float = 0.0
    vehicle_ahead: Optional[Tuple[Any, float]] = None
    traffic_light: Optional[Any] = None
    # Applicable movement lamps; a mixed unselected approach has no single
    # traffic_light. Internal evidence only, not a semantic driver wake.
    traffic_lights_by_connector: Dict[str, Any] = field(default_factory=dict)
    light_end_node: str = ""
    dist_to_light_m: float = 0.0
    road_blocked: bool = False
    observed_flow_speed_kmh: float = 60.0
    has_oncoming: bool = False
    oncoming_gap_m: float = float("inf")
    oncoming_speed_kmh: float = 0.0
    weather_condition: str = "sunny"
    is_night: bool = False
    current_lane: int = 0
    same_dir_lanes: List[Any] = field(default_factory=list)
    config: Any = None


# ── Event types ───────────────────────────────────────────────

@dataclass
class TriggerEvent:
    """Event generated during physics steps that requires agent attention.

    Event types:
      - "intersection_arrival": LLM vehicle reached intersection without waypoints
      - "collision_warning": Collision risk detected ahead
      - "traffic_light_change": Signal ahead changed color (red→green, etc.)
      - "arrived": Vehicle reached destination
      - "congestion_change": Congestion level changed on current segment
    """
    type: str
    vehicle_id: str
    tick: int
    step: int
    time_s: float = 0.0         # precise simulation time in seconds
    details: Dict = field(default_factory=dict)


@dataclass
class CollisionEvent:
    """Record of a collision between two entities.

    Supports any entity pair: vehicle-vehicle, vehicle-pedestrian, etc.
    The entity_a / entity_b fields hold entity IDs (vehicle_id or ped_id).
    """
    entity_a: str
    entity_b: str
    entity_a_type: str = "vehicle"   # "vehicle" | "pedestrian" | ...
    entity_b_type: str = "vehicle"
    location: str = ""               # segment_id or node_id
    collision_type: str = "unknown"  # "head_on" | "rear_end" | "crosswalk" | "node"
    tick: int = 0
    step: int = 0
    time_s: float = 0.0
    # Physical authority that reported the contact. Evaluation consumes this
    # provenance but never changes the physical outcome.
    physics_source: str = "sumo"



# ── Lane change result ─────────────────────────────────────────

@dataclass
class LaneChangeResult:
    """Result of a lane change attempt."""
    success: bool = False
    reason: str = ""
    new_lane: int = 0


# ── Traffic Manager ────────────────────────────────────────────

class TrafficCoordinator:
    """Coordinates decisions and public traffic state on a RoadNetwork.

    Responsibilities:
      - Track public vehicle and pedestrian state synchronized from SUMO
      - Route explicit LLM decisions to the SUMO execution backend
      - Build local observations and perception results
      - Record collision consequences reported by SUMO
      - Produce factual transition events for agent wake-up
    """

    def __init__(self, road_network: RoadNetwork, *,
                 lane_geometry_runtime=None):
        self.road_network = road_network
        self.config = road_network.config    # shorthand for traffic config
        self.vehicles: Dict[str, VehicleState] = {}
        self._move_order: List[str] = []

        # Segment → vehicle index for O(1) lookup
        self._seg_vehicles: Dict[str, Set[str]] = defaultdict(set)

        # Contacts are reported by SUMO and mirrored into this audit log.
        self._collision_log: List[CollisionEvent] = []

        # Weather/daynight state (synced from engine)
        self._current_weather: str = "sunny"
        self._current_wind_speed_mps: float = 0.0
        self._perception_environment_version: int = 0
        self._is_night: bool = False
        self._daylight_level: int = 100

        # Per-tick trigger events
        self._pending_events: List[TriggerEvent] = []

        # Physics clock (seconds) — tracks the current simulation time
        self._physics_time: float = 0.0
        self.horn_events: List[HornEmission] = []
        self.horn_event_log: List[HornEmission] = []
        self.signal_event_log: List[dict] = []
        self._brake_light_engaged_at_s: Dict[str, float] = {}
        self.perception_log: List[dict] = []
        self.perception_log_limit: int = 200_000
        self.perception_log_dropped: int = 0
        self._next_horn_event_id: int = 1
        self.perception_model = PerceptionModel(self)

        # Traffic light state tracking for LLM vehicles: {vid: signal_str}
        # Per-observer movement signal state.  The connector identity is part
        # of the key: changing from one upcoming movement to another is not a
        # colour transition of the same traffic light.
        self._prev_light_signal: Dict[str, Tuple[str, str]] = {}

        # ── Pedestrian management ──
        from simulation.pedestrian_state import PedestrianState
        self.pedestrians: Dict[str, PedestrianState] = {}
        self._ped_move_order: List[str] = []

        lane_level_path = getattr(road_network, "lane_level_path", "")
        if not lane_level_path:
            network_id = getattr(road_network, "network_id", "<unknown>")
            raise ValueError(
                f"Road network {network_id!r} has no lane-level companion "
                "map; lane-level geometry is required")
        if lane_geometry_runtime is None:
            from simulation.lane_level_runtime import LaneGeometryRuntime
            lane_geometry_runtime = LaneGeometryRuntime.load(lane_level_path)
        self._lane_geometry = lane_geometry_runtime
        logger.info("Lane-level geometry enabled: %s", lane_level_path)

    # ── Vehicle lifecycle ──────────────────────────────────────

    def register_vehicle(
        self,
        vehicle_id: str,
        start_node: str,
        destination: str = "",
        destination_name: str = "",
        start_lane: int = 0,
        auto_navigate: bool = True,
        is_llm: bool = False,
        chassis_profile: str = "sedan",
        length_m: float = 4.6,
        width_m: float = 1.9,
        max_acceleration_mps2: float = 3.0,
        max_braking_mps2: float = 6.0,
        lane_change_duration_s: float = 3.5,
        perception_profile: str = "human_driver_standard",
        perception_overrides: Optional[dict] = None,
    ) -> VehicleState:
        """Register a new vehicle and optionally start its navigation."""
        physical_values = {
            "length_m": length_m,
            "width_m": width_m,
            "max_acceleration_mps2": max_acceleration_mps2,
            "max_braking_mps2": max_braking_mps2,
            "lane_change_duration_s": lane_change_duration_s,
        }
        if any(float(value) <= 0 for value in physical_values.values()):
            raise ValueError("Vehicle chassis values must be positive")
        vs = VehicleState(
            current_node=start_node,
            vehicle_id=vehicle_id,
            current_lane=start_lane,
            is_llm=is_llm,
            control_authority="llm" if is_llm else "sumo",
            chassis_profile=chassis_profile,
            length_m=float(length_m),
            width_m=float(width_m),
            max_acceleration_mps2=float(max_acceleration_mps2),
            max_braking_mps2=float(max_braking_mps2),
            lane_change_duration_s=float(lane_change_duration_s),
            perception_profile_name=perception_profile,
            perception_overrides=dict(perception_overrides or {}),
        )
        if is_llm:
            vs.llm_control_command = {
                "target_speed_kmh": 0.0,
                "emergency_brake": False,
                "reason": "awaiting_llm_command",
            }
        if destination and auto_navigate:
            if not self._assign_lane_route(
                    vs, start_node, destination, destination_name):
                raise ValueError(
                    f"vehicle {vehicle_id!r} has no mapped SUMO route from "
                    f"{start_node!r} to {destination!r}")
        else:
            # SUMO vehicles must occupy a concrete edge even before a
            # passenger supplies a destination. Place an idle vehicle on one
            # authored outgoing lane; this is initialization, not motion.
            candidates = sorted([
                lane for lane in self._lane_geometry._lane_by_id.values()
                if lane["start_node"] == start_node
            ], key=lambda lane: (lane["index"], lane["id"]))
            lane = next((
                item for item in candidates
                if int(item["index"]) == int(start_lane)
            ), candidates[0] if candidates else None)
            if lane is None:
                raise ValueError(
                    f"vehicle {vehicle_id!r} cannot be placed at map node "
                    f"{start_node!r}: no outgoing SUMO lane")
            vs.current_node = lane["start_node"]
            vs.current_segment = lane["segment_id"]
            vs.current_lane = int(lane["index"])
            vs.current_lane_id = lane["id"]
            vs.edge_progress = 0.0

        vs._traffic_mgr = self  # back-reference for lane-change queries
        self.vehicles[vehicle_id] = vs
        self._move_order.append(vehicle_id)
        if vs.current_segment:
            self._seg_vehicles[vs.current_segment].add(vehicle_id)
        self._initialize_vehicle_pose(vs)
        self._update_road_network_position(vs)
        logger.info(f"Registered vehicle '{vehicle_id}' at {start_node}"
                     f" (llm={is_llm})")
        return vs

    def update_vehicle_signals(
        self, vehicle_id: str, values: dict, time_s: float,
    ) -> VehicleSignalState:
        """Commit module lamp state into the shared physical world."""
        vehicle = self.vehicles.get(vehicle_id)
        if vehicle is None:
            raise KeyError(f"unknown vehicle: {vehicle_id}")
        state = vehicle.signal_state
        changed = {}
        for key, value in values.items():
            if key not in state.__dataclass_fields__ or key == "updated_at_s":
                continue
            normalized = bool(value)
            if getattr(state, key) != normalized:
                setattr(state, key, normalized)
                changed[key] = normalized
        if state.hazard:
            state.left_indicator = True
            state.right_indicator = True
        state.updated_at_s = float(time_s)
        if changed:
            self._perception_environment_version += 1
            self.signal_event_log.append({
                "time_s": round(float(time_s), 6),
                "vehicle_id": vehicle_id,
                "changed": changed,
                "state": state.as_dict(),
            })
        return state

    def _update_physical_brake_light(
        self, vehicle_id: str, braking_demand_mps2: float, time_s: float,
    ) -> None:
        """Update the automatic brake lamp with physical hysteresis."""
        vehicle = self.vehicles[vehicle_id]
        is_on = vehicle.signal_state.brake_light
        if not is_on and braking_demand_mps2 <= _BRAKE_LIGHT_ON_MPS2:
            self.update_vehicle_signals(
                vehicle_id, {"brake_light": True}, time_s)
            self._brake_light_engaged_at_s[vehicle_id] = float(time_s)
            return
        if not is_on or braking_demand_mps2 < _BRAKE_LIGHT_OFF_MPS2:
            return
        engaged_at = self._brake_light_engaged_at_s.get(
            vehicle_id, float("-inf"))
        if float(time_s) - engaged_at >= _BRAKE_LIGHT_MIN_ON_S - 1e-9:
            self.update_vehicle_signals(
                vehicle_id, {"brake_light": False}, time_s)

    def emit_horn(
        self, vehicle_id: str, *, duration_s: float = 0.3,
        intensity: str = "normal", time_s: Optional[float] = None,
    ) -> HornEmission:
        """Emit a bounded world-visible acoustic event from a vehicle."""
        vehicle = self.vehicles.get(vehicle_id)
        if vehicle is None:
            raise KeyError(f"unknown vehicle: {vehicle_id}")
        duration = max(0.05, min(2.0, float(duration_s)))
        event_time = self._physics_time if time_s is None else float(time_s)
        event = HornEmission(
            event_id=f"horn-{self._next_horn_event_id:012d}",
            source_id=vehicle_id, start_time_s=event_time,
            duration_s=duration, intensity=intensity,
            source_db=self.perception_model.horn_source_db(intensity),
            pose_x_m=float(vehicle.pose_x_m),
            pose_y_m=float(vehicle.pose_y_m),
        )
        self._next_horn_event_id += 1
        self.horn_events.append(event)
        self.horn_event_log.append(event)
        # Retain a bounded recent acoustic history for audit and perception.
        self.horn_events = [
            item for item in self.horn_events
            if item.end_time_s >= event_time - 30.0]
        return event

    def _assign_lane_route(
        self, vs: VehicleState, start_node: str, destination: str,
        destination_name: str = "", current_lane_id: str = "",
    ) -> bool:
        plan = self._lane_geometry.plan_lane_route(
            start_node, destination, current_lane_id=current_lane_id)
        if not plan:
            vs.lane_route_blocked = True
            return False
        lane = self._lane_geometry._lane_by_id[plan["start_lane_id"]]
        vs.current_node = lane["start_node"]
        vs.current_segment = lane["segment_id"]
        vs.current_lane = lane["index"]
        vs.current_lane_id = lane["id"]
        vs.edge_progress = 0.0
        vs.destination_node = destination
        vs.destination_name = destination_name or destination
        vs.lane_route_actions = list(plan["actions"])
        vs.lane_route_action_index = 0
        vs.lane_route_cost = plan["cost"]
        vs.is_navigating = True
        vs.arrived = False
        vs.present_in_physics_world = True
        vs.terminal_crossing_speed_kmh = None
        vs.lane_route_blocked = False
        return True

    def _replan_lane_route(
        self, vs: VehicleState, tick: int,
        extra_blocked: Optional[Set[str]] = None,
    ) -> bool:
        blocked = set(extra_blocked or ())
        blocked.update(
            segment_id for segment_id in self.road_network.edges
            if self.road_network.has_road_closure(segment_id, tick))
        lane = self._lane_geometry._lane_by_id.get(vs.current_lane_id)
        if not lane or not vs.destination_node:
            return False
        plan = self._lane_geometry.plan_lane_route(
            lane["start_node"], vs.destination_node,
            current_lane_id=vs.current_lane_id,
            blocked_segments=blocked)
        if not plan:
            vs.lane_route_blocked = True
            if vs.route_control_authority == "llm_maneuver":
                vs.suggested_lane_route_start_lane_id = vs.current_lane_id
                vs.suggested_lane_route_goal_lane_id = ""
                vs.suggested_lane_route_actions = []
                vs.suggested_lane_route_cost = 0.0
            return False
        if vs.route_control_authority == "llm_maneuver":
            # A reroute is navigation advice only.  It must never select a
            # new intersection movement on behalf of the LLM driver.
            vs.suggested_lane_route_start_lane_id = plan["start_lane_id"]
            vs.suggested_lane_route_goal_lane_id = plan["goal_lane_id"]
            vs.suggested_lane_route_actions = copy.deepcopy(plan["actions"])
            vs.suggested_lane_route_cost = float(plan["cost"])
            vs.lane_route_blocked = False
            return True
        vs.lane_route_actions = list(plan["actions"])
        vs.lane_route_action_index = 0
        vs.lane_route_cost = plan["cost"]
        vs.lane_route_blocked = False
        return True

    def enable_llm_maneuver_authority(self, vehicle_id: str) -> dict:
        """Keep route guidance separate from one-shot steering decisions.

        Registration and deterministic scenario placement need a complete
        lane route to choose a valid initial lane.  This method is called
        after placement but before SUMO materialisation: it preserves that
        route as guidance and removes every not-yet-executed physical action.
        """
        vs = self.vehicles.get(vehicle_id)
        if vs is None:
            return {"success": False, "reason": "unknown_vehicle"}
        if not vs.is_llm:
            return {"success": False, "reason": "vehicle_is_not_llm_controlled"}
        vs.route_control_authority = "llm_maneuver"
        if (vs.destination_node
                and not self._replan_lane_route(
                    vs, round(self._physics_time))):
            return {
                "success": False,
                "reason": "no_lane_level_route_from_current_pose",
            }

        # A scenario may deliberately start a vehicle inside a connector.  It
        # is too late to choose that already-running movement, so retain only
        # its active connector until the body reaches the next lane.
        if vs.active_connector_id:
            connector = self._lane_geometry.connector_record(
                vs.active_connector_id)
            vs.lane_route_actions = ([{
                "type": "connector",
                "connector_id": connector["id"],
                "from_lane_id": connector["from_lane"],
                "to_lane_id": connector["to_lane"],
                "turn": connector["turn"],
            }] if connector is not None else [])
        else:
            vs.lane_route_actions = []
        vs.lane_route_action_index = 0
        vs.planned_connector_id = ""
        vs.planned_maneuver_source = ""
        vs.planned_from_lane_id = ""
        vs.planned_from_lane_index = -1
        vs.planned_turn = ""
        return {
            "success": True,
            "route_control_authority": vs.route_control_authority,
            "suggested_action_count": len(
                vs.suggested_lane_route_actions),
        }

    def navigation_route_preview(
        self, vehicle_id: str, destination_node: str = "",
    ) -> Optional[dict]:
        """Build route-display geometry without selecting physical motion."""
        vs = self.vehicles.get(vehicle_id)
        if vs is None or not vs.current_lane_id:
            return None
        destination = destination_node or vs.destination_node
        if not destination:
            return None
        lane = self._lane_geometry._lane_by_id.get(vs.current_lane_id)
        if lane is None:
            return None
        plan = self._lane_geometry.plan_lane_route(
            lane["start_node"], destination,
            current_lane_id=vs.current_lane_id)
        if plan is None:
            return None
        if (vs.route_control_authority == "llm_maneuver"
                and destination == vs.destination_node):
            vs.suggested_lane_route_start_lane_id = plan["start_lane_id"]
            vs.suggested_lane_route_goal_lane_id = plan["goal_lane_id"]
            vs.suggested_lane_route_actions = copy.deepcopy(plan["actions"])
            vs.suggested_lane_route_cost = float(plan["cost"])
            vs.lane_route_blocked = False
        return {
            "start_lane_id": plan["start_lane_id"],
            "goal_lane_id": plan["goal_lane_id"],
            "actions": copy.deepcopy(plan["actions"]),
            "destination_node": destination,
            "cost": float(plan["cost"]),
        }

    def _current_lane_terminates_at_destination(
        self, vehicle: VehicleState,
    ) -> bool:
        """Whether following the current lane completes the assigned trip."""
        lane = self._lane_geometry._lane_by_id.get(vehicle.current_lane_id)
        if lane is None or not vehicle.destination_node:
            return False
        end = str(lane["end_node"])
        destination = str(vehicle.destination_node)
        junction = self._lane_geometry._node_to_junction
        return junction.get(end, end) == junction.get(
            destination, destination)

    def select_vehicle_maneuver(
        self, vehicle_id: str, direction: str,
    ) -> dict:
        """Select exactly one legal connector from the current lane."""
        vs = self.vehicles.get(vehicle_id)
        if vs is None:
            return {"success": False, "reason": "unknown_vehicle"}
        if not vs.is_llm:
            return {"success": False, "reason": "vehicle_is_sumo_controlled"}
        if vs.route_control_authority != "llm_maneuver":
            return {"success": False, "reason": "llm_maneuver_authority_inactive"}
        if vs.route_failed:
            return {"success": False, "reason": "vehicle_route_failed"}
        if vs.is_crashed:
            return {"success": False, "reason": "vehicle_crashed"}
        if vs.arrived:
            return {"success": False, "reason": "vehicle_arrived"}
        if self._current_lane_terminates_at_destination(vs):
            return {
                "success": False,
                "reason": "destination_is_end_of_current_lane",
            }
        if vs.active_connector_id or (vs.planned_connector_id
                and vs.planned_maneuver_source != "default_straight"):
            return {"success": False, "reason": "maneuver_already_selected"}
        if vs.is_changing_lane:
            return {"success": False, "reason": "lane_change_in_progress"}
        normalized = str(direction or "").strip().lower().replace("-", "_")
        aliases = {
            "left": "left", "straight": "straight", "right": "right",
            "uturn": "uturn", "u_turn": "uturn",
        }
        turn = aliases.get(normalized)
        if turn is None:
            return {
                "success": False,
                "reason": "direction_must_be_left_straight_right_or_u_turn",
            }
        connectors = [
            item for item in self._lane_geometry._connectors_from.get(
                vs.current_lane_id, [])
            if str(item.get("turn", "")) == turn
        ]
        if not connectors:
            return {
                "success": False,
                "reason": "maneuver_unavailable_from_current_lane",
                "requested_direction": turn,
            }

        # Several parallel target lanes may represent the same visible turn.
        # Prefer the connector used by the displayed shortest route; otherwise
        # choose the reachable target with the lowest continuation cost.  The
        # driver still owns the direction decision.
        self._replan_lane_route(vs, round(self._physics_time))
        preferred_connector = next((
            str(action.get("connector_id", ""))
            for action in vs.suggested_lane_route_actions
            if action.get("type") == "connector"
            and action.get("from_lane_id") == vs.current_lane_id
            and action.get("turn") == turn
        ), "")

        def candidate_key(connector: dict) -> tuple:
            target_id = str(connector["to_lane"])
            target = self._lane_geometry._lane_by_id[target_id]
            continuation = self._lane_geometry.plan_lane_route(
                target["start_node"], vs.destination_node,
                current_lane_id=target_id)
            return (
                0 if connector["id"] == preferred_connector else 1,
                0 if continuation is not None else 1,
                float(continuation["cost"])
                if continuation is not None else float("inf"),
                str(connector["id"]),
            )

        connector = min(connectors, key=candidate_key)
        self._clear_authorized_maneuver(vs)
        vs.planned_maneuver_source = "explicit"
        action = {
            "type": "connector",
            "connector_id": connector["id"],
            "from_lane_id": connector["from_lane"],
            "to_lane_id": connector["to_lane"],
            "turn": connector["turn"],
        }
        vs.lane_route_actions = [action]
        vs.lane_route_action_index = 0
        vs.lane_route_blocked = False
        self._prepare_lane_transition(vs)
        if vs.planned_connector_id != connector["id"]:
            vs.lane_route_actions = []
            return {
                "success": False,
                "reason": "connector_selection_failed",
            }
        return {
            "success": True,
            "maneuver": turn,
            "connector_id": connector["id"],
            "execution": "next_0.1s_world_step",
        }

    @staticmethod
    def _clear_authorized_maneuver(vs: VehicleState) -> None:
        """Clear an unentered junction selection without changing guidance."""
        vs.lane_route_actions = []
        vs.lane_route_action_index = 0
        vs.planned_connector_id = ""
        vs.planned_maneuver_source = ""
        vs.planned_from_lane_id = ""
        vs.planned_from_lane_index = -1
        vs.planned_turn = ""

    def _prepare_default_continuation(self, vs: VehicleState) -> bool:
        """Follow only the current lane's unique straight, never route guidance.

        Returns whether a connector was newly installed. Explicit commands and
        movements already inside a junction remain authoritative.
        """
        if (vs.route_control_authority != "llm_maneuver" or vs.is_crashed
                or vs.arrived or vs.route_failed or vs.active_connector_id or vs.is_changing_lane):
            return False
        if (vs.planned_maneuver_source == "default_straight"
                and vs.planned_from_lane_id != vs.current_lane_id):
            self._clear_authorized_maneuver(vs)
        if vs.planned_connector_id or self._current_lane_terminates_at_destination(vs):
            return False
        candidates = [c for c in self._lane_geometry._connectors_from.get(
            vs.current_lane_id, []) if c.get("turn") == "straight"]
        if len(candidates) != 1:
            # No topology-derived wake or automatic braking. The driver must
            # observe the road/navigation during its normal wakes and decide.
            return False
        connector = candidates[0]
        vs.lane_route_actions = [{
            "type": "connector", "connector_id": connector["id"],
            "from_lane_id": connector["from_lane"],
            "to_lane_id": connector["to_lane"], "turn": "straight"}]
        vs.lane_route_action_index = 0
        vs.planned_maneuver_source = "default_straight"
        vs.planned_connector_id = connector["id"]
        vs.planned_from_lane_id = connector["from_lane"]
        vs.planned_from_lane_index = vs.current_lane
        vs.planned_turn = "straight"
        vs.lane_route_blocked = False
        return True

    def get_state(self, vehicle_id: str) -> Optional[VehicleState]:
        return self.vehicles.get(vehicle_id)

    def get_next_intersection_distance_m(self, vehicle_id: str) -> Optional[float]:
        """Distance from the vehicle to its current lane's next connector."""
        vehicle = self.vehicles.get(vehicle_id)
        if vehicle is None or vehicle.arrived or vehicle.route_failed:
            return None
        if vehicle.active_connector_id:
            return 0.0
        if not (vehicle.planned_connector_id
                or self._lane_geometry._connectors_from.get(
                    vehicle.current_lane_id)):
            return None
        length = self._vehicle_path_length(vehicle)
        return round(max(0.0, (1.0 - max(
            0.0, min(1.0, vehicle.edge_progress))) * length), 3)

    def get_next_intersection_exit_distance_m(
            self, vehicle_id: str) -> Optional[float]:
        """Route distance to exiting the next *planned* intersection.

        ``None`` means that the Driver has not selected a connector yet, so a
        passenger-side delayed request cannot claim that its exit event is a
        reachable point on the active route.
        """
        vehicle = self.vehicles.get(vehicle_id)
        if vehicle is None or vehicle.arrived or vehicle.route_failed:
            return None
        if vehicle.active_connector_id:
            length = self._lane_geometry.connector_length(
                vehicle.active_connector_id)
            return round(max(
                0.0,
                (1.0 - max(0.0, min(1.0, vehicle.edge_progress))) * length,
            ), 3)
        connector_id = str(vehicle.planned_connector_id or "")
        if not connector_id:
            return None
        entry = self.get_next_intersection_distance_m(vehicle_id)
        if entry is None:
            return None
        connector_length = self._lane_geometry.connector_length(connector_id)
        if connector_length <= 0.0:
            return None
        return round(float(entry) + connector_length, 3)

    def get_navigation_status(self, vehicle_id: str) -> Dict[str, Any]:
        """Return compact, non-traffic route instrumentation for one vehicle.

        This is the equivalent of an ordinary navigation display: it reports
        remaining route geometry and the next planned maneuver, but never
        exposes traffic, signal, collision-risk, or evaluator truth.
        """
        vehicle = self.vehicles.get(vehicle_id)
        if vehicle is None:
            return {"status": "unavailable"}
        if vehicle.route_failed:
            return {
                "status": "route_failed", "reason": vehicle.route_failure_reason,
                "remaining_distance_m": None, "next_maneuver": None,
            }
        if vehicle.arrived:
            return {
                "status": "arrived",
                "remaining_distance_m": 0.0,
                "next_maneuver": None,
            }
        if not vehicle.destination_node:
            return {"status": "inactive"}
        if (vehicle.lane_route_blocked or (
                vehicle.route_control_authority == "llm_maneuver"
                and not vehicle.suggested_lane_route_goal_lane_id)):
            # The remaining physical lane/one-shot turn is not a route to
            # the destination. Steering can clear lane_route_blocked, but
            # cannot make missing destination guidance valid again.
            return {
                "status": "blocked",
                "remaining_distance_m": None,
                "next_maneuver": None,
            }

        runtime = self._lane_geometry
        if vehicle.route_control_authority == "llm_maneuver":
            actions = vehicle.suggested_lane_route_actions
            action_index = 0
        else:
            actions = vehicle.lane_route_actions
            action_index = min(
                max(0, int(vehicle.lane_route_action_index)), len(actions))
        remaining_m = 0.0
        distance_cursor_m = 0.0
        next_maneuver = None

        if vehicle.active_connector_id:
            connector_length = runtime.connector_length(
                vehicle.active_connector_id)
            connector_remaining = max(
                0.0,
                (1.0 - max(0.0, min(1.0, vehicle.edge_progress)))
                * connector_length,
            )
            remaining_m += connector_remaining
            distance_cursor_m += connector_remaining
            active_action = next(
                (
                    action for action in actions
                    if action.get("type") == "connector"
                    and action.get("connector_id")
                    == vehicle.active_connector_id
                ),
                {},
            )
            connector = runtime.connector_record(
                vehicle.active_connector_id) or {}
            next_maneuver = {
                "type": "connector",
                "turn": str(active_action.get(
                    "turn", connector.get("turn", "straight"))),
                "state": "in_progress",
                "distance_m": 0.0,
            }
            target_lane_id = (
                vehicle.active_connector_to_lane_id
                or str(connector.get("to_lane", "")))
            target_length = runtime.lane_length(target_lane_id, 0.0)
            remaining_m += target_length
            distance_cursor_m += target_length
            if (action_index < len(actions)
                    and actions[action_index].get("type") == "connector"
                    and actions[action_index].get("connector_id")
                    == vehicle.active_connector_id):
                action_index += 1
        else:
            current_length = runtime.lane_length(
                vehicle.current_lane_id, self._vehicle_path_length(vehicle))
            current_remaining = max(
                0.0,
                (1.0 - max(0.0, min(1.0, vehicle.edge_progress)))
                * current_length,
            )
            remaining_m += current_remaining
            distance_cursor_m += current_remaining

        for action in actions[action_index:]:
            action_type = str(action.get("type", ""))
            if action_type == "lane_change":
                if next_maneuver is None:
                    next_maneuver = {
                        "type": "lane_change",
                        "target_lane_index": action.get(
                            "target_lane_index"),
                        "state": "upcoming",
                        "distance_m": round(distance_cursor_m, 1),
                    }
                continue
            if action_type != "connector":
                continue
            if next_maneuver is None:
                next_maneuver = {
                    "type": "connector",
                    "turn": str(action.get("turn", "straight")),
                    "state": "upcoming",
                    "distance_m": round(distance_cursor_m, 1),
                }
            connector_length = runtime.connector_length(
                str(action.get("connector_id", "")))
            target_length = runtime.lane_length(
                str(action.get("to_lane_id", "")), 0.0)
            remaining_m += connector_length + target_length
            distance_cursor_m += connector_length + target_length

        if next_maneuver is None:
            next_maneuver = {
                "type": "arrive",
                "state": "upcoming",
                "distance_m": round(remaining_m, 1),
                "completion": "cross_route_endpoint",
            }

        return {
            "status": "active",
            "remaining_distance_m": round(remaining_m, 1),
            "next_maneuver": next_maneuver,
        }

    def _vehicle_path_length(self, vs: VehicleState) -> float:
        """Authoritative metric length of the vehicle's current path."""
        if vs.active_connector_id:
            length = self._lane_geometry.connector_length(
                vs.active_connector_id)
            if length > 0:
                return length
        length = self._lane_geometry.lane_length(
            vs.current_lane_id, 0.0)
        if length > 0:
            return length
        seg = self.road_network.get_segment(vs.current_segment)
        return float(seg.distance_meters) if seg else 0.0

    def _vehicle_path_position(self, vs: VehicleState) -> float:
        return (
            max(0.0, min(1.0, vs.edge_progress))
            * self._vehicle_path_length(vs))

    def _lane_id_for_index(
        self, vs: VehicleState, lane_index: int,
    ) -> str:
        seg = self.road_network.get_segment(vs.current_segment)
        if not seg:
            return ""
        direction = self._get_travel_direction(vs, seg)
        return next((
            lane["id"] for lane in self._lane_geometry._lanes.get(
                (vs.current_segment, direction), [])
            if lane["index"] == lane_index
        ), "")

    # ── Collision records ─────────────────────────────────

    @property
    def collision_log(self) -> List[CollisionEvent]:
        return list(self._collision_log)

    # ── Mirrored state bookkeeping ─────────────────────

    def _rebuild_seg_index(self):
        """Rebuild segment→vehicles index from scratch. O(N)."""
        self._seg_vehicles.clear()
        for vid, vs in self.vehicles.items():
            if vs.current_segment and not vs.arrived and not vs.route_failed:
                self._seg_vehicles[vs.current_segment].add(vid)

    def _finalize_pending_arrivals(
        self, time_s: float,
    ) -> List[TriggerEvent]:
        """Commit terminal success only after the contact pass."""
        events: List[TriggerEvent] = []
        for vehicle in self.vehicles.values():
            if not vehicle.pending_arrival:
                continue
            vehicle.pending_arrival = False
            if vehicle.is_crashed:
                continue
            lane = self._lane_geometry._lane_by_id.get(
                vehicle.current_lane_id)
            if lane is not None:
                vehicle.current_node = lane["end_node"]
            crossing_speed_kmh = max(0.0, float(vehicle.current_speed_kmh))
            vehicle.terminal_crossing_speed_kmh = crossing_speed_kmh
            vehicle.arrived = True
            vehicle.is_navigating = False
            vehicle.present_in_physics_world = False
            vehicle.current_speed_kmh = 0.0
            vehicle.target_speed_kmh = 0.0
            vehicle.acceleration_mps2 = 0.0
            vehicle.current_segment = ""
            events.append(TriggerEvent(
                type="arrived", vehicle_id=vehicle.vehicle_id,
                tick=round(time_s), step=0, time_s=time_s,
                details={
                    "completion": "route_endpoint_crossed",
                    "terminal_crossing_speed_kmh": round(
                        crossing_speed_kmh, 3),
                    "present_in_physics_world": False,
                    "physically_stopped_at_destination": False,
                }))
        for pedestrian in self.pedestrians.values():
            if not pedestrian.pending_arrival:
                continue
            pedestrian.pending_arrival = False
            if pedestrian.is_crashed:
                continue
            pedestrian.has_arrived = True
            pedestrian.speed = 0.0
            events.append(TriggerEvent(
                type="ped_arrived", vehicle_id=pedestrian.ped_id,
                tick=round(time_s), step=0, time_s=time_s))
        return events

    def _build_env_view(self, vs: VehicleState, tick: int) -> EnvironmentView:
        """Build a read-only environment snapshot for *vs*."""
        env = EnvironmentView(config=self.config)

        # Resolve the environment from the authoritative lane-level pose.
        seg_id = vs.current_segment or ""
        seg = self.road_network.get_segment(seg_id) if seg_id else None
        lane = self._lane_geometry._lane_by_id.get(
            vs.current_lane_id) if vs.current_lane_id else None
        if lane:
            end_node = lane["end_node"]
        elif seg:
            end_node = (
                seg.to_node if seg.from_node == vs.current_node
                else seg.from_node)
        else:
            end_node = ""

        if not seg:
            return env

        env.segment = seg
        env.segment_id = seg_id
        env.speed_limit_kmh = float(seg.speed_limit)
        env.segment_distance_m = self._vehicle_path_length(vs)

        travel_dir = self._get_travel_direction(vs, seg)
        env.travel_direction = travel_dir
        env.current_lane = vs.current_lane
        env.same_dir_lanes = [l for l in seg.lane_config
                              if l.direction == travel_dir] if seg.lane_config else []

        # Vehicle ahead — search on the vehicle's ACTUAL segment,
        # not the route-derived next segment (they may differ for SUMO
        # vehicles that are still on a previous segment approaching a node)
        actual_seg_id = vs.current_segment or seg_id
        actual_seg = self.road_network.get_segment(actual_seg_id) if actual_seg_id else seg
        if actual_seg:
            actual_dir = self._get_travel_direction(vs, actual_seg)
            ahead_vs, gap_m = self._find_vehicle_ahead_m(vs, actual_seg_id, actual_dir)
        else:
            ahead_vs, gap_m = self._find_vehicle_ahead_m(vs, seg_id, travel_dir)
        if ahead_vs is not None:
            env.vehicle_ahead = (ahead_vs, gap_m)

        # Road blocked
        env.road_blocked = (
            self.road_network.has_road_closure(seg_id, tick)
            or vs.lane_route_blocked
        )

        env.observed_flow_speed_kmh = (
            self.road_network.get_segment_mean_speed_kmh(
                seg_id, exclude_vehicle=vs.vehicle_id))

        # Oncoming traffic is an observation; each driver decides whether to
        # yield and SUMO executes the resulting controls.
        if seg.lanes <= 1 and not seg.oneway:
            oncoming_vs, oncoming_gap = self._find_oncoming_vehicle_m(
                vs, actual_seg_id or seg_id, travel_dir)
            if oncoming_vs is not None:
                env.has_oncoming = True
                env.oncoming_gap_m = oncoming_gap
                env.oncoming_speed_kmh = oncoming_vs.current_speed_kmh

        # The same lane-level clock governs SUMO, rendering and evaluation.
        # Destination lanes intentionally have no next maneuver; that must not
        # send their stop evaluation back to the old node-level signal clock.
        if lane and not vs.active_connector_id:
            connectors = self._lane_geometry._connectors_from.get(
                lane["id"], [])
            for connector in connectors:
                cid = connector["id"]
                lane_signal = self._lane_geometry.signal_state(
                    cid, self._physics_time)
                if lane_signal is None:
                    continue
                env.traffic_lights_by_connector[cid] = TrafficLightState(
                    signal=lane_signal.signal,
                    remaining_seconds=int(math.ceil(
                        lane_signal.remaining_seconds)),
                    is_crosswalk_phase=lane_signal.pedestrian_green,
                )
            selected = next((c for c in connectors
                             if c["id"] == vs.planned_connector_id), None)
            if selected is not None:
                # An explicit/default selection may also be unsignalized.
                selected_light = env.traffic_lights_by_connector.get(
                    selected["id"])
                env.traffic_lights_by_connector = (
                    {selected["id"]: selected_light} if selected_light else {})
                env.traffic_light = selected_light
            elif env.traffic_lights_by_connector:
                lamps = list(env.traffic_lights_by_connector.values())
                if len({light.signal for light in lamps}) == 1:
                    env.traffic_light = min(
                        lamps, key=lambda light: light.remaining_seconds)
            if env.traffic_lights_by_connector:
                cid = next(iter(env.traffic_lights_by_connector))
                plan = self._lane_geometry._signal_plan_by_connector.get(
                    cid, {})
                env.light_end_node = plan.get("node_id", end_node)
                stop_progress = self._lane_geometry.stop_progress(
                    vs.current_lane_id)
                lane_length = self._lane_geometry._lane_lengths.get(
                    vs.current_lane_id, seg.distance_meters)
                env.dist_to_light_m = max(
                    0.0, ((stop_progress if stop_progress is not None else 1.0)
                          - vs.edge_progress) * lane_length)

        # Weather / day-night
        env.weather_condition = self._current_weather
        env.is_night = self._is_night

        return env

    def stagger_co_located_vehicles(self, *, excluded_vehicle_ids=None):
        """Space out all vehicles sharing the same starting lane.

        Vehicles on the same segment travelling the same direction are arranged
        in a queue with body clearance between each pair.  This prevents
        the pathological case where multiple entities start at identical
        edge_progress and cannot "see" each other as leaders.
        """
        from collections import defaultdict

        excluded_vehicle_ids = set(excluded_vehicle_ids or ())
        seg_groups: dict = defaultdict(list)
        for vid, vs in self.vehicles.items():
            if vid in excluded_vehicle_ids:
                continue
            if not vs.current_segment:
                continue
            seg = self.road_network.get_segment(vs.current_segment)
            if not seg:
                continue
            d = self._get_travel_direction(vs, seg)
            seg_groups[(
                vs.current_segment, d, vs.current_lane_id)].append(vs)

        for (seg_id, direction, lane_id), vehicles in seg_groups.items():
            if len(vehicles) <= 1:
                continue
            seg = self.road_network.get_segment(seg_id)
            if not seg:
                continue
            path_length = min(
                (self._vehicle_path_length(vehicle)
                 for vehicle in vehicles),
                default=0.0)
            if path_length < 1.0:
                continue
            # Sort by vehicle_id for deterministic ordering
            vehicles.sort(key=lambda v: v.vehicle_id)
            spacing_m = max(
                self.config.min_gap_m
                + first.length_m / 2.0 + second.length_m / 2.0
                for first, second in zip(vehicles, vehicles[1:]))
            required_m = spacing_m * (len(vehicles) - 1)
            if required_m + max(v.length_m for v in vehicles) > path_length:
                raise ValueError(
                    f"initial queue of {len(vehicles)} vehicles does not fit "
                    f"lane {lane_id!r} ({path_length:.1f}m)")
            gap_prog = spacing_m / path_length
            base = max(
                max(vehicle.edge_progress for vehicle in vehicles),
                0.001 + required_m / path_length)
            for i, vs in enumerate(vehicles):
                # Directed lane progress always increases from entry to exit,
                # regardless of the underlying OSM segment orientation.
                vs.edge_progress = max(0.001, base - i * gap_prog)
        for vehicle in self.vehicles.values():
            if vehicle.physical_pose_authority != "sumo":
                self._initialize_vehicle_pose(vehicle)
            self._update_road_network_position(vehicle)

    def recalculate_speeds(self, time_s: float = 0.0):
        """Maintain routes and apply explicit experiment interventions.

        This coordinator does not make background traffic decisions. SUMO
        owns background car-following, junction priority and lane changing; LLM
        targets remain unchanged until that LLM submits a new command.
        """
        self._rebuild_seg_index()
        tick = round(time_s)
        for vs in self.vehicles.values():
            if vs.is_crashed or vs.arrived or vs.route_failed:
                continue

            if ((vs.destination_node or vs.pending_uturn_connector_id)
                    and not vs.active_connector_id):
                if (vs.lane_route_action_index
                        < len(vs.lane_route_actions)):
                    next_action = vs.lane_route_actions[
                        vs.lane_route_action_index]
                    if next_action["type"] == "connector":
                        target_lane = self._lane_geometry._lane_by_id[
                            next_action["to_lane_id"]]
                        target_segment = target_lane["segment_id"]
                        if self.road_network.has_road_closure(
                                target_segment, tick):
                            if vs.route_control_authority == "llm_maneuver":
                                self._clear_authorized_maneuver(vs)
                                self._replan_lane_route(
                                    vs, tick, {target_segment})
                            else:
                                self._replan_lane_route(
                                    vs, tick, {target_segment})
                self._prepare_lane_transition(vs)

    def _build_driver_context(
        self, vs: VehicleState, env: EnvironmentView, *,
        apply_perception: bool = True,
    ):
        from simulation.driving_observations import (
            ConnectorConflictObservation, DriverContext,
            DriverVehicleObservation, PedestrianHazard)
        leader = None
        leader_gap = float("inf")
        # While on a connector, source-segment progress has been reset for
        # connector traversal and must not be compared with approach traffic.
        if not vs.active_connector_id and env.vehicle_ahead is not None:
            leader, leader_gap = env.vehicle_ahead
        connector_id = (
            vs.active_connector_id or vs.planned_connector_id)
        lane_gaps = {
            lane.lane_index: self._lane_neighbor_gap(
                vs, vs.current_segment, lane.lane_index)
            for lane in env.same_dir_lanes
            if lane.lane_index != vs.current_lane
        }
        route_target_lane = -1
        route_lane_change_required = False
        downstream_gap_m = float("inf")
        target_lane_id = (
            vs.active_connector_to_lane_id
            if vs.active_connector_id else "")
        if ((vs.destination_node or vs.pending_uturn_connector_id)
                and vs.lane_route_action_index
                < len(vs.lane_route_actions)):
            route_action = vs.lane_route_actions[
                vs.lane_route_action_index]
            if route_action["type"] == "lane_change":
                route_target_lane = route_action["target_lane_index"]
                route_lane_change_required = True
                following_connector = next((
                    item for item in vs.lane_route_actions[
                        vs.lane_route_action_index + 1:]
                    if item["type"] == "connector"
                ), None)
                if following_connector is not None:
                    connector_id = following_connector["connector_id"]
                    target_lane_id = following_connector["to_lane_id"]
            elif route_action["type"] == "connector":
                connector_id = route_action["connector_id"]
                target_lane_id = route_action["to_lane_id"]
        path_leader, path_gap = self._connector_path_leader(
            vs, connector_id)
        if path_leader is not None and path_gap < leader_gap:
            leader, leader_gap = path_leader, path_gap
        route_leader, route_gap = self._route_path_leader(vs)
        if route_leader is not None and route_gap < leader_gap:
            leader, leader_gap = route_leader, route_gap
        if target_lane_id:
            downstream_gap_m = self._downstream_lane_gap(
                vs, target_lane_id)

        connector_conflicts = []
        if connector_id:
            for item in self._connector_conflict_observations(
                    vs, connector_id):
                connector_conflicts.append(
                    ConnectorConflictObservation(**item))
        connector_conflict_occupied = any(
            item.other_in_conflict_zone
            for item in connector_conflicts)

        pedestrian_hazards = []
        if connector_id:
            for item in self._connector_pedestrian_hazards(
                    vs, connector_id):
                pedestrian_hazards.append(PedestrianHazard(**item))
        for item in self._walking_pedestrian_hazards(vs):
            pedestrian_hazards.append(PedestrianHazard(**item))

        if vs.active_connector_id:
            distance_to_lane_end_m = max(
                0.0,
                self._lane_geometry.connector_length(
                    vs.active_connector_id)
                * (1.0 - vs.edge_progress))
        else:
            lane_length = self._lane_geometry._lane_lengths.get(
                vs.current_lane_id, env.segment_distance_m)
            distance_to_lane_end_m = max(
                0.0, (1.0 - vs.edge_progress) * lane_length)
        oncoming_vehicle = None
        oncoming_gap_m = float("inf")
        if env.segment and not vs.active_connector_id:
            oncoming_vehicle, oncoming_gap_m = (
                self._find_oncoming_vehicle_m(
                    vs, vs.current_segment, env.travel_direction))
            geometric_vehicle, geometric_gap_m = (
                self._find_geometric_oncoming_vehicle_m(vs))
            if geometric_vehicle is not None and geometric_gap_m < oncoming_gap_m:
                oncoming_vehicle = geometric_vehicle
                oncoming_gap_m = geometric_gap_m
        traffic_signal = (
            env.traffic_light.signal if env.traffic_light else "")
        signal_remaining = (
            env.traffic_light.remaining_seconds
            if env.traffic_light else 0.0)
        if apply_perception:
            # Policies receive only detections available to this entity. The
            # evaluator can explicitly request authoritative truth without
            # changing what SUMO actors or LLM tools observe.
            perception = self.perception_model
            if (leader is not None and perception.detect_entity(
                    vs.vehicle_id, leader.vehicle_id,
                    claimed_distance_m=leader_gap) is None):
                leader, leader_gap = None, float("inf")
            connector_conflicts = [
                item for item in connector_conflicts
                if perception.detect_entity(
                    vs.vehicle_id, item.other_vehicle_id,
                    claimed_distance_m=max(
                        item.distance_to_conflict_m,
                        item.other_distance_to_conflict_m)) is not None
            ]
            connector_conflict_occupied = any(
                item.other_in_conflict_zone for item in connector_conflicts)
            pedestrian_hazards = [
                item for item in pedestrian_hazards
                if perception.detect_entity(
                    vs.vehicle_id, item.pedestrian_id,
                    claimed_distance_m=(
                        item.distance_to_crosswalk_m)) is not None
            ]
            if (oncoming_vehicle is not None and perception.detect_entity(
                    vs.vehicle_id, oncoming_vehicle.vehicle_id,
                    claimed_distance_m=oncoming_gap_m) is None):
                oncoming_vehicle, oncoming_gap_m = None, float("inf")
            peripheral_range = perception.visual_envelope(
                vs.vehicle_id, bearing_deg=90.0)["effective_range_m"]
            lane_gaps = {
                lane: (gap if gap <= peripheral_range else float("inf"))
                for lane, gap in lane_gaps.items()
            }
            if downstream_gap_m > peripheral_range:
                downstream_gap_m = float("inf")
            forward_range = perception.visual_envelope(
                vs.vehicle_id, bearing_deg=0.0)["effective_range_m"]
            if (not env.traffic_light
                    or env.dist_to_light_m > forward_range):
                traffic_signal = ""
                signal_remaining = 0.0
        return DriverContext(
            vehicle=DriverVehicleObservation.from_state(vs),
            speed_limit_kmh=env.speed_limit_kmh,
            time_s=self._physics_time,
            leader=(
                DriverVehicleObservation.from_state(leader)
                if leader is not None else None),
            leader_gap_m=leader_gap,
            traffic_light=traffic_signal,
            distance_to_light_m=env.dist_to_light_m,
            lane_gaps_m=lane_gaps,
            current_lane=vs.current_lane,
            route_blocked=env.road_blocked,
            signal_remaining_s=signal_remaining,
            distance_to_lane_end_m=distance_to_lane_end_m,
            route_target_lane=route_target_lane,
            route_lane_change_required=route_lane_change_required,
            connector_conflict_occupied=connector_conflict_occupied,
            downstream_gap_m=downstream_gap_m,
            connector_id=connector_id,
            on_connector=bool(vs.active_connector_id),
            distance_to_connector_end_m=(
                distance_to_lane_end_m
                if vs.active_connector_id else float("inf")),
            connector_conflicts=connector_conflicts,
            pedestrian_hazards=pedestrian_hazards,
            oncoming_vehicle=(
                DriverVehicleObservation.from_state(oncoming_vehicle)
                if oncoming_vehicle is not None else None),
            oncoming_gap_m=oncoming_gap_m,
            heard_horns=self.perception_model.heard_horns(
                vs.vehicle_id, self._physics_time),
        )

    def _distance_to_connector_s(
        self, vs: VehicleState, connector_id: str,
        connector_s_m: float,
    ) -> float:
        """Signed path distance from a vehicle to a connector control point."""
        runtime = self._lane_geometry
        if vs.active_connector_id == connector_id:
            return (
                connector_s_m
                - vs.edge_progress * runtime.connector_length(
                    connector_id))
        if vs.planned_connector_id == connector_id:
            lane_length = runtime._lane_lengths.get(
                vs.current_lane_id, 0.0)
            return (
                max(0.0, 1.0 - vs.edge_progress) * lane_length
                + connector_s_m)
        if any(
                action.get("type") == "connector"
                and action.get("connector_id") == connector_id
                for action in vs.lane_route_actions[
                    vs.lane_route_action_index:]):
            lane_length = runtime._lane_lengths.get(
                vs.current_lane_id, 0.0)
            return (
                max(0.0, 1.0 - vs.edge_progress) * lane_length
                + connector_s_m)
        return float("inf")

    def _downstream_lane_gap(
        self, vs: VehicleState, target_lane_id: str,
    ) -> float:
        """Free longitudinal space at a connector's target-lane mouth."""
        target_length = self._lane_geometry._lane_lengths.get(
            target_lane_id, 0.0)
        best = float("inf")
        for other in self.vehicles.values():
            if (other.vehicle_id == vs.vehicle_id
                    or (other.arrived or other.route_failed) or other.active_connector_id):
                continue
            if not self._vehicle_occupies_lane_id(other, target_lane_id):
                continue
            gap = (
                other.edge_progress * target_length
                - other.length_m / 2.0 - vs.length_m / 2.0)
            best = min(best, max(0.0, gap))
        return best

    def _connector_path_leader(
        self, vs: VehicleState, connector_id: str,
    ) -> Tuple[Optional[VehicleState], float]:
        """Nearest vehicle ahead on approach→connector→exit-lane path."""
        if not connector_id:
            return None, float("inf")
        runtime = self._lane_geometry
        connector_length = runtime.connector_length(connector_id)
        connector = runtime.connector_record(connector_id)
        if not connector or connector_length <= 0:
            return None, float("inf")

        if vs.active_connector_id == connector_id:
            ego_s = vs.edge_progress * connector_length
            distance_to_entry = -ego_s
        elif vs.planned_connector_id == connector_id:
            lane_length = runtime._lane_lengths.get(
                vs.current_lane_id, 0.0)
            distance_to_entry = max(
                0.0, 1.0 - vs.edge_progress) * lane_length
            ego_s = -distance_to_entry
        elif any(
                action.get("type") == "connector"
                and action.get("connector_id") == connector_id
                for action in vs.lane_route_actions[
                    vs.lane_route_action_index:]):
            lane_length = runtime._lane_lengths.get(
                vs.current_lane_id, 0.0)
            distance_to_entry = max(
                0.0, 1.0 - vs.edge_progress) * lane_length
            ego_s = -distance_to_entry
        else:
            return None, float("inf")

        leader = None
        best = float("inf")
        for other in self.vehicles.values():
            if (other.vehicle_id == vs.vehicle_id
                    or (other.arrived or other.route_failed)):
                continue
            if other.active_connector_id == connector_id:
                other_s = other.edge_progress * connector_length
                gap = (
                    other_s - ego_s
                    - other.length_m / 2.0 - vs.length_m / 2.0)
            elif (other.current_lane_id == connector["to_lane"]
                    and not other.active_connector_id):
                target_length = runtime._lane_lengths.get(
                    connector["to_lane"], 0.0)
                other_s = (
                    connector_length
                    + other.edge_progress * target_length)
                gap = (
                    other_s - ego_s
                    - other.length_m / 2.0 - vs.length_m / 2.0)
            else:
                continue
            if -0.1 <= gap < best:
                leader, best = other, max(0.0, gap)
        return leader, best

    def _route_path_leader(
        self, vs: VehicleState, horizon_m: float = 250.0,
    ) -> Tuple[Optional[VehicleState], float]:
        """Nearest entity on the ego vehicle's upcoming lane-route corridor.

        Unlike the immediate connector helper, this projects multiple route
        actions ahead on the lane corridor that the vehicle currently
        occupies. It is essential when a very short lane separates two
        junction connectors: a stopped wreck on the second connector must be
        visible before the ego exits the first one. A planned lane change is
        deliberately not expanded here: a vehicle in the target lane is a
        lane-change risk only once the ego physically occupies that lane, not
        a current longitudinal leader.
        """
        runtime = self._lane_geometry
        lane_offsets: Dict[str, float] = {}
        connector_offsets: Dict[str, float] = {}

        if vs.active_connector_id:
            connector_length = runtime.connector_length(
                vs.active_connector_id)
            connector_offsets[vs.active_connector_id] = (
                -vs.edge_progress * connector_length)
            if vs.active_connector_to_lane_id:
                lane_offsets[vs.active_connector_to_lane_id] = (
                    connector_offsets[vs.active_connector_id]
                    + connector_length)
        elif vs.current_lane_id:
            lane_length = runtime._lane_lengths.get(
                vs.current_lane_id, 0.0)
            lane_offsets[vs.current_lane_id] = (
                -vs.edge_progress * lane_length)

        for action in vs.lane_route_actions[
                vs.lane_route_action_index:]:
            if action["type"] == "lane_change":
                # Do not project an unexecuted target lane into the current
                # rear-end corridor. During an actual lane change,
                # _find_vehicle_ahead_m sees both occupied lane domains and
                # lane_change_risk separately evaluates the target-lane gap.
                break
            if action["type"] != "connector":
                continue
            connector_id = action["connector_id"]
            source_id = action["from_lane_id"]
            target_id = action["to_lane_id"]
            if connector_id not in connector_offsets:
                source_offset = lane_offsets.get(source_id)
                if source_offset is None:
                    continue
                connector_offsets[connector_id] = (
                    source_offset
                    + runtime._lane_lengths.get(source_id, 0.0))
            target_offset = (
                connector_offsets[connector_id]
                + runtime.connector_length(connector_id))
            lane_offsets.setdefault(target_id, target_offset)
            if target_offset > horizon_m:
                break

        leader = None
        best = float("inf")
        for other in self.vehicles.values():
            if other.vehicle_id == vs.vehicle_id or (other.arrived or other.route_failed):
                continue
            if other.active_connector_id:
                offset = connector_offsets.get(
                    other.active_connector_id)
                if offset is None:
                    continue
                distance = (
                    offset
                    + other.edge_progress * runtime.connector_length(
                        other.active_connector_id))
            else:
                offset = lane_offsets.get(other.current_lane_id)
                if offset is None:
                    continue
                distance = (
                    offset
                    + other.edge_progress
                    * runtime._lane_lengths.get(
                        other.current_lane_id, 0.0))
            gap = (
                distance
                - other.length_m / 2.0 - vs.length_m / 2.0)
            if -0.1 <= gap < best and gap <= horizon_m:
                leader, best = other, max(0.0, gap)
        return leader, best

    def _connector_conflict_observations(
        self, vs: VehicleState, connector_id: str,
    ) -> List[dict]:
        """Build local, path-relative arrival facts for crossing vehicles."""
        observations = []
        seen = set()
        for conflict in (
                self._lane_geometry.connector_conflict_points(
                    connector_id)):
            other_connector_id = conflict["other_connector_id"]
            ego_distance = self._distance_to_connector_s(
                vs, connector_id, conflict["self_distance_s_m"])
            if ego_distance < -max(2.0, vs.length_m / 2.0):
                continue
            for other in self.vehicles.values():
                if (other.vehicle_id == vs.vehicle_id
                        or (other.arrived or other.route_failed)
                        or (other.active_connector_id
                            != other_connector_id
                            and other.planned_connector_id
                            != other_connector_id)):
                    continue
                other_distance = self._distance_to_connector_s(
                    other, other_connector_id,
                    conflict["other_distance_s_m"])
                clearance = max(2.0, other.length_m / 2.0 + 0.8)
                if other_distance < -clearance:
                    continue
                key = (
                    other.vehicle_id, other_connector_id,
                    round(conflict["self_distance_s_m"], 3))
                if key in seen:
                    continue
                seen.add(key)
                ego_speed = max(0.0, vs.current_speed_kmh / 3.6)
                other_speed = max(
                    0.0, other.current_speed_kmh / 3.6)
                observations.append({
                    "other_vehicle_id": other.vehicle_id,
                    "other_connector_id": other_connector_id,
                    "distance_to_conflict_m": round(
                        ego_distance, 3),
                    "other_distance_to_conflict_m": round(
                        other_distance, 3),
                    "ego_ttc_s": (
                        max(0.0, ego_distance) / ego_speed
                        if ego_speed > 0.1 else float("inf")),
                    "other_ttc_s": (
                        max(0.0, other_distance) / other_speed
                        if other_speed > 0.1 else float("inf")),
                    "other_speed_kmh": other.current_speed_kmh,
                    "other_in_conflict_zone": (
                        abs(other_distance) <= clearance),
                })
        return observations

    def _connector_pedestrian_hazards(
        self, vs: VehicleState, connector_id: str,
    ) -> List[dict]:
        """Return active crosswalk occupants on this connector path."""
        hazards = []
        for conflict in (
                self._lane_geometry.connector_crosswalk_points(
                    connector_id)):
            for pedestrian in self.pedestrians.values():
                # Authored crosswalk state may exist before SUMO insertion.
                # A scheduled pedestrian is not yet a physical occupant.
                if (not pedestrian.is_spawned or pedestrian.has_arrived
                        or pedestrian.is_crashed):
                    continue
                authored_occupant = bool(
                    pedestrian.authored_crosswalk_path_xy
                    and pedestrian.authored_crosswalk_id
                    == conflict["crosswalk_id"])
                active_occupant = bool(
                    pedestrian.is_on_crosswalk
                    and pedestrian.active_crosswalk_id
                    == conflict["crosswalk_id"])
                if not (active_occupant or authored_occupant):
                    continue
                distance = self._distance_to_connector_s(
                    vs, connector_id,
                    conflict["connector_distance_s_m"])
                if distance < -max(2.0, vs.length_m / 2.0):
                    continue
                speed_mps = max(
                    0.0, vs.current_speed_kmh / 3.6)
                hazards.append({
                    "pedestrian_id": pedestrian.ped_id,
                    "crosswalk_id": conflict["crosswalk_id"],
                    "distance_to_crosswalk_m": round(distance, 3),
                    "vehicle_ttc_s": (
                        max(0.0, distance) / speed_mps
                        if speed_mps > 0.1 else float("inf")),
                    "crossing_progress": (
                        pedestrian.crossing_progress
                        if active_occupant
                        else pedestrian.authored_crosswalk_progress),
                })
        return hazards

    def _vehicle_distance_to_path(
        self, vs: VehicleState, path_xy,
    ) -> Optional[float]:
        """Distance along the immediate vehicle path to a geometric path.

        The current lane and its next connector form the local planning
        horizon.  A small negative distance is retained while the vehicle
        body still occupies the conflict point.
        """
        from simulation.lane_level_runtime import (
            _polyline_intersection_distances, _polyline_length)

        pedestrian_path = [tuple(point) for point in path_xy]
        if len(pedestrian_path) < 2:
            return None

        def hit_distance(line, progress=0.0):
            points = [tuple(point) for point in line]
            if len(points) < 2:
                return None
            hit = _polyline_intersection_distances(
                points, pedestrian_path)
            if hit is None:
                return None
            return (
                hit[1]
                - max(0.0, min(1.0, progress))
                * _polyline_length(points))

        distances = []
        if vs.active_connector_id:
            connector = self._lane_geometry._connector_by_id.get(
                vs.active_connector_id)
            if connector:
                distance = hit_distance(
                    connector.get("centerline_xy", []),
                    vs.edge_progress)
                if distance is not None:
                    distances.append(distance)
        else:
            lane = self._lane_geometry._lane_by_id.get(
                vs.current_lane_id)
            lane_remaining = 0.0
            if lane:
                lane_length = self._lane_geometry._lane_lengths.get(
                    vs.current_lane_id, 0.0)
                lane_remaining = max(
                    0.0, (1.0 - max(0.0, min(
                        1.0, vs.edge_progress))) * lane_length)
                distance = hit_distance(
                    lane.get("centerline_xy", []), vs.edge_progress)
                if distance is not None:
                    distances.append(distance)

            connector_id = vs.planned_connector_id
            if not connector_id:
                for action in vs.lane_route_actions[
                        vs.lane_route_action_index:]:
                    if action.get("type") != "connector":
                        continue
                    candidate = self._lane_geometry._connector_by_id.get(
                        action.get("connector_id", ""))
                    if (candidate is not None and (
                            not vs.current_lane_id
                            or candidate.get("from_lane")
                            == vs.current_lane_id)):
                        connector_id = candidate["id"]
                    break
            connector = self._lane_geometry._connector_by_id.get(
                connector_id)
            if connector:
                distance = hit_distance(
                    connector.get("centerline_xy", []))
                if distance is not None:
                    distances.append(lane_remaining + distance)

        rear_clearance = max(0.5, vs.length_m / 2.0)
        approaching = [
            max(0.0, distance) for distance in distances
            if distance >= -rear_clearance]
        return min(approaching) if approaching else None

    def _walking_pedestrian_hazards(
        self, vs: VehicleState,
    ) -> List[dict]:
        """Expose active unmarked crossings to the vehicle-local policy."""
        hazards = []
        for pedestrian in self.pedestrians.values():
            if (pedestrian.has_arrived or pedestrian.is_crashed
                    or not pedestrian.is_spawned
                    or not pedestrian.is_walking
                    or pedestrian.is_on_crosswalk
                    or len(pedestrian.walking_path_xy) < 2):
                continue
            distance = self._vehicle_distance_to_path(
                vs, pedestrian.walking_path_xy)
            if distance is None:
                continue
            speed_mps = max(0.0, vs.current_speed_kmh / 3.6)
            hazards.append({
                "pedestrian_id": pedestrian.ped_id,
                "crosswalk_id": f"free_path::{pedestrian.ped_id}",
                "distance_to_crosswalk_m": round(distance, 3),
                "vehicle_ttc_s": (
                    distance / speed_mps
                    if speed_mps > 0.1 else float("inf")),
                "crossing_progress": pedestrian.walking_progress,
            })
        return hazards

    def _prepare_lane_transition(self, vs: VehicleState) -> None:
        """Select the legal connector and approach lane for the next route leg."""
        vs.lane_route_blocked = False
        if (vs.destination_node or vs.pending_uturn_connector_id
                or vs.route_control_authority == "llm_maneuver"):
            while vs.lane_route_action_index < len(vs.lane_route_actions):
                action = vs.lane_route_actions[vs.lane_route_action_index]
                if action["type"] == "lane_change":
                    if vs.current_lane_id == action["to_lane_id"]:
                        vs.lane_route_action_index += 1
                        continue
                    # SUMO chooses the timing for background traffic; an LLM
                    # reaches this lane only through an explicit lane command.
                    return
                if action["from_lane_id"] != vs.current_lane_id:
                    if vs.pending_uturn_connector_id:
                        vs.lane_route_blocked = True
                    elif vs.route_control_authority == "llm_maneuver":
                        # A lane change can invalidate an authorised connector.
                        # Revoke it; never replace it with an automatically
                        # generated full route.
                        self._clear_authorized_maneuver(vs)
                        self._replan_lane_route(
                            vs, round(self._physics_time))
                    else:
                        self._replan_lane_route(
                            vs, round(self._physics_time))
                    return
                target = self._lane_geometry._lane_by_id[
                    action["to_lane_id"]]
                vs.planned_from_lane_id = action["from_lane_id"]
                vs.planned_from_lane_index = vs.current_lane
                vs.planned_connector_id = action["connector_id"]
                vs.planned_turn = action["turn"]
                return
            return
    def _apply_collision(self, vid_a: str, vid_b: str,
                         seg_id: str, collision_type: str,
                         tick: int, step: int,
                         time_s: Optional[float] = None,
                         impact_pose_a=None,
                         impact_pose_b=None):
        """Apply consequences of a vehicle-vehicle collision.

        Mark both driving tasks terminal and request a stop. SUMO remains
        authoritative for actual motion, including deceleration after impact;
        recording a zero target must not overwrite measured speed/acceleration.
        Bodies remain obstacles until an explicit removal mechanism applies.
        """
        a = self.vehicles.get(vid_a)
        b = self.vehicles.get(vid_b)
        if not a or not b:
            return

        # Collision consequences are independent of controller type.
        a.crash_pose = impact_pose_a
        a.is_crashed = True
        a.pending_arrival = False
        a.target_speed_kmh = 0.0
        b.crash_pose = impact_pose_b
        b.is_crashed = True
        b.pending_arrival = False
        b.target_speed_kmh = 0.0
        logger.warning(
            f"Collision ({collision_type}): "
            f"{vid_a} vs {vid_b} on {seg_id} at tick={tick} step={step}")

        # Emit per-vehicle 'crashed' trigger events so the collision is visible
        # to the engine message layer (-> "[Traffic] Collision with ...") and
        # countable by the regression V_crash miner. Without this, the hard path
        # set is_crashed but emitted NO event, so V_crash was structurally 0
        # (this also fixes that latent gap for LLM segment crashes).
        for vid_self, vid_other in ((vid_a, vid_b), (vid_b, vid_a)):
            self._pending_events.append(TriggerEvent(
                type="crashed",
                vehicle_id=vid_self,
                tick=tick, step=step,
                time_s=(
                    self._physics_time if time_s is None else time_s),
                details={
                    "other_vehicle": vid_other,
                    "location": seg_id,
                    "collision_type": collision_type,
                },
            ))

    def _apply_entity_collision(
        self,
        a_id: str, a_type: str,
        b_id: str, b_type: str,
        location: str, collision_type: str,
        tick: int, step: int,
        time_s: Optional[float] = None,
        impact_pose_a=None,
        impact_pose_b=None,
    ):
        """Apply consequences of a generic entity collision.

        Consequences are independent of controller type: crashed vehicles
        receive a stop target without replacing SUMO-measured motion, and
        crashed pedestrians retain their existing stop behavior.
        """
        # Apply the same terminal/stop request to every entity pair.
        if a_type == "vehicle":
            vs = self.vehicles.get(a_id)
            if vs:
                vs.crash_pose = impact_pose_a
                vs.is_crashed = True
                vs.pending_arrival = False
                vs.target_speed_kmh = 0.0
        elif a_type == "pedestrian":
            ps = self.pedestrians.get(a_id)
            if ps:
                ps.is_crashed = True
                ps.crash_position_xy = (
                    list(impact_pose_a[:2])
                    if impact_pose_a is not None else None)
                ps.pending_arrival = False
                ps.speed = 0.0
                ps.is_on_crosswalk = False
                ps.is_walking = False
                ps.is_waiting = False
        if b_type == "vehicle":
            vs = self.vehicles.get(b_id)
            if vs:
                vs.crash_pose = impact_pose_b
                vs.is_crashed = True
                vs.pending_arrival = False
                vs.target_speed_kmh = 0.0
        elif b_type == "pedestrian":
            ps = self.pedestrians.get(b_id)
            if ps:
                ps.is_crashed = True
                ps.crash_position_xy = (
                    list(impact_pose_b[:2])
                    if impact_pose_b is not None else None)
                ps.pending_arrival = False
                ps.speed = 0.0
                ps.is_on_crosswalk = False
                ps.is_walking = False
                ps.is_waiting = False
        # Generate trigger events for both entities
        self._pending_events.append(TriggerEvent(
            type="collision",
            vehicle_id=a_id,
            tick=tick, step=step,
            time_s=(
                self._physics_time if time_s is None else time_s),
            details={
                "other_entity": b_id,
                "other_type": b_type,
                "collision_type": collision_type,
                "location": location,
            },
        ))
        self._pending_events.append(TriggerEvent(
            type="collision",
            vehicle_id=b_id,
            tick=tick, step=step,
            time_s=(
                self._physics_time if time_s is None else time_s),
            details={
                "other_entity": a_id,
                "other_type": a_type,
                "collision_type": collision_type,
                "location": location,
            },
        ))

        logger.warning(
            f"Collision ({collision_type}): "
            f"{a_type} '{a_id}' vs {b_type} '{b_id}' "
            f"at {location} tick={tick} step={step}"
        )

    def _get_travel_direction(self, vs: VehicleState, seg) -> str:
        """Determine if vehicle travels forward or backward on segment."""
        lane = self._lane_geometry._lane_by_id.get(vs.current_lane_id)
        if lane is not None:
            return str(lane["direction"])
        return "forward" if seg.from_node == vs.current_node else "backward"

    # ── Gap / avoidance helpers ───────────────────────────────

    def _find_vehicle_ahead_m(
        self, vs: VehicleState, seg_id: str, my_dir: str,
    ) -> Tuple[Optional[VehicleState], float]:
        """Find nearest same-direction vehicle ahead, return (state, gap_meters).

        Skips arrived and crashed vehicles. Returns (None, inf) if no
        vehicle ahead on this segment.
        """
        seg = self.road_network.get_segment(seg_id)
        if not seg:
            return None, float('inf')

        my_pos_m = self._vehicle_path_position(vs)
        best_vs = None
        best_gap = float('inf')

        # Use segment index for O(k) lookup instead of O(N)
        for other_id in self._seg_vehicles.get(seg_id, ()):
            if other_id == vs.vehicle_id:
                continue
            other_vs = self.vehicles[other_id]
            if (other_vs.arrived or other_vs.route_failed):
                continue

            # Crashed vehicles remain physical obstacles only on the lanes
            # occupied by their latest SUMO lane state.
            if other_vs.is_crashed:
                if not self._vehicle_occupies_lane_id(
                        other_vs, vs.current_lane_id):
                    continue
                other_dir = self._get_travel_direction(other_vs, seg)
                if other_dir == my_dir:
                    center_gap = (
                        self._vehicle_path_position(other_vs) - my_pos_m)
                else:
                    my_length = self._vehicle_path_length(vs)
                    other_length = self._vehicle_path_length(other_vs)
                    my_abs = (
                        my_pos_m if my_dir == "forward"
                        else my_length - my_pos_m)
                    other_path_pos = self._vehicle_path_position(other_vs)
                    other_abs = (
                        other_path_pos if other_dir == "forward"
                        else other_length - other_path_pos)
                    center_gap = (
                        other_abs - my_abs if my_dir == "forward"
                        else my_abs - other_abs)
                if center_gap <= 0:
                    continue
                gap = (
                    center_gap
                    - other_vs.length_m / 2.0 - vs.length_m / 2.0)
                if gap < best_gap:
                    best_gap = max(0.0, gap)
                    best_vs = other_vs
                continue

            # A lane-changing vehicle physically occupies both its source
            # and target lanes.  Treat those lane domains as occupied for the
            # complete maneuver so followers react before the rectangles
            # actually overlap.
            if not (
                    self._occupied_lane_indexes(vs)
                    & self._occupied_lane_indexes(other_vs)):
                continue

            other_dir = self._get_travel_direction(other_vs, seg)
            if other_dir != my_dir:
                continue

            other_pos_m = self._vehicle_path_position(other_vs)
            if other_pos_m <= my_pos_m:
                continue
            gap = (
                other_pos_m - my_pos_m
                - other_vs.length_m / 2.0 - vs.length_m / 2.0)

            if gap < best_gap:
                best_gap = max(0.0, gap)
                best_vs = other_vs

        return best_vs, best_gap

    @staticmethod
    def _occupied_lane_indexes(vs: VehicleState) -> set:
        """Lane indexes touched by the latest SUMO lane-change state."""
        lanes = {vs.current_lane}
        if vs.is_changing_lane and vs.target_lane >= 0:
            lanes.add(vs.target_lane)
        return lanes

    def _vehicle_occupies_lane_id(
        self, vehicle: VehicleState, lane_id: str,
    ) -> bool:
        """Interpret a synchronized SUMO lane state for local perception."""
        if not lane_id or vehicle.active_connector_id:
            return False
        if vehicle.current_lane_id == lane_id:
            return True
        requested = self._lane_geometry._lane_by_id.get(lane_id)
        current = self._lane_geometry._lane_by_id.get(
            vehicle.current_lane_id)
        if (requested and current
                and requested.get("shared_bidirectional")
                and current.get("shared_bidirectional")
                and requested.get("segment_id") == current.get("segment_id")):
            return True
        if vehicle.is_changing_lane and vehicle.target_lane >= 0:
            return self._lane_id_for_index(
                vehicle, vehicle.target_lane) == lane_id
        return False

    def _find_oncoming_vehicle_m(
        self, vs: VehicleState, seg_id: str, my_dir: str,
    ) -> Tuple[Optional[VehicleState], float]:
        """Find nearest oncoming vehicle on segment, return (state, gap_meters).

        Gap is the physical distance between the two vehicles.
        edge_progress is direction-relative (0=entry, 1=exit), so to get
        physical positions on the segment we convert to absolute coords
        measured from from_node:
          forward vehicle: abs_pos = edge_progress
          backward vehicle: abs_pos = 1.0 - edge_progress

        On multi-lane segments with separate directional lanes, oncoming
        vehicles are in a different lane and pose no threat — skip detection.
        """
        seg = self.road_network.get_segment(seg_id)
        if not seg:
            return None, float('inf')

        # Multi-lane segments with dedicated forward/backward lanes:
        # oncoming vehicles are in their own lane, no conflict.
        if (seg.lanes > 1 and seg.lane_config
                and len(seg.forward_lanes()) >= 1
                and len(seg.backward_lanes()) >= 1):
            return None, float('inf')

        path_length = self._vehicle_path_length(vs)
        # Absolute position measured from the base segment's from-node.
        if my_dir == "forward":
            my_abs = vs.edge_progress * path_length
        else:
            my_abs = (1.0 - vs.edge_progress) * path_length

        best_vs = None
        best_gap = float('inf')

        for other_id in self._seg_vehicles.get(seg_id, ()):
            if other_id == vs.vehicle_id:
                continue
            other_vs = self.vehicles[other_id]
            if (other_vs.arrived or other_vs.route_failed) or other_vs.is_crashed:
                continue
            other_dir = self._get_travel_direction(other_vs, seg)
            if other_dir == my_dir:
                continue  # same direction, not oncoming

            other_length = self._vehicle_path_length(other_vs)
            if other_dir == "forward":
                other_abs = other_vs.edge_progress * other_length
            else:
                other_abs = (1.0 - other_vs.edge_progress) * other_length

            gap = abs(other_abs - my_abs)
            if gap < best_gap:
                best_gap = gap
                best_vs = other_vs

        return best_vs, best_gap

    def _find_geometric_oncoming_vehicle_m(
        self, vs: VehicleState, horizon_m: float = 100.0,
    ) -> Tuple[Optional[VehicleState], float]:
        """Detect an opposing body in the same physical corridor.

        Imported road data can split two directions into different logical
        segment IDs even when their lane centerlines share one narrow physical
        corridor.  Segment-only lookup then misses a real head-on hazard.  The
        check remains vehicle-local: it uses current poses, headings and body
        widths, and does not reserve or centrally coordinate the road.
        """
        if vs.pose_x_m is None or vs.pose_y_m is None:
            return None, float("inf")
        forward_x = math.cos(vs.yaw_rad)
        forward_y = math.sin(vs.yaw_rad)
        left_x, left_y = -forward_y, forward_x
        best = None
        best_gap = float("inf")
        for other in self.vehicles.values():
            if (other.vehicle_id == vs.vehicle_id or (other.arrived or other.route_failed)
                    or other.is_crashed or other.active_connector_id
                    or other.pose_x_m is None or other.pose_y_m is None
                    or other.z_level != vs.z_level):
                continue
            heading_alignment = math.cos(other.yaw_rad - vs.yaw_rad)
            if heading_alignment > -0.70:
                continue
            dx = other.pose_x_m - vs.pose_x_m
            dy = other.pose_y_m - vs.pose_y_m
            longitudinal = dx * forward_x + dy * forward_y
            if longitudinal <= 0.0 or longitudinal > horizon_m:
                continue
            lateral = abs(dx * left_x + dy * left_y)
            shared_width = (vs.width_m + other.width_m) / 2.0 + 0.6
            if lateral > shared_width:
                continue
            gap = max(
                0.0,
                longitudinal - (vs.length_m + other.length_m) / 2.0)
            if gap < best_gap:
                best, best_gap = other, gap
        return best, best_gap

    # ── LLM driving commands ──────────────────────────────────

    def set_vehicle_speed(
        self, vehicle_id: str, speed_kmh: float, *, reason: str = "",
        acceleration_mps2: Optional[float] = None,
        deceleration_mps2: Optional[float] = None,
    ) -> dict:
        """Set desired speed for an LLM vehicle."""
        vs = self.vehicles.get(vehicle_id)
        if not vs:
            return {"success": False, "reason": "unknown_vehicle"}
        if not vs.is_llm:
            return {"success": False, "reason": "vehicle_is_sumo_controlled"}
        if vs.route_failed:
            return {"success": False, "reason": "vehicle_route_failed"}
        if vs.is_crashed:
            return {"success": False, "reason": "vehicle_crashed"}
        try:
            requested_speed_kmh = max(0.0, float(speed_kmh))
            accel_limit = (
                vs.max_acceleration_mps2 if acceleration_mps2 is None
                else float(acceleration_mps2))
            decel_limit = (
                vs.max_braking_mps2 if deceleration_mps2 is None
                else float(deceleration_mps2))
            if accel_limit <= 0.0 or decel_limit <= 0.0:
                raise ValueError("acceleration limits must be positive")
        except (TypeError, ValueError) as exc:
            return {"success": False, "reason": str(exc)}
        vs.control_acceleration_limit_mps2 = min(
            vs.max_acceleration_mps2, accel_limit)
        vs.control_deceleration_limit_mps2 = min(
            vs.max_braking_mps2, decel_limit)
        vs.target_speed_kmh = requested_speed_kmh
        vs.desired_speed_kmh = requested_speed_kmh
        vs.is_stopped = (requested_speed_kmh <= 0)
        vs.llm_control_command = {
            "target_speed_kmh": requested_speed_kmh,
            "emergency_brake": False,
            "reason": reason or "llm_set_speed",
            "acceleration_limit_mps2": vs.control_acceleration_limit_mps2,
            "deceleration_limit_mps2": vs.control_deceleration_limit_mps2,
        }
        return {
            "success": True,
            "accepted_target_speed_kmh": vs.desired_speed_kmh,
            "current_speed_kmh": vs.current_speed_kmh,
        }

    def emergency_stop_vehicle(
        self, vehicle_id: str, *, reason: str = "",
    ) -> dict:
        """Apply the vehicle's emergency braking envelope continuously."""
        vs = self.vehicles.get(vehicle_id)
        if not vs:
            return {"success": False, "reason": "unknown_vehicle"}
        if not vs.is_llm:
            return {"success": False, "reason": "vehicle_is_not_llm_controlled"}
        if vs.route_failed:
            return {"success": False, "reason": "vehicle_route_failed"}
        if vs.is_crashed:
            return {"success": False, "reason": "vehicle_crashed"}
        vs.control_deceleration_limit_mps2 = vs.max_braking_mps2
        vs.target_speed_kmh = 0.0
        vs.llm_control_command = {
            "target_speed_kmh": 0.0,
            "emergency_brake": True,
            "reason": reason or "llm_emergency_stop",
            "deceleration_limit_mps2": vs.max_braking_mps2,
        }
        vs.desired_speed_kmh = 0.0
        vs.is_stopped = True
        return {
            "success": True,
            "target_speed_kmh": 0.0,
            "emergency_brake": True,
            "execution": "continuous_emergency_braking",
        }

    def set_vehicle_destination(self, vehicle_id: str, target_node: str,
                                speed_kmh: float = -1) -> dict:
        """Set a destination without bypassing LLM junction authority."""
        vs = self.vehicles.get(vehicle_id)
        if not vs:
            return {"success": False, "reason": "unknown_vehicle"}
        if not vs.is_llm:
            return {"success": False, "reason": "vehicle_is_sumo_controlled"}
        if vs.route_failed:
            return {"success": False, "reason": "vehicle_route_failed"}
        if vs.is_crashed:
            return {"success": False, "reason": "vehicle_crashed"}
        if (vs.route_control_authority == "llm_maneuver"
                and vs.active_connector_id):
            return {
                "success": False,
                "reason": "cannot_change_destination_on_connector",
            }

        if target_node not in self.road_network.nodes:
            return {
                "success": False,
                "reason": f"unknown_destination_node: {target_node}",
            }
        previous = copy.deepcopy(vs)
        vs.destination_node = target_node
        vs.destination_name = target_node
        if not self._replan_lane_route(
                vs, round(self._physics_time)):
            vs.__dict__.update(previous.__dict__)
            return {
                "success": False,
                "reason": "no_lane_level_route_from_current_pose",
            }
        if vs.route_control_authority == "llm_maneuver":
            self._clear_authorized_maneuver(vs)
        else:
            self._prepare_lane_transition(vs)
        vs.is_stopped = False
        if speed_kmh >= 0:
            seg = self.road_network.get_segment(vs.current_segment)
            limit = float(seg.speed_limit) if seg else 60.0
            vs.desired_speed_kmh = min(speed_kmh, limit)
        return {
            "success": True,
            "destination_node": target_node,
            "suggested_action_count": len(
                vs.suggested_lane_route_actions
                if vs.route_control_authority == "llm_maneuver"
                else vs.lane_route_actions),
        }

    def u_turn_vehicle(self, vehicle_id: str) -> dict:
        """Schedule a physical U-turn at the next junction.

        The command never flips progress, heading, or lane instantaneously.
        It installs ordinary adjacent-lane actions followed by a map-defined
        curved U-turn connector; the fixed-step dynamics execute that path.
        """
        vs = self.vehicles.get(vehicle_id)
        if not vs:
            return {"success": False, "reason": "unknown_vehicle"}
        if not vs.is_llm:
            return {"success": False, "reason": "vehicle_is_sumo_controlled"}
        if vs.route_failed:
            return {"success": False, "reason": "vehicle_route_failed"}
        if vs.is_crashed:
            return {"success": False, "reason": "vehicle_crashed"}
        if vs.route_control_authority == "llm_maneuver":
            return self.select_vehicle_maneuver(vehicle_id, "uturn")

        if not vs.current_segment or not vs.current_lane_id:
            return {"success": False, "reason": "not_on_segment"}
        if (vs.active_connector_id or vs.is_changing_lane
                or vs.pending_uturn_connector_id):
            return {"success": False, "reason": "maneuver_in_progress"}
        plan = self._lane_geometry.uturn_plan(vs.current_lane_id)
        if not plan:
            return {
                "success": False,
                "reason": "no_legal_uturn_connector_at_next_junction",
            }
        destination = vs.destination_node
        vs.lane_route_actions = list(plan["actions"])
        vs.lane_route_action_index = 0
        vs.pending_uturn_connector_id = plan["connector_id"]
        vs.lane_route_blocked = False
        self._prepare_lane_transition(vs)
        return {
            "success": True,
            "status": "scheduled",
            "connector_id": plan["connector_id"],
            "target_lane_id": plan["target_lane_id"],
            "lane_changes_required": sum(
                action["type"] == "lane_change"
                for action in plan["actions"]),
            "physical_note": (
                "The vehicle will reach the junction, traverse the curved "
                "U-turn connector continuously, then replan."),
        }

    def get_directions_at(self, vehicle_id: str, perception_range_m: float = None) -> dict:
        """Get available directions and nearby vehicles ahead for an LLM vehicle.

        Returns intersection directions with traffic stats, plus vehicles ahead
        within ``perception_range_m`` using interpolated real positions.
        """
        if perception_range_m is None:
            perception_range_m = self.config.perception_range_m
        vs = self.vehicles.get(vehicle_id)
        if not vs:
            return {"current_node": "", "available_directions": [],
                    "vehicles_ahead": []}

        # Route-relative perception uses the same lane/connector geometry as
        # the driver policy and wake monitor.
        context = self._build_driver_context(
            vs, self._build_env_view(vs, round(self._physics_time)))
        vehicles_ahead = []
        if (context.leader is not None
                and context.leader_gap_m <= perception_range_m):
            vehicles_ahead.append({
                "vehicle_id": context.leader.vehicle_id,
                "distance_m": round(context.leader_gap_m, 1),
                "speed_kmh": round(
                    context.leader.current_speed_kmh, 1),
                "lane": context.leader.current_lane,
                "risk": (
                    "stationary_obstacle"
                    if context.leader.is_crashed else "rear_end"),
                "is_crashed": context.leader.is_crashed,
                "path_scope": "planned_lane_route",
            })
        if (context.oncoming_vehicle is not None
                and context.oncoming_gap_m <= perception_range_m
                and all(
                    item["vehicle_id"]
                    != context.oncoming_vehicle.vehicle_id
                    for item in vehicles_ahead)):
            vehicles_ahead.append({
                "vehicle_id": context.oncoming_vehicle.vehicle_id,
                "distance_m": round(context.oncoming_gap_m, 1),
                "speed_kmh": round(
                    context.oncoming_vehicle.current_speed_kmh, 1),
                "lane": context.oncoming_vehicle.current_lane,
                "risk": "head_on",
                "is_crashed": context.oncoming_vehicle.is_crashed,
                "path_scope": "current_lane",
            })
        vehicles_ahead.sort(key=lambda item: item["distance_m"])

        # Legal directions are the connectors reachable from the current lane.
        directions = []
        source_lane_id = (
            vs.active_connector_to_lane_id
            if vs.active_connector_id else vs.current_lane_id)
        source_lane = self._lane_geometry._lane_by_id.get(source_lane_id)
        connectors = self._lane_geometry._connectors_from.get(
            source_lane_id, [])
        for connector in connectors:
            target_lane = self._lane_geometry._lane_by_id.get(
                connector["to_lane"])
            if target_lane is None:
                continue
            seg_id = target_lane["segment_id"]
            seg = self.road_network.get_segment(seg_id)
            if not seg:
                continue
            vcount = self.road_network.get_segment_vehicle_count(seg_id)
            flow_ratio = self.road_network.get_segment_flow_ratio(seg_id)
            observed_speed = (
                self.road_network.get_segment_mean_speed_kmh(seg_id))
            blocked = self.road_network.has_road_closure(
                seg_id, round(self._physics_time))

            # Vehicles on this adjacent segment within perception range
            dir_vehicles = self._get_vehicles_on_segment(
                vs, seg_id, seg, perception_range_m)

            directions.append({
                "target_node": target_lane["end_node"],
                "target_lane_id": target_lane["id"],
                "connector_id": connector["id"],
                "turn": connector.get("turn", ""),
                "road_name": seg.name,
                "road_type": seg.road_type,
                "distance_meters": round(seg.distance_meters, 1),
                "speed_limit": seg.speed_limit,
                "lanes": seg.lanes,
                "oneway": seg.oneway,
                "vehicle_count": vcount,
                "flow_ratio": round(flow_ratio, 3),
                "observed_speed_kmh": round(observed_speed, 1),
                "blocked": blocked,
                "vehicles": dir_vehicles,
            })

        return {
            "current_node": (
                source_lane["end_node"] if source_lane
                else vs.current_node),
            "available_directions": directions,
            "vehicles_ahead": vehicles_ahead,
        }

    # ── Perception helpers ────────────────────────────────────

    def _get_vehicles_on_segment(
        self, ego: VehicleState, seg_id: str, seg, range_m: float = None,
    ) -> List[Dict]:
        """Get vehicles on an adjacent segment within range of ego's node.

        Distance = ego's remaining distance on current segment +
                   vehicle's position on the adjacent segment.
        """
        if range_m is None:
            range_m = self.config.perception_range_m
        if not seg or seg.distance_meters <= 0:
            return []

        # How far is ego from the shared node (intersection)?
        if ego.current_segment:
            ego_seg = self.road_network.get_segment(ego.current_segment)
            if ego_seg:
                ego_to_node_m = (
                    (1.0 - ego.edge_progress)
                    * self._vehicle_path_length(ego))
            else:
                ego_to_node_m = 0.0
        else:
            ego_to_node_m = 0.0  # ego is at the intersection

        result = []
        for vid, other in self.vehicles.items():
            if vid == ego.vehicle_id or other.current_segment != seg_id:
                continue
            if other.is_crashed or (other.arrived or other.route_failed):
                continue

            # Distance from the shared node along this segment
            other_path_length = self._vehicle_path_length(other)
            other_pos_m = self._vehicle_path_position(other)
            other_dir = self._get_travel_direction(other, seg)
            entry_node = (
                seg.from_node if other_dir == "forward" else seg.to_node)
            # Directed progress always starts at the lane entry.
            shared_node = ego.current_node
            if entry_node == shared_node:
                other_from_node_m = other_pos_m
            else:
                other_from_node_m = other_path_length - other_pos_m

            total_dist = ego_to_node_m + other_from_node_m
            if total_dist > range_m:
                continue

            result.append({
                "vehicle_id": vid,
                "distance_m": round(total_dist, 1),
                "speed_kmh": round(other.current_speed_kmh, 1),
                "lane": other.current_lane,
            })

        result.sort(key=lambda x: x["distance_m"])
        return result

    # ── Lane change ────────────────────────────────────────────

    def validate_lane_change(self, vehicle_id: str,
                             target_lane: int) -> LaneChangeResult:
        """Validate local command preconditions without changing state."""
        vs = self.vehicles.get(vehicle_id)
        if vs is None:
            return LaneChangeResult(reason="unknown_vehicle")
        if vs.route_failed:
            return LaneChangeResult(reason="vehicle_route_failed")
        if vs.is_crashed or vs.arrived:
            return LaneChangeResult(reason="vehicle_terminal")
        if not vs.is_llm:
            return LaneChangeResult(reason="vehicle_is_sumo_controlled")
        if (vs.route_control_authority == "llm_maneuver"
                and (vs.active_connector_id or (
                    vs.planned_maneuver_source != "default_straight"
                    and (vs.planned_connector_id or vs.lane_route_actions)))):
            return LaneChangeResult(reason="maneuver_already_selected")

        seg = self.road_network.get_segment(vs.current_segment)
        if seg is None:
            return LaneChangeResult(reason="not_on_segment")

        if target_lane < 0 or target_lane >= seg.lanes:
            return LaneChangeResult(reason="no_such_lane")

        # The public command surface permits adjacent lane changes only.
        if abs(target_lane - vs.current_lane) != 1:
            return LaneChangeResult(reason="non_adjacent_lane")

        if seg.lane_config:
            current_dir = None
            target_dir = None
            for lc in seg.lane_config:
                if lc.lane_index == vs.current_lane:
                    current_dir = lc.direction
                if lc.lane_index == target_lane:
                    target_dir = lc.direction
            if current_dir and target_dir and current_dir != target_dir:
                return LaneChangeResult(reason="wrong_direction")

        if vs.is_changing_lane:
            return LaneChangeResult(reason="lane_change_in_progress")
        return LaneChangeResult(success=True, reason="accepted", new_lane=target_lane)

    def change_lane(self, vehicle_id: str,
                    target_lane: int) -> LaneChangeResult:
        """Submit a locally chosen adjacent-lane maneuver to SUMO."""
        result = self.validate_lane_change(vehicle_id, target_lane)
        if not result.success:
            return result
        vs = self.vehicles[vehicle_id]
        if vs.planned_maneuver_source == "default_straight":
            self._clear_authorized_maneuver(vs)
        vs.target_lane = target_lane
        vs.lane_change_progress = 0.0
        vs.lateral_offset_m = 0.0
        vs.lateral_speed_mps = 0.0
        vs.is_changing_lane = True
        if vs.lane_change_request_outcome == "pending":
            # A new accepted request supersedes an expired maneuver which
            # was still physically settling; never attribute it twice.
            vs.lane_change_uncompleted += 1
        vs.lane_change_attempts += 1
        vs.lane_change_request_outcome = "pending"
        return result

    def _lane_neighbor_gap(self, ego: VehicleState, segment_id: str,
                           lane_index: int) -> float:
        """Distance to the nearest vehicle on (segment, lane) from ego.

        The LLM perception layer uses this value to compare adjacent-lane clearance.
        """
        seg = self.road_network.get_segment(segment_id)
        if seg is None:
            return float('inf')
        ego_pos_m = self._vehicle_path_position(ego)
        lane_id = self._lane_id_for_index(ego, lane_index)
        best = float('inf')
        for vid, other in self.vehicles.items():
            if vid == ego.vehicle_id:
                continue
            if other.current_segment != segment_id:
                continue
            if (other.arrived or other.route_failed):
                continue
            if (lane_id and not self._vehicle_occupies_lane_id(
                    other, lane_id)):
                continue
            d = (
                abs(self._vehicle_path_position(other) - ego_pos_m)
                - other.length_m / 2.0 - ego.length_m / 2.0)
            if d < best:
                best = max(0.0, d)
        return best

    def _adjacent_lane_follower(
        self, ego: VehicleState, segment_id: str, lane_index: int,
    ) -> Tuple[Optional[VehicleState], float, float]:
        """Nearest vehicle BEHIND ego's longitudinal position on
        (segment, lane), travelling the SAME direction as ego.

        Returns (follower_vs, gap_m, closing_speed_ms):
          - gap_m > 0    : follower is behind ego (inf if none).
          - closing_speed_ms > 0 : follower is faster than ego (gap shrinking),
                                   so TTC = gap_m / closing_speed_ms is finite.
        Mirrors _find_vehicle_ahead_m's direction frame but keeps FOLLOWERS
        (behind) instead of leaders. Used for TTC-based cut-in acceptance:
        the cut-in is into the gap *ahead of this follower*.
        """
        seg = self.road_network.get_segment(segment_id)
        if seg is None:
            return None, float('inf'), 0.0

        ego_dir = self._get_travel_direction(ego, seg)
        ego_pos_m = self._vehicle_path_position(ego)
        best_vs = None
        best_gap = float('inf')

        for other_id in self._seg_vehicles.get(segment_id, ()):
            if other_id == ego.vehicle_id:
                continue
            other = self.vehicles[other_id]
            if (other.arrived or other.route_failed) or other.is_crashed:
                continue
            if lane_index not in self._occupied_lane_indexes(other):
                continue
            if self._get_travel_direction(other, seg) != ego_dir:
                continue

            other_pos_m = self._vehicle_path_position(other)
            # Keep only followers (behind ego in the travel-direction frame).
            if other_pos_m >= ego_pos_m:
                continue
            gap = (
                ego_pos_m - other_pos_m
                - other.length_m / 2.0 - ego.length_m / 2.0)

            if gap < best_gap:
                best_gap = max(0.0, gap)
                best_vs = other

        if best_vs is None:
            return None, float('inf'), 0.0
        # Positive closing = follower faster than ego ⇒ the gap shrinks.
        closing_ms = (best_vs.current_speed_kmh - ego.current_speed_kmh) / 3.6
        return best_vs, best_gap, closing_ms

    # ── Perception queries ─────────────────────────────────────

    def get_vehicle_movement_signal(
        self,
        vehicle_id: str,
        *,
        node_id: str = "",
        time_s: Optional[float] = None,
    ) -> Optional[Dict]:
        """Return the signal governing this vehicle's route movement.

        Node-level schedules are insufficient at a multi-movement junction:
        two connectors at the same node can legally show different colours.
        This resolver uses the same connector-level signal plan as physical
        stop-line enforcement, so perception and adjudication cannot disagree.
        """
        vs = self.vehicles.get(vehicle_id)
        if vs is None:
            return None
        connector_id = (
            vs.active_connector_id or vs.planned_connector_id)
        if not connector_id:
            for action in vs.lane_route_actions[
                    vs.lane_route_action_index:]:
                if action.get("type") == "connector":
                    connector_id = str(action.get("connector_id", ""))
                    break
        if not connector_id:
            return None
        plan = self._lane_geometry._signal_plan_by_connector.get(
            connector_id)
        if not plan:
            return None
        movement_node = str(plan.get("node_id", ""))
        if node_id:
            requested_node = self._lane_geometry._node_to_junction.get(
                node_id, node_id)
            if requested_node != movement_node:
                return None
        observed_at = (
            self._physics_time if time_s is None else float(time_s))
        state = self._lane_geometry.signal_state(
            connector_id, observed_at)
        if state is None:
            return None
        connector = self._lane_geometry._connector_by_id.get(
            connector_id, {})
        active = vs.active_connector_id == connector_id
        source_lane_id = str(connector.get("from_lane", ""))
        source_lane = self._lane_geometry._lane_by_id.get(
            source_lane_id, {})
        controls_current_approach = bool(
            active
            or source_lane_id == vs.current_lane_id
            or (source_lane.get("segment_id") == vs.current_segment
                and any(
                    action.get("type") == "lane_change"
                    for action in vs.lane_route_actions[
                        vs.lane_route_action_index:])))
        if active:
            distance_to_stop_line_m = -(
                vs.edge_progress
                * self._lane_geometry.connector_length(connector_id))
        elif controls_current_approach:
            lane_length = self._lane_geometry._lane_lengths.get(
                vs.current_lane_id, 0.0)
            stop_progress = self._lane_geometry.stop_progress(
                vs.current_lane_id)
            control_progress = (
                stop_progress if stop_progress is not None else 1.0)
            distance_to_stop_line_m = (
                control_progress - vs.edge_progress) * lane_length
        else:
            distance_to_stop_line_m = float("inf")
        return {
            "connector_id": connector_id,
            "node_id": movement_node,
            "turn": str(connector.get("turn", "")),
            "signal": state.signal,
            "remaining_seconds": state.remaining_seconds,
            "is_ped_green": state.pedestrian_green,
            "phase_id": state.phase_id,
            "source": "connector_movement_signal",
            "controls_current_approach": controls_current_approach,
            "distance_to_stop_line_m": round(
                distance_to_stop_line_m, 3),
        }

    def get_driving_awareness(
        self, vehicle_id: str, *, ground_truth: bool = False,
    ) -> Dict:
        """Return local physical facts used for driving decisions.

        The method exposes observations, not a prescribed action. SUMO
        SUMO background actors and an LLM driver can therefore consume the same connector,
        crosswalk and leader facts while retaining their own policies.
        """
        vs = self.vehicles.get(vehicle_id)
        if vs is None:
            return {}
        env = self._build_env_view(vs, round(self._physics_time))
        context = self._build_driver_context(
            vs, env, apply_perception=not ground_truth)
        # DriverContext already applies the observer's optical envelope and
        # is built from the same connector-level plan used by enforcement.
        # Do not overwrite it with a future route signal that may be outside
        # sight range or behind the vehicle.
        traffic_signal = context.traffic_light
        signal_remaining_s = context.signal_remaining_s
        governing_connector_id = context.connector_id

        def finite(value: float):
            return (
                round(value, 3)
                if isinstance(value, (int, float))
                and math.isfinite(value) else None)

        return {
            "leader": ({
                "vehicle_id": context.leader.vehicle_id,
                "gap_m": finite(context.leader_gap_m),
                "speed_kmh": round(
                    context.leader.current_speed_kmh, 2),
                "observed_signals": dict(
                    context.leader.observed_signals),
            } if context.leader is not None else None),
            "traffic_light": traffic_signal or None,
            "signal_remaining_s": (
                round(signal_remaining_s, 2)
                if traffic_signal else None),
            "distance_to_light_m": (
                finite(context.distance_to_light_m)
                if traffic_signal else None),
            "distance_to_lane_end_m": finite(
                context.distance_to_lane_end_m),
            "lane_end_semantics": (
                "distance to the current high-precision lane geometry "
                "boundary; this value alone does not mean the lane drops "
                "or that a lane change is required"),
            "connector_id": governing_connector_id or None,
            "on_connector": context.on_connector,
            "distance_to_connector_end_m": finite(
                context.distance_to_connector_end_m),
            "downstream_gap_m": finite(context.downstream_gap_m),
            "connector_conflicts": [{
                "other_vehicle_id": item.other_vehicle_id,
                "other_connector_id": item.other_connector_id,
                "distance_to_conflict_m": finite(
                    item.distance_to_conflict_m),
                "other_distance_to_conflict_m": finite(
                    item.other_distance_to_conflict_m),
                "ego_ttc_s": finite(item.ego_ttc_s),
                "other_ttc_s": finite(item.other_ttc_s),
                "other_speed_kmh": round(
                    item.other_speed_kmh, 2),
                "other_in_conflict_zone":
                    item.other_in_conflict_zone,
            } for item in context.connector_conflicts],
            "pedestrian_hazards": [{
                "pedestrian_id": item.pedestrian_id,
                "crosswalk_id": item.crosswalk_id,
                "distance_to_crosswalk_m": finite(
                    item.distance_to_crosswalk_m),
                "vehicle_ttc_s": finite(item.vehicle_ttc_s),
                "crossing_progress": round(
                    item.crossing_progress, 3),
            } for item in context.pedestrian_hazards],
            "route_blocked": context.route_blocked,
            "route_lane_change_required": bool(
                context.route_lane_change_required),
            "route_target_lane": (
                context.route_target_lane
                if context.route_lane_change_required else None),
            "heard_horns": list(context.heard_horns),
            "visual_envelope": self.perception_model.visual_envelope(
                vehicle_id, bearing_deg=0.0),
            "acoustic_envelope": self.perception_model.acoustic_envelope(
                vehicle_id),
        }

    # ── Internal helpers ───────────────────────────────────────

    def _initialize_vehicle_pose(self, vs: VehicleState) -> None:
        """Publish a static authored spawn pose before SUMO materialisation."""
        pose = self._lane_geometry.initial_vehicle_pose(
            vs, self.road_network)
        if pose is None:
            return
        vs.pose_x_m = float(pose[0])
        vs.pose_y_m = float(pose[1])
        vs.yaw_rad = float(pose[2])
        vs.z_level = int(pose[3])
        vs.heading = (90.0 - math.degrees(vs.yaw_rad)) % 360.0
        vs.physical_pose_authority = "map_initialization"

    def _update_road_network_position(self, vs: VehicleState):
        """Push vehicle position into the RoadNetwork for spatial queries."""
        if not vs.present_in_physics_world:
            self.road_network.remove_vehicle(vs.vehicle_id)
            return
        self.road_network.update_vehicle_position(VehicleRef(
            vehicle_id=vs.vehicle_id,
            node=vs.current_node,
            segment_id=vs.current_segment,
            lane=vs.current_lane,
            speed_kmh=vs.current_speed_kmh,
            heading=vs.heading,
        ))

    # ── Summary ────────────────────────────────────────────────

    def summary(self) -> Dict:
        """Return a summary of all vehicles for debugging."""
        return {
            vid: {
                "node": vs.current_node,
                "segment": vs.current_segment,
                "lane": vs.current_lane,
                "speed": vs.current_speed_kmh,
                "edge_progress": vs.edge_progress,
                "navigating": vs.is_navigating,
                "arrived": vs.arrived,
                "route_failed": vs.route_failed,
                "route_failure_reason": vs.route_failure_reason,
                "crashed": vs.is_crashed,
                "is_llm": vs.is_llm,
            }
            for vid, vs in self.vehicles.items()
        }


    # ══════════════════════════════════════════════════════════════
    # Pedestrian Management
    # ══════════════════════════════════════════════════════════════

    def register_pedestrian(
        self,
        ped_id: str,
        initial_node: str,
        route: List[str],
        speed: float = 1.4,
        is_llm: bool = False,
        start_time: float = 0.0,
        collision_radius_m: float = 0.4,
        perception_profile: str = "pedestrian_standard",
        perception_overrides: Optional[dict] = None,
    ):
        """Register a pedestrian into the traffic system.

        Args:
            ped_id: Unique pedestrian identifier.
            initial_node: Starting node ID.
            route: List of node IDs forming the pedestrian's path.
            speed: Walking speed in m/s.
            is_llm: Whether this pedestrian is LLM-controlled.
            start_time: Simulation time (seconds) when pedestrian appears.

        Returns:
            The created PedestrianState.
        """
        from simulation.pedestrian_state import PedestrianState, PedestrianPosition
        if collision_radius_m <= 0:
            raise ValueError("collision_radius_m must be positive")

        ps = PedestrianState(
            ped_id=ped_id,
            position=PedestrianPosition(at_node=initial_node),
            speed=speed,
            base_speed=speed,
            route=route,
            route_index=0,
            is_llm=is_llm,
            start_time=start_time,
            last_update_time=start_time,
            control_authority="llm" if is_llm else "sumo",
            collision_radius_m=float(collision_radius_m),
            perception_profile_name=perception_profile,
            perception_overrides=dict(perception_overrides or {}),
            is_spawned=(start_time <= self._physics_time + 1e-9),
        )

        self.pedestrians[ped_id] = ps
        self._ped_move_order.append(ped_id)
        logger.info(f"Registered pedestrian '{ped_id}' at {initial_node}"
                    f" (llm={is_llm}, route={route})")
        return ps

    def unregister_pedestrian(self, ped_id: str):
        """Remove a pedestrian from management."""
        self.pedestrians.pop(ped_id, None)
        if ped_id in self._ped_move_order:
            self._ped_move_order.remove(ped_id)

    def collect_pedestrian_lifecycle_events(
        self, target_time_s: float,
    ) -> List[TriggerEvent]:
        """Publish spawn events without integrating a physical position.

        SUMO alone advances pedestrian poses.  The coordinator only mirrors
        the semantic lifecycle needed by agents and evaluation.
        """
        events: List[TriggerEvent] = []
        for ped_id in self._ped_move_order:
            pedestrian = self.pedestrians.get(ped_id)
            if (pedestrian is None or pedestrian.has_arrived
                    or pedestrian.is_crashed
                    or target_time_s < pedestrian.start_time):
                continue
            pedestrian.is_spawned = True
            if not pedestrian.spawn_event_emitted:
                pedestrian.spawn_event_emitted = True
                events.append(TriggerEvent(
                    type="ped_arrive_node",
                    vehicle_id=ped_id,
                    tick=round(pedestrian.start_time),
                    step=0,
                    time_s=pedestrian.start_time,
                    details={"node_id": pedestrian.position.at_node},
                ))
        return events

    def _pedestrian_walking_path(self, pedestrian) -> List[List[float]]:
        """Return a continuous metric path for the next walking leg."""
        start_node = pedestrian.position.at_node
        end_node = pedestrian.next_node
        start = self._lane_geometry.nodes_xy.get(start_node)
        end = self._lane_geometry.nodes_xy.get(end_node)
        if not start or not end:
            return []
        return [list(start), list(end)]

    @staticmethod
    def _crosswalk_pedestrian_path(crosswalk: dict) -> List[List[float]]:
        """Include sidewalk waiting clearance beyond the painted roadway.

        Inferred crosswalk centerlines end at the carriageway boundary.
        Extending the authored pedestrian route to its waiting areas gives
        both ends a sidewalk refuge without changing the painted map.
        """
        line = [list(point) for point in crosswalk.get(
            "centerline_xy", [])]
        if len(line) < 2:
            return line
        start, end = line[0], line[-1]
        dx, dy = end[0] - start[0], end[1] - start[1]
        length = math.hypot(dx, dy)
        if length <= 1e-9:
            return line
        ux, uy = dx / length, dy / length
        clearance_m = 2.5
        wait_a = crosswalk.get("from_wait_area", {}).get("xy")
        wait_b = crosswalk.get("to_wait_area", {}).get("xy")
        # Existing generated maps place wait areas at the painted endpoints.
        # Treat an explicitly offset surveyed wait area as authoritative.
        if (not wait_a
                or math.dist(tuple(wait_a), tuple(start)) < 0.5):
            wait_a = [start[0] - ux * clearance_m,
                      start[1] - uy * clearance_m]
        if (not wait_b
                or math.dist(tuple(wait_b), tuple(end)) < 0.5):
            wait_b = [end[0] + ux * clearance_m,
                      end[1] + uy * clearance_m]
        return [list(wait_a), *line, list(wait_b)]

    def execute_pedestrian_action(self, ped_id: str, action: str, params: dict = None) -> dict:
        """Execute a pedestrian action tool call.

        Called when an LLM pedestrian uses an action tool.

        Args:
            ped_id: Pedestrian ID.
            action: Action name (pedestrian_wait/walk/cross/run/change_route).
            params: Action parameters.

        Returns:
            Dict with action result.
        """
        params = params or {}
        ps = self.pedestrians.get(ped_id)
        if not ps:
            return {"success": False, "error": f"Unknown pedestrian: {ped_id}"}
        if not ps.is_llm:
            return {
                "success": False,
                "reason": "pedestrian_is_sumo_controlled",
            }

        if ps.has_arrived:
            return {"success": False, "error": "Already arrived at destination."}

        if action == "pedestrian_wait":
            ps.set_waiting()
            return {"success": True, "action": "waiting"}

        elif action == "pedestrian_walk":
            speed = params.get("speed", ps.base_speed)
            speed = max(0.5, min(speed, 2.0))  # clamp to reasonable range
            ps.set_walking(
                speed, path_xy=self._pedestrian_walking_path(ps))
            return {"success": True, "action": "walking", "speed_ms": speed}

        elif action == "pedestrian_cross":
            node_id = ps.position.at_node
            if not node_id:
                return {"success": False, "error": "Not at a node, cannot start crossing."}
            # Determine crosswalk target
            next_node = ps.authored_crosswalk_to_node or ps.next_node
            if not next_node:
                return {"success": False, "error": "No next node in route."}
            # Check if crosswalk exists
            node_obj = self.road_network.nodes.get(node_id)
            junction_id = self._lane_geometry._node_to_junction.get(
                node_id, node_id)
            lane_crosswalks = [
                item for item in
                self._lane_geometry.data.get("crosswalks", [])
                if item["node_id"] == junction_id]
            lane_crosswalk = self._lane_geometry._crosswalk_by_id.get(
                ps.authored_crosswalk_id or "")
            start_xy = self._lane_geometry.nodes_xy.get(node_id)
            end_xy = self._lane_geometry.nodes_xy.get(next_node)
            if lane_crosswalk is None and lane_crosswalks:
                if start_xy and end_xy:
                    route_dx = end_xy[0] - start_xy[0]
                    route_dy = end_xy[1] - start_xy[1]
                    route_norm = max(
                        1e-9, (route_dx * route_dx
                               + route_dy * route_dy) ** 0.5)
                    # Crossing direction should align with pedestrian route.
                    def crosswalk_score(item):
                        line = item.get("centerline_xy", [])
                        if len(line) < 2:
                            return float("-inf")
                        dx = line[-1][0] - line[0][0]
                        dy = line[-1][1] - line[0][1]
                        norm = max(1e-9, (dx * dx + dy * dy) ** 0.5)
                        alignment_score = abs(
                            (dx * route_dx + dy * route_dy)
                            / (norm * route_norm))
                        endpoint_distance = min(
                            math.dist(start_xy, tuple(line[0])),
                            math.dist(start_xy, tuple(line[-1])))
                        return alignment_score * 20.0 - endpoint_distance
                    lane_crosswalk = max(
                        lane_crosswalks, key=crosswalk_score)
                else:
                    lane_crosswalk = lane_crosswalks[0]
            if not (
                    (node_obj and getattr(node_obj, 'has_crosswalk', False))
                    or lane_crosswalk):
                return {"success": False, "error": f"No crosswalk at {node_id}."}
            # Start crossing
            crosswalk_length = (
                max(3.0, lane_crosswalk.get(
                    "length_m",
                    lane_crosswalk.get("radius_m", 5.0) * 2.0))
                if lane_crosswalk else 10.0)
            speed = params.get("speed", ps.base_speed)
            ps.speed = max(0.2, min(float(speed), 4.0))
            path_xy = (
                ps.authored_crosswalk_path_xy
                or (self._crosswalk_pedestrian_path(lane_crosswalk)
                    if lane_crosswalk else []))
            # Enter at the physically nearest endpoint. Route alignment is
            # only the tie-breaker used while choosing the crosswalk.
            if path_xy and start_xy:
                if (math.dist(start_xy, tuple(path_xy[-1]))
                        < math.dist(start_xy, tuple(path_xy[0]))):
                    path_xy = list(reversed(path_xy))
                crosswalk_length = sum(
                    math.dist(tuple(first), tuple(second))
                    for first, second in zip(path_xy, path_xy[1:]))
            authored_progress = (
                ps.authored_crosswalk_progress
                if ps.authored_crosswalk_path_xy else 0.0)
            ps.start_crossing(
                node_id, next_node, crosswalk_length,
                crosswalk_id=(
                    lane_crosswalk["id"] if lane_crosswalk else None),
                path_xy=path_xy)
            if ps.authored_crosswalk_path_xy:
                ps.crossing_progress = authored_progress
                ps.authored_crosswalk_id = None
                ps.authored_crosswalk_path_xy = []
                ps.authored_crosswalk_progress = 0.0
                ps.authored_crosswalk_to_node = None
            return {"success": True, "action": "crossing",
                    "from": node_id, "to": next_node}

        elif action == "pedestrian_run":
            speed = params.get("speed")
            if speed is None:
                ps.set_running()
                if not ps.is_on_crosswalk:
                    ps.set_walking(
                        ps.speed,
                        path_xy=self._pedestrian_walking_path(ps))
            else:
                ps.set_walking(
                    max(0.5, min(float(speed), 5.0)),
                    path_xy=self._pedestrian_walking_path(ps))
            if ps.is_on_crosswalk:
                pass  # already crossing, just faster now
            return {"success": True, "action": "running",
                    "speed_ms": ps.speed}

        elif action == "pedestrian_change_route":
            target_node = params.get("target_node")
            if not target_node:
                return {"success": False, "error": "target_node required."}
            if target_node not in self.road_network.nodes:
                return {"success": False, "error": f"Unknown node: {target_node}"}
            # Recompute route from current position
            current = ps.position.at_node or ps.position.crossing_to
            if current:
                new_route = self.road_network.plan_route(current, target_node)
                if new_route:
                    ps.route = new_route
                    ps.route_index = 0
                    return {"success": True, "action": "route_changed",
                            "new_route": new_route}
            return {"success": False, "error": "Cannot compute route."}

        return {"success": False, "error": f"Unknown action: {action}"}


# ── Utility ────────────────────────────────────────────────────

def _bearing(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Compute bearing (degrees, 0=north, 90=east) between two points."""
    dlng = math.radians(lng2 - lng1)
    lat1_r = math.radians(lat1)
    lat2_r = math.radians(lat2)
    x = math.sin(dlng) * math.cos(lat2_r)
    y = (math.cos(lat1_r) * math.sin(lat2_r)
         - math.sin(lat1_r) * math.cos(lat2_r) * math.cos(dlng))
    bearing = math.degrees(math.atan2(x, y))
    return (bearing + 360) % 360
