"""Vehicle control intent plus state mirrored from SUMO."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from simulation.perception_model import VehicleSignalState


@dataclass
class VehicleState:
    """Vehicle decisions and the latest physical state published by SUMO."""
    current_node: str                       # current intersection
    destination_name: str = ""              # human-readable destination
    is_navigating: bool = False
    waiting_red_light: bool = False
    arrived: bool = False
    route_failed: bool = False
    route_failure_reason: str = ""
    route_failure_time_s: Optional[float] = None
    pending_route_failure: bool = False  # finalize after native collision pass
    # SUMO removes an entity after it crosses its configured route endpoint.
    # Keep that lifecycle fact separate from a physical zero-speed sample.
    present_in_physics_world: bool = True
    terminal_crossing_speed_kmh: Optional[float] = None
    # Continuous physical values mirrored from SUMO.
    current_speed_kmh: float = 0.0          # current speed
    target_speed_kmh: float = 0.0           # controller output
    acceleration_mps2: float = 0.0          # continuous longitudinal state
    max_acceleration_mps2: float = 3.0
    max_braking_mps2: float = 6.0
    control_acceleration_limit_mps2: float = 3.0
    control_deceleration_limit_mps2: float = 6.0
    distance_traveled_m: float = 0.0        # cumulative distance
    edge_progress: float = 0.0              # position on current edge 0.0-1.0
    # Multi-vehicle fields (used by TrafficCoordinator)
    vehicle_id: str = ""                    # unique vehicle identifier
    current_segment: str = ""               # current road segment ID
    current_lane: int = 0                   # current lane index (0=rightmost)
    heading: float = 0.0                    # heading in degrees (0=north, 90=east)
    # Driving control fields (new: agent driving control)
    is_llm: bool = False              # True = LLM-controlled, False = SUMO
    desired_speed_kmh: float = -1.0         # -1 = use speed limit
    is_crashed: bool = False                # collision → permanently stopped
    pending_arrival: bool = False           # reached destination; commit after contacts
    is_stopped: bool = False                # agent requested stop
    # Exactly one behavioural authority owns the actor. SUMO controls all
    # background traffic; an LLM controls an ego through explicit commands.
    control_authority: str = "sumo"
    llm_control_command: Dict[str, Any] = field(default_factory=dict)
    # Last successfully committed ego commands by physical control slot.
    # This is actuator state exposed back to the same driver, never a hidden
    # traffic-policy decision or an observation about another entity.
    active_control_commands: Dict[str, Dict[str, Any]] = field(
        default_factory=dict)
    # Physical sensing capability is equipment, independent of control authority.
    perception_profile_name: str = "human_driver_standard"
    perception_overrides: Dict[str, Any] = field(default_factory=dict)
    cabin_open_fraction: float = 0.0
    # World-visible lamps/indicators. Cabin module controls are synchronized
    # into this authoritative state only at an agent-batch boundary.
    signal_state: VehicleSignalState = field(
        default_factory=VehicleSignalState)
    # Physical body dimensions submitted directly to SUMO.
    chassis_profile: str = "sedan"
    length_m: float = 4.6
    width_m: float = 1.9
    # Authoritative SUMO pose plus a derived compass bearing. Rendering and
    # perception consume this synchronized state and never integrate a second
    # vehicle trajectory.
    physical_pose_authority: str = "map_initialization"
    pose_x_m: float = 0.0
    pose_y_m: float = 0.0
    yaw_rad: float = 0.0
    z_level: int = 0
    crash_pose: Any = field(default=None, repr=False)
    # Lane-level route state. Numeric current_lane is the lane's map index.
    current_lane_id: str = ""
    planned_connector_id: str = ""
    planned_maneuver_source: str = ""  # default_straight or explicit
    planned_from_lane_id: str = ""
    planned_from_lane_index: int = -1
    planned_turn: str = ""
    # A junction connector is a first-class physical path. While non-empty,
    # edge_progress measures progress along its curved centerline rather than
    # along current_segment.
    active_connector_id: str = ""
    active_connector_from_lane_id: str = ""
    active_connector_to_lane_id: str = ""
    lane_route_blocked: bool = False
    destination_node: str = ""
    # Route guidance and route execution are deliberately separate.  SUMO
    # background actors keep ``route_control_authority="sumo"`` and may own a
    # complete ``lane_route_actions`` list.  An evaluated LLM vehicle uses
    # ``llm_maneuver``: ``suggested_lane_route_*`` is display/evaluation data,
    # while ``lane_route_actions`` contains at most the connector explicitly
    # authorised by the driver for the current junction.
    route_control_authority: str = "sumo"
    suggested_lane_route_start_lane_id: str = ""
    suggested_lane_route_goal_lane_id: str = ""
    suggested_lane_route_actions: List[Dict[str, Any]] = field(
        default_factory=list)
    suggested_lane_route_cost: float = 0.0
    lane_route_actions: List[Dict[str, Any]] = field(default_factory=list)
    lane_route_action_index: int = 0
    lane_route_cost: float = 0.0
    pending_uturn_connector_id: str = ""
    target_lane: int = -1
    lane_change_progress: float = 0.0
    lane_change_duration_s: float = 3.5
    lateral_offset_m: float = 0.0
    lateral_speed_mps: float = 0.0
    is_changing_lane: bool = False
    # Internal request accounting, not an agent-visible completion signal.
    lane_change_attempts: int = 0
    lane_change_completions: int = 0
    lane_change_uncompleted: int = 0
    lane_change_request_outcome: str = ""

    @property
    def remaining_stops(self) -> int:
        """Number of connector traversals remaining on the lane route."""
        return sum(
            action.get("type") == "connector"
            for action in self.lane_route_actions[
                self.lane_route_action_index:])
