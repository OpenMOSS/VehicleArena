"""Pedestrian decisions plus physical state mirrored from SUMO."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PedestrianPosition:
    """Pedestrian location — either at a node or crossing between nodes."""
    # At a node (sidewalk/waiting area)
    at_node: Optional[str] = None
    # Crossing a road between two nodes
    crossing_from: Optional[str] = None
    crossing_to: Optional[str] = None


@dataclass
class PedestrianState:
    """Complete state of a pedestrian entity in the simulation."""

    ped_id: str = ""
    position: PedestrianPosition = field(default_factory=PedestrianPosition)
    speed: float = 1.4              # current walking speed in m/s
    base_speed: float = 1.4         # default walking speed in m/s
    route: List[str] = field(default_factory=list)   # node ID sequence
    route_index: int = 0            # index of current target node in route

    # Crossing state
    is_on_crosswalk: bool = False
    crossing_progress: float = 0.0  # 0.0 to 1.0 progress across crosswalk
    crosswalk_length_m: float = 10.0  # default crosswalk width
    active_crosswalk_id: Optional[str] = None
    # The physical path follows the marked crossing; the node route remains
    # the topological plan between successive pedestrian waypoints.
    crosswalk_path_xy: List[List[float]] = field(default_factory=list)

    # Walking state between mapped pedestrian edges.
    walking_progress: float = 0.0
    walking_path_xy: List[List[float]] = field(default_factory=list)
    # A policy-controlled mapped crossing also keeps its exact identity and
    # geometry while the pedestrian is still waiting.  It must not be
    # inferred from an arbitrary graph node because one junction can contain
    # several directional crosswalks.
    authored_crosswalk_id: Optional[str] = None
    authored_crosswalk_path_xy: List[List[float]] = field(default_factory=list)
    authored_crosswalk_progress: float = 0.0
    authored_crosswalk_to_node: Optional[str] = None

    # Movement state
    is_waiting: bool = False        # waiting at a signal/decision point
    is_walking: bool = False        # actively walking along route
    has_arrived: bool = False       # reached destination
    pending_arrival: bool = False   # commit after SUMO's contact report
    is_spawned: bool = False
    spawn_event_emitted: bool = False
    is_crashed: bool = False
    crash_position_xy: Optional[List[float]] = field(
        default=None, repr=False)
    # Metric pose mirrored from SUMO. Graph nodes only represent routing
    # decisions and may be far from a local crosswalk endpoint.
    physical_pose_xy: Optional[List[float]] = field(
        default=None, repr=False)
    # Heading mirrored from SUMO in VehicleArena's yaw convention.  This is
    # observational state for rendering/perception, never a second motion
    # authority.
    physical_yaw_rad: Optional[float] = field(default=None, repr=False)
    # SUMO is the sole physical pose authority. The explicit marker helps
    # perception/rendering reject state that has not yet been synchronized.
    physical_pose_authority: str = "sumo"

    # Timing
    last_update_time: float = 0.0   # last physics update time (seconds)
    start_time: float = 0.0         # when this pedestrian enters the simulation

    # LLM control
    is_llm: bool = False
    control_authority: str = "sumo"
    collision_radius_m: float = 0.4
    perception_profile_name: str = "pedestrian_standard"
    perception_overrides: Dict[str, Any] = field(default_factory=dict)

    @property
    def current_node(self) -> Optional[str]:
        """The node the pedestrian is at or heading toward."""
        if self.position.at_node:
            return self.position.at_node
        return self.position.crossing_to

    @property
    def destination(self) -> Optional[str]:
        """Final destination node."""
        if self.route:
            return self.route[-1]
        return None

    @property
    def next_node(self) -> Optional[str]:
        """Next node in the route."""
        if self.route and self.route_index < len(self.route) - 1:
            return self.route[self.route_index + 1]
        return None

    def complete_crossing(self, defer_arrival: bool = False):
        """Finalize a crosswalk crossing — arrive at the target node."""
        target = self.position.crossing_to
        if self.crosswalk_path_xy:
            self.physical_pose_xy = list(self.crosswalk_path_xy[-1])
        self.position = PedestrianPosition(at_node=target)
        self.is_on_crosswalk = False
        self.active_crosswalk_id = None
        self.crosswalk_path_xy = []
        self.crossing_progress = 0.0
        self.is_walking = False
        # Advance route index
        if self.route and target in self.route:
            idx = self.route.index(target)
            if idx >= self.route_index:
                self.route_index = idx
        # Check arrival
        if target == self.destination:
            if defer_arrival:
                self.pending_arrival = True
            else:
                self.has_arrived = True

    def start_crossing(
        self, from_node: str, to_node: str,
        crosswalk_length: float = 10.0,
        crosswalk_id: Optional[str] = None,
        path_xy: Optional[List[List[float]]] = None,
    ):
        """Begin crossing a road from from_node to to_node."""
        self.position = PedestrianPosition(
            at_node=None,
            crossing_from=from_node,
            crossing_to=to_node,
        )
        self.is_on_crosswalk = True
        self.is_waiting = False
        self.crossing_progress = 0.0
        self.crosswalk_length_m = crosswalk_length
        self.active_crosswalk_id = crosswalk_id
        self.crosswalk_path_xy = list(path_xy or [])
        self.physical_pose_xy = None
        self.walking_path_xy = []
        self.walking_progress = 0.0
        # Restore walking speed if it was zeroed by a prior set_waiting()
        if self.speed <= 0:
            self.speed = self.base_speed

    def arrive_at_node(self, node_id: str, defer_arrival: bool = False):
        """Mark arrival at a node (from walking along a segment)."""
        if self.walking_path_xy:
            self.physical_pose_xy = list(self.walking_path_xy[-1])
        self.position = PedestrianPosition(at_node=node_id)
        self.is_on_crosswalk = False
        self.active_crosswalk_id = None
        self.crosswalk_path_xy = []
        self.is_walking = False
        self.crossing_progress = 0.0
        self.walking_progress = 0.0
        self.walking_path_xy = []
        # Advance route index
        if self.route and node_id in self.route:
            idx = self.route.index(node_id)
            if idx >= self.route_index:
                self.route_index = idx
        # Check arrival
        if node_id == self.destination:
            if defer_arrival:
                self.pending_arrival = True
            else:
                self.has_arrived = True

    def set_waiting(self):
        """Set pedestrian to waiting state."""
        self.is_waiting = True
        self.is_walking = False
        self.speed = 0.0

    def set_walking(
        self,
        speed: Optional[float] = None,
        path_xy: Optional[List[List[float]]] = None,
    ):
        """Set pedestrian to walking state."""
        was_walking = (
            not self.is_on_crosswalk
            and (self.is_walking
                 or (self.walking_path_xy
                     and 0.0 < self.walking_progress < 1.0)))
        self.is_waiting = False
        self.is_walking = True
        self.speed = speed if speed is not None else self.base_speed
        if path_xy is not None:
            normalized = [list(point) for point in path_xy]
            if not was_walking or normalized != self.walking_path_xy:
                self.walking_path_xy = normalized
                self.walking_progress = 0.0

    def set_running(self):
        """Set pedestrian to running state (double speed)."""
        self.is_waiting = False
        self.is_walking = True
        self.speed = self.base_speed * 2.0
