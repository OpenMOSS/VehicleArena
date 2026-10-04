"""
RoadNetwork — Real road network for tick-driven driving simulation.

Uses real OpenStreetMap topology with intersections, street segments,
lanes, and traffic lights. Provides routing, observed-flow, and event query
APIs used by SimulationEngine and TrafficCoordinator.

Data model:
  - RoadNode   = intersection (from OSM node)
  - RoadSegment = street between two intersections (from OSM way)
  - Lane        = individual lane within a segment (for multi-vehicle sim)

Topology is loaded once (from pre-built JSON or live OSM import).
Time-varying events are loaded per-scenario via load_scenario_events().

Usage:
    from simulation.road_network import RoadNetwork
    from simulation.osm_import import import_from_osm

    net = import_from_osm(place="Zhongguancun, Beijing")
    net.load_scenario_events(scenario_events_dict)
    route = net.plan_route("n123", "n456", time_sec=0)
"""

from __future__ import annotations

import heapq
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Set


# ── Data classes ──────────────────────────────────────────────

@dataclass
class RoadNode:
    """An intersection or junction on the road network."""
    id: str
    lat: float
    lng: float
    name: str = ""              # intersection name (e.g. "中关村大街 & 北四环")
    street_name: str = ""       # primary street name
    avenue_name: str = ""       # cross street name
    signal: bool = False        # has traffic light
    has_crosswalk: bool = False # pedestrian crossing at this intersection
    poi: List[str] = field(default_factory=list)  # nearby POI names
    osm_id: int = 0             # original OSM node ID

    @property
    def full_name(self) -> str:
        if self.name:
            return self.name
        if self.street_name and self.avenue_name:
            return f"{self.street_name} & {self.avenue_name}"
        return self.id


@dataclass
class Lane:
    """A single lane within a road segment."""
    lane_index: int             # 0 = rightmost
    direction: str = "forward"  # "forward" / "backward"
    lane_type: str = "through"  # "through" / "left_turn" / "right_turn" / "bus"
    width_meters: float = 3.5


@dataclass
class RoadSegment:
    """A street segment connecting two intersections."""
    id: str
    from_node: str
    to_node: str
    name: str = ""                      # street name
    road_type: str = "secondary"        # OSM highway tag
    lanes: int = 2                      # total lane count
    lane_config: List[Lane] = field(default_factory=list)
    # Directional lane counts preserve OSM ``lanes:forward/backward`` when
    # available. ``None`` means the topology source only supplied a total.
    lanes_forward: Optional[int] = None
    lanes_backward: Optional[int] = None
    turn_lanes_forward: List[str] = field(default_factory=list)
    turn_lanes_backward: List[str] = field(default_factory=list)
    lane_width_meters: float = 3.5
    speed_limit: int = 60               # km/h
    distance_meters: float = 500.0
    oneway: bool = False
    geometry: List[Tuple[float, float]] = field(default_factory=list)
    base_travel_time: float = 0.0       # seconds at speed_limit
    osm_way_id: int = 0                 # original OSM way ID
    layer: int = 0
    bridge: bool = False
    tunnel: bool = False
    source_edge_count: int = 1

    def __post_init__(self):
        if self.base_travel_time <= 0 and self.speed_limit > 0:
            # speed_limit is km/h → convert to m/s
            speed_ms = self.speed_limit / 3.6
            self.base_travel_time = self.distance_meters / speed_ms
        if not self.lane_config and self.lanes > 0:
            self._auto_lane_config()

    def _auto_lane_config(self):
        """Generate default lane config from lane count and oneway flag."""
        if self.oneway:
            count = (
                self.lanes_forward
                if self.lanes_forward is not None else self.lanes)
            self.lane_config = [
                Lane(lane_index=i, direction="forward", lane_type="through")
                for i in range(max(1, count))
            ]
        elif (
            self.lanes == 1
            and self.lanes_forward is None
            and self.lanes_backward is None
        ):
            # One physical lane shared by both directions.  Directed lane
            # records deliberately share lane_index=0; the HD geometry stores
            # two opposite-oriented centerlines on the same corridor.
            self.lane_config = [
                Lane(lane_index=0, direction="backward",
                     lane_type="through"),
                Lane(lane_index=0, direction="forward",
                     lane_type="through"),
            ]
        else:
            fwd = (
                self.lanes_forward
                if self.lanes_forward is not None
                else (self.lanes + 1) // 2)
            bwd = (
                self.lanes_backward
                if self.lanes_backward is not None
                else max(1, self.lanes - fwd))
            fwd = max(1, int(fwd))
            bwd = max(1, int(bwd))
            self.lanes = fwd + bwd
            self.lane_config = []
            for i in range(bwd):
                self.lane_config.append(
                    Lane(lane_index=i, direction="backward", lane_type="through")
                )
            for i in range(fwd):
                self.lane_config.append(
                    Lane(lane_index=bwd + i, direction="forward", lane_type="through")
                )

    def forward_lanes(self) -> List[Lane]:
        return [l for l in self.lane_config if l.direction == "forward"]

    def backward_lanes(self) -> List[Lane]:
        return [l for l in self.lane_config if l.direction == "backward"]


@dataclass
class VehicleRef:
    """Lightweight reference to a vehicle's position (for multi-vehicle queries)."""
    vehicle_id: str
    node: str
    segment_id: str = ""
    lane: int = 0
    speed_kmh: float = 0.0
    heading: float = 0.0
    distance_m: float = 0.0     # distance from querying vehicle


# ── Traffic simulation config ────────────────────────────────

DEFAULT_STOPPED_SPEED_THRESHOLD_KMH = 0.5

@dataclass
class TrafficConfig:
    """Configurable parameters for traffic simulation.

    All parameters have sensible defaults.  Override via road-network JSON
    (``"traffic_config": {...}``) or programmatically before running.
    """
    # Perception range for vehicle detection (meters)
    perception_range_m: float = 200.0

    # Shared clearance used by spawn placement and evaluation.
    min_gap_m: float = 5.0              # minimum gap when stopped (meters)
    stop_line_offset_m: float = 5.0     # red-light stop distance before segment end
    # Below this speed a vehicle is treated as having yielded/stopped by
    # world perception and pedestrian gap acceptance.  Keeping the threshold
    # here prevents two subsystems from silently using different semantics.
    stopped_speed_threshold_kmh: float = (
        DEFAULT_STOPPED_SPEED_THRESHOLD_KMH)

    # Default signal cycle for auto-generated traffic lights (durations in seconds)
    default_signal_cycle: List[Tuple[str, int]] = field(
        default_factory=lambda: [("green", 35), ("yellow", 3), ("red", 30)]
    )


    # Agent heartbeat timing; physics runs independently at 0.1 seconds.
    seconds_per_tick: int = 1


# ── Shared event data classes ────────────────────────────────

@dataclass
class TrafficLightState:
    """Signal state at a single tick."""
    signal: str           # "red" | "green" | "yellow" | "flashing_yellow"
    remaining_seconds: int = 0
    is_crosswalk_phase: bool = False  # True when current phase is ped_green


@dataclass
class TrafficPhase:
    """One phase of a traffic light cycle.

    ``allowed_approaches`` lists the node IDs whose approaching traffic
    gets the phase signal.  Empty means *all* approaches (non-directional).
    During ``ped_green`` phases the list should be empty — all vehicle
    approaches see red.
    """
    signal: str                          # "green" | "yellow" | "red" | "ped_green"
    duration_ticks: int                  # how many ticks this phase lasts
    allowed_approaches: List[str] = field(default_factory=list)


@dataclass
class TrafficLightSchedule:
    """Multi-phase, direction-aware traffic light for one intersection.

    Each phase specifies which approaches (incoming node IDs) get green.
    ``phase_offset`` shifts the start of the whole cycle.

    Direction resolution (``at_tick`` with ``from_node``):
      - ``from_node in allowed_approaches`` → return phase signal
      - ``allowed_approaches`` empty (ped phases) → phase signal for
        ``from_node=None``; ``"red"`` for ped_green, phase signal otherwise
      - else → ``"red"``
    """
    phases: List[TrafficPhase]
    phase_offset: int = 0
    intersection_id: str = ""
    has_crosswalk: bool = False          # pedestrian crossing at this node

    def at_tick(self, tick: int, seconds_per_tick: int = 1,
                from_node: Optional[str] = None) -> TrafficLightState:
        """Get signal state at *time_sec* for traffic approaching from *from_node*.

        Phase durations are in seconds.  *tick* is the current simulation
        second.

        Args:
            tick: Current simulation time in seconds.
            seconds_per_tick: Multiplier (should be 1).
            from_node: The node the vehicle is coming from.  ``None`` →
                return the current active-phase signal.
        """
        if not self.phases:
            return TrafficLightState(signal="green", remaining_seconds=0)

        total_cycle = sum(p.duration_ticks for p in self.phases)
        if total_cycle == 0:
            return TrafficLightState(signal="green", remaining_seconds=0)

        pos = (tick + self.phase_offset) % total_cycle

        elapsed = 0
        for phase in self.phases:
            if pos < elapsed + phase.duration_ticks:
                remaining = phase.duration_ticks - (pos - elapsed)
                signal = self._resolve_signal(phase, from_node)
                return TrafficLightState(
                    signal=signal,
                    remaining_seconds=remaining,
                    is_crosswalk_phase=(phase.signal == "ped_green"),
                )
            elapsed += phase.duration_ticks

        return TrafficLightState(signal="green", remaining_seconds=0)

    @staticmethod
    def _resolve_signal(phase: TrafficPhase,
                        from_node: Optional[str]) -> str:
        """Determine what signal *from_node* sees during *phase*."""
        # Pedestrian phase → always red for vehicles
        if phase.signal == "ped_green":
            return "red"
        # Non-directional (empty allowed_approaches): return phase signal
        if not phase.allowed_approaches:
            return phase.signal
        # Directional: caller didn't specify approach → return phase signal
        if from_node is None:
            return phase.signal
        # Directional with from_node
        if from_node in phase.allowed_approaches:
            return phase.signal
        return "red"


@dataclass
class RoadEventSchedule:
    """A road event active on a street during [start_sec, end_sec) in physical seconds."""
    edge_id: str
    event_type: str            # construction | accident | flooding | road_closure
    start_sec: int
    end_sec: int               # exclusive
    severity: str = "moderate"
    description: str = ""
    speed_limit_in_zone: int = 30
    lanes_affected: int = 1
    detour_available: bool = False
    estimated_delay_minutes: int = 10


@dataclass
class CongestionEvent:
    """Congestion on a street during [start_sec, end_sec) with speed penalty."""
    edge_id: str
    start_sec: int
    end_sec: int               # exclusive
    level: str = "heavy"       # "heavy" | "gridlock"
    wait_ticks: int = 1
    speed_factor: float = 0.4  # speed multiplier (ground truth)
    estimated_speed: int = 8   # km/h during congestion
    description: str = ""

    SPEED_FACTOR_RANGES = {
        "heavy":    (0.2, 0.5),
        "gridlock": (0.1, 0.3),
    }


@dataclass
class SpeedCamera:
    """Static speed camera on a street."""
    edge_id: str
    position_ratio: float = 0.5     # 0.0~1.0 along the edge
    speed_limit: int = 60
    camera_type: str = "fixed"      # "fixed" | "temporary"



# ── RoadNetwork ──────────────────────────────────────────────

class RoadNetwork:
    """Real road network with routing, observed-flow, and event query APIs.

    Topology (nodes + segments) is fixed at construction time.
    Time-varying events are loaded per-scenario via load_scenario_events().
    """

    def __init__(self, traffic_config: Optional[TrafficConfig] = None):
        self.config = traffic_config or TrafficConfig()

        self.nodes: Dict[str, RoadNode] = {}
        self.edges: Dict[str, RoadSegment] = {}     # segment_id → RoadSegment
        self.adjacency: Dict[str, List[str]] = {}    # node_id → [neighbor_ids]

        # Time-varying (loaded per scenario)
        self.traffic_lights: Dict[str, TrafficLightSchedule] = {}
        self.road_events: List[RoadEventSchedule] = []
        self.congestion_events: List[CongestionEvent] = []
        self.speed_cameras: List[SpeedCamera] = []

        # Multi-vehicle tracking (populated by TrafficCoordinator)
        self._vehicle_positions: Dict[str, VehicleRef] = {}
        self._seg_to_vehicles: Dict[str, Set[str]] = defaultdict(set)

    # ── Topology construction ──────────────────────────────────

    def add_node(self, node: RoadNode):
        self.nodes[node.id] = node
        if node.id not in self.adjacency:
            self.adjacency[node.id] = []

    def add_segment(self, segment: RoadSegment):
        self.edges[segment.id] = segment
        # Build adjacency
        if segment.from_node not in self.adjacency:
            self.adjacency[segment.from_node] = []
        if segment.to_node not in self.adjacency:
            self.adjacency[segment.to_node] = []
        if segment.to_node not in self.adjacency[segment.from_node]:
            self.adjacency[segment.from_node].append(segment.to_node)
        # If not oneway, add reverse direction
        if not segment.oneway:
            if segment.from_node not in self.adjacency[segment.to_node]:
                self.adjacency[segment.to_node].append(segment.from_node)

    def add_edge(self, segment: RoadSegment):
        """Alias for add_segment."""
        self.add_segment(segment)

    @staticmethod
    def make_edge_id(node_a: str, node_b: str) -> str:
        """Canonical edge id: smaller node first."""
        return f"{min(node_a, node_b)}_{max(node_a, node_b)}"

    def get_node(self, node_id: str) -> Optional[RoadNode]:
        return self.nodes.get(node_id)

    def get_edge_by_nodes(self, from_id: str, to_id: str) -> Optional[RoadSegment]:
        eid = self.make_edge_id(from_id, to_id)
        return self.edges.get(eid)

    def get_edge_obj(self, edge_id: str) -> Optional[RoadSegment]:
        return self.edges.get(edge_id)

    def get_segment(self, segment_id: str) -> Optional[RoadSegment]:
        return self.edges.get(segment_id)

    # ── Event loading ────────────────────────────────────────

    def load_scenario_events(self, scenario_dict: dict, tick_interval_s: float = 300.0):
        """Load time-varying events from a scenario definition.

        Keys: traffic_light_schedule, road_events,
        congestion_events, speed_cameras.

        ``tick_interval_s`` is used to convert start_tick/end_tick
        fields (tick indices) into physical seconds for internal storage.
        """
        self.traffic_lights.clear()
        self.road_events.clear()
        self.congestion_events.clear()
        self.speed_cameras.clear()

        # Traffic lights
        tl_data = scenario_dict.get("traffic_light_schedule", {})
        for node_id, spec in tl_data.items():
            if isinstance(spec, dict):
                raw_cycle = spec.get("cycle", [])
                offset = spec.get("phase_offset", 0)
                raw_phases = spec.get("phases", None)
            elif isinstance(spec, list):
                raw_cycle = spec
                offset = 0
                raw_phases = None
            else:
                continue
            # Support both formats: detailed (phases) and shorthand (cycle)
            if raw_phases:
                phases = [
                    TrafficPhase(
                        signal=p["signal"],
                        duration_ticks=p["duration_ticks"],
                        allowed_approaches=p.get("allowed_approaches", []),
                    )
                    for p in raw_phases
                ]
            else:
                phases = [
                    TrafficPhase(signal=s, duration_ticks=d)
                    for s, d in raw_cycle
                ]
            self.traffic_lights[node_id] = TrafficLightSchedule(
                intersection_id=node_id,
                phases=phases,
                phase_offset=offset,
            )

        # Road events — convert start_tick/end_tick (tick indices) to seconds
        tick_to_sec = tick_interval_s
        for evt in scenario_dict.get("road_events", []):
            self.road_events.append(RoadEventSchedule(
                edge_id=evt["edge_id"],
                event_type=evt["event_type"],
                start_sec=evt["start_tick"] * tick_to_sec,
                end_sec=evt["end_tick"] * tick_to_sec,
                severity=evt.get("severity", "moderate"),
                description=evt.get("description", ""),
                speed_limit_in_zone=evt.get("speed_limit_in_zone", 30),
                lanes_affected=evt.get("lanes_affected", 1),
                detour_available=evt.get("detour_available", False),
                estimated_delay_minutes=evt.get("estimated_delay_minutes", 10),
            ))

        # Speed cameras
        for cam in scenario_dict.get("speed_cameras", []):
            self.speed_cameras.append(SpeedCamera(
                edge_id=cam["edge_id"],
                position_ratio=cam.get("position_ratio", 0.5),
                speed_limit=cam.get("speed_limit", 60),
                camera_type=cam.get("camera_type", "fixed"),
            ))

        # Remember which lights came from the scenario (don't override offsets)
        self._explicit_traffic_lights: set = set(tl_data.keys())

        # Auto-generate traffic lights for signal nodes not already covered
        self._auto_generate_traffic_lights()

    # Road-type weight for green-time allocation
    _ROAD_TYPE_WEIGHT = {
        "motorway": 5, "trunk": 5, "primary": 4, "secondary": 3,
        "tertiary": 2, "residential": 1, "unclassified": 1,
        "living_street": 1, "service": 1,
    }
    # Also match *_link variants
    for _rt in list(_ROAD_TYPE_WEIGHT):
        _ROAD_TYPE_WEIGHT[f"{_rt}_link"] = max(1, _ROAD_TYPE_WEIGHT[_rt] - 1)

    # Total green seconds to distribute among approach groups
    _TOTAL_GREEN_SECONDS = 60

    def _auto_generate_traffic_lights(self):
        """Auto-generate direction-aware traffic light schedules.

        For every node with ``signal=True`` that lacks an explicit
        schedule, this method:

        1. Computes the bearing from the signal node to each neighbor.
        2. Clusters neighbors into opposing-direction groups (±45°
           tolerance, opposite bearings ±180° merged).
        3. Allocates green ticks proportionally to the highest road-type
           weight in each group (total = ``_TOTAL_GREEN_TICKS``).
        4. Builds a phase sequence:
             [group_1 green, yellow(1), ped_green(1),
              group_2 green, yellow(1), ped_green(1), ...]
        5. Marks ``has_crosswalk=True`` on the schedule.
        6. Computes green-wave ``phase_offset`` for chains of signal
           nodes along the same street.
        """
        # Build reverse adjacency: which nodes can REACH each node
        # (incoming edges — the approaches that need a green phase)
        incoming: Dict[str, Set[str]] = defaultdict(set)
        for src, dst_list in self.adjacency.items():
            for dst in dst_list:
                incoming[dst].add(src)

        for node_id, node in self.nodes.items():
            if not node.signal or node_id in self.traffic_lights:
                continue

            # Use incoming (reverse-adjacency) for approach directions
            approach_nodes = incoming.get(node_id, set())
            if not approach_nodes:
                continue

            # ── Step 1: bearing from each approach node to this node ──
            approach_bearings: List[Tuple[str, float, str]] = []
            for nbr_id in approach_nodes:
                nbr = self.nodes.get(nbr_id)
                if not nbr:
                    continue
                # Bearing: approach direction (from nbr toward this node)
                bear = _bearing(nbr.lat, nbr.lng, node.lat, node.lng)
                edge = self.get_edge_by_nodes(node_id, nbr_id)
                rtype = edge.road_type if edge else "residential"
                approach_bearings.append((nbr_id, bear, rtype))

            if not approach_bearings:
                continue

            # ── Step 2: cluster into opposing-direction groups ──
            groups = _cluster_approaches(approach_bearings)

            # ── Step 3: allocate green seconds ──
            total_weight = sum(
                max(self._ROAD_TYPE_WEIGHT.get(rt, 1) for _, _, rt in g)
                for g in groups
            )
            if total_weight == 0:
                total_weight = 1

            green_alloc: List[int] = []
            remaining = self._TOTAL_GREEN_SECONDS
            for i, g in enumerate(groups):
                w = max(self._ROAD_TYPE_WEIGHT.get(rt, 1) for _, _, rt in g)
                if i == len(groups) - 1:
                    secs = remaining  # give remainder to last group
                else:
                    secs = max(5, round(self._TOTAL_GREEN_SECONDS * w / total_weight))
                    remaining -= secs
                green_alloc.append(max(5, secs))

            # ── Step 4: build phase sequence (durations in seconds) ──
            phases: List[TrafficPhase] = []
            for g, green_sec in zip(groups, green_alloc):
                approach_ids = [nbr_id for nbr_id, _, _ in g]
                phases.append(TrafficPhase(
                    signal="green",
                    duration_ticks=green_sec,
                    allowed_approaches=approach_ids,
                ))
                phases.append(TrafficPhase(
                    signal="yellow",
                    duration_ticks=3,       # 3 seconds yellow
                    allowed_approaches=approach_ids,
                ))
                phases.append(TrafficPhase(
                    signal="ped_green",
                    duration_ticks=15,      # 15 seconds pedestrian crossing
                    allowed_approaches=[],
                ))

            self.traffic_lights[node_id] = TrafficLightSchedule(
                phases=phases,
                phase_offset=0,
                intersection_id=node_id,
                has_crosswalk=True,
            )

        # ── Step 5b: mark crosswalks on signal nodes ──
        for node_id in self.traffic_lights:
            node = self.nodes.get(node_id)
            if node:
                node.has_crosswalk = True

        # ── Step 6: green-wave phase offsets ──
        self._compute_green_wave_offsets()

    def _compute_green_wave_offsets(self):
        """Compute phase_offset for green-wave coordination.

        Finds chains of signal nodes on the same named street and sets
        offsets so that a vehicle travelling at the speed limit will
        encounter successive green lights.
        """
        # Build street → ordered signal node chains
        street_signals: Dict[str, List[Tuple[str, RoadNode]]] = defaultdict(list)
        for node_id, sched in self.traffic_lights.items():
            node = self.nodes.get(node_id)
            if not node:
                continue
            # Gather street names from adjacent edges
            for nbr_id in self.adjacency.get(node_id, []):
                edge = self.get_edge_by_nodes(node_id, nbr_id)
                if edge and edge.name:
                    street_signals[edge.name].append((node_id, node))

        for street_name, node_pairs in street_signals.items():
            # Deduplicate and sort by latitude then longitude (rough spatial order)
            seen = set()
            unique: List[Tuple[str, RoadNode]] = []
            for nid, nd in node_pairs:
                if nid not in seen:
                    seen.add(nid)
                    unique.append((nid, nd))
            if len(unique) < 2:
                continue
            # Sort by bearing from first to last (use lat as primary sort)
            unique.sort(key=lambda x: (x[1].lat, x[1].lng))

            # Walk the chain, accumulating travel time (seconds) as offset
            cumulative_offset = 0
            prev_node = unique[0][1]
            for nid, nd in unique[1:]:
                dist = _haversine(prev_node.lat, prev_node.lng, nd.lat, nd.lng)
                edge = self.get_edge_by_nodes(unique[0][0], nid)
                speed_limit = edge.speed_limit if edge else 60
                speed_ms = speed_limit / 3.6
                travel_seconds = dist / max(speed_ms, 1.0)
                cumulative_offset += max(round(travel_seconds), 1)
                sched = self.traffic_lights.get(nid)
                if sched and nid not in getattr(self, '_explicit_traffic_lights', set()):
                    sched.phase_offset = cumulative_offset
                prev_node = nd

    # ── Query APIs ────────────────────────────────────────────

    def get_traffic_light(self, node_id: str, time_sec: int,
                          from_node: Optional[str] = None) -> Optional[TrafficLightState]:
        """Get traffic light state at *time_sec* (physical seconds).

        Phase durations are in seconds; no tick conversion needed.
        """
        schedule = self.traffic_lights.get(node_id)
        if schedule is None:
            return None
        return schedule.at_tick(time_sec, seconds_per_tick=1,
                                from_node=from_node)

    def get_active_events(self, edge_id: str, time_sec: int) -> List[RoadEventSchedule]:
        return [
            evt for evt in self.road_events
            if evt.edge_id == edge_id
            and evt.start_sec <= time_sec < evt.end_sec
        ]

    def get_congestion(self, edge_id: str, time_sec: int) -> Optional[CongestionEvent]:
        """Describe flow using speeds mirrored from the SUMO world."""
        level = self.get_segment_flow_level(edge_id)
        if level == "free":
            return None

        factor = self.get_segment_flow_ratio(edge_id)
        speed = self.get_segment_mean_speed_kmh(edge_id)
        _level_wait = {"moderate": 0, "heavy": 1, "gridlock": 2}
        return CongestionEvent(
            edge_id=edge_id,
            start_sec=time_sec,
            end_sec=time_sec + 1,
            level=level,
            wait_ticks=_level_wait.get(level, 0),
            speed_factor=factor,
            estimated_speed=speed,
            description=f"SUMO-observed {level}",
        )

    def get_congestion_level(self, edge_id: str, time_sec: int) -> str:
        evt = self.get_congestion(edge_id, time_sec)
        return evt.level if evt else "free"

    def get_cameras(self, edge_id: str) -> List[SpeedCamera]:
        return [cam for cam in self.speed_cameras if cam.edge_id == edge_id]

    def has_road_closure(self, edge_id: str, time_sec: int) -> bool:
        return any(
            evt.event_type == "road_closure"
            for evt in self.get_active_events(edge_id, time_sec)
        )

    # get_weather_on_edge removed — weather is global, not per-edge.

    def get_all_events_on_edge(self, edge_id: str, time_sec: int) -> List[dict]:
        events = []
        for evt in self.get_active_events(edge_id, time_sec):
            events.append({
                "type": evt.event_type,
                "severity": evt.severity,
                "speed_limit_in_zone": evt.speed_limit_in_zone,
            })
        cong = self.get_congestion(edge_id, time_sec)
        if cong:
            events.append({
                "type": "congestion",
                "level": cong.level,
                "speed_factor": cong.speed_factor,
                "estimated_speed": cong.estimated_speed,
            })
        # Crosswalk at either endpoint
        seg = self.edges.get(edge_id)
        if seg:
            for nid in (seg.from_node, seg.to_node):
                node = self.nodes.get(nid)
                if node and node.has_crosswalk:
                    light = self.get_traffic_light(nid, tick)
                    events.append({
                        "type": "crosswalk",
                        "node_id": nid,
                        "signal": light.signal if light else "none",
                        "is_pedestrian_phase": light.is_crosswalk_phase if light else False,
                    })
                    break  # one crosswalk event per edge is sufficient
        return events

    # ── Multi-vehicle queries ─────────────────────────────────

    def update_vehicle_position(self, ref: VehicleRef):
        """Update a vehicle's position (called by TrafficCoordinator)."""
        old = self._vehicle_positions.get(ref.vehicle_id)
        if old and old.segment_id and old.segment_id != ref.segment_id:
            self._seg_to_vehicles[old.segment_id].discard(ref.vehicle_id)
        if ref.segment_id:
            self._seg_to_vehicles[ref.segment_id].add(ref.vehicle_id)
        self._vehicle_positions[ref.vehicle_id] = ref

    def remove_vehicle(self, vehicle_id: str):
        old = self._vehicle_positions.pop(vehicle_id, None)
        if old and old.segment_id:
            self._seg_to_vehicles[old.segment_id].discard(vehicle_id)

    def get_vehicles_on_segment(self, segment_id: str) -> List[VehicleRef]:
        """Get all vehicles currently on a road segment."""
        return [
            self._vehicle_positions[vid]
            for vid in self._seg_to_vehicles.get(segment_id, ())
            if vid in self._vehicle_positions
        ]

    def get_lane_occupancy(self, segment_id: str, lane: int) -> List[VehicleRef]:
        """Get vehicles in a specific lane of a segment."""
        return [
            self._vehicle_positions[vid]
            for vid in self._seg_to_vehicles.get(segment_id, ())
            if vid in self._vehicle_positions
            and self._vehicle_positions[vid].lane == lane
        ]

    def get_vehicles_near(self, node_id: str, radius_m: float = 200.0) -> List[VehicleRef]:
        """Get vehicles within radius of a node."""
        center = self.nodes.get(node_id)
        if not center:
            return []
        result = []
        for v in self._vehicle_positions.values():
            v_node = self.nodes.get(v.node)
            if v_node:
                dist = _haversine(center.lat, center.lng, v_node.lat, v_node.lng)
                if dist <= radius_m:
                    v_copy = VehicleRef(
                        vehicle_id=v.vehicle_id,
                        node=v.node,
                        segment_id=v.segment_id,
                        lane=v.lane,
                        speed_kmh=v.speed_kmh,
                        heading=v.heading,
                        distance_m=dist,
                    )
                    result.append(v_copy)
        return result

    def check_lane_available(self, segment_id: str, lane: int,
                             exclude_vehicle: str = "") -> bool:
        """Check if a lane is free (no vehicle occupying it)."""
        for vid in self._seg_to_vehicles.get(segment_id, ()):
            if vid == exclude_vehicle:
                continue
            v = self._vehicle_positions.get(vid)
            if v and v.lane == lane:
                return False
        return True

    # ── SUMO-observed traffic flow ─────────────────────────────────

    def get_segment_vehicle_count(self, segment_id: str,
                                  exclude_vehicle: str = "") -> int:
        """Count vehicles currently on a segment."""
        vids = self._seg_to_vehicles.get(segment_id)
        if not vids:
            return 0
        count = len(vids)
        if exclude_vehicle and exclude_vehicle in vids:
            count -= 1
        return count

    def get_segment_mean_speed_kmh(
        self, segment_id: str, exclude_vehicle: str = "",
    ) -> float:
        """Mean speed reported by synchronized SUMO actors on a segment."""
        seg = self.edges.get(segment_id)
        if not seg:
            return 0.0
        refs = [
            self._vehicle_positions[vehicle_id]
            for vehicle_id in self._seg_to_vehicles.get(segment_id, ())
            if vehicle_id in self._vehicle_positions
            and vehicle_id != exclude_vehicle
        ]
        if not refs:
            return float(seg.speed_limit)
        return sum(max(0.0, ref.speed_kmh) for ref in refs) / len(refs)

    def get_segment_flow_ratio(
        self, segment_id: str, exclude_vehicle: str = "",
    ) -> float:
        """Observed mean-speed ratio against the mapped speed limit."""
        seg = self.edges.get(segment_id)
        if not seg:
            return 1.0
        speed = self.get_segment_mean_speed_kmh(
            segment_id, exclude_vehicle)
        return max(0.0, min(1.0, speed / max(1.0, seg.speed_limit)))

    def get_segment_flow_level(self, segment_id: str) -> str:
        """Classify the current SUMO-observed mean-speed ratio."""
        ratio = self.get_segment_flow_ratio(segment_id)
        if ratio >= 0.8:
            return "free"
        if ratio >= 0.55:
            return "moderate"
        if ratio >= 0.25:
            return "heavy"
        return "gridlock"

    def get_route_traffic_info(self, route: List[str], time_sec: int = 0) -> List[dict]:
        """Get traffic info for each segment in a route."""
        segments = []
        for i in range(len(route) - 1):
            edge_id = self.make_edge_id(route[i], route[i + 1])
            seg = self.edges.get(edge_id)
            flow_ratio = self.get_segment_flow_ratio(edge_id)
            observed_speed = self.get_segment_mean_speed_kmh(edge_id)
            vcount = self.get_segment_vehicle_count(edge_id)
            segments.append({
                "from_node": route[i],
                "to_node": route[i + 1],
                "edge_id": edge_id,
                "road_name": seg.name if seg else "",
                "distance_meters": seg.distance_meters if seg else 0,
                "speed_limit": seg.speed_limit if seg else 60,
                "lanes": seg.lanes if seg else 2,
                "vehicle_count": vcount,
                "flow_ratio": round(flow_ratio, 3),
                "observed_speed_kmh": round(observed_speed, 1),
            })
        return segments

    # ── Road name helpers ─────────────────────────────────────

    def get_segment_road_name(self, edge_id: str) -> str:
        """Get a human-readable road name for a segment.

        Priority: segment.name → to_node.street_name → from_node.street_name → "(支路)"
        """
        seg = self.edges.get(edge_id)
        if seg and seg.name:
            return seg.name
        # edge_id format: "min_max" — check both nodes
        from_id = seg.from_node if seg else ""
        to_id = seg.to_node if seg else ""
        if not from_id and "_" in edge_id:
            parts = edge_id.split("_", 1)
            from_id, to_id = parts[0], parts[1]
        to_node = self.nodes.get(to_id)
        if to_node and to_node.street_name:
            return to_node.street_name
        from_node = self.nodes.get(from_id)
        if from_node and from_node.street_name:
            return from_node.street_name
        if seg:
            return f"({seg.road_type}路)"
        return "(支路)"

    def get_route_as_roads(self, route: List[str], time_sec: int = 0) -> List[dict]:
        """Convert a node-level route into road-level segments.

        Each segment is listed individually with its pre-assigned unique name.
        Returns a list of: {"road", "distance_m", "speed_limit", "traffic",
                            "from_node", "to_node"}
        """
        if len(route) < 2:
            return []

        roads = []
        for i in range(len(route) - 1):
            edge_id = self.make_edge_id(route[i], route[i + 1])
            road_name = self.get_segment_road_name(edge_id)
            seg = self.edges.get(edge_id)
            dist = seg.distance_meters if seg else 0
            limit = seg.speed_limit if seg else 60
            flow_ratio = self.get_segment_flow_ratio(edge_id)
            roads.append({
                "road": road_name,
                "distance_m": round(dist),
                "speed_limit": limit,
                "traffic": self._flow_to_traffic(flow_ratio),
                "from_node": route[i],
                "to_node": route[i + 1],
            })

        return roads

    def compute_bearing(self, from_node_id: str, to_node_id: str) -> float:
        """Compute bearing (degrees, 0=north, clockwise) between two nodes."""
        n1 = self.nodes.get(from_node_id)
        n2 = self.nodes.get(to_node_id)
        if not n1 or not n2:
            return 0.0
        lat1, lng1 = math.radians(n1.lat), math.radians(n1.lng)
        lat2, lng2 = math.radians(n2.lat), math.radians(n2.lng)
        dlng = lng2 - lng1
        x = math.sin(dlng) * math.cos(lat2)
        y = (math.cos(lat1) * math.sin(lat2)
             - math.sin(lat1) * math.cos(lat2) * math.cos(dlng))
        bearing = math.degrees(math.atan2(x, y))
        return bearing % 360

    def compute_turn_angle(self, from_node: str, via_node: str,
                           to_node: str) -> float:
        """Compute the turn angle at via_node.

        Returns degrees [0, 180]. ~0 = straight, ~180 = U-turn.
        """
        incoming = self.compute_bearing(from_node, via_node)
        outgoing = self.compute_bearing(via_node, to_node)
        # The "straight" direction is continuing the incoming bearing
        # Turn angle = difference between outgoing and the continuation of incoming
        continuation = incoming  # straight ahead = same bearing
        diff = abs(outgoing - continuation)
        if diff > 180:
            diff = 360 - diff
        return diff

    @staticmethod
    def _flow_to_traffic(flow_ratio: float) -> str:
        """Convert SUMO-observed speed ratio to a traffic description."""
        if flow_ratio >= 0.8:
            return "畅通"
        if flow_ratio >= 0.55:
            return "缓行"
        return "拥堵"

    # ── Route planning (Dijkstra) ─────────────────────────────

    def edge_cost(self, from_node: str, to_node: str,
                  time_sec: int, oracle: bool = False) -> int:
        """Estimate relative cost using observed SUMO flow and road events."""
        edge_id = self.make_edge_id(from_node, to_node)

        if self.has_road_closure(edge_id, time_sec):
            return 9999

        combined_factor = 1.0

        combined_factor *= self.get_segment_flow_ratio(edge_id)

        edge = self.get_edge_by_nodes(from_node, to_node)
        for evt in self.get_active_events(edge_id, time_sec):
            if evt.event_type != "road_closure" and edge:
                factor = evt.speed_limit_in_zone / edge.speed_limit
                combined_factor *= min(factor, 1.0)

        traversal_cost = math.ceil(1.0 / max(combined_factor, 0.01))

        cost = traversal_cost
        light = self.get_traffic_light(to_node, time_sec, from_node=from_node)
        if light and light.signal == "red":
            cost += 1

        return cost

    def plan_route(self, start: str, end: str,
                   time_sec: int = 0,
                   oracle: bool = False) -> Optional[List[str]]:
        """Dijkstra shortest-cost route."""
        if start not in self.nodes or end not in self.nodes:
            return None
        if start == end:
            return [start]

        dist: Dict[str, int] = {start: 0}
        prev: Dict[str, Optional[str]] = {start: None}
        pq: List[Tuple[int, str]] = [(0, start)]
        visited: Set[str] = set()

        while pq:
            cost_here, node = heapq.heappop(pq)
            if node in visited:
                continue
            visited.add(node)
            if node == end:
                break
            for neighbor in self.adjacency.get(node, []):
                if neighbor in visited:
                    continue
                eval_sec = time_sec + cost_here if oracle else time_sec
                cost = self.edge_cost(node, neighbor, eval_sec, oracle=oracle)
                arrival = cost_here + cost
                if neighbor not in dist or arrival < dist[neighbor]:
                    dist[neighbor] = arrival
                    prev[neighbor] = node
                    heapq.heappush(pq, (arrival, neighbor))

        if end not in prev:
            return None
        path = []
        node = end
        while node is not None:
            path.append(node)
            node = prev[node]
        path.reverse()
        return path

    def estimate_route_cost(self, route: List[str],
                            time_sec: int,
                            oracle: bool = False) -> int:
        """Estimate total cost to traverse a route."""
        if not route or len(route) < 2:
            return 0
        total = 0
        current_sec = time_sec
        for i in range(len(route) - 1):
            eval_sec = current_sec if oracle else time_sec
            cost = self.edge_cost(route[i], route[i + 1], eval_sec,
                                  oracle=oracle)
            total += cost
            current_sec += cost
        return total

    # ── Helpers ─────────────────────────────────────────────────

    def neighbors(self, node_id: str) -> List[str]:
        return self.adjacency.get(node_id, [])

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def edge_count(self) -> int:
        return len(self.edges)

    def __repr__(self):
        return (f"RoadNetwork(nodes={self.node_count}, "
                f"segments={self.edge_count})")

    # ── Serialization ──────────────────────────────────────────

    def to_dict(self) -> dict:
        """Serialize topology to a JSON-friendly dict."""
        result = {
            "nodes": [
                {
                    "id": n.id, "lat": n.lat, "lng": n.lng,
                    "name": n.name, "street_name": n.street_name,
                    "avenue_name": n.avenue_name, "signal": n.signal,
                    "has_crosswalk": n.has_crosswalk,
                    "poi": n.poi, "osm_id": n.osm_id,
                }
                for n in self.nodes.values()
            ],
            "segments": [
                {
                    "id": s.id, "from_node": s.from_node,
                    "to_node": s.to_node, "name": s.name,
                    "road_type": s.road_type, "lanes": s.lanes,
                    "lanes_forward": s.lanes_forward,
                    "lanes_backward": s.lanes_backward,
                    "turn_lanes_forward": s.turn_lanes_forward,
                    "turn_lanes_backward": s.turn_lanes_backward,
                    "lane_width_meters": s.lane_width_meters,
                    "speed_limit": s.speed_limit,
                    "distance_meters": s.distance_meters,
                    "oneway": s.oneway,
                    "geometry": s.geometry,
                    "osm_way_id": s.osm_way_id,
                    "layer": s.layer,
                    "bridge": s.bridge,
                    "tunnel": s.tunnel,
                    "source_edge_count": s.source_edge_count,
                }
                for s in self.edges.values()
            ],
        }
        # Only serialize non-default config values
        default = TrafficConfig()
        tc = {}
        for fld in ("perception_range_m", "min_gap_m",
                     "stop_line_offset_m",
                     "stopped_speed_threshold_kmh",
                     "default_signal_cycle",
                     "seconds_per_tick"):
            val = getattr(self.config, fld)
            if val != getattr(default, fld):
                tc[fld] = val
        if tc:
            result["traffic_config"] = tc
        return result

    def _number_same_name_segments(self):
        """Number segments that share the same road name.

        After this, each segment has a unique display name.
        E.g., three segments named "北四环西路" become "北四环西路-1", "-2", "-3".
        Segments with a unique name are left unchanged.
        Ordering is by geographic position (lat/lng of from_node).
        """
        # Group segments by their resolved road name
        name_groups: Dict[str, List[RoadSegment]] = defaultdict(list)
        for seg in self.edges.values():
            # Use the full name resolution logic
            resolved = self.get_segment_road_name(seg.id)
            name_groups[resolved].append(seg)

        # Number groups with more than one segment
        for name, segs in name_groups.items():
            if len(segs) <= 1:
                continue
            # Sort by from_node geographic position for stable ordering
            def sort_key(s):
                node = self.nodes.get(s.from_node)
                if node:
                    return (node.lat, node.lng)
                return (0, 0)
            segs.sort(key=sort_key)
            for i, seg in enumerate(segs, 1):
                seg.name = f"{name}-{i}"

    @classmethod
    def from_dict(cls, data: dict) -> "RoadNetwork":
        """Deserialize from a dict (e.g. loaded from JSON)."""
        # Build TrafficConfig from JSON if present
        tc_data = data.get("traffic_config", {})
        config = TrafficConfig(**tc_data) if tc_data else TrafficConfig()
        net = cls(traffic_config=config)
        for nd in data.get("nodes", []):
            net.add_node(RoadNode(
                id=nd["id"],
                lat=nd["lat"],
                lng=nd["lng"],
                name=nd.get("name", ""),
                street_name=nd.get("street_name", ""),
                avenue_name=nd.get("avenue_name", ""),
                signal=nd.get("signal", False),
                has_crosswalk=nd.get("has_crosswalk", False),
                poi=nd.get("poi", []),
                osm_id=nd.get("osm_id", 0),
            ))
        for sd in data.get("segments", []):
            net.add_segment(RoadSegment(
                id=sd["id"],
                from_node=sd["from_node"],
                to_node=sd["to_node"],
                name=sd.get("name", ""),
                road_type=sd.get("road_type", "secondary"),
                lanes=sd.get("lanes", 2),
                lanes_forward=sd.get("lanes_forward"),
                lanes_backward=sd.get("lanes_backward"),
                turn_lanes_forward=sd.get(
                    "turn_lanes_forward", []) or [],
                turn_lanes_backward=sd.get(
                    "turn_lanes_backward", []) or [],
                lane_width_meters=sd.get("lane_width_meters", 3.5),
                speed_limit=sd.get("speed_limit", 60),
                distance_meters=sd.get("distance_meters", 500.0),
                oneway=sd.get("oneway", False),
                geometry=sd.get("geometry", []),
                osm_way_id=sd.get("osm_way_id", 0),
                layer=sd.get("layer", 0),
                bridge=sd.get("bridge", False),
                tunnel=sd.get("tunnel", False),
                source_edge_count=sd.get("source_edge_count", 1),
            ))
        net._number_same_name_segments()
        return net

    def save_json(self, path: str):
        """Save topology to a JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load_json(cls, path: str) -> "RoadNetwork":
        """Load topology from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


# ── Utility functions ──────────────────────────────────────────

def _haversine(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Haversine distance in meters between two WGS84 points."""
    R = 6371000  # Earth radius in meters
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1))
         * math.cos(math.radians(lat2))
         * math.sin(dlng / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _bearing(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Initial bearing from (lat1,lng1) to (lat2,lng2) in degrees [0, 360)."""
    dlon = math.radians(lng2 - lng1)
    lat1r, lat2r = math.radians(lat1), math.radians(lat2)
    x = math.sin(dlon) * math.cos(lat2r)
    y = (math.cos(lat1r) * math.sin(lat2r)
         - math.sin(lat1r) * math.cos(lat2r) * math.cos(dlon))
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def _angle_diff(a: float, b: float) -> float:
    """Smallest unsigned angle between two bearings in [0, 180]."""
    d = abs(a - b) % 360
    return d if d <= 180 else 360 - d


def _cluster_approaches(
    approaches: List[Tuple[str, float, str]],
    tolerance: float = 45.0,
) -> List[List[Tuple[str, float, str]]]:
    """Cluster approach bearings into opposing-direction groups.

    Two approaches are "opposing" if their bearings differ by ~180°
    (within *tolerance*).  Two approaches are "same direction" if
    they differ by < *tolerance* (merged into same cluster).

    Args:
        approaches: [(node_id, bearing_degrees, road_type), ...]
        tolerance: angular tolerance in degrees.

    Returns:
        List of groups, each group a list of (node_id, bearing, road_type).
        Typically 1–3 groups for real-world intersections.
    """
    if not approaches:
        return []
    if len(approaches) == 1:
        return [approaches]

    # Sort by bearing for determinism
    sorted_app = sorted(approaches, key=lambda x: x[1])
    assigned = [False] * len(sorted_app)
    groups: List[List[Tuple[str, float, str]]] = []

    for i, (nid_i, bear_i, rt_i) in enumerate(sorted_app):
        if assigned[i]:
            continue
        group = [(nid_i, bear_i, rt_i)]
        assigned[i] = True
        for j, (nid_j, bear_j, rt_j) in enumerate(sorted_app):
            if assigned[j]:
                continue
            # Check if same direction or opposing
            diff = _angle_diff(bear_i, bear_j)
            if diff < tolerance or abs(diff - 180) < tolerance:
                group.append((nid_j, bear_j, rt_j))
                assigned[j] = True
        groups.append(group)

    return groups
