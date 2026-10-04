"""Rule-based automatic ground truth derivation for VehicleArena.

Given a deterministic world state at a wake (weather, day/night and map events),
this module computes expected cabin-rule actions as executable
code strings.  The rules are simple condition→action mappings, reflecting the
tool-calling benchmark philosophy: difficulty comes from discovering the right APIs
and chaining tool calls, not from complex reasoning.

Usage:
    from simulation.ground_truth_rules import derive_ground_truth

    code_lines = derive_ground_truth(vw, prev_snapshot=previous)
    # code_lines is a list of strings like
    #   "vw.wiper.carcontrol_wiperBlade_switch(True)"
    # that can be exec'd on a reference VW to produce the expected state.

Design principles:
    1. Rules are **cumulative** — once an action is triggered (e.g. wipers ON for rain),
       it stays active until an explicit reversal rule fires (rain stops → wipers OFF).
    2. Automatic world-state rules never emit passenger-facing broadcasts;
       broadcast tools are evaluated only for explicit passenger requests.
    3. Each rule produces one or more code lines.  The lines are collected in order and
       concatenated with newlines.
    4. The rule engine is stateless per tick — it looks at the current world state and
       previous state to detect transitions.  The SimulationEngine passes both.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any

from capabilities import action_module_name
from weather_safety import (
    WEATHER_SAFETY_MODULES, apply_weather_safety_transition,
    weather_equipment_action_lines,
)


# ---------------------------------------------------------------------------
# Rule engine
# ---------------------------------------------------------------------------

@dataclass
class AcceptableAction:
    """A primary action with optional alternatives that are also acceptable.

    When verifying agent behavior, the primary is executed on _expect_vw.
    During checkpoint comparison, if the agent's value doesn't match primary
    but matches one of the alternatives, it still counts as correct.
    """
    primary: str                               # executed on _expect_vw
    alternatives: List[str] = field(default_factory=list)  # also correct
    weight: float = 1.0                        # scoring weight


@dataclass
class TrendTolerance:
    """Declares that a numerical field accepts trend-based matching.

    Instead of requiring an exact value, the agent just needs to move the
    value in the correct direction relative to *baseline_value*.

    Example: passenger says "it's too hot" → direction="decrease", the
    agent can set temperature to any value below baseline.
    """
    field_pattern: str         # glob pattern, e.g. "airConditioner.ac_states.*.temperature"
    direction: str             # "decrease", "increase", or "any_change"
    baseline_value: float      # value before the action (agent should move away from this)


@dataclass
class NegativeCheck:
    """A field that should NOT have a specific value at a checkpoint.

    Used to verify the agent doesn't do things it shouldn't
    (e.g., wipers ON when sunny, headlights ON during daytime).
    """
    id: str = ""
    field_path: str = ""         # e.g. "wiper._enabled"
    forbidden_value: Any = True  # the value it should NOT be
    reason: str = ""
    severity: str = "violation"


# ---------------------------------------------------------------------------
# Passenger rule registry — declarative category→action mapping
# ---------------------------------------------------------------------------



# ── Helper functions used by handlers / trend baselines ──────────────

def _resolve_baseline(vw, field_path: str) -> Optional[float]:
    """Read a numeric value from vw by dot-separated field path.

    Supports wildcard '*' for dict/iterable attributes — takes the first match.
    Examples:
        "HUD.brightness_level"                                → vw.HUD.brightness_level
        "airConditioner._ac_states.*.temperature"             → first seat's temperature
        "music._settings.volume"                              → vw.music._settings.volume
        "centerInformationDisplay.brightness_settings.brightness_level" → CID brightness
    """
    if vw is None or not field_path:
        return None
    try:
        obj = vw
        for part in field_path.split("."):
            if part == "*":
                # Iterate dict-like or iterable, take first value
                if hasattr(obj, 'items'):
                    obj = next(iter(obj.values()))
                elif hasattr(obj, '__iter__'):
                    obj = next(iter(obj))
                else:
                    return None
            else:
                obj = getattr(obj, part)
        return float(obj)
    except Exception:
        return None


# ── Handlers for complex rules ───────────────────────────────────────




def _get_rule_loader():
    """Return the single process-wide loader and fail loudly if it is invalid."""
    from rules.rule_loader import get_rule_loader
    return get_rule_loader()






# ---------------------------------------------------------------------------
# World snapshot
# ---------------------------------------------------------------------------

@dataclass
class WorldSnapshot:
    """Snapshot of the world state relevant for rule evaluation."""
    weather_condition: str = "sunny"
    daynight_period: str = "afternoon"
    # Map events (only relevant for grid scenarios)
    road_events_ahead: List[Dict] = field(default_factory=list)
    speed_cameras_ahead: List[Dict] = field(default_factory=list)
    congestion_level: str = "free"  # from current road
    intersections_ahead: List[Dict] = field(default_factory=list)
    # Navigation
    is_navigating: bool = False
    vehicle_arrived: bool = False
    # Driving state (for LLM agent driving GT)
    is_llm_vehicle: bool = False
    is_crashed: bool = False
    vehicle_speed_kmh: float = 0.0
    vehicle_at_intersection: str = ""   # node ID when at intersection, "" when on segment
    vehicles_ahead: List[Dict] = field(default_factory=list)
    # Segment driving context
    segment_speed_limit: float = 60.0       # current segment speed limit
    segment_is_narrow: bool = False         # single-lane or single-direction lane
    has_oncoming_vehicle: bool = False       # oncoming vehicle on narrow road
    next_segment_blocked: bool = False       # next segment closed or collision-blocked
    at_red_light: bool = False              # facing red light at next intersection
    pedestrian_on_crosswalk_ahead: bool = False  # pedestrian crossing at upcoming intersection
    # Computed driving triggers (pre-digested for YAML rule matching)
    vehicle_in_proximity: bool = False      # closest ahead vehicle within threshold
    closest_ahead_speed: float = 0.0        # speed of closest ahead vehicle (km/h)


def take_world_snapshot(vw, vehicle_state=None) -> WorldSnapshot:
    """Extract current world state from VehicleWorld instance."""
    snap = WorldSnapshot()

    # Weather
    try:
        w = vw.externalWorld.weather
        if hasattr(w, '_condition'):
            snap.weather_condition = w._condition.value.lower()
    except Exception:
        pass

    # DayNight
    try:
        dn = vw.externalWorld.dayNight
        snap.daynight_period = dn._time_of_day.value.lower()
    except Exception:
        pass

    # Map events
    try:
        m = vw.externalWorld.map
        if hasattr(m, '_road_events_ahead') and m._road_events_ahead:
            snap.road_events_ahead = list(m._road_events_ahead)
        if hasattr(m, '_speed_cameras_ahead') and m._speed_cameras_ahead:
            snap.speed_cameras_ahead = list(m._speed_cameras_ahead)
        if hasattr(m, '_intersections_ahead') and m._intersections_ahead:
            snap.intersections_ahead = list(m._intersections_ahead)
    except Exception:
        pass

    # Navigation state
    if vehicle_state:
        snap.is_navigating = getattr(vehicle_state, 'is_navigating', False)
        snap.vehicle_arrived = getattr(vehicle_state, 'arrived', False)
        snap.is_llm_vehicle = getattr(vehicle_state, 'is_llm', False)
        snap.is_crashed = getattr(vehicle_state, 'is_crashed', False)
        snap.vehicle_speed_kmh = getattr(vehicle_state, 'current_speed_kmh', 0.0)
        at_intersection = (
            getattr(vehicle_state, 'edge_progress', 0.0) <= 0.01
            and not getattr(vehicle_state, 'current_segment', '')
        )
        snap.vehicle_at_intersection = (
            getattr(vehicle_state, 'current_node', '') if at_intersection else ""
        )

    return snap


def derive_ground_truth(
    vw,
    prev_snapshot: Optional[WorldSnapshot] = None,
    vehicle_state=None,
) -> Tuple[
    List[str], WorldSnapshot, List[AcceptableAction], List[TrendTolerance],
]:
    """Derive expected agent actions for the current tick.

    Args:
        vw: The VehicleWorld instance (already synced to this tick's world state).
        prev_snapshot: WorldSnapshot from the previous tick (None for tick 0).
        vehicle_state: Vehicle state used only for factual cabin snapshot data.

    Returns:
        (code_lines, current_snapshot): code_lines is a list of executable code
        strings; current_snapshot should be passed as prev_snapshot to the next tick.
    """
    current = take_world_snapshot(vw, vehicle_state)
    prev = prev_snapshot or WorldSnapshot()  # default = sunny afternoon, no events

    code_lines: List[str] = []
    acceptable_actions: List[AcceptableAction] = []
    trend_tolerances: List[TrendTolerance] = []

    # -------------------------------------------------------------------------
    # 2. Weather transition rules
    # -------------------------------------------------------------------------
    _apply_weather_rules(prev, current, code_lines,
                         acceptable_actions=acceptable_actions)

    # -------------------------------------------------------------------------
    # 3. DayNight transition rules
    # -------------------------------------------------------------------------
    _apply_daynight_rules(prev, current, code_lines,
                          is_first_tick=(prev_snapshot is None),
                          acceptable_actions=acceptable_actions,
                          trend_tolerances=trend_tolerances,
                          vw=vw)

    # -------------------------------------------------------------------------
    # 4. Map event rules (road events, speed cameras, congestion)
    # -------------------------------------------------------------------------
    _apply_map_event_rules(prev, current, code_lines,
                           acceptable_actions=acceptable_actions)

    # -------------------------------------------------------------------------
    # 7. Negative checks (what should NOT happen)
    # -------------------------------------------------------------------------
    # A missing equipment module makes an automatic cabin action inapplicable.
    code_lines = [
        line for line in code_lines
        if (
            action_module_name(line) is None
            or not hasattr(vw, "has_module")
            or vw.has_module(action_module_name(line))
        )
    ]

    return code_lines, current, acceptable_actions, trend_tolerances

def _apply_weather_rules(prev: WorldSnapshot, current: WorldSnapshot,
                         code_lines: List[str],
                         acceptable_actions: List[AcceptableAction] = None):
    """Use shared external-equipment rules and retain YAML cabin comfort rules."""
    prev_cond = prev.weather_condition
    curr_cond = current.weather_condition
    aa = acceptable_actions

    if prev_cond == curr_cond:
        return

    _, _, _, updates = apply_weather_safety_transition(prev_cond, curr_cond)
    code_lines.extend(weather_equipment_action_lines(
        updates, dark_period=current.daynight_period in {"dusk", "night", "dawn"}))

    loader = _get_rule_loader()
    if loader is None:
        return

    matched = loader.match_weather(prev_cond, curr_cond)
    for rule in matched:
        # Historical YAML external-equipment actions can contradict the
        # current process scorer (e.g. rain -> hail wipers). Only cabin
        # comfort actions still come from those rules.
        code_lines.extend(
            action for action in rule.expected_actions
            if action_module_name(action) not in WEATHER_SAFETY_MODULES)
        if rule.broadcasts and curr_cond in rule.broadcasts:
            warning = rule.broadcasts[curr_cond]
            code_lines.append(f"vw.broadcast.broadcast_warning('{warning}')")
        if rule.ignored_fields and aa is not None:
            aa.append(AcceptableAction(
                primary=(rule.expected_actions[0]
                         if rule.expected_actions else ""),
                alternatives=[f"*:{p}" for p in rule.ignored_fields],
            ))


def _apply_daynight_rules(prev: WorldSnapshot, current: WorldSnapshot,
                          code_lines: List[str], is_first_tick: bool = False,
                          acceptable_actions: List[AcceptableAction] = None,
                          trend_tolerances: List[TrendTolerance] = None,
                          vw=None):
    """Handle daynight period transitions — reads from YAML rules via RuleLoader."""
    prev_period = prev.daynight_period
    curr_period = current.daynight_period
    aa = acceptable_actions

    loader = _get_rule_loader()
    if loader is None:
        return

    matched = loader.match_daynight(prev_period, curr_period,
                                     is_first_tick=is_first_tick)
    for rule in matched:
        code_lines.extend(rule.expected_actions)
        if rule.broadcasts and curr_period in rule.broadcasts:
            warning = rule.broadcasts[curr_period]
            code_lines.append(f"vw.broadcast.broadcast_warning('{warning}')")
        if rule.ignored_fields and aa is not None:
            aa.append(AcceptableAction(
                primary=(rule.expected_actions[0]
                         if rule.expected_actions else ""),
                alternatives=[f"*:{p}" for p in rule.ignored_fields],
            ))
        if rule.trend_spec and trend_tolerances is not None:
            baseline_path = rule.trend_spec.get("baseline_field", "")
            baseline = _resolve_baseline(vw, baseline_path) if baseline_path else None
            if baseline is not None:
                trend_tolerances.append(TrendTolerance(
                    field_pattern=rule.trend_spec["field_pattern"],
                    direction=rule.trend_spec["direction"],
                    baseline_value=baseline,
                ))


def _apply_map_event_rules(prev: WorldSnapshot, current: WorldSnapshot,
                           code_lines: List[str],
                           acceptable_actions: List[AcceptableAction] = None):
    """Handle map events (road events, speed cameras, congestion).

    All logic is driven by YAML rules from map_event_rules.yaml via RuleLoader.
    """
    aa = acceptable_actions
    loader = _get_rule_loader()

    # Collect YAML rule config
    warning_map: Dict[str, str] = {}
    action_template = ""
    skip_on: list = []
    skip_fields: list = []
    cam_template = ""
    cam_template_no_type = ""
    cam_warning = ""

    if loader is not None:
        for rule in loader.match_map_events():
            if rule.warning_map:
                warning_map.update(rule.warning_map)
            template = rule.expect.get("action_template", "")
            if template:
                action_template = template
                skip_on = rule.skip_on
                skip_fields = rule.ignored_fields
            speed_template = rule.expect.get("speed_camera_template", "")
            if speed_template:
                cam_template = speed_template
                cam_template_no_type = rule.expect.get(
                    "speed_camera_template_no_type", "")
                cam_warning = rule.speed_camera_warning

    # Road events: broadcast any newly appeared events ahead
    prev_event_types = set()
    for evt in prev.road_events_ahead:
        prev_event_types.add(evt.get("type", ""))

    for evt in current.road_events_ahead:
        evt_type = evt.get("type", "")
        distance = evt.get("distance_meters", 500)

        if evt_type and evt_type not in prev_event_types:
            if action_template:
                code_lines.append(action_template.format(
                    event_type=evt_type, distance=distance))
            warning = warning_map.get(evt_type)
            if warning:
                code_lines.append(f"vw.broadcast.broadcast_warning('{warning}')")
            if aa is not None and evt_type in skip_on and skip_fields:
                code_lines_last = code_lines[-1] if code_lines else ""
                aa.append(AcceptableAction(
                    primary=code_lines_last,
                    alternatives=[f"*:{p}" for p in skip_fields],
                ))

    # Speed cameras: broadcast any newly visible cameras
    def _cam_sig(cam):
        return (cam.get("distance_meters", 0), cam.get("speed_limit", 0),
                cam.get("type", ""))
    prev_cam_sigs = {_cam_sig(c) for c in prev.speed_cameras_ahead}
    for cam in current.speed_cameras_ahead:
        sig = _cam_sig(cam)
        if sig not in prev_cam_sigs:
            distance = cam.get("distance_meters", 300)
            speed_limit = cam.get("speed_limit", 50)
            cam_type = cam.get("type", "")
            if cam_type and cam_template:
                code_lines.append(cam_template.format(
                    distance=distance, speed_limit=speed_limit, cam_type=cam_type))
            elif cam_template_no_type:
                code_lines.append(cam_template_no_type.format(
                    distance=distance, speed_limit=speed_limit))
            if cam_warning:
                code_lines.append(
                    f"vw.broadcast.broadcast_warning('{cam_warning}')")



# ---------------------------------------------------------------------------
# Convenience: compile code_lines into executable ground truth code string
# ---------------------------------------------------------------------------

def derive_negative_checks(
    current: WorldSnapshot,
    passenger_messages: List[str],
    vw=None,
) -> List[NegativeCheck]:
    """Derive negative checks — things the agent should NOT do.

    Reads declarative rules from global_config.yaml via RuleLoader.
    Each matched rule becomes a NegativeCheck.
    """
    loader = _get_rule_loader()
    if loader is None:
        return []

    matched = loader.match_negative_checks(current, passenger_messages)
    results = []
    for nc in matched:
        module_name = (nc.field or "").split(".", 1)[0]
        if (
            vw is not None
            and hasattr(vw, "has_module")
            and module_name
            and not vw.has_module(module_name)
        ):
            continue
        results.append(NegativeCheck(
            id=nc.id,
            field_path=nc.field,
            forbidden_value=nc.forbidden,
            reason=nc.reason,
            severity=nc.severity,
        ))
    return results
