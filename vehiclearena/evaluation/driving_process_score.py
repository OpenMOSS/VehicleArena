"""Deterministic exam-style score for one vehicle's driving process.

The legacy :mod:`evaluation.driving_evaluator` remains the authority for
trajectory diagnostics and risk episode construction.  This module owns the
100-point, deduction-based metric exposed by the evaluation protocol.
It samples only authoritative simulator/module state and consumes the legacy
risk episodes at finalisation; model prose is never scoring evidence.
"""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from evaluation.trajectory_filters import (
    windowed_longitudinal_acceleration,
)
from evaluation.signal_context import has_restrictive_approach_signal
from weather_safety import (
    FOG_VISIBILITY_CONDITIONS,
    normalize_weather_condition,
    weather_score_requirements,
)

METRIC_REVISION = "at_fault_collision_red_light_hard_gate_v9"
HARD_GATE_POINTS = 100


@dataclass(frozen=True)
class DrivingProcessScoreConfig:
    """Frozen thresholds and point values for the first scoring version."""

    moving_speed_kmh: float = 0.5
    overspeed_tolerance_kmh: float = 3.0
    short_overspeed_min_s: float = 1.0
    long_overspeed_s: float = 5.0
    unexplained_stop_s: float = 3.0
    unexplained_stop_clear_s: float = 1.0
    repeated_duration_block_s: float = 5.0
    green_resume_s: float = 3.0
    signal_lead_s: float = 0.5
    signal_cancel_s: float = 3.0
    signal_turn_approach_m: float = 50.0
    curve_lateral_acceleration_mps2: float = 2.5
    curve_braking_deceleration_mps2: float = 3.0
    curve_braking_reaction_s: float = 1.0
    curve_braking_max_distance_m: float = 100.0
    curve_speed_tolerance_kmh: float = 3.0
    episode_clear_s: float = 0.5
    rule_cooldown_s: float = 3.0
    high_beam_clear_s: float = 1.0
    unsafe_lane_gap_m: float = 6.0
    unsafe_lane_rear_ttc_s: float = 2.0
    intersection_block_s: float = 3.0
    hard_acceleration_mps2: float = 2.8  # fallback when chassis capability is unknown
    hard_acceleration_min_s: float = 0.5
    hard_braking_mps2: float = 4.0
    hard_braking_min_s: float = 0.3
    high_jerk_mps3: float = 5.0
    high_jerk_min_s: float = 0.2
    jerk_smoothing_window_s: float = 1.0
    high_beam_misuse_min_s: float = 1.0
    environment_response_s: float = 2.0
    road_closure_response_s: float = 1.0
    leader_risk_ttc_s: float = 4.5
    connector_conflict_zone_ttc_s: float = 2.0
    connector_horizon_s: float = 5.0
    connector_time_separation_s: float = 1.25
    pedestrian_risk_ttc_s: float = 4.0
    vehicle_near_miss_ttc_s: float = 2.0
    pedestrian_near_miss_clearance_m: float = 1.5
    pedestrian_near_miss_min_speed_kmh: float = 5.0
    risk_response_deadline_s: float = 0.5
    risk_episode_clear_s: float = 1.0
    stop_signal_distance_m: float = 30.0
    stop_leader_gap_m: float = 15.0
    stop_oncoming_gap_m: float = 100.0
    stop_connector_ttc_s: float = 5.0
    stop_pedestrian_distance_m: float = 25.0
    relevant_horn_leader_gap_m: float = 30.0
    high_beam_oncoming_distance_m: float = 120.0
    high_beam_oncoming_bearing_deg: float = 55.0
    high_beam_oncoming_yaw_delta_deg: float = 100.0

    long_overspeed_points: int = 20
    short_overspeed_points: int = 5
    unexplained_stop_points: int = 5
    green_resume_points: int = 5
    unsignaled_lane_change_points: int = 10
    unsignaled_turn_points: int = 10
    signal_not_cancelled_points: int = 2
    unsafe_lane_change_points: int = 10
    risk_response_points: int = 10
    vehicle_near_miss_points: int = 15
    connector_near_miss_points: int = 15
    pedestrian_near_miss_points: int = 20
    intersection_block_points: int = 10
    hard_acceleration_points: int = 3
    unnecessary_hard_brake_points: int = 3
    high_jerk_points: int = 1
    unnecessary_horn_points: int = 2
    high_beam_misuse_points: int = 2
    # Environment compliance is scored by cabin_layer_score_100, not here.
    wet_wiper_points: int = 0
    fog_lights_points: int = 0
    dark_lights_points: int = 0
    severe_weather_opening_points: int = 0
    weather_cleanup_points: int = 0
    road_closure_entry_points: int = 10


@dataclass
class _Trace:
    samples: List[dict] = field(default_factory=list)
    speed_history: List[tuple[float, float]] = field(default_factory=list)
    delivered_events: List[dict] = field(default_factory=list)
    delivered_event_ids: set = field(default_factory=set)


def _finite(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _state_value(value: Any) -> str:
    """Normalize plain strings and Enum-backed module state fields."""
    return str(getattr(value, "value", value)).lower()


_PEDESTRIAN_RISK_DEDUCTION_TYPES = frozenset({
    "pedestrian_response_late",
    "pedestrian_near_miss",
})


def _coalesce_pedestrian_risk_deductions(
    deductions: Sequence[dict], *, clear_s: float,
) -> List[dict]:
    """Charge at most once for one continuous pedestrian-risk encounter.

    The legacy evaluator constructs risk independently for every pedestrian.
    That is useful evidence, but charging each member of a crowd makes one
    physical encounter scale without bound with crowd size.  Here an
    encounter remains open until there has been more than ``clear_s`` without
    pedestrian-risk evidence.  All response-late and near-miss findings in
    that interval are retained as audit evidence, while only the most severe
    deduction is charged.

    Temporal proximity joins findings from a crowd.  Findings for the same
    pedestrian may also join across a time gap when both explicitly identify
    the same physical encounter.  Legacy reports without encounter IDs use
    temporal continuity only: an entity ID alone cannot establish whether
    two findings belong to one crossing or separate appearances.
    """
    if clear_s < 0:
        raise ValueError("pedestrian encounter clear duration must be >= 0")

    indexed = [
        (index, item)
        for index, item in enumerate(deductions)
        if str(item.get("type", "")) in _PEDESTRIAN_RISK_DEDUCTION_TYPES
    ]
    if len(indexed) < 2:
        return list(deductions)

    indexed.sort(key=lambda pair: (
        float(pair[1].get("start_time_s", 0.0)),
        float(pair[1].get("end_time_s", pair[1].get("start_time_s", 0.0))),
        pair[0],
    ))
    def interval(pair: tuple[int, dict]) -> tuple[float, float]:
        item = pair[1]
        start = float(item.get("start_time_s", 0.0))
        return start, max(
            start, float(item.get("end_time_s", start)))

    def hazard(pair: tuple[int, dict]) -> str:
        return str(dict(pair[1].get("evidence", {}) or {}).get(
            "hazard", ""))

    def encounter_identity(pair: tuple[int, dict]) -> str:
        evidence = dict(pair[1].get("evidence", {}) or {})
        # Never substitute the entity ID for missing lifecycle evidence.
        return str(evidence.get("pedestrian_encounter_id") or "")

    # Connected components under either temporal continuity or an explicit
    # physical encounter identity supplied by the live evaluator.
    parents = list(range(len(indexed)))

    def root(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def join(first: int, second: int) -> None:
        first_root, second_root = root(first), root(second)
        if first_root != second_root:
            parents[second_root] = first_root

    for first in range(len(indexed)):
        first_start, first_end = interval(indexed[first])
        first_hazard = hazard(indexed[first])
        first_encounter = encounter_identity(indexed[first])
        for second in range(first + 1, len(indexed)):
            second_start, second_end = interval(indexed[second])
            temporally_connected = (
                second_start <= first_end + clear_s + 1e-9
                and first_start <= second_end + clear_s + 1e-9)
            same_encounter = bool(
                first_hazard.startswith("pedestrian:")
                and first_hazard == hazard(indexed[second])
                and first_encounter
                and first_encounter == encounter_identity(indexed[second]))
            if temporally_connected or same_encounter:
                join(first, second)

    grouped: Dict[int, List[tuple[int, dict]]] = {}
    for index, pair in enumerate(indexed):
        grouped.setdefault(root(index), []).append(pair)
    groups = sorted(
        grouped.values(),
        key=lambda group: min(interval(pair)[0] for pair in group),
    )

    suppressed: set[int] = set()
    replacements: Dict[int, dict] = {}
    for group_index, group in enumerate(groups, 1):
        if len(group) == 1:
            continue
        # Prefer the higher-point finding; on a tie retain the earliest source
        # finding so the selection is deterministic.
        winner_index, winner = max(
            group, key=lambda pair: (int(pair[1].get("points", 0)), -pair[0]))
        merged = copy.deepcopy(winner)
        starts = [float(item.get("start_time_s", 0.0)) for _, item in group]
        ends = [
            max(
                float(item.get("start_time_s", 0.0)),
                float(item.get(
                    "end_time_s", item.get("start_time_s", 0.0))),
            )
            for _, item in group
        ]
        hazards = sorted({
            str(dict(item.get("evidence", {}) or {}).get("hazard", ""))
            for _, item in group
            if dict(item.get("evidence", {}) or {}).get("hazard")
        })
        evidence = dict(merged.get("evidence", {}) or {})
        evidence["pedestrian_encounter_aggregation"] = {
            "policy": "maximum_severity_per_physical_encounter",
            "encounter_index": group_index,
            "source_finding_count": len(group),
            "suppressed_finding_count": len(group) - 1,
            "source_types": sorted({
                str(item.get("type", "")) for _, item in group
            }),
            "pedestrian_hazards": hazards,
            "clear_duration_s": round(float(clear_s), 3),
            "source_findings": [
                {
                    "type": str(item.get("type", "")),
                    "start_time_s": float(item.get("start_time_s", 0.0)),
                    "end_time_s": float(item.get(
                        "end_time_s", item.get("start_time_s", 0.0))),
                    "points": int(item.get("points", 0)),
                    "evidence": copy.deepcopy(dict(
                        item.get("evidence", {}) or {})),
                }
                for _, item in group
            ],
        }
        merged.update({
            "start_time_s": round(min(starts), 3),
            "end_time_s": round(max(ends), 3),
            "evidence": evidence,
        })
        replacements[winner_index] = merged
        suppressed.update(index for index, _ in group if index != winner_index)

    return [
        replacements.get(index, item)
        for index, item in enumerate(deductions)
        if index not in suppressed
    ]


def _episodes(samples: Sequence[dict], key: str) -> List[dict]:
    """Return contiguous true intervals for a sampled boolean field."""
    result: List[dict] = []
    started: Optional[int] = None
    for index, sample in enumerate(samples):
        active = bool(sample.get(key, False))
        if active and started is None:
            started = index
        if not active and started is not None:
            start_t = float(samples[started]["time_s"])
            end_t = float(sample["time_s"])
            result.append({
                "start_time_s": start_t,
                "end_time_s": end_t,
                "duration_s": max(0.0, end_t - start_t),
                "samples": samples[started:index],
            })
            started = None
    if started is not None and samples:
        start_t = float(samples[started]["time_s"])
        end_t = float(samples[-1]["time_s"])
        result.append({
            "start_time_s": start_t,
            "end_time_s": end_t,
            "duration_s": max(0.0, end_t - start_t),
            "samples": samples[started:],
        })
    return result


def _qualified_episodes(
    samples: Sequence[dict], key: str, *, minimum_s: float,
    clear_s: float, strict_minimum: bool = False,
) -> List[dict]:
    """Group qualified continuous runs until the declared clear interval.

    A sub-threshold true run never satisfies the duration requirement by
    being added to a later run.  Once a run has qualified, however, a brief
    false fluctuation shorter than ``clear_s`` does not create a second
    deduction episode.
    """
    raw = _episodes(samples, key)
    result: List[dict] = []
    current: Optional[dict] = None
    for interval in raw:
        duration = float(interval["duration_s"])
        qualifies = (
            duration > minimum_s + 1e-9 if strict_minimum
            else duration + 1e-9 >= minimum_s)
        if current is not None:
            gap_s = max(
                0.0,
                float(interval["start_time_s"])
                - float(current["last_active_end_time_s"]),
            )
            if gap_s + 1e-9 < clear_s:
                current["end_time_s"] = interval["end_time_s"]
                current["duration_s"] = max(
                    0.0,
                    float(current["end_time_s"])
                    - float(current["start_time_s"]),
                )
                current["samples"].extend(interval["samples"])
                current["last_active_end_time_s"] = interval["end_time_s"]
                continue
            result.append(current)
            current = None
        if qualifies:
            current = copy.deepcopy(interval)
            current["last_active_end_time_s"] = interval["end_time_s"]
    if current is not None:
        result.append(current)
    for interval in result:
        interval.pop("last_active_end_time_s", None)
    return result


def _apply_rule_cooldown(
    deductions: Sequence[dict], *, cooldown_s: float,
    exempt_types: Iterable[str] = (),
) -> List[dict]:
    """Suppress repeat charges for the same rule during its cooldown.

    The cooldown is measured from the charged deduction's start time in the
    simulation clock.  Different rule types have independent cooldowns, and
    exempt types retain their original charging behavior.  The candidates are
    considered chronologically for correct cooldown semantics while the
    surviving records retain their original output order for schema stability.
    """
    cooldown = float(cooldown_s)
    if cooldown < 0.0:
        raise ValueError("rule cooldown must be >= 0")
    if cooldown <= 0.0 or len(deductions) < 2:
        return list(deductions)

    exempt = {str(item) for item in exempt_types}
    ordered = sorted(
        enumerate(deductions),
        key=lambda pair: (
            float(pair[1].get("start_time_s", 0.0) or 0.0),
            float(pair[1].get("end_time_s", pair[1].get(
                "start_time_s", 0.0)) or 0.0),
            pair[0],
        ),
    )
    last_charged_at: Dict[str, float] = {}
    suppressed: set[int] = set()
    for index, item in ordered:
        kind = str(item.get("type", ""))
        if kind in exempt:
            continue
        trigger_time = float(item.get("start_time_s", 0.0) or 0.0)
        previous = last_charged_at.get(kind)
        if (previous is not None
                and trigger_time - previous < cooldown - 1e-9):
            suppressed.add(index)
            continue
        last_charged_at[kind] = trigger_time

    return [
        item for index, item in enumerate(deductions)
        if index not in suppressed
    ]


def _module_available(vw: Any, name: str) -> bool:
    try:
        return not hasattr(vw, "has_module") or bool(vw.has_module(name))
    except Exception:
        return False


def _equipment_snapshot(
    vw: Any, npc_equipment: Optional[Mapping[str, Any]] = None,
) -> dict:
    """Read scored module fields, with SUMO-NPC physical-state overrides."""
    value = {
        "front_wiper_on": None,
        "front_fog_on": None,
        "rear_fog_on": None,
        "low_beam_on": None,
        "position_light_on": None,
        "all_windows_closed": None,
        "sunroof_closed": None,
    }
    try:
        if _module_available(vw, "wiper"):
            value["front_wiper_on"] = bool(vw.wiper.front_wiper.is_on)
    except (AttributeError, TypeError):
        pass
    try:
        if _module_available(vw, "fogLight"):
            value["front_fog_on"] = bool(vw.fogLight.front_light.is_on)
            value["rear_fog_on"] = bool(vw.fogLight.rear_light.is_on)
    except AttributeError:
        pass
    try:
        if _module_available(vw, "lowBeamHeadlight"):
            value["low_beam_on"] = (
                _state_value(vw.lowBeamHeadlight.mode) in ("on", "auto"))
    except AttributeError:
        pass
    try:
        if _module_available(vw, "positionLight"):
            value["position_light_on"] = bool(vw.positionLight.is_on)
    except AttributeError:
        pass
    try:
        if _module_available(vw, "window"):
            value["all_windows_closed"] = all(
                not bool(item.is_open)
                for item in vw.window._windows.values())
    except (AttributeError, TypeError):
        pass
    try:
        if _module_available(vw, "sunroof"):
            value["sunroof_closed"] = (
                _state_value(vw.sunroof.state) == "closed")
    except AttributeError:
        pass
    if npc_equipment:
        for key in (
                "front_wiper_on", "front_fog_on", "rear_fog_on",
                "low_beam_on", "position_light_on",
                "all_windows_closed", "sunroof_closed"):
            if key in npc_equipment and npc_equipment[key] is not None:
                value[key] = bool(npc_equipment[key])
    return value


def _legitimate_stop(
    vehicle: Any, env: Any, awareness: Mapping[str, Any],
    config: DrivingProcessScoreConfig,
) -> bool:
    if bool(getattr(vehicle, "arrived", False)) or bool(
            getattr(vehicle, "is_crashed", False)):
        return True
    if awareness.get("route_blocked") or bool(
            getattr(vehicle, "lane_route_blocked", False)):
        return True
    light_distance = _finite(getattr(env, "dist_to_light_m", None))
    if (has_restrictive_approach_signal(env)
            and light_distance is not None
            and light_distance <= config.stop_signal_distance_m):
        return True
    leader = awareness.get("leader") or {}
    leader_gap = _finite(leader.get("gap_m"))
    if (leader_gap is not None
            and leader_gap <= config.stop_leader_gap_m):
        return True
    # Include the remaining connector distance: downstream_gap_m alone is
    # measured from its exit, not from the ego vehicle's current position.
    downstream_gap = _finite(awareness.get("downstream_gap_m"))
    connector_remaining = _finite(awareness.get(
        "distance_to_connector_end_m"))
    if (bool(getattr(vehicle, "active_connector_id", ""))
            and downstream_gap is not None
            and connector_remaining is not None
            and connector_remaining + downstream_gap
            <= config.stop_leader_gap_m):
        return True
    blocker = awareness.get("connector_path_blocker") or {}
    blocker_gap = _finite(blocker.get("gap_m"))
    if (bool(getattr(vehicle, "active_connector_id", ""))
            and blocker_gap is not None
            and blocker_gap <= config.stop_leader_gap_m):
        return True
    oncoming = awareness.get("oncoming") or {}
    oncoming_gap = _finite(oncoming.get("gap_m"))
    if (oncoming_gap is not None
            and oncoming_gap <= config.stop_oncoming_gap_m):
        return True
    if awareness.get("shared_corridor_hazard"):
        return True
    for conflict in awareness.get("connector_conflicts", []):
        ego_ttc = _finite(conflict.get("ego_ttc_s"))
        if conflict.get("other_in_conflict_zone"):
            return True
        if (ego_ttc is not None
                and ego_ttc <= config.stop_connector_ttc_s):
            return True
    for hazard in awareness.get("pedestrian_hazards", []):
        distance = _finite(hazard.get("distance_to_crosswalk_m"))
        if (distance is not None
                and distance <= config.stop_pedestrian_distance_m):
            return True
    return False


def _braking_is_necessary(
        vehicle: Any, env: Any, awareness: Mapping[str, Any],
        config: DrivingProcessScoreConfig) -> bool:
    if _legitimate_stop(vehicle, env, awareness, config):
        return True
    return bool(
        awareness.get("leader")
        or awareness.get("oncoming")
        or awareness.get("shared_corridor_hazard")
        or awareness.get("connector_conflicts")
        or awareness.get("pedestrian_hazards"))


def _curve_reference_speed(connector: Mapping[str, Any], config) -> Optional[float]:
    """Geometry-derived braking context, NOT a new legal speed limit.

    Circumcircle curvature is insensitive to point spacing on circular arcs.
    Ignore sub-0.5 m segments to avoid amplifying map-coordinate noise.
    """
    points = connector.get("centerline_xy", [])
    curvature = 0.0
    for a, b, c in zip(points, points[1:], points[2:]):
        ab, bc, ac = math.dist(a[:2], b[:2]), math.dist(b[:2], c[:2]), math.dist(a[:2], c[:2])
        if min(ab, bc, ac) < .5:
            continue
        cross = abs((b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0]))
        curvature = max(curvature, 2.0 * cross / (ab * bc * ac))
    if curvature < 1e-4:
        return None
    return math.sqrt(config.curve_lateral_acceleration_mps2 / curvature) * 3.6


def _connector_context(manager, vehicle, config, cache) -> dict:
    """Read actual and planned connectors without using indicator intent."""
    runtime = getattr(manager, "_lane_geometry", None)
    connectors = getattr(runtime, "_connector_by_id", {})
    lanes = getattr(runtime, "_lane_by_id", {})
    active_id = str(getattr(vehicle, "active_connector_id", "") or "")
    planned_id = str(getattr(vehicle, "planned_connector_id", "") or "")
    active = connectors.get(active_id, {})
    planned = connectors.get(planned_id, {})
    lane = lanes.get(getattr(vehicle, "current_lane_id", ""), {})
    distance = None
    if planned and lane and not active_id:
        # No exemption for a stale plan on an unrelated approach road.
        source = lanes.get(planned.get("from_lane"), {})
        if (source.get("segment_id"), source.get("direction")) == (
                lane.get("segment_id"), lane.get("direction")):
            distance = max(0.0, 1.0-float(getattr(vehicle, "edge_progress", 0))) * float(lane["length_m"])
    connector = active or (planned if distance is not None else {})
    cid = active_id if active else planned_id
    if connector and cid not in cache:
        cache[cid] = _curve_reference_speed(connector, config)
    reference = cache.get(cid) if connector else None
    remaining = 0.0 if active else distance
    speed = max(0.0, float(getattr(vehicle, "current_speed_kmh", 0)))
    required_distance = None
    if reference is not None:
        v, target = speed / 3.6, reference / 3.6
        required_distance = max(0.0, v*v-target*target) / (2 * config.curve_braking_deceleration_mps2)
        required_distance += v * config.curve_braking_reaction_s + float(getattr(vehicle, "length_m", 0)) / 2
    justified = bool(reference is not None and remaining is not None
                     and speed > reference + config.curve_speed_tolerance_kmh
                     and remaining <= min(config.curve_braking_max_distance_m, required_distance))
    return {
        "active_connector_turn": active.get("turn"),
        "planned_connector_id": planned_id,
        "planned_connector_turn": planned.get("turn"),
        "distance_to_planned_connector_m": distance,
        "curve_braking_context": {
            "connector_id": cid if connector else None,
            "reference_speed_kmh": reference,
            "distance_to_connector_m": remaining,
            "braking_distance_with_reaction_m": required_distance,
            "justified": justified,
        },
    }


def _intersection_blocked(
        manager: Any, vehicle: Any, awareness: Mapping[str, Any],
        config: DrivingProcessScoreConfig) -> bool:
    if not str(getattr(vehicle, "active_connector_id", "") or ""):
        return False
    downstream_gap = _finite(awareness.get("downstream_gap_m"))
    if downstream_gap is None:
        return False
    required = float(getattr(vehicle, "length_m", 0.0)) + float(
        getattr(manager.config, "min_gap_m", 0.0))
    return (downstream_gap < required
            and float(getattr(vehicle, "current_speed_kmh", 0.0))
            <= config.moving_speed_kmh)


def _high_beam_misuse(
        manager: Any, vehicle: Any,
        config: DrivingProcessScoreConfig) -> bool:
    signals = getattr(vehicle, "signal_state", None)
    if not bool(getattr(signals, "high_beam", False)):
        return False
    weather = str(getattr(manager, "_current_weather", "") or "").lower()
    try:
        weather = normalize_weather_condition(weather)
    except ValueError:
        pass
    if weather in FOG_VISIBILITY_CONDITIONS:
        return True
    model = manager.perception_model
    for other_id, other in manager.vehicles.items():
        if (other_id == vehicle.vehicle_id
                or bool(getattr(other, "arrived", False))):
            continue
        geometry = model.relative_geometry(other_id, vehicle.vehicle_id)
        if geometry is None:
            continue
        distance, bearing = geometry
        yaw_delta = abs(model._angle_difference(
            float(other.yaw_rad), float(vehicle.yaw_rad)))
        if (distance <= config.high_beam_oncoming_distance_m
                and abs(bearing)
                <= config.high_beam_oncoming_bearing_deg
                and yaw_delta >= math.radians(
                    config.high_beam_oncoming_yaw_delta_deg)):
            return True
    return False


class DrivingProcessScoreTracker:
    """Collect process-score evidence during a simulation and score at end."""

    def __init__(
        self,
        traffic_manager: Any,
        vehicle_worlds: Mapping[str, Any],
        vehicle_ids: Iterable[str],
        config: Optional[DrivingProcessScoreConfig] = None,
    ):
        self.traffic_manager = traffic_manager
        self.vehicle_worlds = vehicle_worlds
        self.config = config or DrivingProcessScoreConfig()
        self._curve_speed_cache: Dict[str, Optional[float]] = {}
        self._traces: Dict[str, _Trace] = {
            str(vehicle_id): _Trace() for vehicle_id in vehicle_ids
        }

    def observe(self, time_s: float) -> None:
        """Sample authoritative physical and installed-module state."""
        for vehicle_id, trace in self._traces.items():
            vehicle = self.traffic_manager.get_state(vehicle_id)
            vw = self.vehicle_worlds.get(vehicle_id)
            if vehicle is None or vw is None:
                continue
            env = self.traffic_manager._build_env_view(
                vehicle, int(round(float(time_s))))
            awareness = self.traffic_manager.get_driving_awareness(
                vehicle_id, ground_truth=True)
            speed = max(0.0, float(vehicle.current_speed_kmh))
            limit = _finite(getattr(env, "speed_limit_kmh", None))
            acceleration = float(vehicle.acceleration_mps2)
            previous = trace.samples[-1] if trace.samples else None
            dt = (
                max(0.0, float(time_s) - float(previous["time_s"]))
                if previous else 0.0)
            smoothed_acceleration = windowed_longitudinal_acceleration(
                trace.speed_history,
                float(time_s), speed, self.config.jerk_smoothing_window_s,
            )
            previous_smoothed = (
                _finite(previous.get("smoothed_acceleration_mps2"))
                if previous else None)
            jerk = (
                abs(smoothed_acceleration - previous_smoothed) / dt
                if (smoothed_acceleration is not None
                    and previous_smoothed is not None and dt > 1e-9)
                else 0.0)
            signals = getattr(vehicle, "signal_state", None)
            connector_context = _connector_context(
                self.traffic_manager, vehicle, self.config, self._curve_speed_cache)
            otherwise_unnecessary_brake = bool(
                acceleration < -self.config.hard_braking_mps2
                and not _braking_is_necessary(vehicle, env, awareness, self.config))
            # Hard acceleration is judged against the vehicle's own chassis
            # envelope: full-throttle in a compact exceeds a fixed 2.8 m/s²
            # while a freight can never reach it, so the limit scales with
            # installed capability and the config value is only a fallback.
            chassis_acceleration = _finite(
                getattr(vehicle, "max_acceleration_mps2", None))
            hard_acceleration_threshold = (
                0.9 * chassis_acceleration
                if chassis_acceleration is not None and chassis_acceleration > 0
                else self.config.hard_acceleration_mps2)
            npc_equipment = (
                getattr(
                    self.traffic_manager,
                    "_npc_weather_equipment_state", {},
                ).get(vehicle_id)
                if not bool(getattr(vehicle, "is_llm", False)) else None
            )

            target_gap = None
            rear_ttc = None
            target_lane = int(getattr(vehicle, "target_lane", -1))
            current_lane = int(getattr(vehicle, "current_lane", -1))
            current_segment = str(
                getattr(vehicle, "current_segment", "") or "")
            if bool(getattr(vehicle, "is_changing_lane", False)) \
                    and current_segment and target_lane >= 0:
                target_gap = _finite(
                    self.traffic_manager._lane_neighbor_gap(
                        vehicle, current_segment, target_lane))
                follower, follower_gap, closing_mps = (
                    self.traffic_manager._adjacent_lane_follower(
                        vehicle, current_segment, target_lane))
                if follower is not None and closing_mps > 1e-6:
                    rear_ttc = float(follower_gap) / float(closing_mps)

            current_lane_gap = None
            current_lane_rear_ttc = None
            if (current_segment and current_lane >= 0
                    and not getattr(vehicle, "active_connector_id", "")):
                current_lane_gap = _finite(
                    self.traffic_manager._lane_neighbor_gap(
                        vehicle, current_segment, current_lane))
                follower, follower_gap, closing_mps = (
                    self.traffic_manager._adjacent_lane_follower(
                        vehicle, current_segment, current_lane))
                if follower is not None and closing_mps > 1e-6:
                    current_lane_rear_ttc = (
                        float(follower_gap) / float(closing_mps))

            trace.samples.append({
                "time_s": round(float(time_s), 6),
                "speed_kmh": speed,
                "speed_limit_kmh": limit,
                "overspeed": bool(
                    limit is not None and limit > 0.0
                    and speed > limit + self.config.overspeed_tolerance_kmh),
                "acceleration_mps2": acceleration,
                "smoothed_acceleration_mps2": smoothed_acceleration,
                "jerk_mps3": jerk,
                "unexplained_stop": bool(
                    speed <= self.config.moving_speed_kmh
                    and not _legitimate_stop(
                        vehicle, env, awareness, self.config)),
                "intersection_blocked": _intersection_blocked(
                    self.traffic_manager, vehicle, awareness, self.config),
                "hard_acceleration": bool(
                    acceleration > hard_acceleration_threshold),
                "hard_acceleration_threshold_mps2": round(
                    hard_acceleration_threshold, 3),
                "unnecessary_hard_brake": bool(
                    otherwise_unnecessary_brake
                    and not connector_context["curve_braking_context"]["justified"]),
                "curve_braking_exempt": bool(
                    otherwise_unnecessary_brake
                    and connector_context["curve_braking_context"]["justified"]),
                "high_jerk": bool(
                    speed > self.config.moving_speed_kmh
                    and jerk > self.config.high_jerk_mps3),
                "high_beam_misuse": _high_beam_misuse(
                    self.traffic_manager, vehicle, self.config),
                "is_changing_lane": bool(
                    getattr(vehicle, "is_changing_lane", False)),
                "current_lane": current_lane,
                "target_lane": target_lane,
                "target_lane_gap_m": target_gap,
                "target_lane_rear_ttc_s": rear_ttc,
                "current_lane_gap_m": current_lane_gap,
                "current_lane_rear_ttc_s": current_lane_rear_ttc,
                "left_indicator": bool(
                    getattr(signals, "left_indicator", False)),
                "right_indicator": bool(
                    getattr(signals, "right_indicator", False)),
                "current_segment": current_segment,
                "current_lane_id": str(
                    getattr(vehicle, "current_lane_id", "") or ""),
                "active_connector_id": str(
                    getattr(vehicle, "active_connector_id", "") or ""),
                **connector_context,
                "equipment": _equipment_snapshot(vw, npc_equipment),
            })
            trace.speed_history.append((float(time_s), speed))

    def record_delivered_events(
        self, vehicle_id: str, events: Iterable[Mapping[str, Any]],
        delivered_at_s: float,
    ) -> None:
        """Record one agent delivery batch before its callback executes."""
        trace = self._traces.get(str(vehicle_id))
        if trace is None:
            return
        for raw in events:
            event = copy.deepcopy(dict(raw))
            event_id = str(event.get("event_id", "") or "")
            signature = event_id or (
                f"{event.get('event_type')}:{event.get('sequence')}:"
                f"{event.get('occurred_at_s')}:{vehicle_id}")
            if signature in trace.delivered_event_ids:
                continue
            trace.delivered_event_ids.add(signature)
            event["delivered_at_s"] = round(float(
                event.get("delivered_at_s", delivered_at_s)), 6)
            trace.delivered_events.append(event)

    def finalize(self, driving_reports: Mapping[str, Mapping[str, Any]]) -> dict:
        return {
            vehicle_id: calculate_driving_process_score(
                driving_reports.get(vehicle_id, {}),
                samples=trace.samples,
                delivered_events=trace.delivered_events,
                config=self.config,
            )
            for vehicle_id, trace in self._traces.items()
        }


def _sample_at_or_after(
        samples: Sequence[dict], time_s: float) -> Optional[dict]:
    for sample in samples:
        if float(sample["time_s"]) + 1e-9 >= float(time_s):
            return sample
    return None


def _closed_targets(details: Mapping[str, Any]) -> tuple[set, set]:
    lane_keys = ("lane_id", "closed_lane_id", "lane_ids", "closed_lane_ids")
    segment_keys = (
        "segment_id", "closed_segment_id", "segment_ids",
        "closed_segment_ids")

    def values(keys: Sequence[str]) -> set:
        result = set()
        for key in keys:
            value = details.get(key)
            if isinstance(value, str) and value:
                result.add(value)
            elif isinstance(value, (list, tuple, set)):
                result.update(str(item) for item in value if str(item))
        return result

    return values(lane_keys), values(segment_keys)


def _hard_gate_events(driving_report: Mapping[str, Any]) -> List[dict]:
    """Collect at-fault collision/red-light evidence for the score gate."""
    metrics = dict(driving_report.get("metrics", {}) or {})
    violations = list(driving_report.get("hard_violations", []) or [])
    collision_violations = [
        item for item in violations
        if str(item.get("type", "")) in {
            "collision", "vehicle_pedestrian_collision"}
    ]
    red_violations = [
        item for item in violations
        if str(item.get("type", "")) == "red_light_violation"
    ]
    red_entries = [
        item for item in driving_report.get("events", []) or []
        if str(item.get("type", "")) == "red_light_entry"
    ]
    collision_count = max(
        int(metrics.get(
            "at_fault_collision_count", metrics.get("collision_count", 0)) or 0),
        len(collision_violations),
    )
    red_count = max(
        int(metrics.get("red_light_violations", 0) or 0),
        len(red_entries),
        sum(int(item.get("count", 1) or 0) for item in red_violations),
    )
    events = []
    if collision_count > 0:
        events.append({
            "type": "collision",
            "count": collision_count,
            "evidence": {"hard_violations": copy.deepcopy(
                collision_violations)},
        })
    if red_count > 0:
        events.append({
            "type": "red_light_violation",
            "count": red_count,
            "evidence": {"entries": copy.deepcopy(red_entries)},
        })
    return events


def calculate_driving_process_score(
    driving_report: Mapping[str, Any],
    *,
    samples: Sequence[dict] = (),
    delivered_events: Sequence[Mapping[str, Any]] = (),
    config: Optional[DrivingProcessScoreConfig] = None,
) -> dict:
    """Calculate the auditable 100-point process score for one vehicle."""
    cfg = config or DrivingProcessScoreConfig()
    deductions: List[dict] = []
    hard_events = _hard_gate_events(driving_report)
    counters: Dict[str, int] = {}

    def add(
        kind: str, points: int, start: float, end: float,
        evidence: Optional[Mapping[str, Any]] = None,
        thresholds: Optional[Mapping[str, Any]] = None,
    ) -> None:
        # Weather and day/night compliance belongs to the passive cabin
        # metric.  Keep the authoritative checks below for diagnostics, but
        # never let them enter the 100-point driving-process score.
        if kind in {
            "wet_weather_wiper_missing",
            "fog_visibility_lights_missing",
            "snow_visibility_low_beam_missing",
            "weather_opening_not_closed",
            "weather_transition_equipment_not_cleared",
            "dark_period_lights_missing",
        }:
            return
        counters[kind] = counters.get(kind, 0) + 1
        deductions.append({
            "episode_id": f"process:{kind}:{counters[kind]}",
            "type": kind,
            "start_time_s": round(float(start), 3),
            "end_time_s": round(float(end), 3),
            "points": int(points),
            "thresholds": copy.deepcopy(dict(thresholds or {})),
            "evidence": copy.deepcopy(dict(evidence or {})),
        })

    overspeed_episodes = _episodes(samples, "overspeed")
    for episode in overspeed_episodes:
        duration = float(episode["duration_s"])
        episode_samples = episode["samples"]
        evidence = {
            "duration_s": round(duration, 3),
            "max_speed_kmh": round(max(
                float(item["speed_kmh"]) for item in episode_samples), 3),
            "minimum_limit_kmh": round(min(
                float(item["speed_limit_kmh"])
                for item in episode_samples
                if item.get("speed_limit_kmh") is not None), 3),
        }
        if duration + 1e-9 >= cfg.long_overspeed_s:
            add(
                "long_overspeed", cfg.long_overspeed_points,
                episode["start_time_s"], episode["end_time_s"], evidence,
                {
                    "tolerance_kmh": cfg.overspeed_tolerance_kmh,
                    "minimum_duration_s": cfg.long_overspeed_s,
                })
        elif duration + 1e-9 >= cfg.short_overspeed_min_s:
            add(
                "short_overspeed", cfg.short_overspeed_points,
                episode["start_time_s"], episode["end_time_s"], evidence,
                {
                    "tolerance_kmh": cfg.overspeed_tolerance_kmh,
                    "minimum_duration_s": cfg.short_overspeed_min_s,
                    "hard_gate_duration_s": cfg.long_overspeed_s,
                })

    for episode in _qualified_episodes(
            samples, "unexplained_stop",
            minimum_s=cfg.unexplained_stop_s,
            clear_s=cfg.unexplained_stop_clear_s,
            strict_minimum=True):
        duration = float(episode["duration_s"])
        count = 1 + int(math.floor(
            max(0.0, duration - cfg.unexplained_stop_s)
            / cfg.repeated_duration_block_s + 1e-9))
        for index in range(count):
            trigger_t = (
                episode["start_time_s"] + cfg.unexplained_stop_s
                + index * cfg.repeated_duration_block_s)
            # Cooldown is keyed by start_time_s: each sustained-stop block
            # must carry its own charge time, not the shared episode start.
            add(
                "unexplained_stop", cfg.unexplained_stop_points,
                trigger_t, trigger_t,
                {
                    "duration_s": round(duration, 3),
                    "block_index": index + 1,
                    "episode_start_time_s": round(
                        episode["start_time_s"], 3),
                    "episode_end_time_s": round(episode["end_time_s"], 3),
                },
                {
                    "initial_duration_s": cfg.unexplained_stop_s,
                    "repeat_every_s": cfg.repeated_duration_block_s,
                    "moving_speed_kmh": cfg.moving_speed_kmh,
                    "clear_duration_s": cfg.unexplained_stop_clear_s,
                })

    # Legacy evaluator supplies deterministic risk, green-resume and horn
    # episode identities. Reuse them instead of maintaining a second TTC model.
    for episode in driving_report.get("decision_episodes", []) or []:
        if bool(episode.get("reasonable", False)):
            continue
        kind = str(episode.get("type", ""))
        start = float(episode.get("start_time_s", 0.0) or 0.0)
        end = float(episode.get("end_time_s", start) or start)
        if kind in ("leader_response", "connector_response", "pedestrian_response"):
            details = dict(episode.get("details", {}) or {})
            entry_ttc = _finite(details.get("entry_ttc_s"))
            if (kind == "leader_response"
                    and details.get("trigger") == "startup_closing_leader"
                    and entry_ttc is not None
                    and entry_ttc > cfg.leader_risk_ttc_s):
                continue
            risk_thresholds = {
                "response_deadline_s": cfg.risk_response_deadline_s,
                "clear_duration_s": cfg.risk_episode_clear_s,
            }
            if kind == "leader_response":
                risk_thresholds["leader_ttc_s"] = cfg.leader_risk_ttc_s
            elif kind == "connector_response":
                risk_thresholds.update({
                    "occupied_conflict_zone_ego_ttc_s": (
                        cfg.connector_conflict_zone_ttc_s),
                    "horizon_s": cfg.connector_horizon_s,
                    "arrival_time_separation_s": (
                        cfg.connector_time_separation_s),
                })
            else:
                risk_thresholds["pedestrian_ttc_s"] = (
                    cfg.pedestrian_risk_ttc_s)
            add(
                f"{kind}_late", cfg.risk_response_points, start, end,
                details, risk_thresholds)
        elif kind == "green_resume":
            add(
                "green_resume_late", cfg.green_resume_points, start, end,
                episode.get("details", {}),
                {"deadline_s": cfg.green_resume_s})

    for event in driving_report.get("events", []) or []:
        kind = str(event.get("type", ""))
        time_s = float(event.get("time_s", 0.0) or 0.0)
        if kind == "near_miss":
            hazard = str(event.get("hazard", ""))
            if hazard.startswith("pedestrian:"):
                points = cfg.pedestrian_near_miss_points
                score_kind = "pedestrian_near_miss"
                thresholds = {
                    "body_clearance_m": cfg.pedestrian_near_miss_clearance_m,
                    "min_vehicle_speed_kmh": (
                        cfg.pedestrian_near_miss_min_speed_kmh),
                    "clear_duration_s": cfg.risk_episode_clear_s,
                }
            elif hazard.startswith("connector:"):
                points = cfg.connector_near_miss_points
                score_kind = "connector_near_miss"
                thresholds = {
                    "occupied_conflict_zone_ego_ttc_s": (
                        cfg.connector_conflict_zone_ttc_s),
                    "horizon_s": cfg.connector_horizon_s,
                    "arrival_time_separation_s": (
                        cfg.connector_time_separation_s),
                    "clear_duration_s": cfg.risk_episode_clear_s,
                }
            else:
                points = cfg.vehicle_near_miss_points
                score_kind = "vehicle_near_miss"
                thresholds = {
                    "vehicle_ttc_s": cfg.vehicle_near_miss_ttc_s,
                    "clear_duration_s": cfg.risk_episode_clear_s,
                }
            add(
                score_kind, points, time_s, time_s, event,
                thresholds)
        elif kind == "unnecessary_horn":
            add(
                "unnecessary_horn", cfg.unnecessary_horn_points,
                time_s, time_s, event,
                {
                    "relevant_leader_gap_m": (
                        cfg.relevant_horn_leader_gap_m),
                })

    for episode in _episodes(samples, "intersection_blocked"):
        duration = float(episode["duration_s"])
        if duration + 1e-9 < cfg.intersection_block_s:
            continue
        count = 1 + int(math.floor(
            max(0.0, duration - cfg.intersection_block_s)
            / cfg.repeated_duration_block_s + 1e-9))
        for index in range(count):
            add(
                "intersection_blocking", cfg.intersection_block_points,
                episode["start_time_s"], episode["end_time_s"],
                {"duration_s": round(duration, 3), "block_index": index + 1},
                {
                    "initial_duration_s": cfg.intersection_block_s,
                    "repeat_every_s": cfg.repeated_duration_block_s,
                })

    for key, minimum, clear_s, points, kind in (
        ("hard_acceleration", cfg.hard_acceleration_min_s,
         cfg.episode_clear_s,
         cfg.hard_acceleration_points, "hard_acceleration"),
        ("unnecessary_hard_brake", cfg.hard_braking_min_s,
         cfg.episode_clear_s,
         cfg.unnecessary_hard_brake_points, "unnecessary_hard_brake"),
        ("high_jerk", cfg.high_jerk_min_s,
         cfg.episode_clear_s,
         cfg.high_jerk_points, "high_longitudinal_jerk"),
        ("high_beam_misuse", cfg.high_beam_misuse_min_s,
         cfg.high_beam_clear_s,
         cfg.high_beam_misuse_points, "high_beam_misuse"),
    ):
        for episode in _qualified_episodes(
                samples, key, minimum_s=minimum, clear_s=clear_s):
            thresholds = {
                "minimum_duration_s": minimum,
                "clear_duration_s": clear_s,
            }
            if key == "hard_acceleration":
                sampled = [
                    _finite(item.get("hard_acceleration_threshold_mps2"))
                    for item in episode.get("samples", [])
                ]
                sampled = [value for value in sampled if value is not None]
                thresholds["acceleration_greater_than_mps2"] = (
                    sampled[0] if sampled else cfg.hard_acceleration_mps2)
            elif key == "unnecessary_hard_brake":
                thresholds["deceleration_greater_than_mps2"] = (
                    cfg.hard_braking_mps2)
            elif key == "high_jerk":
                thresholds["absolute_jerk_greater_than_mps3"] = (
                    cfg.high_jerk_mps3)
                thresholds["smoothing_window_s"] = (
                    cfg.jerk_smoothing_window_s)
            elif key == "high_beam_misuse":
                thresholds.update({
                    "oncoming_distance_m": (
                        cfg.high_beam_oncoming_distance_m),
                    "oncoming_bearing_deg": (
                        cfg.high_beam_oncoming_bearing_deg),
                    "oncoming_yaw_delta_deg": (
                        cfg.high_beam_oncoming_yaw_delta_deg),
                })
            add(
                kind, points, episode["start_time_s"], episode["end_time_s"],
                {"duration_s": round(float(episode["duration_s"]), 3)},
                thresholds)

    # Lane-change signalling and gap checks require the full 10 Hz trace.
    lane_changes = []
    for index in range(1, len(samples)):
        previous, current = samples[index - 1], samples[index]
        if current.get("is_changing_lane") and not previous.get("is_changing_lane"):
            target = int(current.get("target_lane", -1))
            source = int(current.get("current_lane", -1))
            direction = "left" if target > source else "right"
            lane_changes.append({
                "start_index": index,
                "start_time_s": float(current["time_s"]),
                "direction": direction,
                "source_lane": source,
                "target_lane": target,
                "evidence_index": index,
                "signal_end_index": index,
                "source": "explicit_command",
                "end_index": None,
            })
        if (not current.get("is_changing_lane")
                and previous.get("is_changing_lane") and lane_changes):
            for change in reversed(lane_changes):
                if change["end_index"] is None:
                    change["end_index"] = index
                    break
        # SUMO-native vehicles bypass VehicleArena's explicit maneuver flag.
        # A same-segment lane-ID transition is still authoritative evidence of
        # a physical lane change and must enter the same scoring path.
        previous_segment = str(previous.get("current_segment", "") or "")
        current_segment = str(current.get("current_segment", "") or "")
        previous_lane_id = str(previous.get("current_lane_id", "") or "")
        current_lane_id = str(current.get("current_lane_id", "") or "")
        source = int(previous.get("current_lane", -1))
        target = int(current.get("current_lane", -1))
        native_transition = bool(
            not previous.get("is_changing_lane")
            and not current.get("is_changing_lane")
            and not previous.get("active_connector_id")
            and not current.get("active_connector_id")
            and previous_segment
            and previous_segment == current_segment
            and previous_lane_id
            and current_lane_id
            and previous_lane_id != current_lane_id
            and source >= 0
            and target >= 0
            and source != target)
        if native_transition:
            lane_changes.append({
                "start_index": index,
                "start_time_s": float(current["time_s"]),
                "direction": "left" if target > source else "right",
                "source_lane": source,
                "target_lane": target,
                "evidence_index": index,
                # SUMO may release the indicator on the completion sample.
                "signal_end_index": index - 1,
                "source": "sumo_native_lane_transition",
                "end_index": index,
            })

    turns = []
    for index in range(1, len(samples)):
        previous, current = samples[index - 1], samples[index]
        cid = current.get("active_connector_id")
        direction = current.get("active_connector_turn")
        if cid and cid != previous.get("active_connector_id") and direction in ("left", "right"):
            end_index = next((j for j in range(index + 1, len(samples))
                              if samples[j].get("active_connector_id") != cid), None)
            turns.append({"start_index": index, "start_time_s": float(current["time_s"]),
                          "direction": direction,
                          "connector_id": cid, "turn": direction, "end_index": end_index})

    def continuing_signal(maneuver, end, deadline):
        direction = maneuver["direction"]
        for other in lane_changes + turns:
            if other is maneuver or other["direction"] != direction:
                continue
            start = float(other["start_time_s"])
            other_end = other.get("end_index")
            finish = float(samples[other_end]["time_s"]) if other_end is not None else math.inf
            if end < start <= deadline + 1e-9 or start <= end < finish:
                return True
        sample = _sample_at_or_after(samples, deadline)
        if sample is None:
            return False
        distance = _finite(sample.get("distance_to_planned_connector_m"))
        planned = sample.get("planned_connector_turn")
        return bool(planned == direction and distance is not None
                    and 0 <= distance <= cfg.signal_turn_approach_m)

    for change in lane_changes:
        index = int(change["start_index"])
        current = samples[int(change.get("evidence_index", index))]
        start = float(change["start_time_s"])
        signal_key = (
            "left_indicator" if change["direction"] == "left"
            else "right_indicator")
        lead_start = start - cfg.signal_lead_s
        signal_end_index = int(change.get("signal_end_index", index))
        history = [
            item for item in samples[:signal_end_index + 1]
            if float(item["time_s"]) + 1e-9 >= lead_start
            and float(item["time_s"]) <= start + 1e-9
        ]
        has_full_history = bool(
            samples and float(samples[0]["time_s"]) <= lead_start + 1e-9)
        signaled = bool(
            has_full_history and history
            and all(bool(item.get(signal_key, False))
                    and not bool(item.get("right_indicator" if signal_key == "left_indicator" else "left_indicator", False))
                    for item in history))
        if not signaled:
            add(
                "unsignaled_lane_change", cfg.unsignaled_lane_change_points,
                start, start,
                {
                    "direction": change["direction"],
                    "source_lane": change.get("source_lane"),
                    "target_lane": change.get("target_lane"),
                    "source": change.get("source"),
                    "signal_history": [
                        {
                            "time_s": item["time_s"],
                            "on": bool(item.get(signal_key, False)),
                        }
                        for item in history
                    ],
                },
                {"required_signal_lead_s": cfg.signal_lead_s})

        if change.get("source") == "sumo_native_lane_transition":
            gap = _finite(current.get("current_lane_gap_m"))
            rear_ttc = _finite(current.get("current_lane_rear_ttc_s"))
        else:
            gap = _finite(current.get("target_lane_gap_m"))
            rear_ttc = _finite(current.get("target_lane_rear_ttc_s"))
        unsafe = bool(
            (gap is not None and gap < cfg.unsafe_lane_gap_m)
            or (rear_ttc is not None
                and rear_ttc < cfg.unsafe_lane_rear_ttc_s))
        if unsafe:
            add(
                "unsafe_lane_change", cfg.unsafe_lane_change_points,
                start, start,
                {"nearest_gap_m": gap, "rear_ttc_s": rear_ttc},
                {
                    "minimum_gap_m": cfg.unsafe_lane_gap_m,
                    "minimum_rear_ttc_s": cfg.unsafe_lane_rear_ttc_s,
                })

        end_index = change.get("end_index")
        if end_index is None:
            continue
        end = float(samples[int(end_index)]["time_s"])
        deadline = end + cfg.signal_cancel_s
        deadline_sample = _sample_at_or_after(samples, deadline)
        if (not continuing_signal(change, end, deadline) and deadline_sample is not None
                and bool(deadline_sample.get(signal_key, False))):
            add(
                "turn_signal_not_cancelled",
                cfg.signal_not_cancelled_points,
                end, float(deadline_sample["time_s"]),
                {"direction": change["direction"]},
                {"cancel_deadline_s": cfg.signal_cancel_s})

    for turn in turns:
        index = turn["start_index"]
        start = turn["start_time_s"]
        lead_start = start - cfg.signal_lead_s
        signal_key = turn["direction"] + "_indicator"
        opposite = "right_indicator" if turn["direction"] == "left" else "left_indicator"
        history = [s for s in samples[:index + 1] if float(s["time_s"]) >= lead_start - 1e-9]
        # A spawn already inside a connector or too close to its entry has
        # insufficient pre-entry evidence. Do not invent an earlier offence.
        observable = bool(samples and float(samples[0]["time_s"]) <= lead_start + 1e-9)
        signaled = bool(history and all(s.get(signal_key, False) and not s.get(opposite, False) for s in history))
        turn["signal_evaluable"] = observable
        turn["correctly_signaled"] = signaled if observable else None
        if observable and not signaled:
            add("unsignaled_turn", cfg.unsignaled_turn_points, start, start,
                {"connector_id": turn["connector_id"], "direction": turn["direction"], "turn": turn["turn"],
                 "signal_history": [{"time_s": s["time_s"], "on": bool(s.get(signal_key)),
                                     "opposite_on": bool(s.get(opposite))} for s in history]},
                {"required_signal_lead_s": cfg.signal_lead_s})
        if turn["end_index"] is None:
            continue
        end = float(samples[turn["end_index"]]["time_s"])
        deadline = end + cfg.signal_cancel_s
        after = _sample_at_or_after(samples, deadline)
        if (after is not None and after.get(signal_key, False)
                and not continuing_signal(turn, end, deadline)):
            add("turn_signal_not_cancelled", cfg.signal_not_cancelled_points,
                end, float(after["time_s"]),
                {"connector_id": turn["connector_id"], "direction": turn["direction"], "source": "connector_turn"},
                {"cancel_deadline_s": cfg.signal_cancel_s})

    # Delivered environment events are checked against installed-module state
    # at the declared deadline. Initial state events are intentionally included.
    def dark_period_at(time_s: float) -> bool:
        period = "noon"
        for other in delivered_events:
            if (str(other.get("event_type", "") or "")
                    not in ("daynight_initialized", "daynight_changed")):
                continue
            if float(other.get("delivered_at_s", 0.0) or 0.0) > time_s + 1e-9:
                continue
            period = str(dict(other.get("details", {}) or {}).get(
                "period", period) or period).lower()
        return period in ("dawn", "dusk", "night")

    for event_index, event in enumerate(delivered_events):
        event_type = str(event.get("event_type", "") or "")
        details = dict(event.get("details", {}) or {})
        delivered = float(event.get("delivered_at_s", 0.0) or 0.0)
        event_id = str(event.get("event_id", "") or "")
        if event_type in ("weather_initialized", "weather_changed"):
            condition = normalize_weather_condition(
                details.get("condition", "sunny"))
            previous_condition = details.get("previous_condition")
            if not previous_condition:
                previous_condition = next((
                    str(dict(other.get("details", {}) or {}).get(
                        "condition", "sunny"))
                    for other in reversed(delivered_events[:event_index])
                    if str(other.get("event_type", "") or "")
                    in ("weather_initialized", "weather_changed")
                ), "sunny")
            requirements = weather_score_requirements(
                previous_condition, condition)
            next_weather_at = next((
                float(other.get("delivered_at_s", 0.0) or 0.0)
                for other in delivered_events[event_index + 1:]
                if str(other.get("event_type", "") or "")
                in ("weather_initialized", "weather_changed")
            ), None)
            due = delivered + cfg.environment_response_s
            sample = (
                _sample_at_or_after(samples, due)
                if next_weather_at is None or next_weather_at > due + 1e-9
                else None
            )
            if sample is not None:
                equipment = dict(sample.get("equipment", {}) or {})
                common = {
                    "event_id": event_id,
                    "previous_condition": requirements[
                        "previous_condition"],
                    "condition": condition,
                    "checked_at_s": sample["time_s"],
                }
                if requirements["front_wiper_required"]:
                    if equipment.get("front_wiper_on") is False:
                        add(
                            "wet_weather_wiper_missing", cfg.wet_wiper_points,
                            delivered, float(sample["time_s"]), common,
                            {"response_deadline_s": cfg.environment_response_s})
                if requirements["fog_visibility_lights_required"] and (
                        equipment.get("low_beam_on") is False
                        or equipment.get("front_fog_on") is False
                        or equipment.get("rear_fog_on") is False
                        or equipment.get("position_light_on") is False
                        or equipment.get("high_beam_on") is True):
                    add(
                        "fog_visibility_lights_missing",
                        cfg.fog_lights_points,
                        delivered, float(sample["time_s"]), common,
                        {"response_deadline_s": cfg.environment_response_s})
                if (requirements["low_beam_required"]
                        and equipment.get("low_beam_on") is False):
                    add(
                        "snow_visibility_low_beam_missing",
                        cfg.fog_lights_points,
                        delivered, float(sample["time_s"]), common,
                        {"response_deadline_s": cfg.environment_response_s})
                if requirements["openings_must_be_closed"] and (
                        equipment.get("all_windows_closed") is False
                        or equipment.get("sunroof_closed") is False):
                    add(
                        "weather_opening_not_closed",
                        cfg.severe_weather_opening_points,
                        delivered, float(sample["time_s"]), common,
                        {"response_deadline_s": cfg.environment_response_s})
                fog_cleanup = requirements[
                    "fog_and_position_lights_must_be_off"]
                cleanup_failed = (
                    fog_cleanup
                    and (
                        equipment.get("front_fog_on") is True
                        or equipment.get("rear_fog_on") is True
                        or (
                            equipment.get("position_light_on") is True
                            and not dark_period_at(float(sample["time_s"]))
                        )
                    )
                ) or (
                    requirements["front_wiper_must_be_off"]
                    and equipment.get("front_wiper_on") is True
                )
                if cleanup_failed:
                    add(
                        "weather_transition_equipment_not_cleared",
                        cfg.weather_cleanup_points,
                        delivered, float(sample["time_s"]), common,
                        {"response_deadline_s":
                         cfg.environment_response_s})
        elif event_type in ("daynight_initialized", "daynight_changed"):
            period = str(details.get("period", "") or "").lower()
            if period in ("dusk", "night", "dawn"):
                due = delivered + cfg.environment_response_s
                sample = _sample_at_or_after(samples, due)
                if sample is not None:
                    equipment = dict(sample.get("equipment", {}) or {})
                    if (equipment.get("low_beam_on") is False
                            or equipment.get("position_light_on") is False):
                        add(
                            "dark_period_lights_missing",
                            cfg.dark_lights_points,
                            delivered, float(sample["time_s"]),
                            {
                                "event_id": event_id,
                                "period": period,
                                "checked_at_s": sample["time_s"],
                            },
                            {"response_deadline_s": cfg.environment_response_s})

        lanes, segments = (
            _closed_targets(details)
            if event_type in ("road_closure", "route_blocked")
            and str(event.get("state", "occurred")) != "clear"
            else (set(), set()))
        if lanes or segments:
            entry = None
            for index, item in enumerate(samples):
                if (float(item["time_s"]) + 1e-9
                        < delivered + cfg.road_closure_response_s):
                    continue
                current_closed = (
                    str(item.get("current_lane_id", "")) in lanes
                    or str(item.get("current_segment", "")) in segments)
                previous_closed = False
                if index > 0:
                    previous = samples[index - 1]
                    previous_closed = (
                        str(previous.get("current_lane_id", "")) in lanes
                        or str(previous.get("current_segment", ""))
                        in segments)
                if current_closed and not previous_closed:
                    entry = item
                    break
            if entry is not None:
                add(
                    "entered_closed_road", cfg.road_closure_entry_points,
                    delivered, float(entry["time_s"]),
                    {
                        "event_id": event_id,
                        "lane_id": entry.get("current_lane_id"),
                        "segment_id": entry.get("current_segment"),
                    },
                    {"response_deadline_s": cfg.road_closure_response_s})

    deductions = _coalesce_pedestrian_risk_deductions(
        deductions, clear_s=cfg.risk_episode_clear_s)
    deductions = _apply_rule_cooldown(
        deductions,
        cooldown_s=cfg.rule_cooldown_s,
    )
    total = sum(int(item["points"]) for item in deductions)
    hard_gate = bool(hard_events)
    rule_score = float(max(0, 100 - total))
    score = 0.0 if hard_gate else rule_score
    return {
        "evaluation_type": "exam_style_deduction_v1",
        "metric_revision": METRIC_REVISION,
        "rule_cooldown_s": round(float(cfg.rule_cooldown_s), 3),
        "rule_cooldown_exempt_types": [],
        "driving_process_score_100": round(score, 2),
        "hard_gate_triggered": hard_gate,
        "hard_gate_events": hard_events,
        "hard_gate_penalty_points": HARD_GATE_POINTS if hard_gate else 0,
        "rule_score_100": round(rule_score, 2),
        "deduction_total": total,
        "deductions": deductions,
        "observed_lane_change_count": len(lane_changes),
        "observed_turn_count": len(turns),
        "turn_events": [{
            "time_s": t["start_time_s"], "connector_id": t["connector_id"],
            "direction": t["direction"], "signal_evaluable": t["signal_evaluable"],
            "correctly_signaled": t["correctly_signaled"],
            "end_time_s": samples[t["end_index"]]["time_s"] if t["end_index"] is not None else None,
        } for t in turns],
        "curve_braking_exemptions": [{
            "start_time_s": e["start_time_s"], "end_time_s": e["end_time_s"],
            "evidence": [{"time_s": s["time_s"], "speed_kmh": s.get("speed_kmh"),
                          "acceleration_mps2": s.get("acceleration_mps2"),
                          **s.get("curve_braking_context", {})} for s in e["samples"]],
        } for e in _episodes(samples, "curve_braking_exempt")],
        "lane_change_events": [
            {
                "time_s": round(float(item["start_time_s"]), 3),
                "direction": item["direction"],
                "source_lane": item.get("source_lane"),
                "target_lane": item.get("target_lane"),
                "source": item.get("source"),
            }
            for item in lane_changes
        ],
        "thresholds_and_points": asdict(cfg),
    }
