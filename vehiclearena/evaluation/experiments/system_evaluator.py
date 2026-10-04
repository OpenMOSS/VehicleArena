"""System-level road-world metrics; observes outcomes and never drives actors."""

from __future__ import annotations

from collections import Counter, defaultdict
import math
from statistics import mean
from typing import Dict, Iterable, List, Mapping, Optional

from evaluation.experiments.actor_scope import (
    focal_vehicle_id,
    map_trip_vehicle_ids,
    persistent_obstacle_vehicle_ids,
    required_trip_vehicle_ids,
)
from evaluation.experiments.statistics import gini, percentile


STOPPED_SPEED_KMH = 0.5
DEADLOCK_WINDOW_S = 20.0
DEADLOCK_PROGRESS_M = 1.0
DEADLOCK_CONNECTION_FRACTION = 0.8
QUEUE_MAX_BUMPER_GAP_M = 10.0


def _rows_by_vehicle(rows: Iterable[dict]) -> Dict[str, List[dict]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["vehicle_id"]].append(row)
    return grouped


def _secondary_collisions(collisions: Iterable[dict]) -> int:
    first_contact = {}
    secondary = 0
    for item in sorted(collisions, key=lambda event: event.get("time_s", 0.0)):
        time_s = float(item.get("time_s", 0.0))
        entities = (item.get("entity_a"), item.get("entity_b"))
        if any(entity in first_contact and time_s > first_contact[entity] + 1e-6
               for entity in entities if entity):
            secondary += 1
        for entity in entities:
            if entity:
                first_contact.setdefault(entity, time_s)
    return secondary


def _dominant_reason(rows: List[dict]) -> str:
    reasons = [
        str((row.get("last_control_command") or {}).get("reason", ""))
        for row in rows
    ]
    reasons = [reason for reason in reasons if reason]
    return Counter(reasons).most_common(1)[0][0] if reasons else "unknown"


def _deadlock_kind(details: Dict[str, dict]) -> str:
    reasons = [item["dominant_reason"] for item in details.values()]
    phases = [item["negotiation_phase"] for item in details.values()]
    if sum(
            reason == "connector_conflict_yield" or phase == "yielding"
            for reason, phase in zip(reasons, phases)) >= 2:
        return "connector_mutual_yield"
    if sum(reason.startswith("narrow_road") for reason in reasons) >= 2:
        return "narrow_road_mutual_yield"
    if any(reason == "downstream_spillback" for reason in reasons):
        return "downstream_spillback"
    if any(reason in {"following", "obstacle_ahead"}
           for reason in reasons):
        return "traffic_queue_or_obstacle"
    if any(reason.startswith("pedestrian") for reason in reasons):
        return "pedestrian_priority_wait"
    return "mixed_or_unknown"


def _candidate_deadlocks(
    grouped: Dict[str, List[dict]], window_s=DEADLOCK_WINDOW_S,
    eligible_vehicle_ids: Optional[set[str]] = None,
):
    candidates = []
    times = sorted({row["time_s"] for rows in grouped.values() for row in rows})
    if not times or times[-1] < window_s:
        return candidates
    end_s = times[-1]
    start_s = end_s - window_s
    stalled = []
    details = {}
    for vehicle_id, rows in grouped.items():
        if (eligible_vehicle_ids is not None
                and vehicle_id not in eligible_vehicle_ids):
            continue
        eligible = sorted(
            (row for row in rows if row["time_s"] >= start_s),
            key=lambda row: row["time_s"])
        if not eligible:
            continue
        first, last = eligible[0], eligible[-1]
        if (first["arrived"] or first["crashed"]
                or first.get("route_failed")):
            continue
        lawful_signal_wait = sum(
            bool(row.get("waiting_red_light"))
            or (row.get("last_control_command") or {}).get("reason")
            in ("red_light", "yellow_light")
            for row in eligible)
        if lawful_signal_wait >= max(
                1, math.ceil(len(eligible) * 0.8)):
            continue
        progress = (last["distance_traveled_m"]
                    - first["distance_traveled_m"])
        if (not last["arrived"] and not last["crashed"]
                and not last.get("route_failed")
                and bool(last.get("present_in_physics_world", True))
                and progress < DEADLOCK_PROGRESS_M):
            stalled.append(vehicle_id)
            negotiation = last.get("driver_negotiation") or {}
            details[vehicle_id] = {
                "progress_m": round(float(progress), 3),
                "dominant_reason": _dominant_reason(eligible),
                "last_reason": str(
                    (last.get("last_control_command") or {}).get(
                        "reason", "unknown")),
                "negotiation_phase": str(
                    negotiation.get("phase", "unavailable")),
                "negotiation_yield_count": int(
                    negotiation.get("yield_count", 0)),
                "lane_id": str(last.get("current_lane_id", "")),
                "connector_id": str(
                    last.get("active_connector_id", "")),
            }
    if len(stalled) >= 2:
        candidates.append({
            "start_s": round(start_s, 3), "end_s": round(end_s, 3),
            "entity_ids": sorted(stalled),
            "reason": "at least two active vehicles moved <1 m in 20 s",
            "kind": _deadlock_kind(details),
            "entity_details": {
                vehicle_id: details[vehicle_id]
                for vehicle_id in sorted(details)
            },
        })
    return candidates


def candidate_deadlocks(
    trajectory: Iterable[dict], window_s: float = DEADLOCK_WINDOW_S,
    eligible_vehicle_ids: Optional[Iterable[str]] = None,
) -> List[dict]:
    """Return auditable final-window deadlock candidates.

    This is deliberately an observation-only heuristic.  A candidate is
    classified by the actual local decisions recorded in the trajectory; it
    is not fed back into any driver policy.
    """
    return _candidate_deadlocks(
        _rows_by_vehicle(trajectory), window_s=float(window_s),
        eligible_vehicle_ids=(
            set(eligible_vehicle_ids)
            if eligible_vehicle_ids is not None else None),
    )


def _angle_delta(first: float, second: float) -> float:
    return abs((float(first) - float(second) + math.pi) % (2 * math.pi)
               - math.pi)


def _physically_connected_stop_components(rows: Iterable[dict]) -> List[set[str]]:
    """Return spatially connected groups of stopped physical vehicles.

    A lane ID alone is insufficient: two stopped vehicles can be far apart on
    a long lane.  Conversely, a queue can straddle an incoming lane and its
    connector.  We therefore use authoritative poses, vehicle lengths and
    compatible headings, with a bounded bumper gap.  Crashed vehicles remain
    in these components because they are physical blockers; queue exposure
    filters them out when it counts live queue members.
    """
    stopped = [
        row for row in rows
        if bool(row.get("present_in_physics_world", True))
        and not bool(row.get("arrived", False))
        and float(row.get("speed_kmh", 0.0) or 0.0) < STOPPED_SPEED_KMH
        and row.get("pose_x_m") is not None
        and row.get("pose_y_m") is not None
    ]
    parents = list(range(len(stopped)))

    def root(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def join(first: int, second: int) -> None:
        first_root, second_root = root(first), root(second)
        if first_root != second_root:
            parents[second_root] = first_root

    for first in range(len(stopped)):
        a = stopped[first]
        for second in range(first + 1, len(stopped)):
            b = stopped[second]
            distance = math.hypot(
                float(a["pose_x_m"]) - float(b["pose_x_m"]),
                float(a["pose_y_m"]) - float(b["pose_y_m"]),
            )
            max_distance = (
                0.5 * float(a.get("length_m", 4.6) or 4.6)
                + 0.5 * float(b.get("length_m", 4.6) or 4.6)
                + QUEUE_MAX_BUMPER_GAP_M
            )
            if distance > max_distance:
                continue
            same_path = bool(
                (a.get("active_connector_id")
                 and a.get("active_connector_id")
                 == b.get("active_connector_id"))
                or (a.get("current_lane_id")
                    and a.get("current_lane_id")
                    == b.get("current_lane_id"))
            )
            heading_delta = _angle_delta(
                float(a.get("yaw_rad", 0.0) or 0.0),
                float(b.get("yaw_rad", 0.0) or 0.0),
            )
            compatible_heading = (
                heading_delta <= math.radians(60.0)
                or heading_delta >= math.radians(120.0)
            )
            if same_path or compatible_heading:
                join(first, second)

    grouped: Dict[int, set[str]] = defaultdict(set)
    for index, row in enumerate(stopped):
        grouped[root(index)].add(str(row["vehicle_id"]))
    return [component for component in grouped.values()
            if len(component) >= 2]


def _queue_exposure(
    trajectory: Iterable[dict], eligible_vehicle_ids: set[str],
) -> tuple[Dict[str, float], int, dict]:
    """Integrate live trip-vehicle time in a connected stopped queue.

    A crashed vehicle remains in the physical component so it can block live
    traffic, but it is never a queue member and never contributes to queue
    length or queue exposure after the first crashed observation.
    """
    rows_by_time: Dict[float, List[dict]] = defaultdict(list)
    for row in trajectory:
        rows_by_time[float(row["time_s"])].append(row)
    exposure = {vehicle_id: 0.0 for vehicle_id in eligible_vehicle_ids}
    previous_time = None
    previous_members: set[str] = set()
    max_queue = 0
    max_detail = {}
    crashed_vehicle_ids: set[str] = set()
    for time_s in sorted(rows_by_time):
        if previous_time is not None:
            dt = max(0.0, time_s - previous_time)
            for vehicle_id in previous_members:
                exposure[vehicle_id] += dt
        current_rows = rows_by_time[time_s]
        crashed_vehicle_ids.update(
            str(row["vehicle_id"])
            for row in current_rows
            if bool(row.get("crashed", False))
        )
        components = _physically_connected_stop_components(current_rows)
        queue_components = [
            (component & eligible_vehicle_ids) - crashed_vehicle_ids
            for component in components
        ]
        members = set().union(*queue_components) if queue_components else set()
        largest = max((len(component) for component in queue_components),
                      default=0)
        if largest > max_queue:
            max_queue = largest
            physical_ids = set().union(*components) if components else set()
            max_detail = {
                "time_s": time_s,
                "components": [sorted(component)
                               for component in components],
                "queue_components": [sorted(component)
                                      for component in queue_components],
                "physical_blocker_vehicle_ids": sorted(
                    physical_ids - members),
            }
        previous_time = time_s
        previous_members = members
    return exposure, max_queue, max_detail


def confirmed_deadlocks(
    trajectory: Iterable[dict], eligible_vehicle_ids: Iterable[str],
    window_s: float = DEADLOCK_WINDOW_S,
) -> List[dict]:
    """Confirm only persistent, spatially connected stalled groups.

    The broader candidate detector remains available for diagnostics.  A
    confirmed deadlock requires at least two eligible trip vehicles to remain
    in the same physical stopped component for 80% of the final window;
    unrelated vehicles elsewhere cannot satisfy it.
    """
    trajectory = list(trajectory)
    eligible = set(eligible_vehicle_ids)
    candidates = candidate_deadlocks(
        trajectory, window_s=window_s,
        eligible_vehicle_ids=eligible)
    if not candidates:
        return []
    stalled = set(candidates[0]["entity_ids"])
    times = sorted({float(row["time_s"]) for row in trajectory})
    if not times or times[-1] < window_s:
        return []
    start_s = times[-1] - window_s
    rows_by_time: Dict[float, List[dict]] = defaultdict(list)
    for row in trajectory:
        time_s = float(row["time_s"])
        if time_s >= start_s:
            rows_by_time[time_s].append(row)

    pair_seen: Counter = Counter()
    pair_connected: Counter = Counter()
    for rows in rows_by_time.values():
        present = {
            str(row["vehicle_id"])
            for row in rows
            if str(row["vehicle_id"]) in stalled
        }
        components = _physically_connected_stop_components(rows)
        component_by_vehicle = {
            vehicle_id: index
            for index, component in enumerate(components)
            for vehicle_id in component
        }
        present = sorted(present)
        for first in range(len(present)):
            for second in range(first + 1, len(present)):
                pair = (present[first], present[second])
                pair_seen[pair] += 1
                if (component_by_vehicle.get(pair[0]) is not None
                        and component_by_vehicle.get(pair[0])
                        == component_by_vehicle.get(pair[1])):
                    pair_connected[pair] += 1

    observation_count = len(rows_by_time)
    links = {
        pair for pair, count in pair_connected.items()
        if (pair_seen[pair] / max(1, observation_count)
            >= DEADLOCK_CONNECTION_FRACTION
            and count / max(1, pair_seen[pair])
            >= DEADLOCK_CONNECTION_FRACTION)
    }
    parents = {vehicle_id: vehicle_id for vehicle_id in stalled}

    def root(vehicle_id: str) -> str:
        while parents[vehicle_id] != vehicle_id:
            parents[vehicle_id] = parents[parents[vehicle_id]]
            vehicle_id = parents[vehicle_id]
        return vehicle_id

    for first, second in links:
        first_root, second_root = root(first), root(second)
        if first_root != second_root:
            parents[second_root] = first_root
    groups: Dict[str, set[str]] = defaultdict(set)
    for vehicle_id in stalled:
        groups[root(vehicle_id)].add(vehicle_id)
    confirmed = []
    details = candidates[0].get("entity_details", {})
    for group in sorted(
            (group for group in groups.values() if len(group) >= 2),
            key=lambda item: sorted(item)):
        confirmed.append({
            "start_s": round(start_s, 3),
            "end_s": round(times[-1], 3),
            "entity_ids": sorted(group),
            "kind": "persistent_connected_stall",
            "rule": (
                "each vehicle moved <1 m and remained physically connected "
                "for >=80% of the final 20 s"),
            "entity_details": {
                vehicle_id: details.get(vehicle_id, {})
                for vehicle_id in sorted(group)
            },
        })
    return confirmed


def _stop_go_oscillations(rows: List[dict]) -> int:
    # Count completed move→stop→move cycles; ignore a single normal stop.
    states = []
    for row in rows:
        state = row["speed_kmh"] >= 2.0
        if not states or states[-1] != state:
            states.append(state)
    return sum(1 for index in range(2, len(states))
               if states[index - 2:index + 1] == [True, False, True])


def evaluate_system(
    result, trajectory: List[dict], duration_s: float,
    *, scenario: Optional[Mapping] = None,
) -> dict:
    """Evaluate physical outcomes over declared trip vehicles.

    Non-focal trip vehicles own NPC completion, delay and queue denominators.
    The focal vehicle and persistent authored obstacles remain
    in the physical trajectory, so either can still block or collide with the
    measured traffic.  Supplying ``scenario`` identifies authored obstacles;
    the fallback keeps small callers readable.
    """
    grouped = _rows_by_vehicle(trajectory)
    collisions = list(getattr(result, "_collision_log", []))
    result_vehicles = result.vehicle_results
    if scenario is None:
        persistent_ids: set[str] = set()
        all_trip_ids = set(result_vehicles)
        map_trip_ids = set(result_vehicles)
        focal_id = None
        actor_scope_source = "legacy_all_result_vehicles"
    else:
        persistent_ids = persistent_obstacle_vehicle_ids(scenario)
        all_trip_ids = required_trip_vehicle_ids(scenario)
        map_trip_ids = map_trip_vehicle_ids(scenario)
        focal_id = focal_vehicle_id(scenario)
        actor_scope_source = "frozen_scenario_non_focal_trip_roles"
    map_arrived = sum(
        bool(getattr(result_vehicles.get(vehicle_id), "arrived", False))
        for vehicle_id in map_trip_ids)
    all_arrived = sum(
        bool(getattr(result_vehicles.get(vehicle_id), "arrived", False))
        for vehicle_id in all_trip_ids)
    unexplained_waits = []
    for vehicle_id in sorted(map_trip_ids):
        vehicle = result_vehicles.get(vehicle_id)
        if vehicle is None:
            continue
        report = vehicle.driving_evaluation or {}
        metrics = report.get("metrics", {})
        if "unexplained_idle_s" in metrics:
            unexplained_waits.append(float(metrics["unexplained_idle_s"]))
    stationary_wait_by_vehicle = {
        vehicle_id: 0.0 for vehicle_id in all_trip_ids}
    for vehicle_id in sorted(all_trip_ids):
        rows = grouped.get(vehicle_id, [])
        wait_s = 0.0
        previous = None
        for row in sorted(rows, key=lambda item: item["time_s"]):
            if previous is not None:
                dt = max(0.0, row["time_s"] - previous["time_s"])
                if (not previous["arrived"] and not previous["crashed"]
                        and bool(previous.get(
                            "present_in_physics_world", True))
                        and previous["speed_kmh"] < STOPPED_SPEED_KMH):
                    wait_s += dt
            previous = row
        stationary_wait_by_vehicle[vehicle_id] = wait_s

    queue_wait_by_vehicle, max_queue, queues_at_max = _queue_exposure(
        trajectory, all_trip_ids)
    candidate_items = candidate_deadlocks(
        trajectory, window_s=DEADLOCK_WINDOW_S,
        eligible_vehicle_ids=all_trip_ids)
    all_confirmed_items = confirmed_deadlocks(
        trajectory, all_trip_ids, window_s=DEADLOCK_WINDOW_S)
    confirmed_items = [
        item for item in all_confirmed_items
        if set(item.get("entity_ids", [])) & map_trip_ids
    ]
    trip_vehicle_metrics = {}
    for vehicle_id in sorted(all_trip_ids):
        vehicle = result_vehicles.get(vehicle_id)
        vehicle_arrived = bool(getattr(vehicle, "arrived", False))
        arrival_time = getattr(vehicle, "arrival_time_s", None)
        if arrival_time is not None:
            arrival_time = min(float(duration_s), max(0.0, float(arrival_time)))
        completion_time = (
            arrival_time
            if vehicle_arrived and arrival_time is not None
            else float(duration_s)
        )
        trip_vehicle_metrics[vehicle_id] = {
            "arrived": vehicle_arrived,
            "arrival_time_s": arrival_time,
            "completion_time_s": round(completion_time, 6),
            "stationary_wait_s": round(
                stationary_wait_by_vehicle.get(vehicle_id, 0.0), 6),
            "queue_wait_s": round(
                queue_wait_by_vehicle.get(vehicle_id, 0.0), 6),
        }

    stationary_waits = [
        stationary_wait_by_vehicle[vehicle_id]
        for vehicle_id in map_trip_ids]
    queue_waits = [
        queue_wait_by_vehicle[vehicle_id]
        for vehicle_id in map_trip_ids]

    return {
        "evaluation_type": "system_physical_outcome",
        "definitions": {
            "trip_vehicle": (
                "declared vehicle with an arrival obligation; persistent "
                "authored obstacles are excluded"),
            "map_trip_vehicle": (
                "non-focal trip vehicle measured for NPC impact; the focal "
                "remains in the physical world as a possible blocker but is "
                "not a soft-metric denominator"),
            "throughput": (
                "legacy diagnostic only: trip arrivals / frozen scenario "
                "minute; retained as a diagnostic and not used by the paired "
                "NPC impact metrics"),
            "completion_time": (
                "arrival time from scenario start, or the frozen deadline "
                "for a non-arrival"),
            "stationary_wait": (
                "active time below 0.5 km/h per trip vehicle"),
            "unexplained_wait": "DrivingEvaluator unexplained_idle_s for evaluated vehicles only",
            "queue": (
                "time spent in a spatially connected stopped component; "
                "crashed vehicles and persistent obstacles may be physical "
                "blockers but are not queue members or denominators"),
            "secondary_collision": "collision involving an entity that collided earlier",
            "candidate_deadlock": "at least two active vehicles each move <1 m in the final 20 s",
            "confirmed_deadlock": (
                "candidate trip vehicles remain in one spatially connected "
                "stopped component for >=80% of the final 20 s"),
            "stop_go_oscillation": "completed moving→stopped→moving cycle at a 2 km/h threshold",
        },
        "actor_scope_source": actor_scope_source,
        "focal_vehicle_id": focal_id,
        "required_trip_vehicle_ids": sorted(all_trip_ids),
        "map_trip_vehicle_ids": sorted(map_trip_ids),
        "persistent_obstacle_vehicle_ids": sorted(persistent_ids),
        "frozen_duration_s": float(duration_s),
        "trip_vehicle_metrics": trip_vehicle_metrics,
        "vehicle_count": len(result.vehicle_results),
        "trip_vehicle_count": len(all_trip_ids),
        "map_trip_vehicle_count": len(map_trip_ids),
        "all_required_trip_arrived_count": all_arrived,
        "arrived_count": map_arrived,
        "all_trip_vehicles_arrived": bool(
            all_trip_ids and all_arrived == len(all_trip_ids)),
        "all_map_trip_vehicles_arrived": bool(
            map_trip_ids and map_arrived == len(map_trip_ids)),
        "arrival_rate": (
            map_arrived / len(map_trip_ids) if map_trip_ids else None),
        "throughput_per_min": (
            map_arrived / max(float(duration_s) / 60.0, 1e-9)
            if map_trip_ids else None),
        "collision_count": len(collisions),
        "secondary_collision_count": _secondary_collisions(collisions),
        "stationary_wait_mean_s": (
            mean(stationary_waits) if stationary_waits else None),
        "stationary_wait_p90_s": percentile(stationary_waits, 0.9),
        "stationary_wait_gini": gini(stationary_waits),
        "unexplained_wait_evaluated_vehicle_count": len(
            unexplained_waits),
        "unexplained_wait_mean_s": (
            mean(unexplained_waits) if unexplained_waits else None),
        "unexplained_wait_p90_s": percentile(unexplained_waits, 0.9),
        "unexplained_wait_gini": gini(unexplained_waits),
        "queue_vehicle_seconds_total": round(sum(queue_waits), 6),
        "queue_wait_mean_s": mean(queue_waits) if queue_waits else None,
        "max_queue_length": max_queue,
        "max_queue_detail": queues_at_max,
        "candidate_deadlocks": candidate_items,
        "confirmed_deadlocks": confirmed_items,
        "confirmed_deadlock_count": len(confirmed_items),
        "stop_go_oscillations": {
            vehicle_id: _stop_go_oscillations(rows)
            for vehicle_id, rows in grouped.items()
            if vehicle_id in map_trip_ids
        },
    }
