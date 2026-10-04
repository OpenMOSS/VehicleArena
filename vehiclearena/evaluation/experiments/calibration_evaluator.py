"""Continuous calibration metrics for non-LLM experiment factors."""

from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Iterable, List


def _duration_weights(rows: List[dict]) -> List[float]:
    weights = []
    previous = None
    for row in rows:
        weights.append(
            0.0 if previous is None else
            max(0.0, float(row["time_s"]) - float(previous["time_s"])))
        previous = row
    return weights


def evaluate_traffic_calibration(
    vehicle_trajectory: List[dict], collision_log: Iterable[dict],
) -> dict:
    """Summarize physical behavior by the explicit control authority."""
    grouped = defaultdict(list)
    for row in vehicle_trajectory:
        grouped[str(row.get("control_authority", "unknown"))].append(row)
    collided = defaultdict(set)
    for event in collision_log:
        for key in ("entity_a", "entity_b"):
            entity_id = str(event.get(key, ""))
            if entity_id:
                collided[entity_id].add(float(event.get("time_s", 0.0)))

    reports = {}
    for authority, rows in sorted(grouped.items()):
        rows.sort(key=lambda item: (item["vehicle_id"], item["time_s"]))
        by_vehicle = defaultdict(list)
        for row in rows:
            by_vehicle[row["vehicle_id"]].append(row)
        moving_s = stopped_s = red_wait_s = total_s = 0.0
        speeds = []
        lane_change_entries = 0
        vehicle_reports = {}
        for vehicle_id, entity_rows in by_vehicle.items():
            weights = _duration_weights(entity_rows)
            previous_changing = False
            distance = max(
                (float(row.get("distance_traveled_m", 0.0))
                 for row in entity_rows), default=0.0)
            for row, dt in zip(entity_rows, weights):
                speed = float(row.get("speed_kmh", 0.0))
                speeds.append(speed)
                total_s += dt
                moving_s += dt if speed >= 2.0 else 0.0
                stopped_s += dt if speed < 2.0 else 0.0
                red_wait_s += dt if row.get("waiting_red_light") else 0.0
                changing = bool(row.get("is_changing_lane"))
                if changing and not previous_changing:
                    lane_change_entries += 1
                previous_changing = changing
            vehicle_reports[vehicle_id] = {
                "distance_traveled_m": round(distance, 3),
                "collision_count": len(collided.get(vehicle_id, ())),
            }
        reports[authority] = {
            "vehicle_count": len(by_vehicle),
            "mean_speed_kmh": mean(speeds) if speeds else None,
            "moving_ratio": moving_s / total_s if total_s else 0.0,
            "stopped_ratio": stopped_s / total_s if total_s else 0.0,
            "signal_wait_time_s": round(red_wait_s, 3),
            "lane_change_entries": lane_change_entries,
            "collision_involved_vehicle_count": sum(
                bool(collided.get(vehicle_id)) for vehicle_id in by_vehicle),
            "vehicles": vehicle_reports,
        }
    return {"authorities": reports}


def evaluate_chassis_calibration(
    vehicle_trajectory: List[dict], evaluated_vehicle_ids: Iterable[str],
) -> dict:
    """Expose physical response differences even when terminal outcomes tie."""
    selected = set(evaluated_vehicle_ids)
    grouped = defaultdict(list)
    for row in vehicle_trajectory:
        if row["vehicle_id"] in selected:
            grouped[row["vehicle_id"]].append(row)
    reports = {}
    for vehicle_id, rows in grouped.items():
        rows.sort(key=lambda item: item["time_s"])
        accelerations = [float(row.get("acceleration_mps2", 0.0))
                         for row in rows]
        first_braking = next((row for row in rows
                              if float(row.get("acceleration_mps2", 0.0))
                              <= -0.5), None)
        lane_change_time_s = 0.0
        previous = None
        for row in rows:
            if previous is not None and row.get("is_changing_lane"):
                lane_change_time_s += max(
                    0.0, float(row["time_s"]) - float(previous["time_s"]))
            previous = row
        reports[vehicle_id] = {
            "chassis_profile": rows[-1].get("chassis_profile", ""),
            "peak_speed_kmh": max(
                float(row.get("speed_kmh", 0.0)) for row in rows),
            "max_acceleration_mps2": max(accelerations, default=0.0),
            "max_braking_mps2": abs(min(accelerations, default=0.0)),
            "first_braking_time_s": (
                float(first_braking["time_s"]) if first_braking else None),
            "first_braking_distance_m": (
                float(first_braking["distance_traveled_m"])
                if first_braking else None),
            "lane_change_time_s": round(lane_change_time_s, 3),
            "distance_traveled_m": float(
                rows[-1].get("distance_traveled_m", 0.0)),
            "arrived": bool(rows[-1].get("arrived")),
            "crashed": bool(rows[-1].get("crashed")),
        }
    return {"vehicles": reports}


def evaluate_capability_calibration(result) -> dict:
    reports = {}
    for vehicle_id, vehicle in result.vehicle_results.items():
        if not vehicle.is_evaluated:
            continue
        events = list(vehicle.capability_events)
        reports[vehicle_id] = {
            "equipment_profile": vehicle.capabilities.get(
                "equipment_profile", ""),
            # ``VehicleResult`` exposes the YAML evaluation through derived
            # properties; the serialized ``cabin_evaluation`` mapping only
            # exists in ``MultiSimResult.to_dict()``.
            "cabin_applicable": vehicle.cabin_score is not None,
            "cabin_score": vehicle.cabin_score,
            "capability_probe_count": sum(
                event.get("type") == "capability_probe_delivered"
                for event in events),
            "filtered_request_count": sum(
                event.get("type") == "passenger_request_filtered"
                for event in events),
            "capability_events": events,
        }
    return {"vehicles": reports}
