"""Trajectory-based evaluation for SUMO/LLM pedestrians."""

from __future__ import annotations

import math
from collections import defaultdict
from statistics import mean
from typing import Dict, List


def evaluate_pedestrians(
    result, pedestrian_trajectory: List[dict], vehicle_trajectory: List[dict],
) -> dict:
    ped_rows: Dict[str, List[dict]] = defaultdict(list)
    vehicles_by_time = defaultdict(list)
    for row in pedestrian_trajectory:
        ped_rows[row["ped_id"]].append(row)
    for row in vehicle_trajectory:
        vehicles_by_time[row["time_s"]].append(row)

    reports = {}
    for ped_id, rows in ped_rows.items():
        rows.sort(key=lambda item: item["time_s"])
        observed_s = waiting_s = walking_s = crosswalk_s = 0.0
        minimum_distance = float("inf")
        minimum_moving_distance = float("inf")
        previous = None
        crossing_entries = 0
        first_crossing_entry_s = None
        movement_speeds = []
        for row in rows:
            dt = 0.0 if previous is None else max(
                0.0, row["time_s"] - previous["time_s"])
            if row["spawned"] and not row["arrived"] and not row["crashed"]:
                observed_s += dt
                waiting_s += dt if row["waiting"] else 0.0
                walking_s += dt if row["walking"] else 0.0
                crosswalk_s += dt if row["on_crosswalk"] else 0.0
            if row["on_crosswalk"] and (
                    previous is None or not previous["on_crosswalk"]):
                crossing_entries += 1
                if first_crossing_entry_s is None:
                    first_crossing_entry_s = float(row["time_s"])
            if row["walking"] or row["on_crosswalk"]:
                movement_speeds.append(float(row.get("speed_mps", 0.0)))
            if row["pose_x_m"] is not None:
                for vehicle in vehicles_by_time.get(row["time_s"], []):
                    distance = math.hypot(
                        row["pose_x_m"] - vehicle["pose_x_m"],
                        row["pose_y_m"] - vehicle["pose_y_m"])
                    minimum_distance = min(minimum_distance, distance)
                    if row["walking"] or row["on_crosswalk"]:
                        minimum_moving_distance = min(
                            minimum_moving_distance, distance)
            previous = row
        terminal = result.pedestrian_results.get(ped_id)
        reports[ped_id] = {
            "control_authority": rows[-1]["control_authority"],
            "arrived": bool(terminal.arrived) if terminal else rows[-1]["arrived"],
            "crashed": bool(terminal.crashed) if terminal else rows[-1]["crashed"],
            "observed_time_s": round(observed_s, 3),
            "waiting_time_s": round(waiting_s, 3),
            "walking_time_s": round(walking_s, 3),
            "crosswalk_time_s": round(crosswalk_s, 3),
            "crossing_entries": crossing_entries,
            "first_crossing_entry_s": first_crossing_entry_s,
            "mean_movement_speed_mps": (
                mean(movement_speeds) if movement_speeds else None),
            "min_vehicle_center_distance_m": (
                round(minimum_distance, 3)
                if math.isfinite(minimum_distance) else None),
            "min_vehicle_center_distance_while_moving_m": (
                round(minimum_moving_distance, 3)
                if math.isfinite(minimum_moving_distance) else None),
        }

    by_authority = defaultdict(list)
    for report in reports.values():
        by_authority[report["control_authority"]].append(report)
    summaries = {}
    for authority, items in by_authority.items():
        summaries[authority] = {
            "count": len(items),
            "arrival_rate": sum(item["arrived"] for item in items) / len(items),
            "crash_rate": sum(item["crashed"] for item in items) / len(items),
            "mean_waiting_time_s": mean(
                item["waiting_time_s"] for item in items),
            "mean_crosswalk_time_s": mean(
                item["crosswalk_time_s"] for item in items),
            "mean_movement_speed_mps": mean(values) if (values := [
                item["mean_movement_speed_mps"] for item in items
                if item["mean_movement_speed_mps"] is not None
            ]) else None,
            "mean_first_crossing_entry_s": mean(values) if (values := [
                item["first_crossing_entry_s"] for item in items
                if item["first_crossing_entry_s"] is not None
            ]) else None,
            "mean_min_vehicle_center_distance_m": mean(
                values) if (values := [
                    item["min_vehicle_center_distance_m"] for item in items
                    if item["min_vehicle_center_distance_m"] is not None
                ]) else None,
            "mean_min_vehicle_center_distance_while_moving_m": mean(
                values) if (values := [
                    item["min_vehicle_center_distance_while_moving_m"]
                    for item in items
                    if item[
                        "min_vehicle_center_distance_while_moving_m"]
                    is not None
                ]) else None,
        }
    return {
        "evaluation_type": "pedestrian_physical_trajectory",
        "definitions": {
            "distance": (
                "center-to-center distance; collision domains remain "
                "authoritative; moving distance excludes curb waiting"),
            "times": "integrated from fixed-step post-physics samples",
        },
        "pedestrians": reports,
        "authority_summary": summaries,
    }
