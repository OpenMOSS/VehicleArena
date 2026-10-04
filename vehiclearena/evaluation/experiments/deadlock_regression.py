"""Large SUMO-native background deadlock regression over the scene catalog."""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import os
import time
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable, Sequence

from evaluation.experiments.scene_catalog import iter_catalog_scenes
from evaluation.experiments.system_evaluator import candidate_deadlocks
from simulation.lane_level_runtime import LaneGeometryRuntime
from simulation.multi_sim_engine import MultiScenario, MultiSimEngine
from simulation.road_networks import load_road_network


REPORT_SCHEMA = "vehiclearena-sumo-deadlock-regression-v3"
DEFAULT_EXPERIMENTS = ("Basic", "MultiLLM")
_RUNTIME_CACHE: dict[str, LaneGeometryRuntime] = {}


def _jsonable(value):
    if dataclasses.is_dataclass(value):
        return {
            field.name: _jsonable(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _runtime_for(network_id: str) -> LaneGeometryRuntime:
    """Reuse only the immutable map topology inside one worker process."""
    runtime = _RUNTIME_CACHE.get(network_id)
    if runtime is None:
        network = load_road_network(network_id)
        runtime = LaneGeometryRuntime.load(network.lane_level_path)
        _RUNTIME_CACHE[network_id] = runtime
    return runtime


class DeadlockAuditEngine(MultiSimEngine):
    """Retain only the final observation window, not a full 10 Hz trace."""

    def __init__(
        self, scenario: MultiScenario, *, lane_geometry_runtime,
        window_s: float = 20.0,
    ):
        super().__init__(
            scenario, lane_geometry_runtime=lane_geometry_runtime)
        self.window_s = float(window_s)
        self.audit_rows = deque()

    def _log_per_substep(self, physics_time, tick_index, trigger_events):
        del trigger_events
        for vehicle_id in sorted(self.traffic_mgr.vehicles):
            vehicle = self.traffic_mgr.vehicles[vehicle_id]
            self.audit_rows.append({
                "vehicle_id": vehicle_id,
                "time_s": round(float(physics_time), 6),
                "tick_index": int(tick_index),
                "speed_kmh": round(float(vehicle.current_speed_kmh), 4),
                "distance_traveled_m": round(
                    float(vehicle.distance_traveled_m), 4),
                "arrived": bool(vehicle.arrived),
                "crashed": bool(vehicle.is_crashed),
                "waiting_red_light": bool(vehicle.waiting_red_light),
                "current_lane_id": vehicle.current_lane_id,
                "active_connector_id": vehicle.active_connector_id,
                "control_authority": vehicle.control_authority,
            })
        cutoff = float(physics_time) - self.window_s - 0.2
        while self.audit_rows and self.audit_rows[0]["time_s"] < cutoff:
            self.audit_rows.popleft()

    def _finalize_result(self, result):
        super()._finalize_result(result)
        result._deadlock_audit_rows = list(self.audit_rows)
        result._collision_log = _jsonable(self.traffic_mgr.collision_log)


def _sumo_scenario(payload: dict) -> MultiScenario:
    value = copy.deepcopy(payload)
    value["physics_only_mode"] = True
    value["enable_driving_evaluation"] = False
    value["stop_when_all_vehicles_terminal"] = True
    for vehicle in value.get("vehicles", []):
        vehicle["agent_config"] = {"type": "sumo"}
        # Audit every physical actor and stop only when all are terminal.
        vehicle["is_evaluated"] = True
    for pedestrian in value.get("pedestrians", []):
        pedestrian["agent_config"] = {"type": "sumo"}
        pedestrian["is_evaluated"] = True
    return MultiScenario.from_dict(value)


def _run_one(task: dict) -> dict:
    started = time.perf_counter()
    scenario = _sumo_scenario(task["scenario"])
    engine = DeadlockAuditEngine(
        scenario,
        lane_geometry_runtime=_runtime_for(scenario.road_network_id),
        window_s=task["window_s"],
    )
    result = engine.run({})
    rows = result._deadlock_audit_rows
    candidates = candidate_deadlocks(rows, window_s=task["window_s"])
    collisions = result._collision_log
    vehicles = engine.traffic_mgr.vehicles
    final_states = {
        vehicle_id: {
            "arrived": bool(vehicle.arrived),
            "crashed": bool(vehicle.is_crashed),
            "speed_kmh": round(float(vehicle.current_speed_kmh), 3),
            "distance_traveled_m": round(
                float(vehicle.distance_traveled_m), 3),
            "control_authority": vehicle.control_authority,
        }
        for vehicle_id, vehicle in sorted(vehicles.items())
    }
    return {
        "scene_id": scenario.scenario_id,
        "experiment_id": task["experiment_id"],
        "network": scenario.road_network_id,
        "tags": task["tags"],
        "duration_s": float(scenario.total_time_s),
        "simulated_until_s": max(
            (float(row["time_s"]) for row in rows), default=0.0),
        "wall_time_s": round(time.perf_counter() - started, 4),
        "vehicle_count": len(vehicles),
        "arrived_count": sum(vehicle.arrived for vehicle in vehicles.values()),
        "crashed_count": sum(
            vehicle.is_crashed for vehicle in vehicles.values()),
        "collision_count": len(collisions),
        "collisions": collisions,
        "candidate_deadlocks": candidates,
        "final_states": final_states,
    }


def _safe_run_one(task: dict) -> dict:
    try:
        return {"status": "completed", **_run_one(task)}
    except Exception as exc:
        return {
            "status": "failed",
            "scene_id": task["scenario"].get("scenario_id", "unknown"),
            "experiment_id": task["experiment_id"],
            "network": task["entry"].get("network", ""),
            "error": f"{type(exc).__name__}: {exc}",
        }


def _counter(rows: Iterable[dict], key: str) -> dict:
    return dict(sorted(Counter(row[key] for row in rows).items()))


def _summary(runs: Sequence[dict]) -> dict:
    completed = [run for run in runs if run["status"] == "completed"]
    candidates = [
        (run, item) for run in completed
        for item in run["candidate_deadlocks"]
    ]
    vehicle_count = sum(run["vehicle_count"] for run in completed)
    arrived_count = sum(run["arrived_count"] for run in completed)
    crashed_count = sum(run["crashed_count"] for run in completed)
    return {
        "run_count": len(runs),
        "completed_run_count": len(completed),
        "failed_run_count": len(runs) - len(completed),
        "scene_count": len({run["scene_id"] for run in runs}),
        "network_count": len({run["network"] for run in completed}),
        "vehicle_episode_count": vehicle_count,
        "arrived_vehicle_count": arrived_count,
        "crashed_vehicle_count": crashed_count,
        "active_at_time_limit_count": max(
            0, vehicle_count - arrived_count - crashed_count),
        "collision_run_count": sum(
            run["collision_count"] > 0 for run in completed),
        "candidate_deadlock_run_count": len({
            run["scene_id"] for run, _ in candidates}),
        "candidate_deadlock_kind_counts": dict(sorted(Counter(
            item["kind"] for _, item in candidates).items())),
        "runs_by_experiment": _counter(completed, "experiment_id"),
        "runs_by_network": _counter(completed, "network"),
    }


def _grouped(rows: Iterable[dict], key: str) -> dict[str, list]:
    result = defaultdict(list)
    for row in rows:
        result[row[key]].append(row)
    return result


def run_deadlock_regression(
    catalog_dir: Path, *, output_path: Path | None = None,
    experiments: Sequence[str] = DEFAULT_EXPERIMENTS,
    tags: Sequence[str] = (), scene_ids: Sequence[str] = (),
    window_s: float = 20.0, workers: int = 1, progress: bool = False,
) -> dict:
    selected_experiments = set(experiments)
    selected_tags = set(tags)
    selected_scenes = set(scene_ids)
    tasks = []
    for entry, scenario_path, _ in iter_catalog_scenes(catalog_dir):
        if selected_experiments and entry["experiment_id"] not in selected_experiments:
            continue
        if selected_scenes and entry["scene_id"] not in selected_scenes:
            continue
        if selected_tags and not selected_tags.intersection(entry.get("tags", [])):
            continue
        scenario = json.loads(scenario_path.read_text(encoding="utf-8"))
        tasks.append({
            "entry": entry,
            "experiment_id": entry["experiment_id"],
            "tags": list(entry.get("tags", [])),
            "scenario": scenario,
            "window_s": float(window_s),
        })
    # Keep same-map work adjacent.  Process workers can then reuse one
    # immutable topology index instead of each loading every map in a catalog.
    tasks.sort(key=lambda task: (
        task["entry"].get("network", ""),
        task["scenario"].get("scenario_id", "")))
    started = time.perf_counter()
    pool = None
    if max(1, int(workers)) == 1:
        iterator = map(_safe_run_one, tasks)
    else:
        chunksize = max(1, len(tasks) // (max(1, int(workers)) * 4))
        pool = ProcessPoolExecutor(max_workers=max(1, int(workers)))
        iterator = pool.map(_safe_run_one, tasks, chunksize=chunksize)
    runs = []
    try:
        for index, run in enumerate(iterator, start=1):
            runs.append(run)
            if progress and (index % 25 == 0 or index == len(tasks)):
                print(
                    f"deadlock audit: {index}/{len(tasks)} runs complete",
                    flush=True)
    except BaseException:
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)
            pool = None
        raise
    finally:
        if pool is not None:
            pool.shutdown(wait=True)
    runs.sort(key=lambda item: (
        item.get("experiment_id", ""), item.get("scene_id", "")))
    report = {
        "schema": REPORT_SCHEMA,
        "configuration": {
            "catalog": str(Path(catalog_dir)),
            "experiments": list(experiments),
            "tags_any": list(tags),
            "scene_ids": list(scene_ids),
            "window_s": float(window_s),
            "physics_step_s": 0.1,
            "workers": max(1, int(workers)),
        },
        "wall_time_s": round(time.perf_counter() - started, 4),
        "summary": _summary(runs),
        "runs": runs,
    }
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        temporary.write_text(json.dumps(
            report, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, output_path)
    return report


def _split(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Run large SUMO-background deadlock regressions")
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiments", default=",".join(DEFAULT_EXPERIMENTS))
    parser.add_argument("--tags", default="")
    parser.add_argument("--scene", action="append", default=[])
    parser.add_argument("--window", type=float, default=20.0)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    report = run_deadlock_regression(
        args.catalog, output_path=args.output,
        experiments=_split(args.experiments),
        tags=_split(args.tags), scene_ids=args.scene,
        window_s=args.window, workers=args.workers, progress=True)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 0 if report["summary"]["failed_run_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
