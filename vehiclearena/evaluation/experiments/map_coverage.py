"""Feature-aware physical scene coverage for every bundled road map."""

from __future__ import annotations

import copy
import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from evaluation.experiments.scene_catalog import (
    CATALOG_SCHEMA,
    SceneAsset,
    Topology,
    _approach_connector_vehicle,
    _crosswalk,
    _scene,
    _vehicle,
)
from evaluation.experiments.experiment_parameters import (
    load_experiment_parameters,
)
from evaluation.experiments.scene_validator import SceneValidator
from simulation.multi_sim_engine import MultiScenario, MultiSimEngine


MAP_COVERAGE_SCHEMA = "vehiclearena-map-coverage-v1"
_MAP_COVERAGE_PARAMETERS = load_experiment_parameters()["map_coverage"]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def _all_network_ids(network_dir: Path) -> List[str]:
    return sorted(
        path.stem for path in Path(network_dir).glob("*.json")
        if not path.stem.endswith("_lane_level"))


def _coverage_scene(
    network: str, kind: str, topology: Topology, *, title: str,
    purpose: str, vehicles: Sequence[dict], pedestrians: Sequence[dict] = (),
    tags: Sequence[str], acceptance: Sequence[dict], duration_s: float = 8.0,
    dynamic_duration_s: float = 1.0,
) -> SceneAsset:
    scene_id = f"map__{network}__{kind}"
    asset = _scene(
        "MAP", scene_id, topology, title=title, purpose=purpose,
        vehicles=vehicles, pedestrians=pedestrians,
        duration_s=duration_s, tags=("map_coverage", kind, *tags),
        factors=(), acceptance=acceptance)
    asset.scenario["experiment_scene"].update({
        "coverage_kind": kind,
        "network_id": network,
    })
    asset.expected["smoke"].update({
        "duration_s": float(dynamic_duration_s),
        "allow_initial_collision": False,
        "minimum_initial_collisions": 0,
    })
    asset.expected["dynamic_assertions"] = [
        {"kind": "simulation_reaches_duration"},
        {"kind": "no_collision"},
    ]
    return asset


def _route_probe(topology: Topology) -> SceneAsset:
    try:
        lane, connectors = topology.connector_chains(
            count=1, minimum_connectors=3)[0]
    except ValueError:
        connector = max(
            topology.connectors,
            key=lambda item: (item["length_m"], item["id"]))
        lane = topology.lane[connector["from_lane"]]
        connectors = [connector]
    target = topology.lane[connectors[-1]["to_lane"]]
    vehicle = _vehicle(
        "ego", lane, destination_node=target["end_node"],
        progress=max(0.03, 1.0 - 35.0 / max(lane["length_m"], 1.0)),
        speed=20.0)
    vehicle["initial_physical_state"]["lane_route_actions"] = [{
        "type": "connector", "connector_id": item["id"],
    } for item in connectors]
    asset = _coverage_scene(
        topology.network, "route_chain", topology,
        title=f"{topology.network} frozen route chain",
        purpose="Exercise a connected lane-to-connector route on this map.",
        vehicles=[vehicle], tags=("route", "connector_chain"),
        acceptance=({
            "kind": "minimum_route_connectors", "entity": "ego",
            "count": len(connectors), "minimum_turns": 0,
        },))
    asset.expected["dynamic_assertions"].append({
        "kind": "minimum_vehicle_motion_m", "minimum": 0.05})
    return asset


def _following_probe(topology: Topology) -> SceneAsset:
    lane = max(topology.lanes, key=lambda item: (
        item["length_m"], item["id"]))
    length = max(1.0, float(lane["length_m"]))
    gap_m = min(30.0, max(10.0, length * 0.22))
    lead_progress = min(0.72, max(0.50, 1.0 - 25.0 / length))
    follower_progress = max(0.03, lead_progress - gap_m / length)
    vehicles = [
        _vehicle("ego", lane, progress=follower_progress, speed=24.0),
        _vehicle("lead", lane, progress=lead_progress, speed=18.0,
                 evaluated=False),
    ]
    return _coverage_scene(
        topology.network, "lane_following", topology,
        title=f"{topology.network} lane following",
        purpose="Exercise lane geometry, longitudinal dynamics and leader lookup.",
        vehicles=vehicles, tags=("following",),
        acceptance=(
            {"kind": "same_lane", "entities": ["ego", "lead"]},
            {"kind": "initial_longitudinal_gap_m", "rear": "ego",
             "front": "lead", "maximum": gap_m + 0.5},
        ))


def _conflict_probe(topology: Topology) -> SceneAsset:
    first, second = topology.conflict_pair(
        signalized=None, distinct_source_segments=True,
        minimum_source_length_m=float(_MAP_COVERAGE_PARAMETERS[
            "minimum_connector_source_length_m"]),
        minimum_crossing_angle_deg=float(_MAP_COVERAGE_PARAMETERS[
            "minimum_connector_conflict_angle_deg"]),
        prefer_crossing_angle=True)
    crossing_angle_deg = topology.connector_conflict_angle_deg(first, second)
    controlled = bool(first.get("signal_controlled")
                      or second.get("signal_controlled"))
    vehicles = [
        _approach_connector_vehicle(
            topology, "ego", first, distance_m=25.0,
            speed=15.0),
        _approach_connector_vehicle(
            topology, "cross_traffic", second, distance_m=29.0,
            speed=15.0, evaluated=False),
    ]
    asset = _coverage_scene(
        topology.network, "intersection_conflict", topology,
        title=f"{topology.network} connector conflict",
        purpose="Exercise decentralized negotiation at a real connector conflict.",
        vehicles=vehicles,
        tags=("intersection", "signalized" if controlled else "unsignalized"),
        acceptance=(
            {"kind": "connector_conflict",
             "entities": ["ego", "cross_traffic"]},
            {"kind": "signal_control", "value": controlled,
             "entities": ["ego", "cross_traffic"]},
        ))
    asset.scenario["experiment_scene"]["connector_selection"] = {
        "method": "generic_conflict_angle_ranker",
        "crossing_angle_deg": round(crossing_angle_deg, 3),
        "minimum_crossing_angle_deg": float(_MAP_COVERAGE_PARAMETERS[
            "minimum_connector_conflict_angle_deg"]),
        "connector_ids": [first["id"], second["id"]],
    }
    return asset


def _merge_probe(topology: Topology) -> SceneAsset:
    first, second = topology.merge_pair(
        minimum_source_length_m=35.0)
    vehicles = [
        _approach_connector_vehicle(
            topology, "ego", first, distance_m=24.0, speed=16.0),
        _approach_connector_vehicle(
            topology, "merging_vehicle", second,
            distance_m=28.0, speed=16.0, evaluated=False),
    ]
    return _coverage_scene(
        topology.network, "lane_merge", topology,
        title=f"{topology.network} lane merge",
        purpose="Exercise two connector paths converging on one target lane.",
        vehicles=vehicles, tags=("merge",),
        acceptance=({
            "kind": "same_target_lane",
            "entities": ["ego", "merging_vehicle"],
        },))


def _lane_change_probe(topology: Topology) -> SceneAsset:
    groups: Dict[tuple, List[dict]] = {}
    for lane in topology.lanes:
        groups.setdefault(
            (lane["segment_id"], lane["direction"]), []).append(lane)
    candidates = [
        sorted(lanes, key=lambda item: item["index"])
        for lanes in groups.values() if len(lanes) >= 2]
    if not candidates:
        raise ValueError("no same-direction parallel lanes")
    lanes = max(candidates, key=lambda items: (
        max(item["length_m"] for item in items), len(items), items[0]["id"]))
    first, second = lanes[0], lanes[1]
    length = max(1.0, min(first["length_m"], second["length_m"]))
    ego_progress = 0.25
    lead_progress = min(0.82, ego_progress + min(30.0, length * 0.3) / length)
    adjacent_progress = min(
        0.78, ego_progress + min(16.0, length * 0.16) / length)
    vehicles = [
        _vehicle("ego", first, progress=ego_progress, speed=25.0),
        _vehicle("slow_lead", first, progress=lead_progress,
                 speed=10.0, evaluated=False),
        _vehicle("adjacent_vehicle", second, progress=adjacent_progress,
                 speed=22.0, evaluated=False),
    ]
    return _coverage_scene(
        topology.network, "lane_change", topology,
        title=f"{topology.network} lane change",
        purpose="Exercise adjacent-lane geometry and gap-aware lane changing.",
        vehicles=vehicles, tags=("lane_change", "gap_acceptance"),
        acceptance=(
            {"kind": "same_lane", "entities": ["ego", "slow_lead"]},
            {"kind": "adjacent_lane",
             "entities": ["ego", "adjacent_vehicle"]},
        ))


def _signal_probe(topology: Topology) -> SceneAsset:
    # Some imported maps contain a signal plan for connector approaches that
    # have no usable stop-line mapping.  Such a connector cannot support a
    # physically meaningful stop probe.  Prefer an initially-red approach
    # with a real stop position, then fall back to a valid green approach.
    candidates = []
    for item in topology.connectors:
        if not item.get("signal_controlled"):
            continue
        lane = topology.lane[item["from_lane"]]
        stop = topology.runtime.stop_progress(lane["id"])
        state = topology.runtime.signal_state(item["id"], 0.0)
        if stop is None or state is None or state.signal not in ("red", "green"):
            continue
        candidates.append((
            0 if state.signal == "red" else 1,
            -float(lane["length_m"]), item["id"], item, state.signal,
        ))
    if not candidates:
        raise ValueError(
            f"{topology.network} has no signal connector with a stop line")
    _, _, _, connector, initial_signal = min(candidates)
    lane = topology.lane[connector["from_lane"]]
    target = topology.lane[connector["to_lane"]]
    stop_progress = topology.runtime.stop_progress(lane["id"])
    vehicle = _vehicle(
        "ego", lane, destination_node=target["end_node"],
        progress=max(0.03, stop_progress - 14.0 / max(lane["length_m"], 1.0)),
        speed=15.0)
    vehicle["initial_physical_state"]["lane_route_actions"] = [{
        "type": "connector", "connector_id": connector["id"],
    }]
    asset = _coverage_scene(
        topology.network, "signal_stop", topology,
        title=(f"{topology.network} red-light stop"
               if initial_signal == "red"
               else f"{topology.network} green-light departure"),
        purpose=("Exercise connector-level red signal enforcement."
                 if initial_signal == "red"
                 else "Exercise connector-level green signal departure."),
        vehicles=[vehicle], tags=(
            "signal", "red_stop" if initial_signal == "red"
            else "green_departure"),
        acceptance=({
            "kind": "initial_signal", "connector_id": connector["id"],
            "value": initial_signal,
        },))
    asset.scenario["experiment_scene"]["initial_signal"] = initial_signal
    if initial_signal == "red":
        asset.expected["dynamic_assertions"].append({
            "kind": "connector_not_entered", "entity": "ego"})
    else:
        asset.expected["dynamic_assertions"].append({
            "kind": "minimum_vehicle_motion_m", "minimum": 0.05})
    return asset


def _crosswalk_probe(topology: Topology) -> SceneAsset:
    asset = _crosswalk(
        topology, "MAP", f"map__{topology.network}__crosswalk_yield",
        title=f"{topology.network} crosswalk yield",
        factors=(), vehicle_progress_offset=0.4)
    asset.scenario["experiment_scene"].update({
        "coverage_kind": "crosswalk_yield",
        "network_id": topology.network,
        "interaction_tags": [
            "map_coverage", "crosswalk_yield", "crosswalk",
            "vehicle_pedestrian"],
        "llm_roles": [],
    })
    asset.expected["smoke"].update({
        "duration_s": 1.0,
        "allow_initial_collision": False,
        "minimum_initial_collisions": 0,
    })
    asset.expected["dynamic_assertions"] = [
        {"kind": "simulation_reaches_duration"},
        {"kind": "no_collision"},
    ]
    return asset


def build_map_assets(topology: Topology) -> tuple[List[SceneAsset], dict]:
    """Build all applicable probes and record feature-based omissions."""
    builders = [
        ("route_chain", _route_probe),
        ("lane_following", _following_probe),
        ("intersection_conflict", _conflict_probe),
        ("lane_merge", _merge_probe),
        ("lane_change", _lane_change_probe),
    ]
    if topology.data.get("signal_plans"):
        builders.append(("signal_stop", _signal_probe))
    if topology.data.get("crosswalks"):
        builders.append(("crosswalk_yield", _crosswalk_probe))
    assets = []
    omissions = []
    for kind, builder in builders:
        try:
            assets.append(builder(topology))
        except (KeyError, StopIteration, ValueError) as exc:
            omissions.append({
                "kind": kind,
                "reason": f"{type(exc).__name__}: {exc}",
            })
    if len(assets) < 3:
        raise ValueError(
            f"{topology.network} produced only {len(assets)} coverage probes")
    counts = {
        key: len(topology.data.get(key, []))
        for key in (
            "lanes", "connectors", "connector_conflicts", "stop_lines",
            "crosswalks", "signal_plans")}
    profile = {
        "schema": MAP_COVERAGE_SCHEMA,
        "network_id": topology.network,
        "quality": topology.data.get("quality", {}),
        "feature_counts": counts,
        "scene_ids": [asset.scene_id for asset in assets],
        "omissions": omissions,
        "manual_overrides": {},
        "connector_conflict_selection": {
            "method": "generic_conflict_angle_ranker",
            **copy.deepcopy(_MAP_COVERAGE_PARAMETERS),
        },
    }
    return assets, profile


def generate_map_coverage(output_dir: Path, *, source_root: Path | None = None,
                          network_ids: Iterable[str] | None = None) -> dict:
    """Generate feature-aware scene directories for all selected maps."""
    output_dir = Path(output_dir)
    source_root = source_root or Path(__file__).resolve().parents[2]
    network_dir = source_root / "simulation" / "road_networks"
    selected = list(network_ids or _all_network_ids(network_dir))
    entries = []
    maps = []
    for network in selected:
        topology = Topology(network, network_dir)
        assets, profile = build_map_assets(topology)
        map_root = output_dir / "maps" / network
        _write_json(map_root / "profile.json", profile)
        maps.append({
            "network_id": network,
            "profile": str(Path("maps") / network / "profile.json"),
            "scene_count": len(assets),
            "feature_counts": profile["feature_counts"],
            "omission_count": len(profile["omissions"]),
        })
        for asset in assets:
            kind = asset.scenario["experiment_scene"]["coverage_kind"]
            relative = Path("maps") / network / kind
            _write_json(output_dir / relative / "scenario.json", asset.scenario)
            _write_json(output_dir / relative / "expected.json", asset.expected)
            entries.append({
                "network_id": network,
                "scene_id": asset.scene_id,
                "kind": kind,
                "scenario": str(relative / "scenario.json"),
                "expected": str(relative / "expected.json"),
            })
    catalog = {
        "schema": MAP_COVERAGE_SCHEMA,
        "map_count": len(maps),
        "scene_count": len(entries),
        "maps": maps,
        "entries": entries,
    }
    _write_json(output_dir / "catalog.json", catalog)
    _write_text(
        output_dir / "README.md",
        "# VehicleArena 全地图动态覆盖目录\n\n"
        f"包含 {len(maps)} 张程序化车道级地图和 {len(entries)} 个冻结场景。"
        "每张图固定覆盖路线、跟车、路口冲突、汇入和变道；仅在地图包含对应设施时"
        "增加信号与斑马线场景。`profile.json` 记录能力、遗漏和通用几何"
        "选点参数。路口冲突使用连接器交角排序，不依赖地图名称特判。\n\n"
        "生成：`python -m evaluation.experiments.cli prepare-map-coverage "
        "--output vehiclearena/evaluation/experiments/scenarios/map_coverage`\n\n"
        "验证：`python -m evaluation.experiments.cli validate-map-coverage "
        "--catalog vehiclearena/evaluation/experiments/scenarios/map_coverage "
        "--report evaluation/map_coverage_validation/latest --workers 3`")
    return catalog


def _run_dynamic_scene(payload: dict, *, lane_geometry_runtime=None) -> dict:
    """Worker entry point for one isolated map-coverage scene."""
    logging.disable(logging.CRITICAL)
    scenario = copy.deepcopy(payload["scenario"])
    expected = payload["expected"]
    duration_s = float(expected.get("smoke", {}).get("duration_s", 1.0))
    scenario["total_time_s"] = duration_s
    scenario["tick_interval_s"] = duration_s
    for vehicle in scenario.get("vehicles", []):
        vehicle["agent_config"] = {"type": "sumo"}
    for pedestrian in scenario.get("pedestrians", []):
        pedestrian["agent_config"] = {"type": "sumo"}
    engine = MultiSimEngine(
        MultiScenario.from_dict(scenario),
        lane_geometry_runtime=lane_geometry_runtime)
    engine.run({})
    collisions = [{
        "entity_a": item.entity_a,
        "entity_b": item.entity_b,
        "collision_type": item.collision_type,
        "time_s": item.time_s,
    } for item in engine.traffic_mgr.collision_log]
    vehicle_motion = {
        vehicle_id: round(state.distance_traveled_m, 6)
        for vehicle_id, state in engine.traffic_mgr.vehicles.items()}
    failures = []
    kind = scenario["experiment_scene"]["coverage_kind"]
    if collisions:
        failures.append("unexpected_collision")
    if engine.traffic_mgr._physics_time < duration_s - 1e-6:
        failures.append("simulation_ended_early")
    if kind == "route_chain" and max(vehicle_motion.values(), default=0.0) < 0.05:
        failures.append("route_vehicle_did_not_move")
    if (kind == "signal_stop"
            and scenario["experiment_scene"].get("initial_signal") == "red"):
        ego = engine.traffic_mgr.vehicles["ego"]
        if ego.active_connector_id or ego.lane_route_action_index > 0:
            failures.append("red_signal_connector_entered")
    if (kind == "signal_stop"
            and scenario["experiment_scene"].get("initial_signal") == "green"
            and max(vehicle_motion.values(), default=0.0) < 0.05):
        failures.append("green_signal_vehicle_did_not_move")
    return {
        "network_id": payload["network_id"],
        "scene_id": scenario["scenario_id"],
        "kind": kind,
        "status": "passed" if not failures else "failed",
        "duration_s": duration_s,
        "collisions": collisions,
        "vehicle_motion_m": vehicle_motion,
        "failures": failures,
    }


def _run_map_batch(payload: dict) -> List[dict]:
    """Validate and execute one map at a time to bound topology memory."""
    catalog_dir = Path(payload["catalog_dir"])
    network_dir = Path(payload["network_dir"])
    validator = SceneValidator(network_dir)
    results = []
    for entry in payload["entries"]:
        scenario = json.loads(
            (catalog_dir / entry["scenario"]).read_text(encoding="utf-8"))
        expected = json.loads(
            (catalog_dir / entry["expected"]).read_text(encoding="utf-8"))
        scene_payload = {
            "network_id": entry["network_id"],
            "scenario": scenario,
            "expected": expected,
        }
        try:
            validator.validate(scenario, expected)
        except Exception as exc:
            results.append({
                "network_id": entry["network_id"],
                "scene_id": entry["scene_id"],
                "kind": entry["kind"],
                "status": "failed",
                "failures": [
                    f"static: {type(exc).__name__}: {exc}"],
            })
            continue
        try:
            results.append(_run_dynamic_scene(
                scene_payload,
                lane_geometry_runtime=validator.runtime(
                    entry["network_id"])))
        except Exception as exc:
            results.append({
                "network_id": entry["network_id"],
                "scene_id": entry["scene_id"],
                "kind": entry["kind"],
                "status": "failed",
                "failures": [
                    f"dynamic: {type(exc).__name__}: {exc}"],
            })
    return results


def validate_map_coverage(
    catalog_dir: Path, *, source_root: Path | None = None,
    workers: int = 1, report_dir: Path | None = None,
) -> dict:
    """Statically and dynamically validate every generated coverage scene."""
    catalog_dir = Path(catalog_dir)
    report_dir = Path(report_dir or catalog_dir / "validation")
    source_root = source_root or Path(__file__).resolve().parents[2]
    catalog = json.loads(
        (catalog_dir / "catalog.json").read_text(encoding="utf-8"))
    if catalog.get("schema") != MAP_COVERAGE_SCHEMA:
        raise ValueError("unsupported map-coverage catalog")
    by_network: Dict[str, List[dict]] = {}
    for entry in catalog["entries"]:
        by_network.setdefault(entry["network_id"], []).append(entry)
    payloads = [{
        "catalog_dir": str(catalog_dir),
        "network_dir": str(source_root / "simulation" / "road_networks"),
        "entries": entries,
    } for _, entries in sorted(by_network.items())]
    results = []
    if workers <= 1:
        for payload in payloads:
            results.extend(_run_map_batch(payload))
    else:
        # One map per child guarantees that large geometry/conflict indexes
        # are returned to the OS instead of lingering in Python's allocator.
        with ProcessPoolExecutor(
                max_workers=workers, max_tasks_per_child=1) as pool:
            futures = {
                pool.submit(_run_map_batch, payload): payload
                for payload in payloads}
            for future in as_completed(futures):
                payload = futures[future]
                try:
                    results.extend(future.result())
                except Exception as exc:
                    for entry in payload["entries"]:
                        results.append({
                            "network_id": entry["network_id"],
                            "scene_id": entry["scene_id"],
                            "kind": entry["kind"],
                            "status": "failed",
                            "failures": [
                                f"map batch: {type(exc).__name__}: {exc}"],
                        })
    results.sort(key=lambda item: (item["network_id"], item["kind"]))
    by_map: Dict[str, List[dict]] = {}
    for item in results:
        by_map.setdefault(item["network_id"], []).append(item)
    map_reports = []
    for network_id, scene_results in sorted(by_map.items()):
        status = (
            "passed" if all(item["status"] == "passed"
                            for item in scene_results) else "failed")
        report = {
            "schema": MAP_COVERAGE_SCHEMA,
            "network_id": network_id,
            "status": status,
            "scene_count": len(scene_results),
            "passed": sum(item["status"] == "passed"
                          for item in scene_results),
            "failed": sum(item["status"] != "passed"
                          for item in scene_results),
            "scenes": scene_results,
        }
        _write_json(report_dir / "maps" / f"{network_id}.json", report)
        map_reports.append({key: report[key] for key in (
            "network_id", "status", "scene_count", "passed", "failed")})
    summary = {
        "schema": MAP_COVERAGE_SCHEMA,
        "map_count": len(map_reports),
        "scene_count": len(results),
        "passed_maps": sum(item["status"] == "passed"
                           for item in map_reports),
        "failed_maps": sum(item["status"] != "passed"
                           for item in map_reports),
        "passed_scenes": sum(item["status"] == "passed" for item in results),
        "failed_scenes": sum(item["status"] != "passed" for item in results),
        "maps": map_reports,
    }
    _write_json(report_dir / "summary.json", summary)
    return summary
