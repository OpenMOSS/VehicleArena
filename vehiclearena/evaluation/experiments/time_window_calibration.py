"""Reference-policy calibration of frozen experiment time windows.

Time limits are benchmark infrastructure, not a treatment outcome.  This
module calibrates Basic with all-SUMO traffic and can calibrate MultiLLM with
the paired-reference policy: SUMO focal plus fixed peer LLMs.  Each case
executes the same authoritative 0.1 s physics engine used by evaluation.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from evaluation.experiments.scene_catalog import iter_catalog_scenes


CALIBRATION_SCHEMA = "vehiclearena-time-window-calibration-v6"
CALIBRATION_PROTOCOL = (
    "paired-reference-max-sumo-terminal-or-last-event-plus-10s-2026-09")
NPC_ARRIVAL_MARGIN_S = 10.0
DEFAULT_REGISTRY_PATH = Path(__file__).with_name(
    "time_window_calibration.json")

_DRIVER_ROLE_PROMPTS = (
    "You are a cautious driver. Prefer generous following gaps, early "
    "observation, and gradual speed changes. Yield when right of way or "
    "visibility is uncertain; proceed when the situation is safely resolved. "
    "Complete your own route and passenger tasks.",
    "You are an efficient driver. Make timely progress along your own route, "
    "choose legal useful gaps, and avoid unnecessary stops. Observe speed "
    "limits and right of way, and never trade safety for speed. Complete "
    "passenger tasks when safe.",
    "You are a cooperative driver. Make your actions predictable, signal "
    "intentions, and accommodate safe merges and vulnerable road users. "
    "Resolve interactions using only your own observations and available "
    "tools; do not assume access to other agents' plans. Complete your own "
    "route and passenger tasks.",
    "You are a comfort-focused driver. Prefer smooth acceleration, "
    "progressive braking and stable lane choice. Balance passenger comfort "
    "with timely legal progress, following distances and road safety. "
    "Complete passenger tasks when safe.",
)


def _public_peer_protocol(config: dict | None) -> dict | None:
    """Return behavioural peer settings, excluding endpoint credentials."""
    if not config:
        return None
    protocol = {
        "model": config.get("model"),
        "temperature": config.get("temperature", 0.7),
        "max_tokens": config.get("max_tokens", 32768),
        "thinking_mode": config.get("thinking_mode", "default"),
        "reasoning_effort": config.get("reasoning_effort"),
        "chat_template_enable_thinking": config.get(
            "chat_template_enable_thinking"),
        "context_window_tokens": config.get(
            "context_window_tokens", 1000000),
        "todo_max_ttl_s": config.get("todo_max_ttl_s", 3600.0),
        "heartbeat_interval_s": config.get("heartbeat_interval_s", 3.0),
        "driver_role_policy": (
            "ordered-cautious-efficient-cooperative-comfortable-v1"),
    }
    personal_config, judge_config = _peer_in_cabin_runtime_configs(config)
    protocol["in_cabin_agents"] = {
        "scope": "all_fixed_peer_llm_vehicles",
        "personal_agent": {
            key: personal_config.get(key) for key in (
                "model", "temperature", "max_tokens", "thinking_mode",
                "reasoning_effort", "chat_template_enable_thinking",
                "context_window_tokens", "trigger_mode", "seed")
        },
        "passenger_judge": {
            key: judge_config.get(key) for key in (
                "model", "temperature", "max_tokens", "thinking_mode",
                "reasoning_effort", "chat_template_enable_thinking",
                "context_window_tokens", "check_offsets_s", "max_checks",
                "acceptance_timeout_s")
        },
    }
    return protocol


def _peer_in_cabin_runtime_configs(config: dict) -> tuple[dict, dict]:
    """Build the PA/Judge settings used by every fixed background LLM.

    The SUMO focal is deliberately excluded later by ``build_callbacks``;
    only vehicles whose driving agent remains LLM receive these roles.
    """
    role_fields = {
        "api_base", "api_key", "model", "context_window_tokens",
        "max_tokens", "thinking_mode", "reasoning_effort",
        "chat_template_enable_thinking",
    }
    role = {
        key: copy.deepcopy(value)
        for key, value in config.items()
        if key in role_fields and value is not None
    }
    role.setdefault("context_window_tokens", 1000000)
    role.setdefault("max_tokens", 32768)
    role.setdefault("thinking_mode", "enabled")
    role.setdefault("reasoning_effort", "xhigh")
    role.setdefault("chat_template_enable_thinking", True)
    personal = {
        **role,
        "enabled": True,
        "temperature": 0.7,
        "trigger_mode": "event_random",
        "seed": 0,
    }
    judge = {
        **role,
        "enabled": True,
        "temperature": 0.0,
        "check_offsets_s": [0.1, 1.0, 3.0],
        "max_checks": 3,
        "acceptance_timeout_s": 3.0,
    }
    return personal, judge


def _actor_configuration(experiment_id: str, config: dict | None) -> str:
    if experiment_id != "MultiLLM" or not config:
        return "all_sumo"
    payload = json.dumps(
        _public_peer_protocol(config), ensure_ascii=False, sort_keys=True,
        separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"focal_sumo_fixed_peers:{digest}"


@lru_cache(maxsize=1)
def calibration_runtime_fingerprint() -> str:
    """Hash code that can change calibration physics or policy outcomes."""
    root = Path(__file__).resolve().parents[2]
    files = sorted((root / "simulation").rglob("*.py"))
    files.extend([
        Path(__file__).resolve(),
        root / "evaluation" / "agent_client.py",
        root / "evaluation" / "multi_agent_runner.py",
        root / "evaluation" / "personal_agent.py",
        root / "evaluation" / "passenger_orchestration.py",
    ])
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    temporary.replace(path)


def scenario_physical_fingerprint(scenario: dict) -> str:
    """Hash everything that can affect a calibration run except its limit."""
    payload = copy.deepcopy(scenario)
    for key in (
            "total_time_s", "time_window_calibration",
            "stop_when_all_vehicles_terminal", "enable_driving_evaluation",
            "physics_only_mode"):
        payload.pop(key, None)
    payload.pop("terminal_vehicle_ids", None)
    network_id = str(payload.get("road_network_id", ""))
    lane_level_path = (
        Path(__file__).resolve().parents[2]
        / "simulation" / "road_networks"
        / f"{network_id}_lane_level.json")
    payload["_lane_level_map_sha256"] = (
        hashlib.sha256(lane_level_path.read_bytes()).hexdigest()
        if lane_level_path.exists() else None)
    payload["_calibration_runtime_sha256"] = (
        calibration_runtime_fingerprint())
    return hashlib.sha256(json.dumps(
        payload, ensure_ascii=False, sort_keys=True,
        separators=(",", ":")).encode("utf-8")).hexdigest()


def load_calibration_registry(path: Path = DEFAULT_REGISTRY_PATH) -> dict:
    path = Path(path)
    if not path.exists():
        return {"schema": CALIBRATION_SCHEMA, "entries": {}}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != CALIBRATION_SCHEMA:
        raise ValueError(
            f"unsupported time-window calibration schema: "
            f"{payload.get('schema')!r}")
    if not isinstance(payload.get("entries"), dict):
        raise ValueError("time-window calibration entries must be an object")
    return payload


def apply_calibration_entry(scenario: dict, entry: dict) -> bool:
    """Apply one registry entry only when its physical fingerprint matches."""
    if entry.get("physical_fingerprint") != scenario_physical_fingerprint(
            scenario):
        return False
    scenario["total_time_s"] = float(entry["time_limit_s"])
    scenario["time_window_calibration"] = {
        "schema": CALIBRATION_SCHEMA,
        "method": entry.get(
            "method", "sumo_reference_max_arrival_plus_margin"),
        "max_successful_completion_s": entry.get(
            "max_successful_completion_s"),
        "last_scheduled_event_s": entry.get("last_scheduled_event_s", 0.0),
        "time_limit_s": float(entry["time_limit_s"]),
        "case_count": int(entry.get("case_count", 0)),
        "successful_case_count": int(entry.get(
            "successful_case_count", 0)),
        "reference_policy": entry.get("reference_policy"),
        "actor_configurations": list(entry.get(
            "actor_configurations", [])),
        "registry_generated_at": entry.get("generated_at"),
    }
    return True


def _last_scheduled_event_s(scenario: dict) -> float:
    values = [0.0]
    for pedestrian in scenario.get("pedestrians", []):
        values.append(60.0 * float(pedestrian.get("start_time", 0.0)))
    for keyframe in scenario.get("weather_keyframes", []):
        values.append(60.0 * float(keyframe.get("t", 0.0)))
    for keyframe in scenario.get("daynight_keyframes", []):
        values.append(60.0 * float(keyframe.get("t", 0.0)))
    return max(values)


def _environment_schedule_basis_s(scenario: dict) -> float | None:
    """Preserve the authored horizon used to place environment keyframes."""
    metadata = scenario.get("experiment_scene", {})
    values = [
        float(profile["schedule_basis_s"])
        for profile in (
            metadata.get("weather_profile", {}),
            metadata.get("daynight_profile", {}),
        )
        if profile.get("schedule_basis_s") is not None
    ]
    if not values:
        return None
    if max(values) - min(values) > 1e-6:
        raise ValueError(
            f"{scenario.get('scenario_id')} uses inconsistent environment "
            "schedule bases")
    return round(values[0], 6)


def _ceil_five(value: float) -> float:
    return float(5 * math.ceil(float(value) / 5.0))


def _time_limit_from_npc_arrival(completion_time_s: float) -> float:
    """Return the fixed deadline: NPC arrival plus ten simulation seconds."""
    return round(float(completion_time_s) + NPC_ARRIVAL_MARGIN_S, 6)


def _scenario_vehicle_ids(scenario: dict) -> set[str]:
    """Return every vehicle whose trip defines whole-scene completion."""
    return {
        str(vehicle["vehicle_id"])
        for vehicle in scenario.get("vehicles", [])
    }


def _persistent_obstacle_vehicle_ids(scenario: dict) -> set[str]:
    """Identify declared vehicles that intentionally do not finish a trip.

    SUMO NPCs cannot receive runtime speed commands. A persistent obstacle is
    therefore an explicit initial crash state, not a scripted stop.
    """
    return {
        str(vehicle["vehicle_id"])
        for vehicle in scenario.get("vehicles", [])
        if bool((vehicle.get("initial_physical_state") or {}).get("crashed"))
    }


def _calibration_cases(
    catalog_dir: Path, *, quick: bool,
    scene_ids: Iterable[str] | None = None,
    multi_peer_runtime_config: dict | None = None,
) -> list[dict]:
    cases = []
    selected_scene_ids = set(scene_ids or ())
    for entry, scenario_path, _ in iter_catalog_scenes(catalog_dir):
        if (selected_scene_ids
                and entry["scene_id"] not in selected_scene_ids):
            continue
        scenario = json.loads(scenario_path.read_text(encoding="utf-8"))
        experiment_id = str(entry["experiment_id"])
        peer_config = (
            copy.deepcopy(multi_peer_runtime_config)
            if experiment_id == "MultiLLM" else None)
        cases.append({
            "catalog_dir": str(Path(catalog_dir).resolve()),
            "scenario_path": str(scenario_path.resolve()),
            "experiment_id": experiment_id,
            "scene_id": entry["scene_id"],
            "actor_configuration": _actor_configuration(
                experiment_id, peer_config),
            "reference_policy": (
                "focal_sumo_fixed_peers"
                if peer_config else "all_sumo"),
            "fixed_peer_protocol": _public_peer_protocol(peer_config),
            "multi_peer_runtime_config": peer_config,
            "authored_time_limit_s": float(
                scenario.get("total_time_s", 0.0)),
            "last_scheduled_event_s": _last_scheduled_event_s(scenario),
        })
    return cases


def _configure_case_actors(
    scenario: dict, case: dict,
) -> tuple[set[str], set[str], dict[str, str]]:
    """Apply the selected reference policy and return fixed-peer prompts."""
    vehicle_ids = _scenario_vehicle_ids(scenario)
    original_llm_ids = [
        str(vehicle["vehicle_id"])
        for vehicle in scenario.get("vehicles", [])
        if vehicle.get("agent_config", {}).get("type") == "llm"
    ]
    peer_ids = set()
    if case.get("reference_policy") == "focal_sumo_fixed_peers":
        scene = scenario.get("experiment_scene", {})
        focal_id = str(scene.get("focal_vehicle_id", ""))
        peer_ids = set(map(str, scene.get("fixed_peer_vehicle_ids", [])))
        if not focal_id or not peer_ids:
            raise ValueError(
                f"{case['scene_id']}: MultiLLM calibration requires focal "
                "and fixed peer IDs")
        if focal_id in peer_ids:
            raise ValueError(
                f"{case['scene_id']}: focal vehicle cannot be a fixed peer")

    for vehicle in scenario.get("vehicles", []):
        vehicle["is_evaluated"] = True
        vehicle_id = str(vehicle["vehicle_id"])
        vehicle["agent_config"] = {
            "type": "llm" if vehicle_id in peer_ids else "sumo"}

    for pedestrian in scenario.get("pedestrians", []):
        pedestrian["agent_config"] = {"type": "sumo"}
        pedestrian["is_evaluated"] = False
    prompts = {
        vehicle_id: _DRIVER_ROLE_PROMPTS[
            index % len(_DRIVER_ROLE_PROMPTS)]
        for index, vehicle_id in enumerate(original_llm_ids)
        if vehicle_id in peer_ids
    }
    if peer_ids != set(prompts):
        raise ValueError(
            f"{case['scene_id']}: fixed peers are not declared LLM vehicles")
    sumo_vehicle_ids = {
        str(vehicle["vehicle_id"])
        for vehicle in scenario.get("vehicles", [])
        if vehicle.get("agent_config", {}).get("type") == "sumo"
    }
    return vehicle_ids, sumo_vehicle_ids, prompts


def _terminal_time_by_vehicle(
    required_vehicle_ids: set[str], vehicle_results_by_id: dict,
    vehicle_states_by_id: dict, collision_events: list,
) -> tuple[dict[str, str], dict[str, float]]:
    """Resolve arrival/crash terminal kinds and their physical timestamps."""
    kinds: dict[str, str] = {}
    times: dict[str, float] = {}
    first_collision_s: dict[str, float] = {}
    for event in collision_events:
        for entity_id in (event.entity_a, event.entity_b):
            if entity_id not in required_vehicle_ids:
                continue
            event_time = float(event.time_s)
            first_collision_s[entity_id] = min(
                event_time, first_collision_s.get(entity_id, event_time))
    for vehicle_id in sorted(required_vehicle_ids):
        result = vehicle_results_by_id.get(vehicle_id)
        state = vehicle_states_by_id.get(vehicle_id)
        if result is not None and bool(result.arrived):
            kinds[vehicle_id] = "arrived"
            if result.arrival_time_s is not None:
                times[vehicle_id] = float(result.arrival_time_s)
        elif state is not None and bool(state.is_crashed):
            kinds[vehicle_id] = "crashed"
            if vehicle_id in first_collision_s:
                times[vehicle_id] = first_collision_s[vehicle_id]
        elif state is not None and bool(state.route_failed):
            kinds[vehicle_id] = "route_failed"
        else:
            kinds[vehicle_id] = "unfinished"
    return kinds, times


@lru_cache(maxsize=1)
def _worker_lane_runtime(network_id: str):
    """Reuse one immutable lane topology inside each calibration worker."""
    from simulation.lane_level_runtime import LaneGeometryRuntime

    path = (
        Path(__file__).resolve().parents[2]
        / "simulation" / "road_networks"
        / f"{network_id}_lane_level.json")
    return LaneGeometryRuntime.load(str(path))


def _run_case(case: dict, max_duration_s: float) -> dict:
    # Imports stay inside the worker so multiprocessing never serializes the
    # simulator's runtime objects.
    from evaluation.multi_agent_runner import (
        apply_resolved_agent_authorities, build_callbacks,
        resolve_agent_specs,
    )
    from simulation.multi_sim_engine import MultiScenario, MultiSimEngine

    scenario_path = Path(case["scenario_path"])
    scenario = json.loads(scenario_path.read_text(encoding="utf-8"))
    vehicle_ids, sumo_vehicle_ids, peer_prompts = _configure_case_actors(
        scenario, case)
    persistent_obstacle_ids = (
        _persistent_obstacle_vehicle_ids(scenario) & sumo_vehicle_ids)
    required_terminal_ids = sumo_vehicle_ids - persistent_obstacle_ids
    # The deadline covers the whole shared road world, not only the focal
    # model vehicle.  This lets later system-level evaluation observe every
    # background vehicle through its own terminal boundary.
    scenario["total_time_s"] = float(max_duration_s)
    scenario["physics_step_s"] = 0.1
    scenario["stop_when_all_vehicles_terminal"] = True
    scenario["terminal_vehicle_ids"] = sorted(required_terminal_ids)
    scenario["enable_driving_evaluation"] = False
    scenario["physics_only_mode"] = not bool(peer_prompts)

    outcome = {
        key: case[key] for key in (
            "experiment_id", "scene_id", "actor_configuration",
            "reference_policy", "fixed_peer_protocol")
    }
    outcome["run_duration_s"] = float(max_duration_s)
    try:
        parsed_scenario = MultiScenario.from_dict(scenario)
        callbacks = {}
        if peer_prompts:
            runtime_config = dict(case["multi_peer_runtime_config"])
            overrides = {
                peer_id: {
                    "type": "llm",
                    "model": runtime_config["model"],
                    "driver_prompt": prompt,
                }
                for peer_id, prompt in peer_prompts.items()
            }
            specs = resolve_agent_specs(
                parsed_scenario, overrides,
                llm_runtime_config=runtime_config)
            apply_resolved_agent_authorities(parsed_scenario, specs)
            personal_config, judge_config = (
                _peer_in_cabin_runtime_configs(runtime_config))
            callbacks, _ = build_callbacks(
                specs,
                personal_agent_runtime_config=personal_config,
                passenger_judge_runtime_config=judge_config,
            )
        engine = MultiSimEngine(
            parsed_scenario,
            lane_geometry_runtime=_worker_lane_runtime(
                str(scenario["road_network_id"])),
        )
        result = engine.run(callbacks)
        collision_events = list(engine.traffic_mgr.collision_log)
        callback_errors = list(engine.agent_callback_errors)
        vehicle_results_by_id = {
            vehicle_id: result.vehicle_results[vehicle_id]
            for vehicle_id in sorted(vehicle_ids)
            if vehicle_id in result.vehicle_results
        }
        vehicle_results = list(vehicle_results_by_id.values())
        vehicle_states_by_id = {
            vehicle_id: engine.traffic_mgr.get_state(vehicle_id)
            for vehicle_id in sorted(required_terminal_ids)
        }
        terminal_kinds, terminal_times = _terminal_time_by_vehicle(
            required_terminal_ids, vehicle_results_by_id,
            vehicle_states_by_id, collision_events)
        success = (
            bool(required_terminal_ids)
            and set(terminal_kinds) == required_terminal_ids
            and all(kind in {"arrived", "crashed"}
                    for kind in terminal_kinds.values())
            and set(terminal_times) == required_terminal_ids)
        completion_s = (
            max(terminal_times.values()) if success else None)
        completion_kind = "all_sumo_trip_vehicles_terminal"
        outcome.update({
            "success": bool(success),
            "completion_time_s": (
                round(completion_s, 6)
                if completion_s is not None else None),
            "completion_kind": completion_kind,
            "vehicle_arrived": {
                item.vehicle_id: bool(item.arrived)
                for item in vehicle_results},
            "required_terminal_vehicle_ids": sorted(required_terminal_ids),
            "required_vehicle_terminal_kind": terminal_kinds,
            "required_vehicle_terminal_time_s": {
                vehicle_id: round(value, 6)
                for vehicle_id, value in terminal_times.items()},
            "persistent_obstacle_vehicle_ids": sorted(
                persistent_obstacle_ids),
            "collision_count": len(collision_events),
            "agent_callback_error_count": len(callback_errors),
            "simulated_until_s": round(float(engine._sim_time), 6),
            "error": (
                json.dumps(callback_errors, ensure_ascii=False)[:4000]
                if callback_errors else ""),
        })
    except Exception as exc:  # keep the full matrix auditable
        outcome.update({
            "success": False,
            "completion_time_s": None,
            "completion_kind": "error",
            "vehicle_arrived": {},
            "required_terminal_vehicle_ids": sorted(required_terminal_ids),
            "required_vehicle_terminal_kind": {},
            "required_vehicle_terminal_time_s": {},
            "persistent_obstacle_vehicle_ids": sorted(
                persistent_obstacle_ids),
            "collision_count": None,
            "agent_callback_error_count": None,
            "simulated_until_s": None,
            "error": f"{type(exc).__name__}: {exc}",
        })
    return outcome


def _case_key(value: dict) -> tuple:
    return tuple(value[key] for key in (
        "experiment_id", "scene_id", "actor_configuration"))


def _initial_run_duration(case: dict, max_duration_s: float) -> float:
    """Use a generous scene-local window before the 600 s fallback."""
    return min(float(max_duration_s), max(
        60.0,
        2.0 * float(case.get("authored_time_limit_s", 0.0)),
        float(case.get("last_scheduled_event_s", 0.0)) + 30.0,
    ))


def _attempt_summary(outcome: dict) -> dict:
    return {
        "run_duration_s": outcome.get("run_duration_s"),
        "success": bool(outcome.get("success")),
        "completion_time_s": outcome.get("completion_time_s"),
        "collision_count": outcome.get("collision_count"),
        "agent_callback_error_count": outcome.get(
            "agent_callback_error_count"),
        "simulated_until_s": outcome.get("simulated_until_s"),
        "error": outcome.get("error", ""),
    }


def _execute_cases(
    cases: list[dict], durations: dict[tuple, float], workers: int,
    *, progress_label: str, checkpoint_dir: Path | None = None,
) -> list[dict]:
    def save_checkpoint(outcome: dict) -> None:
        if checkpoint_dir is None:
            return
        key = json.dumps(
            _case_key(outcome), ensure_ascii=False,
            separators=(",", ":"))
        suffix = hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]
        filename = f"{outcome['scene_id']}--{suffix}.json"
        _json_write(Path(checkpoint_dir) / filename, outcome)

    outcomes = []
    if max(1, int(workers)) == 1:
        for index, case in enumerate(cases, 1):
            outcome = _run_case(case, durations[_case_key(case)])
            outcomes.append(outcome)
            save_checkpoint(outcome)
            if index % 25 == 0 or index == len(cases):
                print(
                    f"{progress_label} {index}/{len(cases)}", flush=True)
        return outcomes

    with ProcessPoolExecutor(max_workers=max(1, int(workers))) as pool:
        futures = {
            pool.submit(
                _run_case, case, durations[_case_key(case)]): case
            for case in cases}
        for index, future in enumerate(as_completed(futures), 1):
            outcome = future.result()
            outcomes.append(outcome)
            save_checkpoint(outcome)
            if index % 25 == 0 or index == len(cases):
                print(
                    f"{progress_label} {index}/{len(cases)}", flush=True)
    return outcomes


def _load_case_checkpoints(
    cases: list[dict], durations: dict[tuple, float], checkpoint_dir: Path,
) -> dict[tuple, dict]:
    """Recover compatible per-case outcomes written by an interrupted run."""
    expected = {_case_key(case) for case in cases}
    recovered: dict[tuple, dict] = {}
    if not Path(checkpoint_dir).is_dir():
        return recovered
    for path in sorted(Path(checkpoint_dir).glob("*.json")):
        try:
            outcome = json.loads(path.read_text(encoding="utf-8"))
            key = _case_key(outcome)
            duration = float(outcome["run_duration_s"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        if key not in expected:
            continue
        if abs(duration - float(durations[key])) > 1e-6:
            continue
        if not all(field in outcome for field in (
                "required_terminal_vehicle_ids",
                "required_vehicle_terminal_kind",
                "required_vehicle_terminal_time_s")):
            continue
        recovered[key] = outcome
    return recovered


def _should_retry(outcome: dict, max_duration_s: float) -> bool:
    """Retry any incomplete SUMO terminal set when more horizon is available."""
    return (
        not bool(outcome.get("success"))
        and (
            bool(outcome.get("error"))
            or float(outcome["run_duration_s"]) < float(max_duration_s)
        )
    )


def calibrate_time_windows(
    catalog_dir: Path, *, registry_path: Path = DEFAULT_REGISTRY_PATH,
    report_path: Path | None = None,
    max_duration_s: float = 600.0, workers: int = 1,
    quick: bool = False, scene_ids: Iterable[str] | None = None,
    multi_peer_runtime_config: dict | None = None,
) -> dict:
    """Execute paired references, freeze limits, and update current scenes."""
    catalog_dir = Path(catalog_dir)
    if max_duration_s <= 0:
        raise ValueError("max_duration_s must be positive")
    cases = _calibration_cases(
        catalog_dir, quick=quick,
        scene_ids=scene_ids,
        multi_peer_runtime_config=multi_peer_runtime_config)
    if not cases:
        raise ValueError("no calibration cases matched the selected scenes")
    initial_durations = {
        _case_key(case): _initial_run_duration(case, max_duration_s)
        for case in cases}
    checkpoint_root = (
        Path(report_path).parent / "case-outcomes"
        if report_path is not None
        else Path(registry_path).parent / "case-outcomes")
    initial_by_key = _load_case_checkpoints(
        cases, initial_durations, checkpoint_root / "initial")
    pending_initial_cases = [
        case for case in cases if _case_key(case) not in initial_by_key]
    if initial_by_key:
        print(
            f"calibration checkpoint reuse "
            f"{len(initial_by_key)}/{len(cases)}", flush=True)
    new_initial_outcomes = _execute_cases(
        pending_initial_cases, initial_durations, workers,
        progress_label="calibration",
        checkpoint_dir=checkpoint_root / "initial") \
        if pending_initial_cases else []
    initial_by_key.update({
        _case_key(outcome): outcome for outcome in new_initial_outcomes})
    initial_outcomes = [
        initial_by_key[_case_key(case)] for case in cases]
    retry_cases = [
        case for case in cases
        if _should_retry(
            initial_by_key[_case_key(case)], max_duration_s)
    ]
    retry_durations = {
        _case_key(case): float(max_duration_s) for case in retry_cases}
    retry_by_key = _load_case_checkpoints(
        retry_cases, retry_durations, checkpoint_root / "retry")
    pending_retry_cases = [
        case for case in retry_cases if _case_key(case) not in retry_by_key]
    if retry_by_key:
        print(
            f"calibration retry checkpoint reuse "
            f"{len(retry_by_key)}/{len(retry_cases)}", flush=True)
    new_retry_outcomes = _execute_cases(
        pending_retry_cases, retry_durations, workers,
        progress_label="calibration retry",
        checkpoint_dir=checkpoint_root / "retry") \
        if pending_retry_cases else []
    retry_by_key.update({
        _case_key(outcome): outcome for outcome in new_retry_outcomes})
    retry_outcomes = [
        retry_by_key[_case_key(case)] for case in retry_cases]
    outcomes = []
    for case in cases:
        key = _case_key(case)
        initial = initial_by_key[key]
        final = retry_by_key.get(key, initial)
        final["attempts"] = [_attempt_summary(initial)]
        if key in retry_by_key:
            final["attempts"].append(_attempt_summary(retry_by_key[key]))
        outcomes.append(final)

    now = datetime.now(timezone.utc).isoformat()
    by_scene: dict[str, list[dict]] = {}
    for outcome in outcomes:
        by_scene.setdefault(outcome["scene_id"], []).append(outcome)

    # A full recalibration replaces the registry and therefore must not load
    # an obsolete schema. Partial runs merge only with the current schema.
    previous_registry = (
        load_calibration_registry(registry_path)
        if scene_ids else {"entries": {}})
    registry = {
        "schema": CALIBRATION_SCHEMA,
        "protocol": CALIBRATION_PROTOCOL,
        "generated_at": now,
        "method": "sumo_reference_max_trip_vehicle_terminal_plus_margin",
        "formula": (
            "max(Tmax(SUMO trip vehicle arrival or collision), last "
            "scheduled environment/event) + 10 simulation seconds"),
        "execution_policy": (
            "min(max_duration, max(60, 2*authored_limit, "
            "last_event+30)); retry non-collision timeout at max_duration"),
        "multi_llm_reference_policy": (
            "focal_sumo_fixed_peers"
            if multi_peer_runtime_config else "all_sumo"),
        "fixed_peer_protocol": _public_peer_protocol(
            multi_peer_runtime_config),
        "max_duration_s": float(max_duration_s),
        "quick": bool(quick),
        "partial_update": bool(scene_ids),
        "entries": dict(previous_registry.get("entries", {}))
        if scene_ids else {},
    }
    scenario_paths = {
        entry["scene_id"]: path
        for entry, path, _ in iter_catalog_scenes(catalog_dir)}
    for scene_id, scene_outcomes in sorted(by_scene.items()):
        scenario_path = scenario_paths[scene_id]
        scenario = json.loads(scenario_path.read_text(encoding="utf-8"))
        successful_times = [
            float(item["completion_time_s"])
            for item in scene_outcomes
            if item.get("success")
            and item.get("completion_time_s") is not None]
        last_event_s = _last_scheduled_event_s(scenario)
        max_success_s = max(successful_times) if successful_times else None
        if max_success_s is None:
            # A total failure is kept visible in the registry and cannot
            # silently shrink a previously authored observation window.
            time_limit_s = _ceil_five(max(
                float(scenario.get("total_time_s", 0.0)),
                last_event_s + NPC_ARRIVAL_MARGIN_S))
        else:
            # The reference trip may finish before a deliberately scheduled
            # weather/day-night transition. Keep the full authored event
            # observable instead of ending the task as soon as traffic exits.
            time_limit_s = _time_limit_from_npc_arrival(max(
                max_success_s, last_event_s))
        entry = {
            "scene_id": scene_id,
            "protocol": CALIBRATION_PROTOCOL,
            "physical_fingerprint": scenario_physical_fingerprint(scenario),
            "method": "sumo_reference_max_trip_vehicle_terminal_plus_margin",
            "reference_policy": scene_outcomes[0].get(
                "reference_policy", "all_sumo"),
            "fixed_peer_protocol": scene_outcomes[0].get(
                "fixed_peer_protocol"),
            "generated_at": now,
            "max_successful_completion_s": (
                round(max_success_s, 6) if max_success_s is not None else None),
            "last_scheduled_event_s": round(last_event_s, 6),
            "environment_schedule_basis_s": (
                _environment_schedule_basis_s(scenario)),
            "time_limit_s": time_limit_s,
            "case_count": len(scene_outcomes),
            "successful_case_count": len(successful_times),
            "failed_case_count": len(scene_outcomes) - len(successful_times),
            "actor_configurations": sorted({
                item["actor_configuration"] for item in scene_outcomes}),
            "outcomes": sorted(
                scene_outcomes,
                key=lambda item: item["actor_configuration"]),
        }
        registry["entries"][scene_id] = entry
        if not apply_calibration_entry(scenario, entry):
            raise RuntimeError(
                f"fresh calibration fingerprint mismatch for {scene_id}")
        _json_write(scenario_path, scenario)

    _json_write(Path(registry_path), registry)
    report = {
        "schema": CALIBRATION_SCHEMA,
        "generated_at": now,
        "catalog": str(catalog_dir),
        "registry": str(Path(registry_path)),
        "scene_count": len(by_scene),
        "registry_scene_count": len(registry["entries"]),
        "case_count": len(outcomes),
        "attempt_count": len(initial_outcomes) + len(retry_outcomes),
        "retry_case_count": len(retry_outcomes),
        "successful_case_count": sum(
            bool(item.get("success")) for item in outcomes),
        "failed_case_count": sum(
            not bool(item.get("success")) for item in outcomes),
        "collision_case_count": sum(
            bool(item.get("collision_count")) for item in outcomes),
        "scenes_without_success": sorted(
            scene_id for scene_id, entry in registry["entries"].items()
            if not entry["successful_case_count"]),
        "time_limit_min_s": min(
            entry["time_limit_s"] for entry in registry["entries"].values()),
        "time_limit_max_s": max(
            entry["time_limit_s"] for entry in registry["entries"].values()),
    }
    if report_path is not None:
        _json_write(Path(report_path), report)
    return report


def _entry_covers_cases(
    entry: dict | None, cases: list[dict], scenario: dict,
) -> bool:
    if not entry or entry.get("protocol") != CALIBRATION_PROTOCOL:
        return False
    if entry.get("physical_fingerprint") != scenario_physical_fingerprint(
            scenario):
        return False
    expected = {_case_key(case) for case in cases}
    outcomes = entry.get("outcomes", [])
    actual = {_case_key(outcome) for outcome in outcomes}
    return (
        expected == actual
        and int(entry.get("case_count", -1)) == len(expected)
        and all(
            "vehicle_arrived" in outcome
            and "required_terminal_vehicle_ids" in outcome
            and "required_vehicle_terminal_kind" in outcome
            and "required_vehicle_terminal_time_s" in outcome
            and "persistent_obstacle_vehicle_ids" in outcome
            and "collision_count" in outcome
            and isinstance(outcome.get("attempts"), list)
            for outcome in outcomes)
    )


def _registry_report(
    registry: dict, *, catalog_dir: Path, selected_scene_ids: set[str],
) -> dict:
    entries = {
        scene_id: entry for scene_id, entry in registry["entries"].items()
        if scene_id in selected_scene_ids}
    outcomes = [
        outcome for entry in entries.values()
        for outcome in entry.get("outcomes", [])]
    limits = [float(entry["time_limit_s"]) for entry in entries.values()]
    return {
        "schema": CALIBRATION_SCHEMA,
        "protocol": CALIBRATION_PROTOCOL,
        "generated_at": registry.get("generated_at"),
        "catalog": str(catalog_dir),
        "scene_count": len(entries),
        "expected_scene_count": len(selected_scene_ids),
        "complete": len(entries) == len(selected_scene_ids),
        "case_count": len(outcomes),
        "attempt_count": sum(
            len(outcome.get("attempts", ())) for outcome in outcomes),
        "retry_case_count": sum(
            len(outcome.get("attempts", ())) > 1 for outcome in outcomes),
        "successful_case_count": sum(
            bool(outcome.get("success")) for outcome in outcomes),
        "failed_case_count": sum(
            not bool(outcome.get("success")) for outcome in outcomes),
        "collision_case_count": sum(
            bool(outcome.get("collision_count")) for outcome in outcomes),
        "scenes_without_success": sorted(
            scene_id for scene_id, entry in entries.items()
            if not entry.get("successful_case_count")),
        "time_limit_min_s": min(limits) if limits else None,
        "time_limit_max_s": max(limits) if limits else None,
    }


def calibrate_time_windows_checkpointed(
    catalog_dir: Path, *, registry_path: Path = DEFAULT_REGISTRY_PATH,
    report_path: Path | None = None,
    max_duration_s: float = 600.0, workers: int = 1,
    scene_ids: Iterable[str] | None = None, resume: bool = True,
    multi_peer_runtime_config: dict | None = None,
) -> dict:
    """Calibrate and atomically checkpoint every completed physical scene."""
    catalog_dir = Path(catalog_dir)
    selected = set(scene_ids or ())
    scenes = [
        (entry, path, json.loads(path.read_text(encoding="utf-8")))
        for entry, path, _ in iter_catalog_scenes(catalog_dir)
        if not selected or entry["scene_id"] in selected]
    if not scenes:
        raise ValueError("no calibration scenes matched")
    selected_ids = {entry["scene_id"] for entry, _, _ in scenes}

    skipped = completed = 0
    for index, (catalog_entry, _, scenario) in enumerate(scenes, 1):
        scene_id = catalog_entry["scene_id"]
        cases = _calibration_cases(
            catalog_dir, quick=False,
            scene_ids=[scene_id],
            multi_peer_runtime_config=multi_peer_runtime_config)
        current = load_calibration_registry(registry_path)
        existing = current.get("entries", {}).get(scene_id)
        if resume and _entry_covers_cases(existing, cases, scenario):
            skipped += 1
            print(
                f"checkpoint {index}/{len(scenes)} skip {scene_id}",
                flush=True)
            continue
        print(
            f"checkpoint {index}/{len(scenes)} run {scene_id} "
            f"({len(cases)} cases)", flush=True)
        calibrate_time_windows(
            catalog_dir, registry_path=registry_path,
            max_duration_s=max_duration_s, workers=workers,
            quick=False, scene_ids=[scene_id],
            multi_peer_runtime_config=multi_peer_runtime_config)
        completed += 1

    registry = load_calibration_registry(registry_path)
    registry["schema"] = CALIBRATION_SCHEMA
    registry["protocol"] = CALIBRATION_PROTOCOL
    registry["max_duration_s"] = float(max_duration_s)
    all_complete = True
    for _, path, _ in scenes:
        scenario = json.loads(path.read_text(encoding="utf-8"))
        scene_id = scenario["scenario_id"]
        cases = _calibration_cases(
            catalog_dir, quick=False,
            scene_ids=[scene_id],
            multi_peer_runtime_config=multi_peer_runtime_config)
        if not _entry_covers_cases(
                registry.get("entries", {}).get(scene_id), cases, scenario):
            all_complete = False
            break
    registry["quick"] = False
    registry["partial_update"] = not all_complete
    registry["checkpointed"] = True
    _json_write(Path(registry_path), registry)
    report = _registry_report(
        registry, catalog_dir=catalog_dir,
        selected_scene_ids=selected_ids)
    report["scenes_run_this_invocation"] = completed
    report["scenes_skipped_by_resume"] = skipped
    if report_path is not None:
        _json_write(Path(report_path), report)
    return report
