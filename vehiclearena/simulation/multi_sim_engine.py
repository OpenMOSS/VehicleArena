"""
MultiSimEngine — Multi-vehicle multi-agent simulation engine.

Extends the single-vehicle SimulationEngine to support multiple vehicles,
each driven by an independent Agent with its own VehicleWorld instance.
The TrafficCoordinator coordinates all vehicles on a shared RoadNetwork.

Each vehicle/agent:
  - Has its own VehicleWorld (module states, EventBus, ConstraintEngine)
  - Can only perceive nearby vehicles (position + speed), not their intent
  - Makes independent decisions each tick
  - Is evaluated independently (FieldAcc per vehicle)

Usage:
    from simulation.multi_sim_engine import MultiSimEngine, MultiScenario

    scenario = MultiScenario.from_dict({...})
    engine = MultiSimEngine(scenario)

    # agent_callbacks: {vehicle_id: callback_fn}
    result = engine.run({
        "ego": my_ego_agent,
        "v2":  my_other_agent,
    })
"""

from __future__ import annotations

import copy
import json
import logging
import math
import random
from collections import defaultdict, deque
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Callable, Dict, List, Optional, Tuple

from vehiclearena import VehicleWorld
from utils import execute
from simulation.memory import SessionHistory
from simulation.model_concurrency import ModelCallScheduler, on_simulation_owner
from simulation.scenario import WeatherKeyframe, DayNightKeyframe
from simulation.road_network import RoadNetwork, TrafficConfig
from simulation.traffic_manager import TrafficCoordinator, TriggerEvent
from simulation.vehicle_state import VehicleState
from simulation.pedestrian_state import PedestrianState
from simulation.perception_model import (
    resolve_perception_profile,
    resolve_radar_spec,
)
from module.lidar import resolve_lidar_spec
from simulation.world_state import (
    AgentWorldState, FrozenStateView, WorldState)
from simulation.wake_events import (
    WakeEvent, WakeEventBroker, WakePriority, group_wake_events)
from simulation.ground_truth_rules import (
    derive_ground_truth, take_world_snapshot, WorldSnapshot,
    AcceptableAction, derive_negative_checks,
)
from evaluation.cabin_evaluator import (
    CabinYamlEvaluator,
    execute_yaml_action,
)
from evaluation.driving_evaluator import (
    DrivingEvaluationConfig, DrivingEvaluator,
)
from evaluation.driving_process_score import DrivingProcessScoreTracker
from evaluation.scoring_awareness import ScoringTrafficView
from evaluation.layer_scoring import (
    cabin_layer_score_100, single_vehicle_layer_score_100,
)
from evaluation.snapshot_utils import (
    snapshot_modules, deep_diff, get_by_path, INTERNAL_MODULES,
)
from evaluation.check_utils import _FULL_INIT_PREFIX, _match_announcements

from module.weather import Weather
from module.daynight import DayNight
from capabilities import (
    VehicleCapabilitySet,
    action_module_name,
    resolve_vehicle_capabilities,
)


_NPC_WORLD_VISIBLE_RULE_MODULES = frozenset({
    "fogLight",
    "hazardLight",
    "highBeamHeadlight",
    "lowBeamHeadlight",
    "positionLight",
    "tailLight",
    "turnSignal",
    # Weather-driven closures and wipers are scored for SUMO vehicles too;
    # without the oracle applying them, NPC baselines are charged for
    # equipment no actor ever set.
    "wiper",
    "window",
    "sunroof",
})


def _redact_scenario_for_audit(value):
    """Serialize an executable scenario without credential-bearing fields."""
    if isinstance(value, dict):
        return {
            str(key): (
                "[redacted]"
                if str(key).lower() in {
                    "api_key", "authorization", "password", "token"}
                else _redact_scenario_for_audit(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_redact_scenario_for_audit(item) for item in value]
    return copy.deepcopy(value)

logger = logging.getLogger(__name__)
_WAKE_COMMAND_BUFFER = ContextVar("vehicle_wake_command_buffer", default=None)

# Caution warnings that flicker across the radar threshold do not wake the
# model again until this much simulation time has elapsed after a clear.
# Critical risk bypasses the cooldown below.
RADAR_WARNING_REENTRY_COOLDOWN_S = 1.0


def _matches_any_pattern(path: str, patterns: List[str]) -> bool:
    """Check if a dot-separated field path matches any wildcard pattern.

    Pattern syntax: ``*`` matches any single path component.
    E.g. ``wiper.wipers.front.speed_setting.*`` matches
    ``wiper.wipers.front.speed_setting.value``.
    """
    from fnmatch import fnmatch
    for pat in patterns:
        if fnmatch(path, pat):
            return True
    return False


# ── Multi-vehicle scenario ────────────────────────────────────

_AGENT_CONFIG_FIELDS = {
    "type", "model", "api_base", "api_key", "max_turns", "temperature",
    "max_tokens", "context_window_tokens", "todo_max_ttl_s",
    "heartbeat_interval_s", "thinking_mode", "reasoning_effort",
    "chat_template_enable_thinking",
    # Per-vehicle in-cabin ablation switches.  They default to enabled and
    # are intentionally independent of the global runtime PA/Judge config.
    "personal_agent_enabled", "passenger_judge_enabled",
}

_LLM_SENSOR_OR_LIFECYCLE_WAKE_EVENTS = {
    "simulation_start", "driving_task_assigned",
    "heartbeat", "scheduled_wake", "passenger_request",
    "passenger_judge_due",
    "personal_agent_due", "personal_agent_update",
    "command_result", "acoustic_cue",
    "weather_initialized", "weather_changed",
    "daynight_initialized", "daynight_changed",
    "front_collision_warning", "rear_collision_warning",
    "arrived", "route_failed", "crashed", "collision", "simulation_ended",
    "agent_callback_failed", "control_constraint_overrun",
}
_SUMO_CONFIG_FIELDS = {
    "cache_root", "step_length_s", "lateral_resolution_m",
    "collision_action", "collision_stop_time_s", "time_to_teleport_s",
    "suppress_warnings", "gui", "auto_start_xvfb", "gui_width",
    "gui_height", "gui_settings_file", "gui_schema",
    "npc_behavior",
}
_SCENARIO_FIELDS = {
    "scenario_id", "name", "road_network_id", "vehicles", "difficulty",
    "total_time_s", "tick_interval_s", "physics_step_s",
    "sumo_config",
    "traffic_config_overrides", "inits_code",
    "weather_keyframes", "daynight_keyframes", "pedestrians",
    "entities",
    "traffic_light_schedule", "congestion_events",
    "road_events", "speed_cameras", "driving_evaluation",
    "enable_driving_evaluation",
    "stop_when_all_vehicles_terminal", "terminal_vehicle_ids",
    "physics_only_mode", "max_parallel_model_calls",
    # Experiment provenance is carried with the executable scene but never
    # interpreted by the physical world.
    "experiment_scene", "time_window_calibration", "base_scene_catalog_id",
}

@dataclass
class VehicleConfig:
    """Configuration for one vehicle in a multi-vehicle scenario."""
    vehicle_id: str
    initial_node: str
    destination_node: str = ""
    destination_name: str = ""
    initial_lane: int = 0
    is_evaluated: bool = True   # whether to score this vehicle
    equipment_profile: str = "executive"
    chassis_profile: str = "sedan"
    enable_modules: List[str] = field(default_factory=list)
    disable_modules: List[str] = field(default_factory=list)
    chassis_overrides: Dict[str, float] = field(default_factory=dict)
    perception_profile: str = "human_driver_standard"
    perception_overrides: Dict[str, float] = field(default_factory=dict)
    sensor_overrides: Dict[str, Dict[str, float]] = field(
        default_factory=dict)
    agent_config: Dict = field(default_factory=dict)
    # Optional deterministic micro-scenario placement. Normal generated
    # scenarios omit it and start at route origin as before.
    initial_physical_state: Dict = field(default_factory=dict)
    # agent_config schema (all optional):
    #   type: "llm" | "sumo"
    #   model: str         (e.g. "gpt-5.4", "qwen3-235b")
    #   api_base: str      (API endpoint URL)
    #   api_key: str       (API key)
    #   max_turns: int     (max tool-call turns per tick, default 10)
    #   max_tokens: int    (max completion tokens per turn, default 2048)
    #   context_window_tokens: int (explicit model context window)
    #   todo_max_ttl_s: float (maximum finite Todo horizon)
    #   temperature: float (LLM temperature, default 0.7)
    #   personal_agent_enabled: bool (per-vehicle PA switch, default true)
    #   passenger_judge_enabled: bool (per-vehicle Judge switch, default true)

    @property
    def agent_type(self) -> str:
        return str(self.agent_config.get("type", "sumo"))


@dataclass
class PedestrianConfig:
    """Configuration for one pedestrian in a multi-agent scenario."""
    ped_id: str
    initial_node: str
    destination_node: str = ""
    speed: float = 1.4               # walking speed in m/s
    start_time: float = 0.0          # simulation time (minutes) when pedestrian appears
    is_evaluated: bool = True        # whether to score this pedestrian
    collision_radius_m: float = 0.4
    perception_profile: str = "pedestrian_standard"
    perception_overrides: Dict[str, float] = field(default_factory=dict)
    agent_config: Dict = field(default_factory=dict)
    initial_physical_state: Dict = field(default_factory=dict)
    # agent_config schema (same as vehicle): type/model/api_base/api_key,
    #   max_turns/max_tokens/context_window_tokens/todo_max_ttl_s/temperature/
    #   heartbeat_interval_s

    @property
    def agent_type(self) -> str:
        return str(self.agent_config.get("type", "sumo"))


@dataclass
class MultiScenario:
    """Authoritative multi-entity VehicleArena scenario."""
    scenario_id: str
    name: str
    road_network_id: str
    vehicles: List[VehicleConfig]
    difficulty: str = "medium"
    total_time_s: float = 3600.0       # duration in seconds
    tick_interval_s: float = 300.0     # heartbeat interval in seconds
    # Experiment/debug override. Production scenarios normally omit this and
    # therefore use the engine's authoritative 0.1 s step.
    physics_step_s: float = 0.1
    # SUMO is the only physical execution engine.  This object contains only
    # SUMO runtime settings; scenarios cannot select a second physics model.
    sumo_config: Dict[str, Any] = field(default_factory=dict)
    # Per-scenario physical semantics. Keys must be TrafficConfig fields;
    # this is intentionally separate from experiment metadata so any runtime
    # threshold change is explicit and goes through the same physics engine.
    traffic_config_overrides: Dict[str, Any] = field(default_factory=dict)
    inits_code: str = ""
    weather_keyframes: List[WeatherKeyframe] = field(default_factory=list)
    daynight_keyframes: List[DayNightKeyframe] = field(default_factory=list)
    # Pedestrian configurations
    pedestrians: List[PedestrianConfig] = field(default_factory=list)
    # Shared events (weather/congestion/road events on the network)
    _grid_events: Dict = field(default_factory=dict, repr=False)
    driving_evaluation_config: Dict = field(default_factory=dict)
    enable_driving_evaluation: bool = True
    # Deterministic exogenous interventions used by controlled experiments.
    # These alter the physical world (for example, a scripted lead-vehicle
    # brake) and are deliberately separate from any driver's policy.
    # A task is complete only after every configured vehicle has either
    # arrived or reached a physical failure state.  The switch is useful for
    # diagnostic fixtures that intentionally need to observe a later time.
    stop_when_all_vehicles_terminal: bool = True
    # Optional subset used by reference/calibration runs.  Empty preserves the
    # normal whole-scene policy; otherwise the episode can end as soon as the
    # listed physical vehicles are terminal.  Controller bookkeeping is never
    # part of this predicate.
    terminal_vehicle_ids: List[str] = field(default_factory=list)
    # SUMO time-window calibration does not need agent wake/context/cabin
    # scheduling.  This mode preserves the same 0.1 s traffic, pedestrian,
    # signal and collision world while omitting those non-physical services.
    physics_only_mode: bool = False
    # Network requests only; tools/physics/rendering remain on the owner thread.
    max_parallel_model_calls: int = 4

    @classmethod
    def from_dict(cls, d: dict) -> "MultiScenario":
        if not isinstance(d, dict):
            raise ValueError("Scenario must be an object")
        unknown_scenario_fields = sorted(set(d) - _SCENARIO_FIELDS)
        if unknown_scenario_fields:
            raise ValueError(
                f"Unknown scenario fields: {unknown_scenario_fields!r}")
        raw_vehicles = d.get("vehicles", [])
        raw_pedestrians = d.get("pedestrians", [])
        if not isinstance(raw_vehicles, list):
            raise ValueError("vehicles must be a list")
        if not isinstance(raw_pedestrians, list):
            raise ValueError("pedestrians must be a list")
        raw_vehicles = list(raw_vehicles)
        raw_pedestrians = list(raw_pedestrians)
        from simulation.runtime_entities import expand_runtime_entities
        for physical_kind, config in expand_runtime_entities(
                d.get("entities", [])):
            if physical_kind == "vehicle":
                raw_vehicles.append(config)
            else:
                raw_pedestrians.append(config)
        for vehicle in raw_vehicles:
            if not isinstance(vehicle, dict):
                raise ValueError("Each vehicles entry must be an object")
            unknown = sorted(
                set(vehicle) - set(VehicleConfig.__dataclass_fields__))
            if unknown:
                raise ValueError(
                    f"Unknown VehicleConfig fields: {unknown!r}")
            if not isinstance(vehicle.get("agent_config", {}), dict):
                raise ValueError("Vehicle agent_config must be an object")
            unknown_agent_fields = sorted(
                set(vehicle.get("agent_config", {}))
                - _AGENT_CONFIG_FIELDS)
            if unknown_agent_fields:
                raise ValueError(
                    "Unknown vehicle agent_config fields: "
                    f"{unknown_agent_fields!r}")
        for pedestrian in raw_pedestrians:
            if not isinstance(pedestrian, dict):
                raise ValueError("Each pedestrians entry must be an object")
            unknown = sorted(
                set(pedestrian) - set(PedestrianConfig.__dataclass_fields__))
            if unknown:
                raise ValueError(
                    f"Unknown PedestrianConfig fields: {unknown!r}")
            if not isinstance(pedestrian.get("agent_config", {}), dict):
                raise ValueError("Pedestrian agent_config must be an object")
            unknown_agent_fields = sorted(
                set(pedestrian.get("agent_config", {}))
                - _AGENT_CONFIG_FIELDS)
            if unknown_agent_fields:
                raise ValueError(
                    "Unknown pedestrian agent_config fields: "
                    f"{unknown_agent_fields!r}")
        vehicles = [
            VehicleConfig(
                vehicle_id=v["vehicle_id"],
                initial_node=v["initial_node"],
                destination_node=v.get("destination_node", ""),
                destination_name=v.get("destination_name", ""),
                initial_lane=v.get("initial_lane", 0),
                is_evaluated=v.get("is_evaluated", True),
                equipment_profile=v.get("equipment_profile", "executive"),
                chassis_profile=v.get("chassis_profile", "sedan"),
                enable_modules=v.get("enable_modules", []) or [],
                disable_modules=v.get("disable_modules", []) or [],
                chassis_overrides=v.get("chassis_overrides", {}) or {},
                perception_profile=v.get(
                    "perception_profile", "human_driver_standard"),
                perception_overrides=v.get(
                    "perception_overrides", {}) or {},
                sensor_overrides=v.get("sensor_overrides", {}) or {},
                agent_config=v.get("agent_config", {}),
                initial_physical_state=v.get(
                    "initial_physical_state", {}) or {},
            )
            for v in raw_vehicles
        ]
        # Fail before simulation startup when a profile, module override or
        # chassis value is invalid.
        for vehicle in vehicles:
            if vehicle.agent_type not in {"sumo", "llm"}:
                raise ValueError(
                    "Vehicle agent_config.type must be one of "
                    "'sumo' or 'llm'")
            capabilities = resolve_vehicle_capabilities(
                equipment_profile=vehicle.equipment_profile,
                chassis_profile=vehicle.chassis_profile,
                enable_modules=vehicle.enable_modules,
                disable_modules=vehicle.disable_modules,
                chassis_overrides=vehicle.chassis_overrides,
            )
            resolve_perception_profile(
                vehicle.perception_profile,
                vehicle.perception_overrides,
                entity_type="vehicle")
            if not isinstance(vehicle.sensor_overrides, dict):
                raise ValueError("Vehicle sensor_overrides must be an object")
            for module_name, overrides in vehicle.sensor_overrides.items():
                if not isinstance(overrides, dict):
                    raise ValueError(
                        f"sensor_overrides[{module_name!r}] must be an object")
                if module_name == "lidar":
                    resolve_lidar_spec(module_name, overrides)
                else:
                    resolve_radar_spec(module_name, overrides)
                if not capabilities.has_module(module_name):
                    raise ValueError(
                        f"Cannot override unavailable sensor {module_name!r}")

        # Parse pedestrian configs
        pedestrians = [
            PedestrianConfig(
                ped_id=p["ped_id"],
                initial_node=p["initial_node"],
                destination_node=p.get("destination_node", ""),
                speed=p.get("speed", 1.4),
                start_time=p.get("start_time", 0.0),
                is_evaluated=p.get("is_evaluated", True),
                collision_radius_m=p.get("collision_radius_m", 0.4),
                perception_profile=p.get(
                    "perception_profile", "pedestrian_standard"),
                perception_overrides=p.get(
                    "perception_overrides", {}) or {},
                agent_config=p.get("agent_config", {}),
                initial_physical_state=p.get(
                    "initial_physical_state", {}) or {},
            )
            for p in raw_pedestrians
        ]
        for pedestrian in pedestrians:
            if pedestrian.agent_type not in {"sumo", "llm"}:
                raise ValueError(
                    "Pedestrian agent_config.type must be 'sumo' or 'llm'")
            resolve_perception_profile(
                pedestrian.perception_profile,
                pedestrian.perception_overrides,
                entity_type="pedestrian")
            if not isinstance(pedestrian.initial_physical_state, dict):
                raise ValueError(
                    "Pedestrian initial_physical_state must be an object")
            if "walking_path_xy" in pedestrian.initial_physical_state:
                raise ValueError(
                    f"walking_path_xy is unsupported for "
                    f"{pedestrian.ped_id}; SUMO-only physics requires a "
                    "mapped pedestrian route or crosswalk_id")

        entity_ids = [vehicle.vehicle_id for vehicle in vehicles]
        entity_ids.extend(pedestrian.ped_id for pedestrian in pedestrians)
        duplicate_ids = sorted({
            entity_id for entity_id in entity_ids
            if entity_ids.count(entity_id) > 1
        })
        if duplicate_ids:
            raise ValueError(
                f"Duplicate runtime entity ids: {duplicate_ids!r}")

        # Collect grid-event data
        grid_events = {}
        for key in ("traffic_light_schedule", "congestion_events",
                     "road_events", "speed_cameras"):
            if key in d:
                grid_events[key] = d[key]

        traffic_config_overrides = d.get("traffic_config_overrides", {}) or {}
        if not isinstance(traffic_config_overrides, dict):
            raise ValueError("traffic_config_overrides must be an object")
        valid_traffic_fields = {item.name for item in fields(TrafficConfig)}
        unknown_traffic_fields = (
            set(traffic_config_overrides) - valid_traffic_fields)
        if unknown_traffic_fields:
            raise ValueError(
                "unknown traffic_config_overrides: "
                f"{sorted(unknown_traffic_fields)}")
        if float(traffic_config_overrides.get(
                "stopped_speed_threshold_kmh", 0.0)) < 0.0:
            raise ValueError(
                "stopped_speed_threshold_kmh must be non-negative")
        sumo_config = d.get("sumo_config", {}) or {}
        if not isinstance(sumo_config, dict):
            raise ValueError("sumo_config must be an object")
        unknown_sumo_fields = sorted(set(sumo_config) - _SUMO_CONFIG_FIELDS)
        if unknown_sumo_fields:
            raise ValueError(
                f"unknown SUMO physics settings: {unknown_sumo_fields!r}")
        from simulation.native_npc_behavior import validate_behavior
        behavior = sumo_config.get("npc_behavior", {})
        validate_behavior(behavior)
        if behavior:
            native_ids = {v.vehicle_id for v in vehicles if v.agent_type == "sumo"}
            if set(behavior["vehicle_ids"]) - native_ids:
                raise ValueError("npc_behavior must target existing SUMO vehicles only")
        parallel_calls = d.get("max_parallel_model_calls", 4)
        if type(parallel_calls) is not int or parallel_calls < 1:
            raise ValueError("max_parallel_model_calls must be a positive integer")
        terminal_vehicle_ids = d.get("terminal_vehicle_ids", []) or []
        if (not isinstance(terminal_vehicle_ids, list)
                or any(not isinstance(item, str)
                       for item in terminal_vehicle_ids)):
            raise ValueError("terminal_vehicle_ids must be a list of strings")
        if len(terminal_vehicle_ids) != len(set(terminal_vehicle_ids)):
            raise ValueError("terminal_vehicle_ids must not contain duplicates")
        unknown_terminal_ids = sorted(
            set(terminal_vehicle_ids) - {item.vehicle_id for item in vehicles})
        if unknown_terminal_ids:
            raise ValueError(
                "terminal_vehicle_ids contains unknown vehicles: "
                f"{unknown_terminal_ids!r}")
        return cls(
            scenario_id=d["scenario_id"],
            name=d.get("name", d["scenario_id"]),
            road_network_id=d["road_network_id"],
            vehicles=vehicles,
            pedestrians=pedestrians,
            difficulty=d.get("difficulty", "medium"),
            total_time_s=d.get("total_time_s", 3600.0),
            tick_interval_s=d.get("tick_interval_s", 300.0),
            physics_step_s=d.get("physics_step_s", 0.1),
            sumo_config=dict(sumo_config),
            max_parallel_model_calls=parallel_calls,
            traffic_config_overrides=dict(traffic_config_overrides),
            inits_code=d.get("inits_code", ""),
            weather_keyframes=[
                WeatherKeyframe(**kf)
                for kf in d.get("weather_keyframes", [])
            ],
            daynight_keyframes=[
                DayNightKeyframe(**kf)
                for kf in d.get("daynight_keyframes", [])
            ],
            _grid_events=grid_events,
            driving_evaluation_config=d.get(
                "driving_evaluation", {}) or {},
            enable_driving_evaluation=bool(
                d.get("enable_driving_evaluation", True)),
            stop_when_all_vehicles_terminal=bool(
                d.get("stop_when_all_vehicles_terminal", True)),
            terminal_vehicle_ids=list(terminal_vehicle_ids),
            physics_only_mode=bool(d.get("physics_only_mode", False)),
        )


# ── Per-vehicle result ─────────────────────────────────────────

@dataclass
class VehicleResult:
    """Simulation result for one vehicle."""
    vehicle_id: str
    is_evaluated: bool
    checkpoints: List[dict] = field(default_factory=list)
    tick_interactions: List[dict] = field(default_factory=list)
    arrived: bool = False
    route_failed: bool = False
    route_failure_reason: str = ""
    route_failure_time_s: Optional[float] = None
    arrival_tick: Optional[int] = None
    arrival_time_s: Optional[float] = None
    arrival_completion: Optional[str] = None
    terminal_crossing_speed_kmh: Optional[float] = None
    physically_stopped_at_destination: Optional[bool] = None
    driving_evaluation: Optional[dict] = None
    passenger_evaluation: Optional[dict] = None
    capabilities: Dict = field(default_factory=dict)
    capability_events: List[dict] = field(default_factory=list)

    @property
    def cabin_score(self) -> Optional[float]:
        total = sum(c.get("total_fields", 0) for c in self.checkpoints)
        if total <= 0:
            return None
        correct = sum(c.get("correct_fields", 0) for c in self.checkpoints)
        return correct / total

    @property
    def cabin_rule_events(self) -> int:
        return sum(
            1 for checkpoint in self.checkpoints
            if checkpoint.get("applicable", True))

    @property
    def cabin_task_count(self) -> int:
        """Number of applicable YAML cabin tasks with scorable fields."""
        return sum(
            1 for checkpoint in self.checkpoints
            if checkpoint.get("applicable", True)
            and int(checkpoint.get("total_fields", 0)) > 0)

    @property
    def cabin_tasks_completed(self) -> int:
        return sum(
            1 for checkpoint in self.checkpoints
            if checkpoint.get("applicable", True)
            and int(checkpoint.get("total_fields", 0)) > 0
            and int(checkpoint.get("correct_fields", 0))
            == int(checkpoint.get("total_fields", 0)))

    @property
    def cabin_task_completion_rate(self) -> Optional[float]:
        total = self.cabin_task_count
        return self.cabin_tasks_completed / total if total else None


@dataclass
class PedestrianResult:
    """Simulation result for one pedestrian."""
    ped_id: str
    is_evaluated: bool
    tick_interactions: List[dict] = field(default_factory=list)
    arrived: bool = False
    crashed: bool = False
    actions_correct: int = 0
    actions_total: int = 0

    @property
    def action_accuracy(self) -> float:
        return self.actions_correct / self.actions_total if self.actions_total > 0 else 1.0


@dataclass
class MultiSimResult:
    """Complete result of a multi-vehicle simulation."""
    scenario_id: str
    vehicle_results: Dict[str, VehicleResult]
    pedestrian_results: Dict[str, PedestrianResult] = field(default_factory=dict)
    total_ticks: int = 0
    signal_events: List[dict] = field(default_factory=list)
    horn_events: List[dict] = field(default_factory=list)
    npc_equipment_rule_events: List[dict] = field(default_factory=list)
    npc_behavior_assignments: Dict[str, dict] = field(default_factory=dict)
    perception_log: List[dict] = field(default_factory=list)
    perception_log_dropped: int = 0
    physics_engine: Dict[str, Any] = field(default_factory=dict)

    @property
    def all_vehicles_arrived(self) -> bool:
        return bool(self.vehicle_results) and all(
            result.arrived for result in self.vehicle_results.values())

    @property
    def _focal_llm_result(self) -> Optional[VehicleResult]:
        """Return the declared focal LLM result when this is an LLM run.

        MultiLLM peers are evaluated vehicles for interaction purposes, but
        they are not task subjects.  SUMO reference runs deliberately retain
        the historical all-evaluated fallback because the focal vehicle has
        been replaced by SUMO and is not a model score.
        """
        scenario_config = getattr(self, "_scenario_config", {}) or {}
        experiment_scene = scenario_config.get("experiment_scene", {}) or {}
        focal_id = str(
            experiment_scene.get("focal_vehicle_id")
            or scenario_config.get("focal_vehicle_id")
            or "")
        focal_result = self.vehicle_results.get(focal_id)
        focal_config = next(
            (vehicle for vehicle in scenario_config.get("vehicles", [])
             if str(vehicle.get("vehicle_id")) == focal_id),
            None,
        )
        focal_is_llm = (
            focal_config is not None
            and str((focal_config.get("agent_config", {}) or {}).get(
                "type", "")).lower() == "llm")
        if (focal_result is not None and focal_result.is_evaluated
                and focal_is_llm):
            return focal_result
        return None

    @property
    def overall_cabin_score(self) -> Optional[float]:
        """Return the focal cabin score for LLM tasks.

        Fixed peer LLMs are interaction actors and must not be averaged into
        the task-level cabin metric.  Older/non-LLM callers retain the
        all-evaluated fallback.
        """
        focal_result = self._focal_llm_result
        if focal_result is not None:
            return focal_result.cabin_score
        evaluated = [
            vr for vr in self.vehicle_results.values()
            if vr.is_evaluated and vr.cabin_score is not None
        ]
        if not evaluated:
            return None
        return sum(vr.cabin_score for vr in evaluated) / len(evaluated)

    @property
    def overall_trajectory_quality_score(self) -> Optional[float]:
        """Mean physical-trajectory quality over evaluated vehicles."""
        scores = [
            vr.driving_evaluation["trajectory_quality_score"]
            for vr in self.vehicle_results.values()
            if vr.is_evaluated and vr.driving_evaluation
            and vr.driving_evaluation.get("applicable", False)
        ]
        return sum(scores) / len(scores) if scores else None

    @property
    def overall_cabin_layer_score_100(self) -> Optional[float]:
        return cabin_layer_score_100(self.overall_cabin_score)

    @property
    def overall_single_vehicle_layer_score_100(self) -> Optional[float]:
        evaluated_results = [
            vr for vr in self.vehicle_results.values() if vr.is_evaluated]
        # MultiLLM contains evaluated fixed-peer LLMs for interaction, but
        # the task driving score belongs to its declared focal vehicle. The
        # redacted scenario is attached before serialization; older callers
        # without that metadata retain the all-evaluated fallback.
        scenario_config = getattr(self, "_scenario_config", {}) or {}
        experiment_scene = scenario_config.get("experiment_scene", {}) or {}
        focal_id = str(
            experiment_scene.get("focal_vehicle_id")
            or scenario_config.get("focal_vehicle_id")
            or "")
        focal_result = self.vehicle_results.get(focal_id)
        focal_config = next(
            (vehicle for vehicle in scenario_config.get("vehicles", [])
             if str(vehicle.get("vehicle_id")) == focal_id),
            None,
        )
        focal_is_llm = (
            focal_config is not None
            and str((focal_config.get("agent_config", {}) or {}).get(
                "type", "")).lower() == "llm")
        if (focal_result is not None and focal_result.is_evaluated
                and focal_is_llm):
            evaluated_results = [focal_result]
        scores = [
            single_vehicle_layer_score_100(vr.driving_evaluation)
            for vr in evaluated_results
        ]
        scores = [score for score in scores if score is not None]
        return round(sum(scores) / len(scores), 2) if scores else None

    @property
    def overall_cabin_task_completion_rate(self) -> Optional[float]:
        focal_result = self._focal_llm_result
        if focal_result is not None:
            return focal_result.cabin_task_completion_rate
        completed = sum(
            vr.cabin_tasks_completed for vr in self.vehicle_results.values()
            if vr.is_evaluated)
        total = sum(
            vr.cabin_task_count for vr in self.vehicle_results.values()
            if vr.is_evaluated)
        return completed / total if total else None

    @property
    def hard_safety_passed(self) -> bool:
        return all(
            not vr.is_evaluated
            or not vr.driving_evaluation
            or vr.driving_evaluation.get("hard_safety_passed", False)
            for vr in self.vehicle_results.values()
        )

    @property
    def driving_process_hard_gate_triggered(self) -> bool:
        focal_result = self._focal_llm_result
        results = (
            [focal_result] if focal_result is not None
            else list(self.vehicle_results.values()))
        return any(
            vr.is_evaluated
            and vr.driving_evaluation
            and (vr.driving_evaluation.get("driving_process") or {}).get(
                "hard_gate_triggered", False)
            for vr in results
        )

    @property
    def passenger_request_summary(self) -> dict:
        from evaluation.personal_agent import aggregate_passenger_judgements
        rows = []
        request_count = 0
        for vid, vehicle in self.vehicle_results.items():
            if not vehicle.is_evaluated or vehicle.passenger_evaluation is None:
                continue
            evaluation = vehicle.passenger_evaluation
            request_count += evaluation.get("request_count", 0)
            for index, item in enumerate(evaluation.get("judgements", [])):
                rows.append({**item, "request_id": f"{vid}:{item.get('request_id', index)}"})
        return aggregate_passenger_judgements(rows, request_count=request_count)

    def to_dict(self) -> dict:
        d = {
            "scenario_id": self.scenario_id,
            "total_ticks": self.total_ticks,
            "physics_engine": copy.deepcopy(self.physics_engine),
            "all_vehicles_arrived": self.all_vehicles_arrived,
            "world_communication": {
                "signal_events": self.signal_events,
                "horn_events": self.horn_events,
                "npc_equipment_rule_events": (
                    self.npc_equipment_rule_events),
                "npc_behavior_assignments": self.npc_behavior_assignments,
            },
            "perception": {
                "detections": self.perception_log,
                "dropped_records": self.perception_log_dropped,
            },
            "evaluation": {
                "cabin": {
                    "evaluation_type": "yaml_rules",
                    "score": self.overall_cabin_score,
                    "layer_score_100": (
                        self.overall_cabin_layer_score_100),
                    "task_completion_rate": (
                        self.overall_cabin_task_completion_rate),
                },
                "driving": {
                    "evaluation_type": "physical_trajectory",
                    "layer_score_100": (
                        self.overall_single_vehicle_layer_score_100),
                    "driving_process_score_100": (
                        self.overall_single_vehicle_layer_score_100),
                    "driving_process_hard_gate_triggered": (
                        self.driving_process_hard_gate_triggered),
                    "trajectory_quality_score": (
                        self.overall_trajectory_quality_score),
                    "hard_safety_passed": self.hard_safety_passed,
                },
                "passenger_interaction": {
                    **self.passenger_request_summary,
                    "vehicles": {
                        vid: copy.deepcopy(vr.passenger_evaluation)
                        for vid, vr in self.vehicle_results.items()
                        if vr.is_evaluated
                        and vr.passenger_evaluation is not None
                    },
                },
            },
            "vehicles": {
                vid: {
                    "is_evaluated": vr.is_evaluated,
                    "cabin_evaluation": {
                        "evaluation_type": "yaml_rules",
                        "applicable": vr.cabin_score is not None,
                        "score": vr.cabin_score,
                        "layer_score_100": cabin_layer_score_100(
                            vr.cabin_score),
                        "rule_events": vr.cabin_rule_events,
                        "task_count": vr.cabin_task_count,
                        "tasks_completed": vr.cabin_tasks_completed,
                        "task_completion_rate": (
                            vr.cabin_task_completion_rate),
                    },
                    "driving_evaluation": vr.driving_evaluation,
                    "passenger_evaluation": vr.passenger_evaluation,
                    "capabilities": vr.capabilities,
                    "capability_events": vr.capability_events,
                    "arrived": vr.arrived,
                    "route_failed": vr.route_failed,
                    "route_failure_reason": vr.route_failure_reason,
                    "route_failure_time_s": vr.route_failure_time_s,
                    "arrival_tick": vr.arrival_tick,
                    "arrival_time_s": vr.arrival_time_s,
                    "arrival_completion": vr.arrival_completion,
                    "terminal_crossing_speed_kmh": (
                        vr.terminal_crossing_speed_kmh),
                    "physically_stopped_at_destination": (
                        vr.physically_stopped_at_destination),
                    "checkpoints": vr.checkpoints,
                }
                for vid, vr in self.vehicle_results.items()
            },
        }
        if self.pedestrian_results:
            d["pedestrians"] = {
                pid: {
                    "is_evaluated": pr.is_evaluated,
                    "action_accuracy": pr.action_accuracy,
                    "arrived": pr.arrived,
                    "crashed": pr.crashed,
                    "interactions": pr.tick_interactions,
                }
                for pid, pr in self.pedestrian_results.items()
            }
        return d


# ── Multi-vehicle simulation engine ───────────────────────────

class MultiSimEngine:
    """Multi-vehicle multi-agent simulation engine (fixed-step physics loop).

    Each vehicle gets its own VehicleWorld.  The TrafficCoordinator
    coordinates spatial movement.  Weather/daynight are shared
    (all vehicles experience the same global weather, but edge-level
    weather can differ based on position).

    SUMO advances physics in fixed 0.1-second sub-steps. Each step first
    collects LLM controls, then advances SUMO and synchronizes the public
    state. Agents are woken whenever a relevant event fires
    (proximity/weather/daynight/trigger) or on their independent heartbeat.
    """

    # Single global physics clock for vehicles, pedestrians and collisions.
    PHYSICS_STEP_SEC: float = 0.1

    def __init__(self, scenario: MultiScenario, *,
                 lane_geometry_runtime=None, max_parallel_model_calls=None):
        from rules.rule_loader import get_rule_loader

        self.scenario = scenario
        self._model_scheduler = ModelCallScheduler(
            scenario.max_parallel_model_calls if max_parallel_model_calls is None
            else max_parallel_model_calls)
        self._rule_loader = get_rule_loader()
        # Validation and batch runners may safely reuse one immutable HD-map
        # topology index across independent simulations of the same map.
        self._lane_geometry_runtime = lane_geometry_runtime
        self.traffic_mgr: Optional[TrafficCoordinator] = None
        self.road_network: Optional[RoadNetwork] = None
        self._agent_visual_renderer = None
        self._camera_visual_renderer = None
        # Read-only consumers (for example Web3D) observe synchronized SUMO
        # boundaries. Their failures are isolated from physical execution.
        self._world_observers: List[Any] = []
        self.world_observer_errors: List[dict] = []

        # Per-vehicle state
        self._vw: Dict[str, VehicleWorld] = {}           # vehicle_id → agent VW
        self._expect_vw: Dict[str, VehicleWorld] = {}     # vehicle_id → reference VW
        self._memory: Dict[str, SessionHistory] = {}
        self._prev_snapshot: Dict[str, Optional[WorldSnapshot]] = {}
        self._capabilities: Dict[str, VehicleCapabilitySet] = {}
        self._vehicle_results: Dict[str, VehicleResult] = {}
        self._gt_lines_by_tick: Dict[str, dict] = {}      # vid → {tick: [lines]}
        self._prev_snap_agent: Dict[str, Optional[dict]] = {}
        self._cabin_evaluator = CabinYamlEvaluator()
        self._driving_evaluator: Optional[DrivingEvaluator] = None
        self._driving_process_scorer: Optional[
            DrivingProcessScoreTracker] = None
        self._npc_environment_signatures: Dict[str, Tuple[str, str]] = {}
        self._pending_npc_equipment_actions: Dict[str, List[str]] = defaultdict(list)
        self.npc_equipment_rule_log: List[dict] = []

        # Global state
        self._last_weather_t: Optional[int] = None
        self._last_daynight_t: Optional[int] = None
        self._sim_time: float = 0.0  # current physics time (seconds)
        # LLM callbacks decide from one pre-commit world. Their physical
        # commands are collected during the wake batch and committed only
        # after every callback has returned.
        self._collect_agent_commands: bool = False
        self._agent_command_queue: List[dict] = []
        self._next_agent_command_id: int = 1
        self._pending_command_receipts: List[dict] = []
        # Latest front/rear radar frame for every equipped vehicle. Frames are
        # refreshed exactly once per authoritative 0.1-second world boundary;
        # agent tools only read this cache.
        self._radar_frames: Dict[str, Dict[str, dict]] = {}
        self._radar_sampled_at_s: Optional[float] = None
        # Authoritative 10 Hz motion history supplied to passenger/evaluator
        # runtimes. It is observational only and never affects control.
        self._vehicle_motion_history: Dict[str, deque] = {}
        self.agent_callback_errors: List[dict] = []

    @staticmethod
    def _module_bool(module, *names: str) -> bool:
        for name in names:
            if module is not None and hasattr(module, name):
                value = getattr(module, name)
                if hasattr(value, "value"):
                    value = value.value
                if isinstance(value, str):
                    return value.lower() in ("on", "true", "active")
                return bool(value)
        return False

    def _sync_vehicle_external_signals(self, time_s: float) -> None:
        """Publish per-VehicleWorld controls at one batch boundary."""
        if self.traffic_mgr is None:
            return
        for vehicle_id, vw in self._vw.items():
            available = set(vw.available_module_names())
            module = lambda name: (getattr(vw, name, None)
                                   if name in available else None)
            turn = module("turnSignal")
            direction = getattr(turn, "direction", "off")
            low_beam = module("lowBeamHeadlight")
            low_mode = getattr(low_beam, "mode", "off")
            if hasattr(low_mode, "value"):
                low_mode = low_mode.value
            low_on = str(low_mode).lower() == "on"
            if str(low_mode).lower() == "auto":
                low_on = bool(self.traffic_mgr._is_night)
            fog = module("fogLight")
            vehicle = self.traffic_mgr.get_state(vehicle_id)
            signal_values = {
                "low_beam": low_on,
                "high_beam": self._module_bool(
                    module("highBeamHeadlight"), "high_beam_on"),
                "front_fog_light": self._module_bool(
                    getattr(fog, "front_light", None), "is_on"),
                "rear_fog_light": self._module_bool(
                    getattr(fog, "rear_light", None), "is_on"),
                "position_light": self._module_bool(
                    module("positionLight"), "is_on"),
                "tail_light": self._module_bool(
                    module("tailLight"), "is_on"),
            }
            # Background indicators/hazards are synchronized from SUMO.
            # LLM vehicles publish the explicit cabin-module state instead.
            if vehicle is not None and vehicle.is_llm:
                signal_values.update({
                    "left_indicator": direction == "left",
                    "right_indicator": direction == "right",
                    "hazard": self._module_bool(
                        module("hazardLight"), "is_active"),
                })
            self.traffic_mgr.update_vehicle_signals(
                vehicle_id, signal_values, time_s)
            horn = module("horn")
            pending = list(getattr(horn, "pending_emissions", []) or [])
            if horn is not None:
                horn.pending_emissions = []
            for request in pending:
                self.traffic_mgr.emit_horn(
                    vehicle_id,
                    duration_s=request.get("duration_s", 0.3),
                    intensity=request.get("intensity", "normal"),
                    time_s=time_s)
            window = module("window")
            window_states = list(
                getattr(window, "_windows", {}).values())
            if vehicle is not None:
                vehicle.cabin_open_fraction = (
                    sum(float(getattr(state, "open_degree", 0.0))
                        for state in window_states)
                    / (100.0 * len(window_states))
                    if window_states else 0.0)

    def _apply_npc_external_equipment_rules(
        self,
        vehicle_id: str,
        expected_actions: List[str],
    ) -> List[str]:
        """Execute YAML-oracle actions that control an NPC's exterior equipment.

        SUMO remains authoritative for motion, braking and manoeuvre signals.
        The shared YAML oracle owns environment-driven equipment such as
        headlamps, fog lamps, wipers and closed openings. Evaluated agents
        receive the same oracle actions only as an isolated scoring
        reference; they are never applied to an LLM vehicle here.
        """
        vw = self._vw[vehicle_id]
        applied: List[str] = []
        for action in expected_actions:
            module_name = action_module_name(action)
            if module_name not in _NPC_WORLD_VISIBLE_RULE_MODULES:
                continue
            if not vw.has_module(module_name):
                continue
            result = execute_yaml_action(vw, action)
            if isinstance(result, dict) and result.get("success") is False:
                raise RuntimeError(
                    "NPC exterior-equipment rule failed: "
                    f"{vehicle_id}: {action}: {result}")
            applied.append(action)
        return applied

    def _sync_npc_environment_equipment_rules(self, time_s: float) -> None:
        """Apply NPC environment-equipment rules independently of agent wakeups."""
        for config in self.scenario.vehicles:
            if config.agent_type != "sumo":
                continue
            vehicle_id = config.vehicle_id
            vw = self._vw.get(vehicle_id)
            if vw is None:
                continue
            snapshot = take_world_snapshot(
                vw, self.traffic_mgr.get_state(vehicle_id))
            signature = (
                snapshot.weather_condition,
                snapshot.daynight_period,
            )
            if self._npc_environment_signatures.get(vehicle_id) == signature:
                continue
            expected_actions, current, _, _ = derive_ground_truth(
                vw=vw,
                prev_snapshot=self._prev_snapshot.get(vehicle_id),
                vehicle_state=self.traffic_mgr.get_state(vehicle_id),
            )
            self._prev_snapshot[vehicle_id] = current
            applied = self._apply_npc_external_equipment_rules(
                vehicle_id, expected_actions)
            self._npc_environment_signatures[vehicle_id] = signature
            if not applied:
                continue
            self._pending_npc_equipment_actions[vehicle_id].extend(applied)
            self.npc_equipment_rule_log.append({
                "time_s": round(float(time_s), 6),
                "vehicle_id": vehicle_id,
                "environment": {
                    "weather": signature[0],
                    "daynight": signature[1],
                },
                "actions": list(applied),
            })

    def _submit_agent_command(
        self,
        entity_id: str,
        command: str,
        executor: Callable[[], dict],
        accepted_result: Optional[dict] = None,
        control_slots: Optional[List[str]] = None,
    ) -> dict:
        """Queue an agent command during a wake batch, or execute directly."""
        vehicle = (self.traffic_mgr.get_state(entity_id)
                   if self.traffic_mgr is not None else None)
        if vehicle is not None and vehicle.route_failed:
            return {"success": False, "reason": "vehicle_route_failed"}
        if not self._collect_agent_commands:
            result = dict(executor())
            result["status"] = (
                "accepted" if result.get("success", True) else "rejected")
            return result
        command_id = f"command-{self._next_agent_command_id:012d}"
        self._next_agent_command_id += 1
        active_buffer = _WAKE_COMMAND_BUFFER.get()
        queue = (active_buffer[1] if active_buffer is not None
                 and active_buffer[0] is self else self._agent_command_queue)
        queue.append({
            "command_id": command_id,
            "entity_id": entity_id,
            "command": command,
            "executor": executor,
            "control_slots": tuple(dict.fromkeys(control_slots or ())),
        })
        result = dict(accepted_result or {"success": True})
        result.setdefault("success", True)
        result["status"] = "queued"
        result["queued_for_world_commit"] = True
        result["command_id"] = command_id
        if control_slots:
            result["control_slots"] = list(dict.fromkeys(control_slots))
        return result

    def _commit_agent_commands(self, time_s: float) -> None:
        """Commit one wake batch after all agents observed the same world."""
        commands = self._agent_command_queue
        self._agent_command_queue = []
        self._collect_agent_commands = False
        last_owner = {}
        for index, item in enumerate(commands):
            for slot in item.get("control_slots", ()):
                last_owner[(item["entity_id"], slot)] = index
        for index, item in enumerate(commands):
            slots = item.get("control_slots", ())
            retained_slots = [
                slot for slot in slots
                if last_owner[(item["entity_id"], slot)] == index
            ]
            if slots and not retained_slots:
                superseding_indices = sorted({
                    last_owner[(item["entity_id"], slot)]
                    for slot in slots
                })
                self._pending_command_receipts.append({
                    "command_id": item["command_id"],
                    "entity_id": item["entity_id"],
                    "command": item["command"],
                    "committed_at_s": round(float(time_s), 6),
                    "status": "superseded",
                    "result": {
                        "success": True,
                        "applied": False,
                        "reason": "later_command_won_same_control_slot",
                        "control_slots": list(slots),
                        "superseded_by": [
                            commands[owner]["command_id"]
                            for owner in superseding_indices
                        ],
                    },
                })
                continue
            try:
                result = item["executor"]()
                normalized = (
                    dict(result) if isinstance(result, dict)
                    else {"success": True, "value": str(result)})
                if normalized.get("success", True):
                    vehicle = (
                        self.traffic_mgr.get_state(item["entity_id"])
                        if self.traffic_mgr is not None else None)
                    if vehicle is not None:
                        for slot in retained_slots:
                            control = {
                                "command_id": item["command_id"],
                                "command": item["command"],
                                "committed_at_s": round(
                                    float(time_s), 6),
                            }
                            if slot == "longitudinal":
                                control.update({
                                    "target_speed_kmh": round(float(
                                        vehicle.desired_speed_kmh), 3),
                                    "emergency_brake": bool(
                                        vehicle.llm_control_command.get(
                                            "emergency_brake", False)),
                                })
                            elif slot == "lateral":
                                control["target_lane"] = int(
                                    vehicle.target_lane)
                            elif slot == "route_maneuver":
                                control["maneuver"] = str(
                                    vehicle.planned_turn or "")
                            vehicle.active_control_commands[slot] = control
                receipt = {
                    "command_id": item["command_id"],
                    "entity_id": item["entity_id"],
                    "command": item["command"],
                    "committed_at_s": round(float(time_s), 6),
                    "status": (
                        "committed" if normalized.get("success", True)
                        else "rejected"),
                    "result": normalized,
                }
                self._pending_command_receipts.append(receipt)
                if isinstance(result, dict) and not result.get(
                        "success", True):
                    self.agent_callback_errors.append({
                        "entity_id": item["entity_id"],
                        "time_s": time_s,
                        "phase": "command_rejected",
                        "command": item["command"],
                        "error": result.get(
                            "reason", result.get("error", "rejected")),
                    })
            except Exception as exc:
                self._pending_command_receipts.append({
                    "command_id": item["command_id"],
                    "entity_id": item["entity_id"],
                    "command": item["command"],
                    "committed_at_s": round(float(time_s), 6),
                    "status": "failed",
                    "result": {
                        "success": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                })
                self.agent_callback_errors.append({
                    "entity_id": item["entity_id"],
                    "time_s": time_s,
                    "phase": "command_commit",
                    "command": item["command"],
                    "error": f"{type(exc).__name__}: {exc}",
                })

    @staticmethod
    def _normalize_agent_actions(actions) -> List[str]:
        """Validate the callback contract before committing its commands."""
        if actions is None:
            return []
        if not isinstance(actions, (list, tuple)):
            raise TypeError(
                "agent callback must return list[str] or tuple[str, ...]")
        if not all(isinstance(action, str) for action in actions):
            raise TypeError("every returned agent action must be a string")
        return list(actions)

    def _agent_world_view(
        self,
        entity_id: str,
        world: WorldState,
    ) -> AgentWorldState:
        def submit(
            submitted_entity_id: str, action: str, params: dict,
        ) -> dict:
            pedestrian = self.traffic_mgr.pedestrians.get(
                submitted_entity_id)
            if pedestrian is None:
                return {"success": False, "error": "unknown pedestrian"}
            if pedestrian.is_crashed or pedestrian.has_arrived:
                return {"success": False, "error": "pedestrian inactive"}
            return self._submit_agent_command(
                submitted_entity_id,
                action,
                lambda: self.traffic_mgr.execute_pedestrian_action(
                    submitted_entity_id, action, params),
                {
                    "success": True,
                    "action": action,
                    "execution": "next_0.1s_world_step",
                },
                control_slots=["pedestrian_motion"],
            )

        return AgentWorldState(world, entity_id, submit)

    def _sample_continuous_radars(self, time_s: float) -> None:
        """Refresh both installed radar modules on the fixed world clock."""
        if (self._radar_sampled_at_s is not None
                and abs(self._radar_sampled_at_s - float(time_s)) <= 1e-9):
            return
        for vehicle_id, vw in self._vw.items():
            vehicle = self.traffic_mgr.get_state(vehicle_id)
            if vehicle is None or vehicle.arrived or vehicle.route_failed:
                continue
            frames = self._radar_frames.setdefault(vehicle_id, {})
            for module_name in ("frontRadar", "rearRadar"):
                if not vw.has_module(module_name):
                    frames.pop(module_name, None)
                    continue
                module = getattr(vw, module_name)
                frame = self.traffic_mgr.perception_model.radar_scan(
                    vehicle_id, module_name,
                    getattr(module, "_overrides", {}),
                    include_monitor_metadata=True)
                frames[module_name] = copy.deepcopy(frame)
        self._radar_sampled_at_s = round(float(time_s), 6)

    def _read_radar_frame(
        self, vehicle_id: str, module_name: str, overrides: dict,
    ) -> dict:
        """Return the public current frame without performing a private scan."""
        frame = self._radar_frames.get(vehicle_id, {}).get(module_name)
        if frame is None:
            # This is only a startup/direct-call fallback. During the normal
            # loop a frame is always sampled before any callback can run.
            self._sample_continuous_radars(self._sim_time)
            frame = self._radar_frames.get(vehicle_id, {}).get(module_name)
        if frame is None:
            return {
                "success": False,
                "error": "radar_frame_unavailable",
                "sensor": module_name,
            }
        result = copy.deepcopy(frame)
        for track in result.get("tracks", []):
            for key in list(track):
                if key.startswith("_warning_"):
                    track.pop(key, None)
        result["measurement_age_s"] = round(max(
            0.0,
            float(self._sim_time) - float(result.get(
                "sample_time_s", self._sim_time))), 6)
        result["source"] = "continuous_0.1s_radar_cache"
        return result

    def _radar_warning_events(
        self, time_s: float, callback_entity_ids,
    ) -> List[WakeEvent]:
        """Generate debounced front/rear proximity warnings from cached data."""
        events = []
        rank = {"clear": 0, "caution": 1, "critical": 2}
        for vehicle_id in callback_entity_ids:
            vehicle = self.traffic_mgr.get_state(vehicle_id)
            if (vehicle is None or vehicle.arrived or vehicle.route_failed or vehicle.is_crashed):
                continue
            speed_mps = max(0.0, vehicle.current_speed_kmh / 3.6)
            for module_name, event_type in (
                    ("frontRadar", "front_collision_warning"),
                    ("rearRadar", "rear_collision_warning")):
                frame = self._radar_frames.get(
                    vehicle_id, {}).get(module_name)
                if not frame or not frame.get("success"):
                    continue
                level = "clear"
                for track in frame.get("tracks", []):
                    # A radar beam may see a vehicle in the adjacent lane.
                    # Keep it in the scan, but only turn an occupied or
                    # predicted ego corridor into a collision warning.
                    if not track.get("_warning_corridor_relevant", True):
                        continue
                    distance = float(track.get("distance_m", float("inf")))
                    ttc = track.get("ttc_s")
                    ttc = float(ttc) if ttc is not None else float("inf")
                    if module_name == "frontRadar":
                        critical_gap = max(1.5, speed_mps * 0.35)
                        caution_gap = max(3.0, speed_mps * 0.8)
                    else:
                        critical_gap = 1.5
                        caution_gap = 3.0
                    candidate = (
                        "critical" if ttc <= 2.0 or distance <= critical_gap
                        else "caution"
                        if ttc <= 4.5 or distance <= caution_gap
                        else "clear")
                    if rank[candidate] > rank[level]:
                        level = candidate
                priority = (
                    WakePriority.CRITICAL
                    if level == "critical" else
                    WakePriority.IMPORTANT
                    if level == "caution" else WakePriority.INFO)
                event = self.wake_broker.transition(
                    event_type, vehicle_id, "vehicle", time_s, level,
                    dedupe_key=f"radar_warning:{vehicle_id}:{module_name}",
                    priority=priority,
                    source="continuous_radar_warning_monitor",
                    details={
                        "sensor": module_name,
                        "direction": (
                            "front" if module_name == "frontRadar"
                            else "rear"),
                        "observation_tool": f"{module_name}__scan",
                    },
                    repeat_after_s=(2.0 if level == "critical" else 4.0),
                    reenter_cooldown_s=(
                        0.0 if level == "critical"
                        else RADAR_WARNING_REENTRY_COOLDOWN_S),
                )
                if event is not None:
                    events.append(event)
        return events

    def _heartbeat_interval_for(
        self,
        entity_id: str,
        base_interval_s: float,
        agent_callbacks: Dict[str, Callable],
    ) -> float:
        """Return the model-selected cadence at fixed-step resolution."""
        callback = agent_callbacks.get(entity_id)
        callback_state = getattr(callback, "_state", None)
        selected = None
        if isinstance(callback_state, dict):
            selected = callback_state.get("heartbeat_interval_s")
            if selected is not None:
                try:
                    selected = float(selected)
                except (TypeError, ValueError):
                    selected = None
        requested = (
            selected if selected is not None and math.isfinite(selected)
            else base_interval_s)
        return max(self.PHYSICS_STEP_SEC, requested)

    def _destination_display_name(
        self, destination_node: str, configured_name: str = "",
    ) -> str:
        """Resolve one stable human-readable label for a driving task."""
        configured = str(configured_name or "").strip()
        if configured:
            return configured
        node_id = str(destination_node or "").strip()
        node = self.road_network.nodes.get(node_id)
        if node is None:
            return node_id
        poi = getattr(node, "poi", None) or []
        if isinstance(poi, str) and poi.strip():
            return poi.strip()
        if poi:
            return str(poi[0])
        name = str(getattr(node, "name", "") or "").strip()
        if name:
            return name
        road_names = {
            str(value).strip()
            for value in (
                getattr(node, "street_name", ""),
                getattr(node, "avenue_name", ""),
            )
            if str(value or "").strip()
        }
        for segment in self.road_network.edges.values():
            if (segment.from_node == node_id or segment.to_node == node_id):
                road_name = str(segment.name or "").strip()
                if road_name:
                    road_names.add(road_name)
        if len(road_names) >= 2:
            return " & ".join(sorted(road_names)[:2])
        if road_names:
            return f"{next(iter(road_names))}沿线目的地"
        return node_id

    def _event_can_wake_agent(
        self, event: WakeEvent, agent_callbacks: Dict[str, Callable],
    ) -> bool:
        """Keep evaluator truth out of an LLM driver's wake schedule."""
        callback = agent_callbacks.get(event.entity_id)
        if callback is None:
            return True
        is_llm = getattr(callback, "_agent_type", None) == "llm"
        if not is_llm:
            config = next((
                item for item in self.scenario.vehicles
                if item.vehicle_id == event.entity_id), None)
            is_llm = bool(config and config.agent_type == "llm")
        if not is_llm or event.entity_type != "vehicle":
            return True
        if event.event_type == "pedestrian_crosswalk_conflict":
            return (
                event.source == "connector_crosswalk_monitor"
                and event.state in {"entered", "updated"}
                and event.details.get("level") in {"caution", "critical"}
            )
        return event.event_type in _LLM_SENSOR_OR_LIFECYCLE_WAKE_EVENTS

    def _entity_is_physically_terminal(self, entity_id: str) -> bool:
        """Return terminal body state without relying on callback bookkeeping."""
        vehicle = self.traffic_mgr.vehicles.get(entity_id)
        if vehicle is not None:
            return bool(vehicle.arrived or getattr(vehicle, "route_failed", False)
                        or vehicle.is_crashed)
        pedestrian = self.traffic_mgr.pedestrians.get(entity_id)
        return bool(pedestrian and (
            pedestrian.has_arrived or pedestrian.is_crashed))

    # ── Main run loop (fixed-step physics) ────────────────────

    def run(
        self,
        agent_callbacks: Dict[str, Callable],
    ) -> MultiSimResult:
        """Run one episode and always release its SUMO engine instance."""
        try:
            with self._model_scheduler:
                return self._run_episode(agent_callbacks)
        finally:
            if self._camera_visual_renderer is not None:
                try:
                    self._camera_visual_renderer.close()
                except Exception as error:
                    logger.warning("Web3D camera close failed: %s", error)
            for observer in self._world_observers:
                close = getattr(observer, "close", None)
                if not callable(close):
                    continue
                try:
                    close()
                except Exception as error:  # visualization must not own physics
                    logger.warning(
                        "world observer close failed: %s", error)
            traffic_mgr = getattr(self, "traffic_mgr", None)
            if traffic_mgr is not None:
                traffic_mgr.close()

    def _run_episode(
        self,
        agent_callbacks: Dict[str, Callable],
    ) -> MultiSimResult:
        """Run the multi-vehicle simulation.

        Fixed-step physics loop: every ``PHYSICS_STEP_SEC`` seconds we
        advance traffic + pedestrians, then wake agents on heartbeat or
        any detected event (proximity / weather / daynight / trigger).

        Args:
            agent_callbacks: {vehicle_id: callback_fn}
                Each callback: fn(vw, t, passenger_messages, memory, tick_index,
                                  _vehicle_state=,
                                  _destination_name=, _destination_node=,
                                  _road_network=)
                    -> list[str] (actions taken)

        Returns:
            MultiSimResult with per-vehicle accuracy data.
        """
        # 1. Load road network
        from simulation.road_networks import load_road_network
        self.road_network = load_road_network(self.scenario.road_network_id)
        for key, value in self.scenario.traffic_config_overrides.items():
            setattr(self.road_network.config, key, copy.deepcopy(value))
        # Traffic lights now use 1-second resolution (phases in seconds)
        self.road_network.config.seconds_per_tick = 1
        self.road_network.load_scenario_events(
            self.scenario._grid_events,
            tick_interval_s=self.scenario.tick_interval_s)
        from simulation.sumo_traffic_manager import SumoTrafficManager
        sumo_config = dict(self.scenario.sumo_config)
        configured_step = float(sumo_config.get(
            "step_length_s", self.scenario.physics_step_s))
        if abs(configured_step - self.scenario.physics_step_s) > 1e-9:
            raise ValueError(
                "SUMO step_length_s must equal scenario.physics_step_s")
        sumo_config["step_length_s"] = configured_step
        self.traffic_mgr = SumoTrafficManager(
            self.road_network,
            lane_geometry_runtime=self._lane_geometry_runtime,
            sumo_config=sumo_config)
        from visualization.agent_visual_renderer import AgentVisualRenderer
        self._agent_visual_renderer = AgentVisualRenderer(self.traffic_mgr)
        self.wake_broker = WakeEventBroker()
        self._radar_frames = {}
        self._radar_sampled_at_s = None

        # 2. Initialize per-vehicle state
        vehicle_results: Dict[str, VehicleResult] = {}

        for vcfg in self.scenario.vehicles:
            vid = vcfg.vehicle_id
            capabilities = resolve_vehicle_capabilities(
                equipment_profile=vcfg.equipment_profile,
                chassis_profile=vcfg.chassis_profile,
                enable_modules=vcfg.enable_modules,
                disable_modules=vcfg.disable_modules,
                chassis_overrides=vcfg.chassis_overrides,
            )
            self._capabilities[vid] = capabilities
            chassis = capabilities.chassis

            is_llm = vcfg.agent_type == "llm"
            vs = self.traffic_mgr.register_vehicle(
                vehicle_id=vid,
                start_node=vcfg.initial_node,
                destination=vcfg.destination_node,
                destination_name=self._destination_display_name(
                    vcfg.destination_node, vcfg.destination_name),
                start_lane=vcfg.initial_lane,
                auto_navigate=bool(vcfg.destination_node),
                is_llm=is_llm,
                chassis_profile=capabilities.chassis_profile,
                length_m=chassis.length_m,
                width_m=chassis.width_m,
                max_acceleration_mps2=chassis.max_acceleration_mps2,
                max_braking_mps2=chassis.max_braking_mps2,
                lane_change_duration_s=chassis.lane_change_duration_s,
                perception_profile=vcfg.perception_profile,
                perception_overrides=vcfg.perception_overrides,
            )

            self._vw[vid] = VehicleWorld(capability_set=capabilities)
            self._expect_vw[vid] = VehicleWorld(
                capability_set=capabilities)
            sensor_specs = {}
            for module_name in ("frontRadar", "rearRadar"):
                if not capabilities.has_module(module_name):
                    continue
                overrides = vcfg.sensor_overrides.get(module_name, {})
                spec = resolve_radar_spec(module_name, overrides)
                sensor_specs[module_name] = spec.as_dict()
                for world in (self._vw[vid], self._expect_vw[vid]):
                    getattr(world, module_name).configure(overrides)
                getattr(self._vw[vid], module_name).bind_scan_provider(
                    lambda sensor_name, sensor_overrides, vehicle_id=vid:
                    self._read_radar_frame(
                        vehicle_id, sensor_name, sensor_overrides))
            if capabilities.has_module("lidar"):
                lidar_overrides = vcfg.sensor_overrides.get("lidar", {})
                lidar_spec = resolve_lidar_spec("lidar", lidar_overrides)
                sensor_specs["lidar"] = lidar_spec.as_dict()
                for world in (self._vw[vid], self._expect_vw[vid]):
                    world.lidar.configure(lidar_overrides)
            capability_events = []
            if self.scenario.inits_code.strip():
                full_init = _FULL_INIT_PREFIX + self.scenario.inits_code
                filtered_init, skipped = self._vw[
                    vid].filter_capability_code(full_init)
                capability_events.extend({
                    **event,
                    "type": "initial_condition_skipped",
                } for event in skipped)
                execute(filtered_init, local_vars={'vw': self._vw[vid]},
                        global_vars=None)
                execute(filtered_init,
                        local_vars={'vw': self._expect_vw[vid]},
                        global_vars=None)

            self._memory[vid] = SessionHistory()
            self._prev_snapshot[vid] = None
            self._gt_lines_by_tick[vid] = {}
            self._prev_snap_agent[vid] = None

            capability_snapshot = capabilities.to_dict()
            capability_snapshot["sensor_specs"] = sensor_specs
            vehicle_results[vid] = VehicleResult(
                vehicle_id=vid,
                is_evaluated=vcfg.is_evaluated,
                capabilities=capability_snapshot,
                capability_events=capability_events,
            )
        self._vehicle_results = vehicle_results

        # 2b. Initialize pedestrians
        for pcfg in self.scenario.pedestrians:
            pid = pcfg.ped_id
            is_ped_llm = pcfg.agent_type == "llm"
            # Compute route from initial to destination
            ped_route = [pcfg.initial_node]
            if pcfg.destination_node:
                computed = self.road_network.plan_route(
                    pcfg.initial_node, pcfg.destination_node)
                if computed:
                    ped_route = computed
                else:
                    ped_route = [pcfg.initial_node, pcfg.destination_node]

            self.traffic_mgr.register_pedestrian(
                ped_id=pid,
                initial_node=pcfg.initial_node,
                route=ped_route,
                speed=pcfg.speed,
                is_llm=is_ped_llm,
                start_time=pcfg.start_time * 60.0,  # minutes → seconds
                collision_radius_m=pcfg.collision_radius_m,
                perception_profile=pcfg.perception_profile,
                perception_overrides=pcfg.perception_overrides,
            )
            self._memory[pid] = SessionHistory()

        # Initialize pedestrian results
        pedestrian_results: Dict[str, PedestrianResult] = {}
        for pcfg in self.scenario.pedestrians:
            pedestrian_results[pcfg.ped_id] = PedestrianResult(
                ped_id=pcfg.ped_id,
                is_evaluated=pcfg.is_evaluated,
            )

        self._agent_callbacks = agent_callbacks

        result = MultiSimResult(
            scenario_id=self.scenario.scenario_id,
            vehicle_results=vehicle_results,
            pedestrian_results=pedestrian_results,
        )
        result._scenario_config = _redact_scenario_for_audit(
            asdict(self.scenario))

        # 3. Stagger only generic spawns. Exactly placed experiment entities
        # already carry authoritative lane progress and must not be rejected
        # or moved by the coarse segment-level queue initializer.
        exactly_placed = {
            config.vehicle_id for config in self.scenario.vehicles
            if config.initial_physical_state
        }
        self.traffic_mgr.stagger_co_located_vehicles(
            excluded_vehicle_ids=exactly_placed)

        # Exact placements are applied after generic queue staggering. This is
        # an experiment authoring feature, not a central traffic policy.
        self._apply_initial_physical_states()

        # 3b. Subclass hook — run once per vehicle after self._vw[vid] is built
        for vcfg in self.scenario.vehicles:
            self._post_vehicle_init(vcfg.vehicle_id, vcfg)

        # Driving is evaluated from the authoritative physical world at 10 Hz,
        # independently of YAML cabin rules and independently of driver type.
        self._driving_evaluator = (
            DrivingEvaluator(
                ScoringTrafficView(self.traffic_mgr),
                self.road_network,
                [
                    vcfg.vehicle_id for vcfg in self.scenario.vehicles
                    if vcfg.is_evaluated
                ],
                config=DrivingEvaluationConfig(
                    **self.scenario.driving_evaluation_config),
            )
            if self.scenario.enable_driving_evaluation else None)
        self._driving_process_scorer = (
            DrivingProcessScoreTracker(
                ScoringTrafficView(self.traffic_mgr),
                self._vw,
                [
                    vcfg.vehicle_id for vcfg in self.scenario.vehicles
                    if vcfg.is_evaluated
                ],
            )
            if self.scenario.enable_driving_evaluation else None)

        # 4. Fixed-step physics loop (0.1s sub-steps).
        physics_time = 0.0
        self._sim_time = 0.0
        tick_index = 0
        heartbeat_intervals: Dict[str, float] = {}
        for vcfg in self.scenario.vehicles:
            if vcfg.vehicle_id not in agent_callbacks:
                continue
            callback = agent_callbacks[vcfg.vehicle_id]
            default_interval = (
                1.0
                if getattr(callback, "_agent_type", None) == "llm"
                or vcfg.agent_type == "llm"
                else self.scenario.tick_interval_s)
            heartbeat_intervals[vcfg.vehicle_id] = max(
                self.PHYSICS_STEP_SEC, float(vcfg.agent_config.get(
                    "heartbeat_interval_s", default_interval)))
        for pcfg in self.scenario.pedestrians:
            if pcfg.ped_id not in agent_callbacks:
                continue
            default_interval = min(
                self.scenario.tick_interval_s, 1.0)
            heartbeat_intervals[pcfg.ped_id] = max(
                self.PHYSICS_STEP_SEC, float(pcfg.agent_config.get(
                    "heartbeat_interval_s", default_interval)))
        next_heartbeat = {}
        pedestrian_start_times = {
            pcfg.ped_id: pcfg.start_time * 60.0
            for pcfg in self.scenario.pedestrians
        }
        for entity_id, interval in heartbeat_intervals.items():
            next_heartbeat[entity_id] = (
                max(0.0, pedestrian_start_times.get(entity_id, 0.0))
                + interval)
        startup_pending = True
        started_agent_entities = set()
        terminal_entities = set()
        simulation_end_emitted = False
        active_proximity_keys = set()
        active_crosswalk_keys = set()
        active_route_lane_keys = set()
        active_road_block_keys = set()
        active_connector_conflict_keys = set()
        active_pedestrian_risk_keys = set()
        active_pedestrian_path_keys = set()
        active_vehicle_pedestrian_path_keys = set()
        active_pedestrian_signal_keys = set()
        active_lane_change_keys = set()
        delivered_horn_event_keys = set()
        processed_signal_event_count = 0
        deferred_success_receipts: Dict[str, List[dict]] = defaultdict(list)
        fatal_agent_callback = False
        step_s = float(self.scenario.physics_step_s)
        if not 0.01 <= step_s <= 1.0:
            raise ValueError("physics_step_s must be in [0.01, 1.0]")
        total_time = self.scenario.total_time_s
        motion_capacity = int(math.ceil(35.0 / step_s)) + 2
        self._vehicle_motion_history = {
            vcfg.vehicle_id: deque(maxlen=motion_capacity)
            for vcfg in self.scenario.vehicles
        }

        prev_weather = None
        prev_daynight = None

        # Proximity threshold for triggering LLM warning.
        # 100m ≈ ~3-4 seconds at urban speeds (40 km/h).
        proximity_threshold_m = min(
            self.traffic_mgr.config.perception_range_m * 0.5,
            100.0,
        )

        while physics_time <= total_time + 1e-9:
            # ── Advance physics by one sub-step ──────────────────
            target = (
                physics_time if startup_pending
                else min(physics_time + step_s, total_time))
            if target > physics_time + 1e-9:
                self.traffic_mgr.recalculate_speeds(time_s=physics_time)
                trigger_events = list(
                    self.traffic_mgr.advance_world_to(target))
            else:
                # Materialize and publish SUMO's authoritative t=0 pose before
                # any evaluator or telemetry samples it.  This private SUMO
                # bootstrap does not advance VehicleArena simulation time.
                trigger_events = list(
                    self.traffic_mgr.advance_world_to(physics_time))
            physics_time = round(target, 6)
            self._sim_time = physics_time
            self._record_vehicle_motion(physics_time)

            # Environment-driven exterior equipment belongs to background
            # world simulation, not to the Driving Agent wake schedule.
            self._sync_global_world(physics_time)
            self._sync_npc_environment_equipment_rules(physics_time)
            self._sync_vehicle_external_signals(physics_time)

            if self._driving_evaluator is not None:
                self._driving_evaluator.observe(physics_time)
            if self._driving_process_scorer is not None:
                self._driving_process_scorer.observe(physics_time)

            # Per-substep subclass hook (used by Tracked for fine-grained
            # encounter logging).
            self._log_per_substep(physics_time, tick_index, trigger_events)
            self._notify_world_observers(
                physics_time, tick_index, trigger_events)

            if self.scenario.physics_only_mode:
                # The full agent path clears this after emitting startup
                # wakes.  Physics-only mode has no such wake, but must still
                # release the clock from its intentional t=0 sample.
                startup_pending = False
                for trigger_event in trigger_events:
                    if (trigger_event.type == "arrived"
                            and trigger_event.vehicle_id in vehicle_results):
                        vehicle_result = vehicle_results[
                            trigger_event.vehicle_id]
                        vehicle_result.arrived = True
                        vehicle_result.arrival_time_s = round(
                            float(trigger_event.time_s), 6)
                        vehicle_result.arrival_completion = (
                            trigger_event.details.get("completion"))
                        vehicle_result.terminal_crossing_speed_kmh = (
                            trigger_event.details.get(
                                "terminal_crossing_speed_kmh"))
                        vehicle_result.physically_stopped_at_destination = (
                            trigger_event.details.get(
                                "physically_stopped_at_destination"))
                if (self.scenario.stop_when_all_vehicles_terminal
                        and self._all_scenario_vehicles_terminal()):
                    break
                if physics_time >= total_time - 1e-9:
                    break
                continue

            # ── Detect dynamic events this sub-step ──────────────
            vehicle_events: Dict[str, List[Tuple[str, Dict]]] = {}
            wake_events: List[WakeEvent] = []
            # Rejections, failures and same-wake supersession wake immediately.
            # A routine successful commit is piggybacked on the entity's next
            # natural event/heartbeat instead of purchasing a model call only
            # to say that an already accepted command committed successfully.
            command_receipts = self._pending_command_receipts
            self._pending_command_receipts = []
            for receipt in command_receipts:
                entity_id = receipt["entity_id"]
                if receipt["status"] == "committed":
                    deferred_success_receipts[entity_id].append(receipt)
                    continue
                entity_type = (
                    "pedestrian"
                    if entity_id in self.traffic_mgr.pedestrians
                    else "vehicle")
                wake_events.append(self.wake_broker.discrete(
                    "command_result", entity_id, entity_type,
                    float(receipt["committed_at_s"]),
                    detected_at_s=physics_time,
                    priority=WakePriority.IMPORTANT,
                    source="agent_command_commit",
                    details=receipt))
            pending_terminal_entities = set()
            if startup_pending:
                for vcfg in self.scenario.vehicles:
                    if vcfg.vehicle_id in agent_callbacks:
                        wake_events.append(self.wake_broker.discrete(
                            "simulation_start", vcfg.vehicle_id, "vehicle",
                            physics_time, priority=WakePriority.IMPORTANT,
                            source="multi_sim_engine",
                            details={
                                "initial_node": vcfg.initial_node,
                            }))
                        if vcfg.destination_node:
                            destination_name = (
                                self._destination_display_name(
                                    vcfg.destination_node,
                                    vcfg.destination_name))
                            wake_events.append(self.wake_broker.discrete(
                                "driving_task_assigned",
                                vcfg.vehicle_id, "vehicle", physics_time,
                                priority=WakePriority.IMPORTANT,
                                source="scenario_driving_task",
                                details={
                                    "task_type": "drive_to_destination",
                                    "destination_name": destination_name,
                                    "destination_node": (
                                        vcfg.destination_node),
                                    "instruction": (
                                        "Drive safely to the assigned "
                                        "destination and complete the trip."),
                                }))
                        started_agent_entities.add(vcfg.vehicle_id)
                for pcfg in self.scenario.pedestrians:
                    pedestrian = self.traffic_mgr.pedestrians.get(
                        pcfg.ped_id)
                    if (pcfg.agent_type == "llm"
                            and pcfg.ped_id in agent_callbacks
                            and pedestrian is not None
                            and pedestrian.is_spawned):
                        wake_events.append(self.wake_broker.discrete(
                            "simulation_start", pcfg.ped_id, "pedestrian",
                            physics_time, priority=WakePriority.IMPORTANT,
                            source="multi_sim_engine",
                            details={
                                "initial_node": pcfg.initial_node,
                                "destination_node": pcfg.destination_node,
                            }))
                        started_agent_entities.add(pcfg.ped_id)
                startup_pending = False
            # Delayed pedestrians receive their start event when they actually
            # enter the physical world, never at an undeliverable t=0.
            for pcfg in self.scenario.pedestrians:
                pedestrian = self.traffic_mgr.pedestrians.get(pcfg.ped_id)
                if (pcfg.agent_type != "llm"
                        or pcfg.ped_id not in agent_callbacks
                        or pcfg.ped_id in started_agent_entities
                        or pedestrian is None
                        or not pedestrian.is_spawned):
                    continue
                wake_events.append(self.wake_broker.discrete(
                    "simulation_start", pcfg.ped_id, "pedestrian",
                    pedestrian.start_time,
                    detected_at_s=physics_time,
                    priority=WakePriority.IMPORTANT,
                    source="pedestrian_lifecycle",
                    details={
                        "initial_node": pcfg.initial_node,
                        "destination_node": pcfg.destination_node,
                    }))
                started_agent_entities.add(pcfg.ped_id)
            for te in trigger_events:
                vehicle_events.setdefault(te.vehicle_id, []).append(
                    (te.type, te.details))
                if (te.type == "arrived"
                        and te.vehicle_id in vehicle_results):
                    vehicle_result = vehicle_results[te.vehicle_id]
                    vehicle_result.arrived = True
                    vehicle_result.arrival_time_s = round(
                        float(te.time_s), 6)
                    vehicle_result.arrival_completion = te.details.get(
                        "completion")
                    vehicle_result.terminal_crossing_speed_kmh = (
                        te.details.get("terminal_crossing_speed_kmh"))
                    vehicle_result.physically_stopped_at_destination = (
                        te.details.get(
                            "physically_stopped_at_destination"))
                if te.vehicle_id not in agent_callbacks:
                    continue
                entity_type = (
                    "pedestrian" if te.vehicle_id in self.traffic_mgr.pedestrians
                    else "vehicle")
                priority = (
                    WakePriority.CRITICAL
                    if te.type in ("crashed", "collision")
                    else WakePriority.IMPORTANT
                    if te.type in (
                        "traffic_light_change", "intersection_arrival",
                        "ped_arrive_node", "ped_crossing_complete")
                    else WakePriority.NORMAL)
                wake_events.append(self.wake_broker.discrete(
                    te.type, te.vehicle_id, entity_type, te.time_s,
                    detected_at_s=physics_time,
                    priority=priority, source="traffic_manager",
                    details=te.details))
                if te.type in ("arrived", "route_failed", "ped_arrived",
                               "crashed", "collision"):
                    pending_terminal_entities.add(te.vehicle_id)

            # Front and rear radars sample continuously at the same 0.1 s
            # boundary as physics. Ordinary frames stay cached; only a
            # debounced warning transition becomes an agent wake event.
            self._sample_continuous_radars(physics_time)
            wake_events.extend(self._radar_warning_events(
                physics_time, agent_callbacks))

            # Build one high-definition local context per LLM vehicle. The
            # same lane-route/connector facts drive both autonomous control
            # and wake monitoring; the event system no longer maintains a
            # second segment-only interpretation of the map.
            driver_contexts = {}
            driver_environments = {}
            for vehicle in self.traffic_mgr.vehicles.values():
                if (not vehicle.is_llm
                        or vehicle.vehicle_id not in agent_callbacks
                        or vehicle.arrived or vehicle.route_failed or vehicle.is_crashed):
                    continue
                env = self.traffic_mgr._build_env_view(
                    vehicle, round(physics_time))
                driver_environments[vehicle.vehicle_id] = env
                driver_contexts[vehicle.vehicle_id] = (
                    self.traffic_mgr._build_driver_context(vehicle, env))

            # Route-relative leader/obstacle risk. Includes crashed vehicles,
            # connector occupants and leaders multiple lane-route legs ahead.
            current_proximity_keys = set()
            for vid, context in driver_contexts.items():
                vs = self.traffic_mgr.get_state(vid)
                leader = context.leader
                if leader is None:
                    continue
                gap_m = context.leader_gap_m
                if gap_m >= proximity_threshold_m:
                    continue
                closing_kmh = max(
                    0.0,
                    vs.current_speed_kmh - leader.current_speed_kmh)
                ttc_s = (
                    gap_m / (closing_kmh / 3.6)
                    if closing_kmh > 0.1 else float("inf"))
                braking_distance_m = (
                    (vs.current_speed_kmh / 3.6) ** 2
                    / (2.0 * max(vs.max_braking_mps2, 0.1)))
                if (ttc_s <= 2.0
                        or gap_m <= braking_distance_m + 1.0):
                    level = "critical"
                elif ttc_s <= 5.0:
                    level = "caution"
                else:
                    level = "observed"
                event_type = (
                    "lane_obstacle_ahead"
                    if leader.is_crashed else "vehicle_proximity_risk")
                key = (vid, leader.vehicle_id, event_type)
                current_proximity_keys.add(key)
                dedupe_key = ":".join(key)
                priority = (
                    WakePriority.CRITICAL
                    if level == "critical"
                    else WakePriority.IMPORTANT
                    if level == "caution"
                    else WakePriority.NORMAL)
                details = {
                    "vehicle_id": leader.vehicle_id,
                    "distance_m": round(gap_m, 2),
                    "speed_kmh": round(
                        leader.current_speed_kmh, 2),
                    "is_crashed_obstacle": leader.is_crashed,
                    "risk": (
                        "static_obstacle"
                        if leader.is_crashed else "route_leader"),
                    "ttc_s": (
                        round(ttc_s, 2)
                        if ttc_s != float("inf") else None),
                    "braking_distance_m": round(
                        braking_distance_m, 2),
                }
                event = self.wake_broker.transition(
                    event_type, vid, "vehicle",
                    physics_time, level, dedupe_key=dedupe_key,
                    priority=priority, source="lane_route_risk_monitor",
                    details=details,
                    related_entities=[leader.vehicle_id],
                    repeat_after_s=5.0)
                if event:
                    wake_events.append(event)
                    vehicle_events.setdefault(vid, []).append(
                        (event_type, event.as_dict()))
            for key in sorted(
                    active_proximity_keys - current_proximity_keys):
                vid, other, event_type = key
                event = self.wake_broker.transition(
                    event_type, vid, "vehicle",
                    physics_time, "clear",
                    dedupe_key=":".join(key),
                    priority=WakePriority.INFO,
                    source="lane_route_risk_monitor",
                    related_entities=[other])
                if event:
                    wake_events.append(event)
                    vehicle_events.setdefault(vid, []).append(
                        (event_type, event.as_dict()))
            active_proximity_keys = current_proximity_keys

            # Active crosswalk occupants are projected onto the exact conflict
            # point of each LLM vehicle's planned/active connector.
            current_crosswalk_keys = set()
            for vid, context in driver_contexts.items():
                for hazard in context.pedestrian_hazards:
                    ttc_s = hazard.vehicle_ttc_s
                    level = (
                        "critical" if ttc_s <= 3.0
                        else "caution" if ttc_s <= 7.0
                        else "observed")
                    key = (
                        vid, hazard.pedestrian_id,
                        hazard.crosswalk_id)
                    current_crosswalk_keys.add(key)
                    event = self.wake_broker.transition(
                        "pedestrian_crosswalk_conflict",
                        vid, "vehicle", physics_time,
                        level,
                        dedupe_key=(
                            "crosswalk:" + ":".join(key)),
                        priority=(
                            WakePriority.CRITICAL
                            if level == "critical"
                            else WakePriority.IMPORTANT),
                        source="connector_crosswalk_monitor",
                        details={
                            "pedestrian_id":
                                hazard.pedestrian_id,
                            "crosswalk_id": hazard.crosswalk_id,
                            "crossing_progress": round(
                                hazard.crossing_progress, 3),
                            "connector_id": context.connector_id,
                            "distance_to_conflict_m": round(
                                hazard.distance_to_crosswalk_m, 2),
                            "vehicle_ttc_s": (
                                round(ttc_s, 2)
                                if ttc_s != float("inf") else None),
                        },
                        related_entities=[hazard.pedestrian_id],
                        repeat_after_s=3.0)
                    if event:
                        wake_events.append(event)
                        vehicle_events.setdefault(vid, []).append(
                            ("pedestrian_crosswalk_conflict",
                             event.as_dict()))
            for key in sorted(
                    active_crosswalk_keys - current_crosswalk_keys):
                vid, ped_id, crosswalk_id = key
                event = self.wake_broker.transition(
                    "pedestrian_crosswalk_conflict", vid, "vehicle",
                    physics_time, "clear",
                    dedupe_key=(
                        "crosswalk:" + ":".join(key)),
                    priority=WakePriority.INFO,
                    source="connector_crosswalk_monitor",
                    details={"pedestrian_id": ped_id,
                             "crosswalk_id": crosswalk_id},
                    related_entities=[ped_id])
                if event:
                    wake_events.append(event)
                    vehicle_events.setdefault(vid, []).append(
                        ("pedestrian_crosswalk_conflict",
                         event.as_dict()))
            active_crosswalk_keys = current_crosswalk_keys

            # Route-lane requirement and physical road/lane blockage.
            current_route_lane_keys = set()
            current_road_block_keys = set()
            for vehicle in self.traffic_mgr.vehicles.values():
                if (not vehicle.is_llm
                        or vehicle.vehicle_id not in agent_callbacks
                        or vehicle.arrived or vehicle.route_failed or vehicle.is_crashed
                        or vehicle.vehicle_id in terminal_entities
                        or vehicle.vehicle_id
                        in pending_terminal_entities):
                    continue
                required_lane_index = vehicle.planned_from_lane_index
                required_lane_id = vehicle.planned_from_lane_id
                if (vehicle.lane_route_action_index
                        < len(vehicle.lane_route_actions)):
                    route_action = vehicle.lane_route_actions[
                        vehicle.lane_route_action_index]
                    if route_action.get("type") == "lane_change":
                        required_lane_id = route_action["to_lane_id"]
                        required_lane_index = (
                            self.traffic_mgr._lane_geometry._lane_by_id[
                                required_lane_id]["index"])
                if (required_lane_index >= 0
                        and required_lane_index != vehicle.current_lane):
                    key = f"route_lane:{vehicle.vehicle_id}"
                    current_route_lane_keys.add(key)
                    event = self.wake_broker.transition(
                        "route_lane_required", vehicle.vehicle_id,
                        "vehicle", physics_time, "required",
                        dedupe_key=key, priority=WakePriority.IMPORTANT,
                        source="lane_route_planner",
                        details={
                            "current_lane_index": vehicle.current_lane,
                            "required_lane_index": required_lane_index,
                            "required_lane_id": required_lane_id,
                            "planned_connector_id":
                                vehicle.planned_connector_id,
                            "planned_turn": vehicle.planned_turn,
                        })
                    if event:
                        wake_events.append(event)
                        vehicle_events.setdefault(
                            vehicle.vehicle_id, []).append(
                                ("route_lane_required", event.as_dict()))
                env = self.traffic_mgr._build_env_view(
                    vehicle, round(physics_time))
                blocked = bool(env.road_blocked)
                key = f"route_blocked:{vehicle.vehicle_id}"
                if blocked:
                    current_road_block_keys.add(key)
                    event = self.wake_broker.transition(
                        "route_blocked", vehicle.vehicle_id, "vehicle",
                        physics_time, "blocked", dedupe_key=key,
                        priority=WakePriority.CRITICAL,
                        source="road_condition_monitor",
                        details={
                            "segment_id": vehicle.current_segment,
                            "lane_id": vehicle.current_lane_id,
                        })
                    if event:
                        wake_events.append(event)
                        vehicle_events.setdefault(
                            vehicle.vehicle_id, []).append(
                                ("route_blocked", event.as_dict()))
            for key in sorted(
                    active_route_lane_keys - current_route_lane_keys):
                vid = key.split(":", 1)[1]
                event = self.wake_broker.transition(
                    "route_lane_required", vid, "vehicle",
                    physics_time, "clear", dedupe_key=key,
                    priority=WakePriority.INFO,
                    source="lane_route_planner")
                if event:
                    wake_events.append(event)
                    vehicle_events.setdefault(vid, []).append(
                        ("route_lane_required", event.as_dict()))
            for key in sorted(
                    active_road_block_keys - current_road_block_keys):
                vid = key.split(":", 1)[1]
                event = self.wake_broker.transition(
                    "route_blocked", vid, "vehicle",
                    physics_time, "clear", dedupe_key=key,
                    priority=WakePriority.INFO,
                    source="road_condition_monitor")
                if event:
                    wake_events.append(event)
                    vehicle_events.setdefault(vid, []).append(
                        ("route_blocked", event.as_dict()))
            active_route_lane_keys = current_route_lane_keys
            active_road_block_keys = current_road_block_keys

            # Connector conflicts use exact path-relative conflict distances
            # and TTCs from the same context supplied to driver policies.
            current_connector_conflict_keys = set()
            for vid, context in driver_contexts.items():
                for conflict in context.connector_conflicts:
                    ego_ttc = conflict.ego_ttc_s
                    other_ttc = conflict.other_ttc_s
                    finite_pair = (
                        ego_ttc != float("inf")
                        and other_ttc != float("inf"))
                    arrival_delta = (
                        abs(ego_ttc - other_ttc)
                        if finite_pair else float("inf"))
                    if (not conflict.other_in_conflict_zone
                            and min(ego_ttc, other_ttc) > 10.0):
                        continue
                    if (conflict.other_in_conflict_zone
                            or (max(ego_ttc, other_ttc) <= 3.0
                                and arrival_delta <= 2.0)):
                        level = "critical"
                    elif (max(ego_ttc, other_ttc) <= 7.0
                          and arrival_delta <= 3.0):
                        level = "caution"
                    else:
                        level = "observed"
                    key = (
                        vid, conflict.other_vehicle_id,
                        context.connector_id,
                        conflict.other_connector_id)
                    current_connector_conflict_keys.add(key)
                    event = self.wake_broker.transition(
                        "connector_conflict", vid,
                        "vehicle", physics_time, level,
                        dedupe_key=(
                            "connector_conflict:" + ":".join(key)),
                        priority=(
                            WakePriority.CRITICAL
                            if level == "critical"
                            else WakePriority.IMPORTANT
                            if level == "caution"
                            else WakePriority.NORMAL),
                        source="connector_conflict_monitor",
                        details={
                            "connector_id": context.connector_id,
                            "other_vehicle_id":
                                conflict.other_vehicle_id,
                            "other_connector_id":
                                conflict.other_connector_id,
                            "distance_to_conflict_m":
                                conflict.distance_to_conflict_m,
                            "other_distance_to_conflict_m":
                                conflict.other_distance_to_conflict_m,
                            "ego_ttc_s": (
                                round(ego_ttc, 2)
                                if ego_ttc != float("inf") else None),
                            "other_ttc_s": (
                                round(other_ttc, 2)
                                if other_ttc != float("inf") else None),
                            "other_in_conflict_zone":
                                conflict.other_in_conflict_zone,
                        },
                        related_entities=[
                            conflict.other_vehicle_id],
                        repeat_after_s=3.0)
                    if event:
                        wake_events.append(event)
                        vehicle_events.setdefault(
                            vid, []).append(
                                ("connector_conflict", event.as_dict()))
            for key in sorted(
                    active_connector_conflict_keys
                    - current_connector_conflict_keys):
                vid, other, _, _ = key
                event = self.wake_broker.transition(
                    "connector_conflict", vid, "vehicle",
                    physics_time, "clear",
                    dedupe_key=(
                        "connector_conflict:" + ":".join(key)),
                    priority=WakePriority.INFO,
                    source="connector_conflict_monitor",
                    related_entities=[other])
                if event:
                    wake_events.append(event)
                    vehicle_events.setdefault(vid, []).append(
                        ("connector_conflict", event.as_dict()))
            active_connector_conflict_keys = (
                current_connector_conflict_keys)

            # A lane-changing LLM is woken when the nearest target-lane body
            # gap crosses a meaningful level. This is a geometric trajectory
            # risk, not a global veto of the maneuver.
            current_lane_change_keys = set()
            for vid, context in driver_contexts.items():
                vehicle = self.traffic_mgr.get_state(vid)
                if (vehicle is None or not vehicle.is_changing_lane
                        or vehicle.target_lane < 0):
                    continue
                gap_m = context.lane_gaps_m.get(
                    vehicle.target_lane, float("inf"))
                if gap_m == float("inf"):
                    continue
                closing_envelope_m = (
                    vehicle.current_speed_kmh / 3.6
                    * 1.0)
                level = (
                    "critical" if gap_m <= 2.0
                    else "caution"
                    if gap_m <= max(8.0, closing_envelope_m)
                    else "observed")
                key = (vid, str(vehicle.target_lane))
                current_lane_change_keys.add(key)
                event = self.wake_broker.transition(
                    "lane_change_risk", vid, "vehicle",
                    physics_time, level,
                    dedupe_key="lane_change:" + ":".join(key),
                    priority=(
                        WakePriority.CRITICAL
                        if level == "critical"
                        else WakePriority.IMPORTANT
                        if level == "caution"
                        else WakePriority.NORMAL),
                    source="lane_change_trajectory_monitor",
                    details={
                        "source_lane_index": vehicle.current_lane,
                        "target_lane_index": vehicle.target_lane,
                        "body_gap_m": round(gap_m, 2),
                    },
                    repeat_after_s=3.0)
                if event:
                    wake_events.append(event)
                    vehicle_events.setdefault(vid, []).append(
                        ("lane_change_risk", event.as_dict()))
            for key in sorted(
                    active_lane_change_keys - current_lane_change_keys):
                vid, _ = key
                event = self.wake_broker.transition(
                    "lane_change_risk", vid, "vehicle",
                    physics_time, "clear",
                    dedupe_key="lane_change:" + ":".join(key),
                    priority=WakePriority.INFO,
                    source="lane_change_trajectory_monitor")
                if event:
                    wake_events.append(event)
                    vehicle_events.setdefault(vid, []).append(
                        ("lane_change_risk", event.as_dict()))
            active_lane_change_keys = current_lane_change_keys

            # SUMO-observed flow changes stay local to the current road.
            for vid, env in driver_environments.items():
                speed_limit = max(1.0, env.speed_limit_kmh)
                ratio = env.observed_flow_speed_kmh / speed_limit
                level = (
                    "critical" if ratio < 0.25
                    else "congested" if ratio < 0.55
                    else "slowing" if ratio < 0.8
                    else "clear")
                event = self.wake_broker.transition(
                    "traffic_flow_state", vid, "vehicle",
                    physics_time, level,
                    dedupe_key=f"traffic_flow:{vid}",
                    priority=(
                        WakePriority.IMPORTANT
                        if level in ("critical", "congested")
                        else WakePriority.NORMAL
                        if level == "slowing"
                        else WakePriority.INFO),
                    source="sumo_flow_monitor",
                    details={
                        "segment_id": env.segment_id,
                        "speed_limit_kmh": round(speed_limit, 2),
                        "observed_flow_speed_kmh": round(
                            env.observed_flow_speed_kmh, 2),
                        "speed_ratio": round(ratio, 3),
                    },
                    repeat_after_s=10.0)
                if event:
                    wake_events.append(event)
                    vehicle_events.setdefault(vid, []).append(
                        ("traffic_flow_state", event.as_dict()))

            # Vehicles approaching a pedestrian's candidate/active
            # crosswalk wake the pedestrian LLM with factual TTC data.
            current_pedestrian_risk_keys = set()
            runtime = self.traffic_mgr._lane_geometry
            for pedestrian in self.traffic_mgr.pedestrians.values():
                if (not pedestrian.is_llm
                        or pedestrian.ped_id not in agent_callbacks
                        or not pedestrian.is_spawned
                        or pedestrian.has_arrived
                        or pedestrian.is_crashed):
                    continue
                candidate_crosswalks = []
                if pedestrian.active_crosswalk_id:
                    item = runtime._crosswalk_by_id.get(
                        pedestrian.active_crosswalk_id)
                    if item:
                        candidate_crosswalks = [item]
                elif pedestrian.position.at_node:
                    junction_id = runtime._node_to_junction.get(
                        pedestrian.position.at_node,
                        pedestrian.position.at_node)
                    candidates = (
                        runtime._crosswalks_by_junction.get(
                            junction_id, []))
                    # Before crossing, monitor only the crosswalk aligned with
                    # the pedestrian's actual next route leg.
                    start = runtime.nodes_xy.get(
                        pedestrian.position.at_node)
                    end = runtime.nodes_xy.get(pedestrian.next_node)
                    if candidates and start and end:
                        route_dx = end[0] - start[0]
                        route_dy = end[1] - start[1]
                        route_norm = max(
                            1e-9, math.hypot(route_dx, route_dy))

                        def alignment(item):
                            line = item.get("centerline_xy", [])
                            if len(line) < 2:
                                return float("-inf")
                            dx = line[-1][0] - line[0][0]
                            dy = line[-1][1] - line[0][1]
                            norm = max(1e-9, math.hypot(dx, dy))
                            aligned = abs(
                                (dx * route_dx + dy * route_dy)
                                / (norm * route_norm))
                            endpoint_gap = min(
                                math.dist(start, tuple(line[0])),
                                math.dist(start, tuple(line[-1])))
                            return aligned * 20.0 - endpoint_gap

                        candidate_crosswalks = [
                            max(candidates, key=alignment)]
                    else:
                        candidate_crosswalks = list(candidates[:1])
                crosswalk_ids = {
                    item["id"] for item in candidate_crosswalks}
                connector_ids = {
                    connector_id
                    for item in candidate_crosswalks
                    for connector_id in item.get(
                        "conflicting_connectors", [])}
                for vehicle in self.traffic_mgr.vehicles.values():
                    connector_id = (
                        vehicle.active_connector_id
                        or vehicle.planned_connector_id)
                    if connector_id not in connector_ids \
                            or vehicle.arrived or vehicle.route_failed or vehicle.is_crashed:
                        continue
                    conflict = next((
                        item for item in
                        runtime.connector_crosswalk_points(connector_id)
                        if item["crosswalk_id"] in crosswalk_ids
                    ), None)
                    if conflict is None:
                        continue
                    distance_m = (
                        self.traffic_mgr._distance_to_connector_s(
                            vehicle, connector_id,
                            conflict["connector_distance_s_m"]))
                    if distance_m < -max(
                            2.0, vehicle.length_m / 2.0):
                        continue
                    speed_mps = vehicle.current_speed_kmh / 3.6
                    ttc_s = (
                        max(0.0, distance_m) / speed_mps
                        if speed_mps > 0.1 else float("inf"))
                    level = (
                        "critical" if ttc_s <= 3.0
                        else "caution" if ttc_s <= 7.0
                        else "observed")
                    key = (
                        pedestrian.ped_id,
                        vehicle.vehicle_id,
                        connector_id,
                        conflict["crosswalk_id"])
                    current_pedestrian_risk_keys.add(key)
                    event = self.wake_broker.transition(
                        "vehicle_crosswalk_risk", pedestrian.ped_id,
                        "pedestrian", physics_time, level,
                        dedupe_key=(
                            "ped_vehicle_risk:" + ":".join(key)),
                        priority=(
                            WakePriority.CRITICAL
                            if level == "critical"
                            else WakePriority.IMPORTANT),
                        source="crosswalk_conflict_monitor",
                        details={
                            "vehicle_id": vehicle.vehicle_id,
                            "connector_id": connector_id,
                            "crosswalk_id":
                                conflict["crosswalk_id"],
                            "distance_to_conflict_m": round(
                                distance_m, 2),
                            "vehicle_speed_kmh": round(
                                vehicle.current_speed_kmh, 2),
                            "ttc_s": (
                                round(ttc_s, 2)
                                if ttc_s != float("inf") else None),
                        },
                        related_entities=[vehicle.vehicle_id],
                        repeat_after_s=3.0)
                    if event:
                        wake_events.append(event)
            for key in sorted(
                    active_pedestrian_risk_keys
                    - current_pedestrian_risk_keys):
                ped_id, vehicle_id, _, _ = key
                event = self.wake_broker.transition(
                    "vehicle_crosswalk_risk", ped_id, "pedestrian",
                    physics_time, "clear",
                    dedupe_key=(
                        "ped_vehicle_risk:" + ":".join(key)),
                    priority=WakePriority.INFO,
                    source="crosswalk_conflict_monitor",
                    related_entities=[vehicle_id])
                if event:
                    wake_events.append(event)
            active_pedestrian_risk_keys = current_pedestrian_risk_keys

            # Free-walking/jaywalking pedestrians are not represented by a
            # connector-crosswalk relation. Use a short local constant-
            # velocity closest-approach monitor so they still receive risk
            # wakes before the physical collision pass.
            current_pedestrian_path_keys = set()
            current_vehicle_pedestrian_path_keys = set()
            for pedestrian in self.traffic_mgr.pedestrians.values():
                pedestrian_needs_wake = (
                    pedestrian.is_llm
                    and pedestrian.ped_id in agent_callbacks)
                if (not pedestrian.is_spawned
                        or pedestrian.has_arrived
                        or pedestrian.is_crashed
                        or pedestrian.is_on_crosswalk
                        or not pedestrian.is_walking):
                    continue
                if not pedestrian_needs_wake and not driver_contexts:
                    continue
                ped_xy = runtime.pedestrian_pose(pedestrian)
                path = pedestrian.walking_path_xy
                if ped_xy is None or len(path) < 2:
                    continue
                dx = path[-1][0] - path[0][0]
                dy = path[-1][1] - path[0][1]
                norm = max(1e-9, math.hypot(dx, dy))
                ped_velocity = (
                    dx / norm * pedestrian.speed,
                    dy / norm * pedestrian.speed)
                for vehicle in self.traffic_mgr.vehicles.values():
                    if vehicle.arrived or vehicle.route_failed or vehicle.is_crashed:
                        continue
                    if (not pedestrian_needs_wake
                            and vehicle.vehicle_id
                            not in driver_contexts):
                        continue
                    pose = runtime.vehicle_pose(vehicle)
                    if pose is None or pose[3] != 0:
                        continue
                    rx = ped_xy[0] - pose[0]
                    ry = ped_xy[1] - pose[1]
                    current_distance = math.hypot(rx, ry)
                    if current_distance > 60.0:
                        continue
                    speed_mps = vehicle.current_speed_kmh / 3.6
                    vehicle_velocity = (
                        math.cos(pose[2]) * speed_mps,
                        math.sin(pose[2]) * speed_mps)
                    rvx = ped_velocity[0] - vehicle_velocity[0]
                    rvy = ped_velocity[1] - vehicle_velocity[1]
                    rel_speed_sq = rvx * rvx + rvy * rvy
                    if rel_speed_sq <= 1e-9:
                        continue
                    closest_t = max(
                        0.0, min(
                            8.0,
                            -(rx * rvx + ry * rvy)
                            / rel_speed_sq))
                    closest_distance = math.hypot(
                        rx + rvx * closest_t,
                        ry + rvy * closest_t)
                    contact_radius = (
                        math.hypot(
                            vehicle.length_m / 2.0,
                            vehicle.width_m / 2.0)
                        + pedestrian.collision_radius_m)
                    if closest_distance > contact_radius + 1.0:
                        continue
                    level = (
                        "critical" if closest_t <= 2.0
                        else "caution" if closest_t <= 5.0
                        else "observed")
                    key = (
                        pedestrian.ped_id, vehicle.vehicle_id)
                    details = {
                        "vehicle_id": vehicle.vehicle_id,
                        "pedestrian_id": pedestrian.ped_id,
                        "current_distance_m": round(
                            current_distance, 2),
                        "closest_approach_s": round(
                            closest_t, 2),
                        "closest_distance_m": round(
                            closest_distance, 2),
                        "vehicle_speed_kmh": round(
                            vehicle.current_speed_kmh, 2),
                    }
                    if pedestrian_needs_wake:
                        current_pedestrian_path_keys.add(key)
                        event = self.wake_broker.transition(
                            "vehicle_pedestrian_path_risk",
                            pedestrian.ped_id, "pedestrian",
                            physics_time, level,
                            dedupe_key=(
                                "ped_path_risk:" + ":".join(key)),
                            priority=(
                                WakePriority.CRITICAL
                                if level == "critical"
                                else WakePriority.IMPORTANT),
                            source="local_trajectory_monitor",
                            details=details,
                            related_entities=[vehicle.vehicle_id],
                            repeat_after_s=3.0)
                        if event:
                            wake_events.append(event)
                    if vehicle.vehicle_id in driver_contexts:
                        vehicle_key = (
                            vehicle.vehicle_id, pedestrian.ped_id)
                        current_vehicle_pedestrian_path_keys.add(
                            vehicle_key)
                        event = self.wake_broker.transition(
                            "pedestrian_path_conflict",
                            vehicle.vehicle_id, "vehicle",
                            physics_time, level,
                            dedupe_key=(
                                "vehicle_ped_path_risk:"
                                + ":".join(vehicle_key)),
                            priority=(
                                WakePriority.CRITICAL
                                if level == "critical"
                                else WakePriority.IMPORTANT),
                            source="local_trajectory_monitor",
                            details=details,
                            related_entities=[pedestrian.ped_id],
                            repeat_after_s=3.0)
                        if event:
                            wake_events.append(event)
                            vehicle_events.setdefault(
                                vehicle.vehicle_id, []).append(
                                    ("pedestrian_path_conflict",
                                     event.as_dict()))
            for key in sorted(
                    active_pedestrian_path_keys
                    - current_pedestrian_path_keys):
                ped_id, vehicle_id = key
                event = self.wake_broker.transition(
                    "vehicle_pedestrian_path_risk",
                    ped_id, "pedestrian",
                    physics_time, "clear",
                    dedupe_key=(
                        "ped_path_risk:" + ":".join(key)),
                    priority=WakePriority.INFO,
                    source="local_trajectory_monitor",
                    related_entities=[vehicle_id])
                if event:
                    wake_events.append(event)
            active_pedestrian_path_keys = (
                current_pedestrian_path_keys)
            for key in sorted(
                    active_vehicle_pedestrian_path_keys
                    - current_vehicle_pedestrian_path_keys):
                vehicle_id, ped_id = key
                event = self.wake_broker.transition(
                    "pedestrian_path_conflict",
                    vehicle_id, "vehicle", physics_time, "clear",
                    dedupe_key=(
                        "vehicle_ped_path_risk:" + ":".join(key)),
                    priority=WakePriority.INFO,
                    source="local_trajectory_monitor",
                    related_entities=[ped_id])
                if event:
                    wake_events.append(event)
                    vehicle_events.setdefault(
                        vehicle_id, []).append(
                            ("pedestrian_path_conflict",
                             event.as_dict()))
            active_vehicle_pedestrian_path_keys = (
                current_vehicle_pedestrian_path_keys)

            # Pedestrian signal state changes are independent wake reasons.
            current_pedestrian_signal_keys = set()
            for pedestrian in self.traffic_mgr.pedestrians.values():
                if (not pedestrian.is_llm
                        or pedestrian.ped_id not in agent_callbacks
                        or not pedestrian.is_spawned
                        or pedestrian.has_arrived
                        or pedestrian.is_crashed):
                    continue
                node_id = (
                    pedestrian.position.at_node
                    or pedestrian.position.crossing_to)
                if not node_id:
                    continue
                signal = runtime.pedestrian_signal_state(
                    node_id, physics_time)
                if signal is None:
                    continue
                level = (
                    "ped_green"
                    if signal.pedestrian_green else "ped_red")
                key = (pedestrian.ped_id, node_id)
                current_pedestrian_signal_keys.add(key)
                event = self.wake_broker.transition(
                    "pedestrian_signal_state",
                    pedestrian.ped_id, "pedestrian",
                    physics_time, level,
                    dedupe_key="ped_signal:" + ":".join(key),
                    priority=WakePriority.IMPORTANT,
                    source="pedestrian_signal_monitor",
                    details={
                        "node_id": node_id,
                        "signal": level,
                        "remaining_seconds": round(
                            signal.remaining_seconds, 2),
                    })
                if event:
                    wake_events.append(event)
            for key in sorted(
                    active_pedestrian_signal_keys
                    - current_pedestrian_signal_keys):
                ped_id, node_id = key
                event = self.wake_broker.transition(
                    "pedestrian_signal_state",
                    ped_id, "pedestrian",
                    physics_time, "clear",
                    dedupe_key="ped_signal:" + ":".join(key),
                    priority=WakePriority.INFO,
                    source="pedestrian_signal_monitor",
                    details={"node_id": node_id})
                if event:
                    wake_events.append(event)
            active_pedestrian_signal_keys = (
                current_pedestrian_signal_keys)
            # Evaluate only new signal transitions. This scales with actual
            # lamp changes instead of scanning every entity pair each step.
            signal_events = self.traffic_mgr.signal_event_log[
                processed_signal_event_count:]
            processed_signal_event_count = len(
                self.traffic_mgr.signal_event_log)
            for signal_event in signal_events:
                source_id = signal_event["vehicle_id"]
                for entity_id in agent_callbacks:
                    if (entity_id == source_id
                            or entity_id not in started_agent_entities
                            or entity_id in terminal_entities):
                        continue
                    detection = self.traffic_mgr.perception_model.detect_entity(
                        entity_id, source_id)
                    if detection is None:
                        continue
                    visible_changed = {
                        key: value
                        for key, value in signal_event["changed"].items()
                        if key in detection.observed_signals
                    }
                    if not visible_changed:
                        continue
                    entity_type = (
                        "pedestrian"
                        if entity_id in self.traffic_mgr.pedestrians
                        else "vehicle")
                    wake_events.append(self.wake_broker.discrete(
                        "vehicle_signal_change", entity_id, entity_type,
                        float(signal_event["time_s"]),
                        detected_at_s=physics_time,
                        priority=WakePriority.NORMAL,
                        source="optical_perception_model",
                        details={
                            "source_id": source_id,
                            "changed": visible_changed,
                            "observed_signals": detection.observed_signals,
                            "distance_m": detection.distance_m,
                            "bearing_deg": detection.bearing_deg,
                            "confidence": detection.confidence,
                        }))
            # Horn pulses are physical emissions. At the next fixed world
            # boundary, only entities whose acoustic profile can hear them
            # receive a factual wake event. Short pulses are not lost between
            # 0.1-second boundaries because the detector covers one step.
            for entity_id in agent_callbacks:
                if (entity_id not in started_agent_entities
                        or entity_id in terminal_entities):
                    continue
                for cue in self.traffic_mgr.perception_model.heard_horns(
                        entity_id, physics_time, lookback_s=step_s):
                    key = (entity_id, cue["event_id"])
                    if key in delivered_horn_event_keys:
                        continue
                    delivered_horn_event_keys.add(key)
                    entity_type = (
                        "pedestrian"
                        if entity_id in self.traffic_mgr.pedestrians
                        else "vehicle")
                    wake_events.append(self.wake_broker.discrete(
                        "acoustic_cue", entity_id, entity_type,
                        float(cue["start_time_s"]),
                        detected_at_s=physics_time,
                        priority=WakePriority.IMPORTANT,
                        source="acoustic_perception_model",
                        details=cue))
            # Passenger judging has its own simulation-time deadline.  It is
            # intentionally independent of a model-selected heartbeat and
            # therefore cannot be postponed by reducing Driving Agent wakes.
            for entity_id, callback in agent_callbacks.items():
                if (entity_id not in started_agent_entities
                        or entity_id in terminal_entities):
                    continue
                due_at_s = getattr(
                    callback, "pending_judge_due_at_s", None)
                if due_at_s is None or physics_time < float(due_at_s) - 1e-9:
                    continue
                wake_events.append(self.wake_broker.discrete(
                    "passenger_judge_due", entity_id, "vehicle",
                    float(due_at_s), detected_at_s=physics_time,
                    priority=WakePriority.NORMAL,
                    source="passenger_judge_timer",
                    details={
                        "judge_due_at_s": float(due_at_s),
                        "observation_window_s": float(getattr(
                            callback, "judge_window_s", 1.0)),
                    }))
            wake_events_by_entity = group_wake_events(wake_events)

            # ── Decide whether to wake LLM agents ────────────────
            # The time-limit event is already a lifecycle wake. Emitting a
            # heartbeat at the same boundary duplicates context and used to
            # trigger PA/Driver calls whose only valid answer was ``finish``.
            at_time_limit = physics_time >= total_time - 1e-9
            scheduled_wake_entities = set()
            for entity_id, callback in agent_callbacks.items():
                if (at_time_limit or entity_id in terminal_entities
                        or entity_id in pending_terminal_entities):
                    continue
                callback_state = getattr(callback, "_state", None)
                if not isinstance(callback_state, dict):
                    continue
                scheduled_at = callback_state.get("scheduled_wake_at_s")
                try:
                    scheduled_at = float(scheduled_at)
                except (TypeError, ValueError):
                    continue
                if physics_time >= scheduled_at - 1e-9:
                    scheduled_wake_entities.add(entity_id)
            heartbeat_entities = {
                entity_id for entity_id, wake_time
                in next_heartbeat.items()
                if not at_time_limit
                and entity_id not in terminal_entities
                and entity_id not in pending_terminal_entities
                and physics_time >= wake_time - 1e-9}
            is_heartbeat = bool(heartbeat_entities)
            clock_wake_entities = (
                heartbeat_entities | scheduled_wake_entities)
            for entity_id in sorted(clock_wake_entities):
                entity_type = (
                    "pedestrian"
                    if entity_id in self.traffic_mgr.pedestrians
                    else "vehicle")
                callback_state = getattr(
                    agent_callbacks.get(entity_id), "_state", None)
                scheduled = entity_id in scheduled_wake_entities
                details = {}
                if scheduled and isinstance(callback_state, dict):
                    details = {
                        "scheduled_for_s": callback_state.get(
                            "scheduled_wake_at_s"),
                        "reason": callback_state.get(
                            "scheduled_wake_reason", ""),
                    }
                    # Consume before the callback so a schedule made during
                    # this wake is unambiguously a new one-shot observation.
                    callback_state["scheduled_wake_at_s"] = None
                    callback_state["scheduled_wake_reason"] = ""
                wake_events.append(self.wake_broker.discrete(
                    "scheduled_wake" if scheduled else "heartbeat",
                    entity_id, entity_type, physics_time,
                    priority=WakePriority.INFO,
                    source=("model_scheduled_wake" if scheduled
                            else "model_heartbeat"),
                    details=details))
            cur_weather = self._get_weather_at(physics_time)
            cur_daynight = self._get_daynight_at(physics_time)
            weather_changed = (cur_weather is not None
                               and (prev_weather is None
                                    or cur_weather.condition != prev_weather.condition))
            daynight_changed = (cur_daynight is not None
                                and (prev_daynight is None
                                     or cur_daynight.period != prev_daynight.period))

            if weather_changed:
                event_type = (
                    "weather_initialized" if prev_weather is None
                    else "weather_changed")
                for entity_id in agent_callbacks:
                    if (entity_id not in started_agent_entities
                            or entity_id in terminal_entities
                            or self._entity_is_physically_terminal(
                                entity_id)):
                        continue
                    entity_type = (
                        "pedestrian"
                        if entity_id in self.traffic_mgr.pedestrians
                        else "vehicle")
                    wake_events.append(self.wake_broker.discrete(
                        event_type, entity_id, entity_type, physics_time,
                        priority=WakePriority.NORMAL,
                        source="weather_timeline",
                        details={
                            "condition": cur_weather.condition,
                            "previous_condition": (
                                prev_weather.condition
                                if prev_weather else None),
                        }))
            if daynight_changed:
                event_type = (
                    "daynight_initialized" if prev_daynight is None
                    else "daynight_changed")
                for entity_id in agent_callbacks:
                    if (entity_id not in started_agent_entities
                            or entity_id in terminal_entities):
                        continue
                    entity_type = (
                        "pedestrian"
                        if entity_id in self.traffic_mgr.pedestrians
                        else "vehicle")
                    wake_events.append(self.wake_broker.discrete(
                        event_type, entity_id, entity_type, physics_time,
                        priority=WakePriority.NORMAL,
                        source="daynight_timeline",
                        details={
                            "period": cur_daynight.period,
                            "daylight_level": DayNight._DAYLIGHT_MAP.get(
                                DayNight.TimePeriod(cur_daynight.period), 50),
                            "is_dark": cur_daynight.period in {"dawn", "dusk", "night"},
                            "previous_period": (
                                prev_daynight.period
                                if prev_daynight else None),
                        }))
            if (physics_time >= total_time - 1e-9
                    and not simulation_end_emitted):
                for entity_id in agent_callbacks:
                    if (entity_id not in started_agent_entities
                            or entity_id in terminal_entities
                            or self._entity_is_physically_terminal(
                                entity_id)):
                        continue
                    entity_type = (
                        "pedestrian"
                        if entity_id in self.traffic_mgr.pedestrians
                        else "vehicle")
                    wake_events.append(self.wake_broker.discrete(
                        "simulation_ended", entity_id, entity_type,
                        physics_time, priority=WakePriority.IMPORTANT,
                        source="multi_sim_engine",
                        details={
                            "reason": "time_limit",
                            "total_time_s": total_time,
                        }))
                simulation_end_emitted = True
            # Passenger clocks and state-edge detectors run at physics
            # boundaries, not driver heartbeats. Shared raw events are filtered
            # by the passenger subscriber before any PA-visible event is emitted.
            for entity_id, callback in agent_callbacks.items():
                poll = getattr(callback, "poll_personal_events", None)
                if (poll is None or entity_id not in started_agent_entities
                        or entity_id in terminal_entities
                        or entity_id in pending_terminal_entities
                        or physics_time >= total_time - 1e-9):
                    continue
                state = self.traffic_mgr.get_state(entity_id)
                if state is None:
                    continue
                waiting_conditions = {
                    runtime["spec"]["condition"]
                    for pending in getattr(callback, "pending_requests", {}).values()
                    if (runtime := pending.get("trigger_runtime"))
                    and runtime.get("status") == "waiting"
                }
                trigger_context = (
                    self._passenger_trigger_context(
                        entity_id, physics_time, include_future=False,
                        conditions=waiting_conditions)
                    if waiting_conditions else {})
                details = poll(physics_time, {
                    "current_speed_kmh": state.current_speed_kmh,
                    "acceleration_mps2": state.acceleration_mps2,
                    "current_lane_id": state.current_lane_id,
                    "current_segment": state.current_segment,
                    "active_connector_id": state.active_connector_id,
                    "planned_connector_id": state.planned_connector_id,
                    "next_intersection_distance_m": trigger_context.get(
                        "next_intersection_distance_m"),
                    "remaining_distance_m": trigger_context.get(
                        "remaining_distance_m"),
                    "arrived": state.arrived, "is_crashed": state.is_crashed,
                    "route_failed": state.route_failed,
                }, [event.as_dict() for event in wake_events
                    if event.entity_id == entity_id], episode_id=self.scenario.scenario_id)
                if details:
                    wake_events.append(self.wake_broker.discrete(
                        "personal_agent_due", entity_id, "vehicle", physics_time,
                        source="passenger_event_random_scheduler", details=details))
            naturally_waking = {
                event.entity_id for event in wake_events
                if self._event_can_wake_agent(event, agent_callbacks)
            }
            for entity_id in sorted(naturally_waking):
                receipts = deferred_success_receipts.pop(entity_id, [])
                entity_type = (
                    "pedestrian"
                    if entity_id in self.traffic_mgr.pedestrians
                    else "vehicle")
                for receipt in receipts:
                    wake_events.append(self.wake_broker.discrete(
                        "command_result", entity_id, entity_type,
                        float(receipt["committed_at_s"]),
                        detected_at_s=physics_time,
                        priority=WakePriority.INFO,
                        source="agent_command_commit_piggyback",
                        details=receipt))
            deliverable_wake_events = [
                event for event in wake_events
                if self._event_can_wake_agent(event, agent_callbacks)]
            wake_events_by_entity = group_wake_events(
                deliverable_wake_events)

            has_event = bool(deliverable_wake_events)
            if not has_event and not is_heartbeat:
                if (self.scenario.stop_when_all_vehicles_terminal
                        and self._all_scenario_vehicles_terminal()):
                    break
                if physics_time >= total_time - 1e-9:
                    break
                continue

            # ── Agent wake ────────────────────────────────────────
            self._sync_global_world(physics_time)
            self._sync_vehicle_external_signals(physics_time)

            # Per-wake subclass hook (used by Tracked for position/traffic
            # logs at heartbeat).
            self._log_per_wake(physics_time, tick_index, is_heartbeat,
                               trigger_events)

            for entity_id in (
                    heartbeat_entities
                    | set(wake_events_by_entity)):
                if (entity_id not in heartbeat_intervals
                        or entity_id in pending_terminal_entities):
                    continue
                entity_events = wake_events_by_entity.get(entity_id, [])
                if (entity_id not in heartbeat_entities
                        and entity_events
                        and all(event.event_type in {
                            "passenger_judge_due", "scheduled_wake",
                        } for event in entity_events)):
                    # Evaluator timers and additive one-shot observations must
                    # not shift the persistent heartbeat schedule.
                    continue
                interval = self._heartbeat_interval_for(
                    entity_id, heartbeat_intervals[entity_id],
                    agent_callbacks)
                next_heartbeat[entity_id] = physics_time + interval

            self._agent_command_queue = []
            self._collect_agent_commands = True

            # ── Per-vehicle processing ────────────────────────────
            def process_vehicle(vcfg):
                nonlocal fatal_agent_callback
                vid = vcfg.vehicle_id
                vw = self._vw[vid]
                expect_vw = self._expect_vw[vid]
                vs = self.traffic_mgr.get_state(vid)
                callback = agent_callbacks.get(vid)

                # Track arrival + crash from trigger events
                vevents = vehicle_events.get(vid, [])
                for etype, event_details in vevents:
                    if etype == "arrived":
                        vehicle_results[vid].arrived = True
                        vehicle_results[vid].arrival_tick = tick_index
                        if vehicle_results[vid].arrival_time_s is None:
                            vehicle_results[vid].arrival_time_s = physics_time
                        vehicle_results[vid].arrival_completion = (
                            event_details.get("completion"))
                        vehicle_results[vid].terminal_crossing_speed_kmh = (
                            event_details.get(
                                "terminal_crossing_speed_kmh"))
                        vehicle_results[
                            vid].physically_stopped_at_destination = (
                            event_details.get(
                                    "physically_stopped_at_destination"))

                # Arrival/crash is a physical terminal boundary.  Record the
                # event for audit, but never purchase another Driving Agent
                # call or accept a post-terminal driving command.
                if vid in pending_terminal_entities:
                    entity_wake_events = wake_events_by_entity.get(vid, [])
                    finalize_passenger = getattr(callback, "finalize_passenger", None)
                    if finalize_passenger is not None:
                        finalize_passenger(vw, physics_time, self._memory[vid], tick_index,
                            _vehicle_state=FrozenStateView.from_state(vs),
                            _wake_events=[event.as_dict() for event in entity_wake_events],
                            _recent_motion=self._recent_vehicle_motion(vid, physics_time, window_s=30.0),
                            _passenger_judge_world=self._passenger_judge_world_snapshot(vid),
                            _episode_total_time_s=total_time,
                            _navigation_status=(
                                self.traffic_mgr.get_navigation_status(vid)))
                    if entity_wake_events:
                        self.wake_broker.record_delivery(
                            vid, entity_wake_events, physics_time,
                            status="terminal_no_callback", actions=[])
                    return

                # Skip vehicles with no relevant events (unless heartbeat).
                # Non-evaluated SUMO actors without callbacks never get woken.
                entity_wake_events = wake_events_by_entity.get(vid, [])
                if self._driving_process_scorer is not None:
                    self._driving_process_scorer.record_delivered_events(
                        vid,
                        [
                            event.as_dict(delivered_at_s=physics_time)
                            for event in entity_wake_events
                        ],
                        physics_time,
                    )
                has_vehicle_event = bool(entity_wake_events)
                entity_heartbeat = vid in heartbeat_entities
                judge_only_wake = bool(
                    not entity_heartbeat and entity_wake_events
                    and all(event.event_type == "passenger_judge_due"
                            for event in entity_wake_events))
                # Weather/day-night changes have already been converted to
                # entity-scoped WakeEvents above.  Do not use the global
                # flags here: doing so would re-wake terminal or not-yet-
                # spawned agents that deliberately received no event.
                if not entity_heartbeat and not has_vehicle_event:
                    return
                if not vcfg.is_evaluated and not callback:
                    return

                self._sync_map_to_vehicle(vid, tick_index)
                passenger_messages = []
                npc_rule_actions = list(
                    self._pending_npc_equipment_actions.pop(vid, []))

                # Judge-only timer wakes are evaluator infrastructure, not a
                # new task checkpoint or a Driving Agent control opportunity.
                if judge_only_wake:
                    gt_lines = []
                    acceptable_actions = []
                    trend_tolerances = []
                    negative_checks = []
                    pre_agent_vw = None
                else:
                    (gt_lines, cur_snapshot, acceptable_actions,
                     trend_tolerances) = derive_ground_truth(
                        vw=vw,
                        prev_snapshot=self._prev_snapshot[vid],
                        vehicle_state=vs,
                    )
                    self._prev_snapshot[vid] = cur_snapshot
                    self._gt_lines_by_tick[vid][physics_time] = list(gt_lines)
                    negative_checks = derive_negative_checks(
                        cur_snapshot, passenger_messages, vw=vw)
                    pre_agent_vw = (
                        copy.deepcopy(vw) if vcfg.is_evaluated else None)

                # ── Wake agent ───────────────────────────────────
                callback_error = ""
                if callback:
                    core_world = WorldState(
                        traffic_mgr=self.traffic_mgr,
                        road_network=self.road_network,
                        weather=cur_weather,
                        daynight=cur_daynight,
                        tick=tick_index,
                        time_s=physics_time,
                    )
                    world_state = self._agent_world_view(
                        vid, core_world)
                    command_buffer = _WAKE_COMMAND_BUFFER.get()[1]
                    queue_checkpoint = len(command_buffer)
                    try:
                        camera_visual = None
                        lidar_visual = None
                        if (not judge_only_wake
                                and (
                                    getattr(callback, "_agent_type", None)
                                    == "llm"
                                    or vcfg.agent_type == "llm")):
                            if self._camera_visual_renderer is None:
                                from visualization.web3d_camera import (
                                    Web3DCameraRenderer)
                                self._camera_visual_renderer = (
                                    Web3DCameraRenderer())
                            if vw.has_module("lidar"):
                                camera_visual, lidar_visual = on_simulation_owner(
                                    self._camera_visual_renderer.render_observations,
                                    self, vid, physics_time, tick_index,
                                    getattr(vw.lidar, "_overrides", {}))
                            else:
                                camera_visual = on_simulation_owner(
                                    self._camera_visual_renderer.render,
                                    self, vid, physics_time, tick_index)
                        def emit_personal_update(details):
                            event = self.wake_broker.discrete(
                                "personal_agent_update", vid, "vehicle", physics_time,
                                source="personal_agent", details=details)
                            self.wake_broker.record_delivery(
                                vid, [event], physics_time, status="delivered_to_driver")
                            return event.as_dict(delivered_at_s=physics_time)

                        callback_actions = self._normalize_agent_actions(callback(
                            vw, physics_time, passenger_messages,
                            self._memory[vid], tick_index,
                            _destination_name=vcfg.destination_name,
                            _destination_node=vcfg.destination_node,
                            _vehicle_state=(
                                FrozenStateView.from_state(vs)),
                            _road_network=self.road_network,
                            _camera_visual=camera_visual,
                            _lidar_visual=lidar_visual,
                            _render_navigation_minimap=(
                                self._agent_visual_renderer.render_minimap),
                            world_state=world_state,
                            _wake_events=[
                                event.as_dict(
                                    delivered_at_s=physics_time)
                                for event in entity_wake_events
                            ],
                            _recent_motion=self._recent_vehicle_motion(
                                vid, physics_time, window_s=30.0),
                            _passenger_judge_world=(
                                self._passenger_judge_world_snapshot(vid)),
                            _episode_total_time_s=total_time,
                            _navigation_status=(
                                self.traffic_mgr.get_navigation_status(vid)),
                            _trigger_context=(
                                self._passenger_trigger_context(
                                    vid, physics_time, include_future=True)),
                            _emit_personal_update=emit_personal_update,
                        ))
                        actions = npc_rule_actions + callback_actions
                    except Exception as exc:
                        del command_buffer[queue_checkpoint:]
                        callback_error = (
                            f"{type(exc).__name__}: {exc}")
                        actions = []
                        self.agent_callback_errors.append({
                            "entity_id": vid,
                            "time_s": physics_time,
                            "phase": "callback",
                            "error": callback_error,
                        })
                        self.wake_broker.discrete(
                            "agent_callback_failed", vid,
                            "vehicle", physics_time,
                            priority=WakePriority.CRITICAL,
                            source="multi_sim_engine",
                            details={"error": callback_error})
                        fatal_agent_callback = True
                else:
                    actions = list(npc_rule_actions)
                self.wake_broker.record_delivery(
                    vid, entity_wake_events, physics_time,
                    status=(
                        "failed" if callback_error else
                        "delivered" if callback else "no_callback"),
                    actions=list(actions),
                    error=callback_error)

                vehicle_results[vid].tick_interactions.append({
                    "tick_index": tick_index,
                    "time_s": round(physics_time, 4),
                    "passenger_messages": passenger_messages,
                    "actions_taken": actions,
                    "gt_lines": list(gt_lines),
                    "callback_error": callback_error,
                })

                # ── Checkpoint verification ──────────────────────
                if vcfg.is_evaluated and pre_agent_vw is not None:
                    cp_result = self._verify_vehicle_checkpoint(
                        vid, physics_time, tick_index, pre_agent_vw, gt_lines,
                        acceptable_actions=acceptable_actions,
                        trend_tolerances=trend_tolerances,
                        negative_checks=negative_checks,
                    )
                    if cp_result:
                        vehicle_results[vid].checkpoints.append(cp_result)

            # Each cooperative callback owns a command buffer. Model requests
            # overlap, but SUMO and tools run on this thread at frozen time.
            # Completion order never controls physical command commit order.
            vehicle_command_buffers = [[] for _ in self.scenario.vehicles]

            def vehicle_job(vcfg, commands):
                token = _WAKE_COMMAND_BUFFER.set((self, commands))
                try:
                    return process_vehicle(vcfg)
                finally:
                    _WAKE_COMMAND_BUFFER.reset(token)

            self._model_scheduler.run([
                lambda vcfg=vcfg, commands=commands: vehicle_job(vcfg, commands)
                for vcfg, commands in zip(self.scenario.vehicles, vehicle_command_buffers)
            ])
            self._agent_command_queue.extend(
                command for commands in vehicle_command_buffers for command in commands)

            # One exhausted callback request makes the run an infrastructure
            # failure. Continuing the simulated world would repeatedly call
            # the same unavailable provider and produce a meaningless
            # no-action trajectory, so terminate this run immediately.
            if fatal_agent_callback:
                # Failed callbacks were rolled back to their queue
                # checkpoint. Preserve commands from other agents that made
                # valid decisions against the same frozen world snapshot.
                self._commit_agent_commands(physics_time)
                self._sync_vehicle_external_signals(physics_time)
                self._collect_agent_commands = False
                break

            # ── Per-pedestrian LLM processing ───────────────────
            for pcfg in self.scenario.pedestrians:
                if pcfg.agent_type != "llm":
                    continue
                pedestrian = self.traffic_mgr.pedestrians.get(pcfg.ped_id)
                callback = agent_callbacks.get(pcfg.ped_id)
                if not pedestrian or not callback:
                    continue
                entity_events = wake_events_by_entity.get(pcfg.ped_id, [])
                terminal_event = any(
                    event.event_type in (
                        "ped_arrived", "collision",
                        "pedestrian_incapacitated",
                        "simulation_ended")
                    for event in entity_events)
                if pedestrian.has_arrived and not terminal_event:
                    continue
                entity_heartbeat = pcfg.ped_id in heartbeat_entities
                if not entity_heartbeat and not entity_events:
                    continue
                if physics_time + 1e-9 < pedestrian.start_time:
                    continue
                wake_event_payloads = [
                    event.as_dict(delivered_at_s=physics_time)
                    for event in entity_events]
                core_world = WorldState(
                    traffic_mgr=self.traffic_mgr,
                    road_network=self.road_network,
                    weather=cur_weather,
                    daynight=cur_daynight,
                    tick=tick_index,
                    time_s=physics_time,
                )
                pedestrian_world = self._agent_world_view(
                    pcfg.ped_id, core_world)
                queue_checkpoint = len(self._agent_command_queue)
                callback_error = ""
                try:
                    actions = self._normalize_agent_actions(callback(
                        FrozenStateView.from_state(pedestrian),
                        physics_time, wake_event_payloads,
                        self._memory[pcfg.ped_id], tick_index,
                        pedestrian_world))
                except Exception as exc:
                    del self._agent_command_queue[
                        queue_checkpoint:]
                    callback_error = (
                        f"{type(exc).__name__}: {exc}")
                    actions = []
                    self.agent_callback_errors.append({
                        "entity_id": pcfg.ped_id,
                        "time_s": physics_time,
                        "phase": "callback",
                        "error": callback_error,
                    })
                    self.wake_broker.discrete(
                        "agent_callback_failed", pcfg.ped_id,
                        "pedestrian", physics_time,
                        priority=WakePriority.CRITICAL,
                        source="multi_sim_engine",
                        details={"error": callback_error})
                    fatal_agent_callback = True
                self.wake_broker.record_delivery(
                    pcfg.ped_id, entity_events,
                    physics_time,
                    status=(
                        "failed" if callback_error
                        else "delivered"),
                    actions=list(actions or []),
                    error=callback_error)
                pedestrian_results[
                    pcfg.ped_id].tick_interactions.append({
                    "tick_index": tick_index,
                    "time_s": round(physics_time, 4),
                    "wake_events": wake_event_payloads,
                    "actions_taken": actions or [],
                    "callback_error": callback_error,
                })

            if fatal_agent_callback:
                self._commit_agent_commands(physics_time)
                self._sync_vehicle_external_signals(physics_time)
                self._collect_agent_commands = False
                break

            if (simulation_end_emitted
                    and physics_time >= total_time - 1e-9):
                # ``simulation_ended`` is a notification, not another
                # control opportunity.  Commands returned by that final
                # callback must not mutate the completed world.
                self._agent_command_queue = []
                self._collect_agent_commands = False
            else:
                self._commit_agent_commands(physics_time)
                self._sync_vehicle_external_signals(physics_time)

                # The model owns its periodic observation cadence. External
                # events and Todo deadlines may still wake it earlier.
                cadence_woken_entities = (
                    heartbeat_entities | set(wake_events_by_entity))
                cadence_woken_entities = {
                    entity_id for entity_id in cadence_woken_entities
                    if entity_id in heartbeat_entities or any(
                        event.event_type not in {
                            "passenger_judge_due", "scheduled_wake",
                        }
                        for event in wake_events_by_entity.get(entity_id, []))}
                for entity_id in agent_callbacks:
                    if (entity_id not in heartbeat_intervals
                            or entity_id in pending_terminal_entities):
                        continue
                    callback_state = getattr(
                        agent_callbacks[entity_id], "_state", None)
                    model_selected = (
                        isinstance(callback_state, dict)
                        and callback_state.get(
                            "heartbeat_interval_s") is not None)
                    interval = self._heartbeat_interval_for(
                        entity_id, heartbeat_intervals[entity_id],
                        agent_callbacks)
                    candidate = physics_time + interval
                    scheduled = next_heartbeat.get(entity_id)
                    if (model_selected
                            and entity_id in cadence_woken_entities):
                        next_heartbeat[entity_id] = candidate
                    elif scheduled is None or candidate < scheduled:
                        next_heartbeat[entity_id] = candidate

            # A model-owned Todo deadline is a wake-up deadline, not a
            # driving command. LLM callbacks expose only their nearest future
            # deadline; the fixed-step scheduler wakes the entity at the
            # first physics boundary at or after it. Todo contents remain
            # private to the callback and are never interpreted by physics.
            for entity_id, callback in agent_callbacks.items():
                if entity_id in pending_terminal_entities:
                    continue
                callback_state = getattr(callback, "_state", None)
                if not isinstance(callback_state, dict):
                    continue
                todo_deadline = callback_state.get(
                    "next_todo_deadline_s")
                if todo_deadline is None:
                    continue
                try:
                    todo_deadline = float(todo_deadline)
                except (TypeError, ValueError):
                    continue
                if todo_deadline <= physics_time + 1e-9:
                    continue
                scheduled = next_heartbeat.get(entity_id)
                if scheduled is None or todo_deadline < scheduled:
                    next_heartbeat[entity_id] = todo_deadline
            for entity_id in pending_terminal_entities:
                terminal_entities.add(entity_id)
                next_heartbeat.pop(entity_id, None)
            prev_weather = cur_weather
            prev_daynight = cur_daynight
            tick_index += 1

            if (self.scenario.stop_when_all_vehicles_terminal
                    and self._all_scenario_vehicles_terminal()):
                break

            if physics_time >= total_time - 1e-9:
                break

        result.total_ticks = tick_index
        self._finalize_result(result)
        result.physics_engine = copy.deepcopy(
            getattr(self.traffic_mgr, "physics_metadata", {
                "name": "sumo",
            }))
        return result

    def _all_scenario_vehicles_terminal(self) -> bool:
        """Whether every required vehicle has ended physically."""
        required_ids = (
            list(self.scenario.terminal_vehicle_ids)
            if self.scenario.terminal_vehicle_ids
            else [item.vehicle_id for item in self.scenario.vehicles])
        if not required_ids:
            return False
        for vehicle_id in required_ids:
            vehicle = self.traffic_mgr.get_state(vehicle_id)
            if vehicle is None or not (vehicle.arrived or vehicle.route_failed or vehicle.is_crashed):
                return False
        return True

    # ── SUMO pedestrian polling ───────────────────────────────────

    def _apply_initial_physical_states(self) -> None:
        """Apply validated lane/crosswalk placements for controlled scenes."""
        runtime = self.traffic_mgr._lane_geometry
        for config in self.scenario.vehicles:
            placement = dict(config.initial_physical_state or {})
            if not placement:
                continue
            vehicle = self.traffic_mgr.get_state(config.vehicle_id)
            if vehicle is None:
                raise ValueError(f"unknown placed vehicle {config.vehicle_id}")
            # Registration may have prepared a transition for its generic
            # spawn lane. Exact placement replaces that physical state, so
            # every cached transition derived from the spawn lane must be
            # invalidated and rebuilt from the authored route below.
            vehicle.planned_connector_id = ""
            vehicle.planned_from_lane_id = ""
            vehicle.planned_from_lane_index = -1
            vehicle.planned_turn = ""
            lane_id = str(placement.get("lane_id", ""))
            connector_id = str(placement.get("connector_id", ""))
            if lane_id:
                lane = runtime._lane_by_id.get(lane_id)
                if lane is None:
                    raise ValueError(
                        f"unknown initial lane_id {lane_id!r} for "
                        f"{config.vehicle_id}")
                vehicle.current_lane_id = lane_id
                vehicle.current_lane = int(lane["index"])
                vehicle.current_segment = lane["segment_id"]
                vehicle.current_node = lane["start_node"]
                vehicle.active_connector_id = ""
                vehicle.active_connector_from_lane_id = ""
                vehicle.active_connector_to_lane_id = ""
            if connector_id:
                connector = runtime.connector_record(connector_id)
                if connector is None:
                    raise ValueError(
                        f"unknown initial connector_id {connector_id!r} for "
                        f"{config.vehicle_id}")
                source = runtime._lane_by_id[connector["from_lane"]]
                vehicle.current_lane_id = source["id"]
                vehicle.current_lane = int(source["index"])
                vehicle.current_segment = source["segment_id"]
                vehicle.current_node = source["start_node"]
                vehicle.active_connector_id = connector_id
                vehicle.active_connector_from_lane_id = connector["from_lane"]
                vehicle.active_connector_to_lane_id = connector["to_lane"]
                vehicle.lane_route_actions = [{
                    "type": "connector",
                    "connector_id": connector_id,
                    "from_lane_id": connector["from_lane"],
                    "to_lane_id": connector["to_lane"],
                    "turn": connector["turn"],
                }]
                vehicle.lane_route_action_index = 0
            progress = float(placement.get("progress", vehicle.edge_progress))
            if not 0.0 <= progress <= 1.0:
                raise ValueError(
                    f"initial progress outside [0,1] for {config.vehicle_id}")
            vehicle.edge_progress = progress
            vehicle.current_speed_kmh = max(
                0.0, float(placement.get("speed_kmh", 0.0)))
            vehicle.target_speed_kmh = max(
                0.0, float(placement.get(
                    "target_speed_kmh", vehicle.current_speed_kmh)))
            vehicle.desired_speed_kmh = float(placement.get(
                "desired_speed_kmh", vehicle.desired_speed_kmh))
            vehicle.is_stopped = bool(placement.get("stopped", False))
            target_lane_id = str(
                placement.get("lane_change_target_lane_id", ""))
            if target_lane_id:
                target_lane = runtime._lane_by_id.get(target_lane_id)
                source_lane = runtime._lane_by_id.get(vehicle.current_lane_id)
                if (target_lane is None or source_lane is None
                        or target_lane["segment_id"]
                        != source_lane["segment_id"]
                        or target_lane["direction"]
                        != source_lane["direction"]):
                    raise ValueError(
                        f"invalid initial lane-change target {target_lane_id!r} "
                        f"for {config.vehicle_id}")
                lane_change_progress = float(
                    placement.get("lane_change_progress", 0.5))
                if not 0.0 <= lane_change_progress <= 1.0:
                    raise ValueError(
                        f"lane-change progress outside [0,1] for "
                        f"{config.vehicle_id}")
                vehicle.is_changing_lane = True
                vehicle.target_lane = int(target_lane["index"])
                vehicle.lane_change_progress = lane_change_progress
            route_actions = placement.get("lane_route_actions")
            if route_actions is not None:
                if not isinstance(route_actions, list):
                    raise ValueError(
                        f"lane_route_actions must be a list for "
                        f"{config.vehicle_id}")
                cursor_lane_id = (
                    vehicle.active_connector_to_lane_id
                    if vehicle.active_connector_id
                    else vehicle.current_lane_id)
                normalized_actions = []
                for index, raw_action in enumerate(route_actions):
                    action = dict(raw_action)
                    action_type = action.get("type")
                    if action_type == "connector":
                        connector = runtime.connector_record(
                            str(action.get("connector_id", "")))
                        if connector is None:
                            raise ValueError(
                                f"unknown route connector at index {index} "
                                f"for {config.vehicle_id}")
                        if connector["from_lane"] != cursor_lane_id:
                            raise ValueError(
                                f"disconnected route connector at index "
                                f"{index} for {config.vehicle_id}")
                        action.update({
                            "from_lane_id": connector["from_lane"],
                            "to_lane_id": connector["to_lane"],
                            "turn": connector["turn"],
                        })
                        cursor_lane_id = connector["to_lane"]
                    elif action_type == "lane_change":
                        target_id = str(action.get("to_lane_id", ""))
                        source = runtime._lane_by_id.get(cursor_lane_id)
                        target = runtime._lane_by_id.get(target_id)
                        if (source is None or target is None
                                or source["segment_id"] != target["segment_id"]
                                or source["direction"] != target["direction"]
                                or abs(source["index"] - target["index"]) != 1):
                            raise ValueError(
                                f"invalid route lane change at index {index} "
                                f"for {config.vehicle_id}")
                        action.update({
                            "from_lane_id": cursor_lane_id,
                            "target_lane_index": target["index"],
                        })
                        cursor_lane_id = target_id
                    else:
                        raise ValueError(
                            f"unknown route action {action_type!r} at index "
                            f"{index} for {config.vehicle_id}")
                    normalized_actions.append(action)
                if vehicle.active_connector_id:
                    active = runtime.connector_record(
                        vehicle.active_connector_id)
                    normalized_actions.insert(0, {
                        "type": "connector",
                        "connector_id": active["id"],
                        "from_lane_id": active["from_lane"],
                        "to_lane_id": active["to_lane"],
                        "turn": active["turn"],
                    })
                vehicle.lane_route_actions = normalized_actions
                vehicle.lane_route_action_index = 0
            if not vehicle.active_connector_id:
                self.traffic_mgr._prepare_lane_transition(vehicle)
            pose = runtime.initial_vehicle_pose(
                vehicle, self.road_network)
            if pose is not None:
                vehicle.pose_x_m, vehicle.pose_y_m = pose[:2]
                vehicle.yaw_rad = pose[2]
                vehicle.z_level = pose[3]
            if placement.get("crashed", False):
                vehicle.is_crashed = True
                vehicle.current_speed_kmh = 0.0
                vehicle.target_speed_kmh = 0.0
                vehicle.crash_pose = tuple(pose) if pose is not None else None

            # Placement may move the vehicle away from its route origin.
            # Keep both the spatial lookup and the public road-network view
            # consistent before the first physics sub-step.
            self.traffic_mgr._update_road_network_position(vehicle)

        self.traffic_mgr._rebuild_seg_index()

        for config in self.scenario.pedestrians:
            placement = dict(config.initial_physical_state or {})
            if not placement:
                continue
            pedestrian = self.traffic_mgr.pedestrians.get(config.ped_id)
            if pedestrian is None:
                raise ValueError(
                    f"unknown placed pedestrian {config.ped_id}")
            crosswalk_id = str(placement.get("crosswalk_id", ""))
            walking_path_xy = placement.get("walking_path_xy")
            policy_controlled = (
                pedestrian.is_llm
                and bool(placement.get("policy_controlled", False)))
            if crosswalk_id and walking_path_xy:
                raise ValueError(
                    f"pedestrian {config.ped_id} cannot start on both a "
                    "crosswalk and a free walking path")
            if crosswalk_id:
                crosswalk = runtime._crosswalk_by_id.get(crosswalk_id)
                if crosswalk is None:
                    raise ValueError(
                        f"unknown initial crosswalk_id {crosswalk_id!r} for "
                        f"{config.ped_id}")
                progress = float(placement.get("progress", 0.0))
                if not 0.0 <= progress <= 1.0:
                    raise ValueError(
                        f"pedestrian progress outside [0,1] for {config.ped_id}")
                if not policy_controlled:
                    pedestrian.start_crossing(
                        config.initial_node, config.destination_node,
                        crosswalk["length_m"], crosswalk_id=crosswalk_id,
                        path_xy=crosswalk["centerline_xy"])
                    pedestrian.crossing_progress = progress
                else:
                    pedestrian.authored_crosswalk_id = crosswalk_id
                    pedestrian.authored_crosswalk_path_xy = (
                        self.traffic_mgr._crosswalk_pedestrian_path(
                            crosswalk))
                    pedestrian.authored_crosswalk_progress = progress
                    pedestrian.authored_crosswalk_to_node = (
                        config.destination_node)
                    # While the policy is still waiting, publish a real
                    # sidewalk pose instead of the road-graph lane mouth.
                    wait_path = pedestrian.authored_crosswalk_path_xy
                    node_xy = runtime.nodes_xy.get(config.initial_node)
                    if wait_path:
                        waiting_candidates = [wait_path[0], wait_path[-1]]
                        waiting_pose = (
                            min(waiting_candidates, key=lambda point:
                                math.dist(tuple(node_xy), tuple(point)))
                            if node_xy else waiting_candidates[0])
                        pedestrian.physical_pose_xy = list(waiting_pose)
            elif walking_path_xy:
                raise ValueError(
                    f"walking_path_xy is unsupported for {config.ped_id}; "
                    "SUMO-only physics requires crosswalk_id or a mapped "
                    "pedestrian route")
            base_speed = max(
                0.0, float(placement.get("speed_mps", pedestrian.base_speed)))
            pedestrian.speed = base_speed
            pedestrian.is_waiting = (
                bool(placement.get("waiting", pedestrian.is_waiting))
                if pedestrian.is_llm else False)
            pedestrian.is_spawned = bool(
                placement.get("spawned", pedestrian.start_time <= 0.0))

    # ── Subclass hooks (no-ops in base) ───────────────────────────

    def add_world_observer(self, observer: Any) -> None:
        """Attach a read-only observer called after each synchronized step."""
        callback = getattr(observer, "on_world_step", None)
        if not callable(callback):
            raise TypeError("world observer must define on_world_step()")
        self._world_observers.append(observer)

    def _notify_world_observers(
        self, physics_time: float, tick_index: int, trigger_events: Any,
    ) -> None:
        active = []
        for observer in self._world_observers:
            try:
                observer.on_world_step(
                    self, physics_time, tick_index, trigger_events)
                active.append(observer)
            except Exception as error:  # observers cannot change the episode
                record = {
                    "observer": type(observer).__name__,
                    "time_s": round(float(physics_time), 6),
                    "error": f"{type(error).__name__}: {error}",
                }
                self.world_observer_errors.append(record)
                logger.warning(
                    "disabled world observer %s at t=%.3f: %s",
                    type(observer).__name__, physics_time, error)
                close = getattr(observer, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception:
                        pass
        self._world_observers = active

    def _post_vehicle_init(self, vid: str, vcfg) -> None:
        """Called once per vehicle after ``self._vw[vid]`` is created.

        Subclasses (e.g. TrackedMultiSimEngine) override this to install
        driving-API intercepts on evaluated vehicles' VehicleWorlds.
        """
        vehicle = self.traffic_mgr.get_state(vid)
        if (vehicle is not None and vehicle.is_llm
                and vid in self._agent_callbacks):
            authority = self.traffic_mgr.enable_llm_maneuver_authority(vid)
            if not authority.get("success"):
                raise ValueError(
                    f"cannot enable LLM maneuver authority for {vid}: "
                    f"{authority.get('reason', 'unknown_error')}")
            self._install_driving_intercepts(vid)

    def _log_per_substep(self, physics_time: float, tick_index: int,
                         trigger_events) -> None:
        """Called every fixed physics sub-step (10Hz). No-op in base."""
        pass

    def _record_vehicle_motion(self, time_s: float) -> None:
        """Record physical motion at every SUMO boundary for PA/Judge input."""
        for vehicle_id, vehicle in self.traffic_mgr.vehicles.items():
            history = self._vehicle_motion_history.get(vehicle_id)
            if history is None:
                continue
            history.append({
                "time_s": round(float(time_s), 6),
                "speed_kmh": round(float(vehicle.current_speed_kmh), 3),
                "acceleration_mps2": round(
                    float(vehicle.acceleration_mps2), 3),
            })

    def _recent_vehicle_motion(
        self, vehicle_id: str, time_s: float, *, window_s: float,
    ) -> List[dict]:
        cutoff = float(time_s) - max(0.0, float(window_s))
        return [
            dict(sample)
            for sample in self._vehicle_motion_history.get(vehicle_id, ())
            if float(sample["time_s"]) >= cutoff - 1e-9
        ]

    def _passenger_judge_world_snapshot(self, vehicle_id: str) -> dict:
        """Compact evaluator-only local truth; never exposed to either agent."""
        ego = self.traffic_mgr.vehicles.get(vehicle_id)
        if ego is None:
            return {}
        forward_x = math.cos(float(ego.yaw_rad))
        forward_y = math.sin(float(ego.yaw_rad))
        left_x, left_y = -forward_y, forward_x
        nearby = []
        for other_id, other in self.traffic_mgr.vehicles.items():
            if other_id == vehicle_id or other.arrived or other.route_failed:
                continue
            dx = float(other.pose_x_m) - float(ego.pose_x_m)
            dy = float(other.pose_y_m) - float(ego.pose_y_m)
            center_distance = math.hypot(dx, dy)
            if center_distance > 100.0:
                continue
            longitudinal = dx * forward_x + dy * forward_y
            lateral = dx * left_x + dy * left_y
            same_lane_corridor = abs(lateral) <= (
                (float(ego.width_m) + float(other.width_m)) / 2.0 + 0.75)
            bumper_clearance = (
                max(
                    0.0,
                    abs(longitudinal)
                    - float(ego.length_m) / 2.0
                    - float(other.length_m) / 2.0,
                )
                if same_lane_corridor else None
            )
            nearby.append({
                "entity_id": other_id,
                "entity_type": "vehicle",
                "center_distance_m": round(center_distance, 2),
                "relative_longitudinal_m": round(longitudinal, 2),
                "lateral_offset_m": round(lateral, 2),
                "relative_position": (
                    "front" if longitudinal > 0.5
                    else "rear" if longitudinal < -0.5 else "side"),
                "same_lane_bumper_clearance_m": (
                    round(bumper_clearance, 2)
                    if bumper_clearance is not None else None),
                "speed_kmh": round(float(other.current_speed_kmh), 2),
                "acceleration_mps2": round(
                    float(other.acceleration_mps2), 2),
                "is_crashed": bool(other.is_crashed),
            })
        for pedestrian_id, pedestrian in self.traffic_mgr.pedestrians.items():
            if pedestrian.has_arrived or not pedestrian.is_spawned:
                continue
            pose = self.traffic_mgr.perception_model.entity_pose(pedestrian_id)
            if pose is None:
                continue
            center_distance = math.hypot(
                float(pose[0]) - float(ego.pose_x_m),
                float(pose[1]) - float(ego.pose_y_m))
            if center_distance > 100.0:
                continue
            nearby.append({
                "entity_id": pedestrian_id,
                "entity_type": "pedestrian",
                "center_distance_m": round(center_distance, 2),
                "speed_mps": round(float(pedestrian.speed), 2),
                "is_on_crosswalk": bool(pedestrian.is_on_crosswalk),
                "is_crashed": bool(pedestrian.is_crashed),
            })
        nearby.sort(key=lambda item: (
            float(item["center_distance_m"]), str(item["entity_id"])))
        return {
            "schema_version": "passenger-judge-world-v2",
            "distance_semantics": {
                "center_distance_m": "Euclidean pose-center range",
                "same_lane_bumper_clearance_m": (
                    "dimension-adjusted longitudinal clearance; null outside "
                    "the ego lane corridor"),
            },
            "ego": {
                "speed_kmh": round(float(ego.current_speed_kmh), 2),
                "target_speed_kmh": round(float(ego.target_speed_kmh), 2),
                "acceleration_mps2": round(
                    float(ego.acceleration_mps2), 2),
                "is_crashed": bool(ego.is_crashed),
                "active_agent_control": copy.deepcopy(
                    ego.active_control_commands),
            },
            "nearby_entities": nearby[:20],
        }

    def _log_per_wake(self, physics_time: float, tick_index: int,
                      is_heartbeat: bool, trigger_events) -> None:
        """Called once per agent-wake tick. No-op in base."""
        pass

    def _finalize_result(self, result: "MultiSimResult") -> None:
        """Called once after the physics loop ends.

        Backfills pedestrian arrival state from traffic_mgr (vehicle.arrived
        is set on event during the loop, pedestrians have no equivalent
        event hook so we sync here).
        """
        if self.traffic_mgr is None:
            return
        driving_reports = (
            self._driving_evaluator.finalize()
            if self._driving_evaluator is not None else {})
        process_reports = (
            self._driving_process_scorer.finalize(driving_reports)
            if self._driving_process_scorer is not None else {})
        for vid, process_report in process_reports.items():
            report = driving_reports.get(vid)
            if report is None:
                continue
            report["driving_process"] = process_report
            report["driving_process_score_100"] = process_report[
                "driving_process_score_100"]
            report["environment_accuracy"] = process_report.get(
                "environment_accuracy")
            report["environment_acc"] = process_report.get(
                "environment_acc")
            report["arrival_gate_triggered"] = process_report.get(
                "arrival_gate_triggered", False)
            report["single_vehicle_layer_score_100"] = (
                single_vehicle_layer_score_100(report))
        for vid, vs in self.traffic_mgr.vehicles.items():
            vr = result.vehicle_results.get(vid)
            if vr is not None:
                vr.arrived = bool(vs.arrived)
                vr.route_failed = bool(vs.route_failed)
                vr.route_failure_reason = vs.route_failure_reason
                vr.route_failure_time_s = vs.route_failure_time_s
                if vs.route_failed:
                    vr.terminal_crossing_speed_kmh = (
                        vs.terminal_crossing_speed_kmh)
                vr.driving_evaluation = driving_reports.get(vid)
        for pid, ps in self.traffic_mgr.pedestrians.items():
            pr = result.pedestrian_results.get(pid)
            if pr is not None:
                pr.arrived = bool(getattr(ps, "has_arrived", False))
                pr.crashed = bool(getattr(ps, "is_crashed", False))
        result.signal_events = copy.deepcopy(
            self.traffic_mgr.signal_event_log)
        result.npc_equipment_rule_events = copy.deepcopy(
            self.npc_equipment_rule_log)
        result.npc_behavior_assignments = copy.deepcopy(
            self.traffic_mgr.npc_behavior_assignments)
        result.horn_events = [
            {
                "event_id": item.event_id,
                "source_id": item.source_id,
                "start_time_s": item.start_time_s,
                "duration_s": item.duration_s,
                "intensity": item.intensity,
                "source_db": item.source_db,
                "pose_x_m": item.pose_x_m,
                "pose_y_m": item.pose_y_m,
            }
            for item in self.traffic_mgr.horn_event_log]
        result.perception_log = copy.deepcopy(
            self.traffic_mgr.perception_log)
        result.perception_log_dropped = int(
            self.traffic_mgr.perception_log_dropped)

    # ── World sync ─────────────────────────────────────────────

    def _sync_global_world(self, t: float):
        """Sync weather/daynight to all VehicleWorld instances."""
        # Weather
        wkf = self._get_weather_at(t)
        if wkf and wkf.t != self._last_weather_t:
            for vw in list(self._vw.values()) + list(self._expect_vw.values()):
                w = vw.externalWorld.weather
                w._condition = Weather.Condition(wkf.condition)
                w._temperature = wkf.temperature
                w._humidity = wkf.humidity
                w._wind_speed = wkf.wind_speed
                w._visibility = Weather._VISIBILITY_MAP.get(w._condition, 10000)
                w._rain_intensity = (wkf.rain_intensity
                                     if wkf.rain_intensity > 0
                                     else Weather._RAIN_INTENSITY_MAP.get(w._condition, 0.0))
                w._snow_intensity = (wkf.snow_intensity
                                     if wkf.snow_intensity > 0
                                     else Weather._SNOW_INTENSITY_MAP.get(w._condition, 0.0))
                w._fog_density = (wkf.fog_density
                                  if wkf.fog_density > 0
                                  else Weather._FOG_DENSITY_MAP.get(w._condition, 0.0))
            self._last_weather_t = wkf.t
            # Sync weather to SUMO traffic coordination inputs.
            if self.traffic_mgr:
                self.traffic_mgr._current_weather = wkf.condition
                self.traffic_mgr._current_wind_speed_mps = float(
                    wkf.wind_speed)
                self.traffic_mgr._perception_environment_version += 1

        # DayNight
        dkf = self._get_daynight_at(t)
        if dkf and dkf.t != self._last_daynight_t:
            for vw in list(self._vw.values()) + list(self._expect_vw.values()):
                dn = vw.externalWorld.dayNight
                dn._time_of_day = DayNight.TimePeriod(dkf.period)
                dn._daylight_level = DayNight._DAYLIGHT_MAP.get(
                    dn._time_of_day, 50)
                dn._is_dark = dn._daylight_level < 30
            self._last_daynight_t = dkf.t
            # Sync night state to SUMO traffic coordination inputs.
            if self.traffic_mgr:
                self.traffic_mgr._daylight_level = int(dn._daylight_level)
                self.traffic_mgr._is_night = (
                    dkf.period in ("night", "late_night"))
                self.traffic_mgr._perception_environment_version += 1

    def _sync_map_to_vehicle(self, vehicle_id: str, tick: int):
        """Update a vehicle's MapModule based on its current position."""
        vs = self.traffic_mgr.get_state(vehicle_id)
        if not vs:
            return

        node = self.road_network.get_node(vs.current_node)
        if not node:
            return

        # Current road info comes from the authoritative physical segment.
        edge = (
            self.road_network.get_segment(vs.current_segment)
            if vs.current_segment else None)

        location = {"lat": node.lat, "lng": node.lng, "name": node.full_name}

        weather = self._vw[vehicle_id].externalWorld.weather
        surface = Weather._ROAD_SURFACE_MAP.get(weather._condition, "dry")

        if edge:
            seg_id = RoadNetwork.make_edge_id(edge.from_node, edge.to_node)
            road = {
                "name": edge.name,
                "type": edge.road_type,
                "lanes": edge.lanes,
                "speed_limit": edge.speed_limit,
                "surface": surface,
                "vehicle_count": self.road_network.get_segment_vehicle_count(seg_id),
                "flow_ratio": round(
                    self.road_network.get_segment_flow_ratio(seg_id), 3),
                "observed_speed": round(
                    self.road_network.get_segment_mean_speed_kmh(seg_id), 1),
            }
        else:
            road = {
                "name": node.street_name,
                "type": "urban",
                "lanes": 4,
                "speed_limit": 60,
                "surface": surface,
                "vehicle_count": 0,
                "flow_ratio": 1.0,
                "observed_speed": 60,
            }

        for vw in (self._vw[vehicle_id], self._expect_vw[vehicle_id]):
            m = vw.externalWorld.map
            m._current_location = dict(location)
            m._current_road = dict(road)
            m._is_navigating = vs.is_navigating and not vs.arrived

    # ── Checkpoint verification ────────────────────────────────

    def _verify_vehicle_checkpoint(
        self,
        vehicle_id: str,
        t: int,
        tick_index: int,
        pre_agent_vw: VehicleWorld,
        gt_lines: List[str],
        acceptable_actions: List[AcceptableAction] = None,
        trend_tolerances: list = None,
        negative_checks: list = None,
    ) -> Optional[dict]:
        """Evaluate one cabin wake from YAML-derived expectations."""
        vw = self._vw[vehicle_id]
        return self._cabin_evaluator.evaluate(
            vehicle_id=vehicle_id,
            time_s=t,
            pre_agent_vw=pre_agent_vw,
            post_agent_vw=vw,
            ground_truth_lines=gt_lines,
            acceptable_actions=acceptable_actions,
            trend_tolerances=trend_tolerances,
            negative_checks=negative_checks,
            global_skip_fields=self._rule_loader.global_skip_fields,
        )

    # ── Helpers ─────────────────────────────────────────────────

    def _passenger_trigger_context(
            self, vehicle_id, time_s, *, include_future, conditions=None):
        """Observable geometry plus registerable future environment edges."""
        state = self.traffic_mgr.get_state(vehicle_id)
        if state is None:
            return {}
        conditions = set(conditions) if conditions is not None else None
        context = {}
        if conditions is None or "distance_to_destination_below" in conditions:
            navigation = self.traffic_mgr.get_navigation_status(vehicle_id)
            context["remaining_distance_m"] = navigation.get(
                "remaining_distance_m")
        if (conditions is None or conditions & {
                "approach_next_intersection", "enter_next_intersection",
                "exit_next_intersection"}):
            intersection_distance = (
                self.traffic_mgr.get_next_intersection_distance_m(vehicle_id))
            context["next_intersection_distance_m"] = intersection_distance
            context["intersection_ahead_known"] = (
                intersection_distance is not None)
            context["next_intersection_exit_distance_m"] = (
                self.traffic_mgr.get_next_intersection_exit_distance_m(
                    vehicle_id))
        if not include_future:
            return context
        horizon = min(float(self.scenario.total_time_s), float(time_s) + 60.0)
        weather = self._get_weather_at(time_s)
        previous_weather = weather.condition if weather else None
        future_weather = []
        future_weather_events = []
        for keyframe in self.scenario.weather_keyframes:
            when = float(keyframe.t) * 60.0
            if when <= time_s + 1e-9:
                continue
            if when > horizon:
                break
            if (keyframe.condition != previous_weather
                    and keyframe.condition in {"rainy", "heavy_rain", "foggy"}
                    and keyframe.condition not in future_weather):
                future_weather.append(keyframe.condition)
                future_weather_events.append({
                    "weather_condition": keyframe.condition,
                    "after_s": round(when - float(time_s), 6),
                })
            previous_weather = keyframe.condition
        context["future_weather_conditions"] = future_weather
        context["future_weather_events"] = future_weather_events
        daynight = self._get_daynight_at(time_s)
        previous_dark = (
            daynight.period in {"dawn", "dusk", "night"}
            if daynight else None)
        future_dark = False
        future_dark_after_s = None
        for keyframe in self.scenario.daynight_keyframes:
            when = float(keyframe.t) * 60.0
            if when <= time_s + 1e-9:
                continue
            if when > horizon:
                break
            dark = keyframe.period in {"dawn", "dusk", "night"}
            if dark and previous_dark is False:
                future_dark = True
                future_dark_after_s = round(when - float(time_s), 6)
                break
            previous_dark = dark
        context["future_dark_transition"] = future_dark
        context["future_dark_after_s"] = future_dark_after_s
        return context

    def _get_weather_at(self, t_sec: float) -> Optional[WeatherKeyframe]:
        active = None
        t_min = t_sec / 60.0  # keyframe.t is in minutes
        for kf in self.scenario.weather_keyframes:
            if kf.t <= t_min:
                active = kf
            else:
                break
        return active

    def _get_daynight_at(self, t_sec: float) -> Optional[DayNightKeyframe]:
        active = None
        t_min = t_sec / 60.0  # keyframe.t is in minutes
        for kf in self.scenario.daynight_keyframes:
            if kf.t <= t_min:
                active = kf
            else:
                break
        return active

    # ── Driving API intercepts ────────────────────────────────

    def _install_driving_intercepts(self, vehicle_id: str):
        """Install intercept handlers for driving APIs on a vehicle's VW.

        These replace the placeholder navigation methods with real
        implementations backed by TrafficCoordinator.
        """
        vw = self._vw[vehicle_id]
        vs = self.traffic_mgr.get_state(vehicle_id)
        if not vs:
            return

        nav = vw.navigation
        speed_limit_module = vw.speedLimit
        mgr = self.traffic_mgr
        rn = self.road_network
        engine = self  # capture for nested closures

        # The scenario assigns only a destination.  This module state powers
        # the route display; it grants no connector to the physical vehicle.
        if vs.destination_node:
            nav.current_route = nav.RouteInfo(
                destination=(vs.destination_name or vs.destination_node))
            nav.is_active = True
            nav._preview_destination_node = vs.destination_node
            nav._lane_route_preview = mgr.navigation_route_preview(
                vehicle_id, vs.destination_node)
        nav._lane_route_preview_provider = (
            lambda destination_node="": mgr.navigation_route_preview(
                vehicle_id, destination_node or vs.destination_node)
            if nav.is_active else None)

        # The standalone module defaults to 120 km/h. Agent queries must read
        # the physical road instead, and a vehicle cannot rewrite map rules.
        def _physical_speed_limit_get(self_module=None):
            seg = (
                rn.get_segment(vs.current_segment)
                if vs.current_segment else None)
            if seg is None and vs.active_connector_from_lane_id:
                lane = mgr._lane_geometry._lane_by_id.get(
                    vs.active_connector_from_lane_id, {})
                seg = rn.get_segment(str(lane.get("segment_id", "")))
            if seg is None:
                return {
                    "success": False,
                    "reason": "speed_limit_unavailable_at_current_position",
                }
            return {
                "success": True,
                "current_limit": float(seg.speed_limit),
                "zone_type": "road_network",
                "is_active": True,
                "segment_id": vs.current_segment,
                "source": "authoritative_road_network",
            }

        def _reject_speed_limit_mutation(*args, **kwargs):
            return {
                "success": False,
                "reason": "road_speed_limit_is_read_only",
            }

        speed_limit_module.speed_limit_get = _physical_speed_limit_get
        speed_limit_module.speed_limit_set = _reject_speed_limit_mutation
        speed_limit_module.speed_limit_clear = _reject_speed_limit_mutation

        # These standalone map methods contain demo POIs / a zero-valued
        # road placeholder, not observations from the loaded offline map.
        def _unavailable_road_query(*args, **kwargs):
            return {"success": False, "reason": "road_query_backend_unavailable"}

        def _unavailable_poi_query(*args, **kwargs):
            return {"success": False, "reason": "poi_data_unavailable"}

        vw.map.map_query_road = _unavailable_road_query
        vw.map.map_search_poi = _unavailable_poi_query
        vw.map.map_get_nearby = _unavailable_poi_query

        # ── navigation_set_speed ─────────────────────────────────
        def _set_speed(
            speed_kmh: float, reason: str = "",
            acceleration_mps2: Optional[float] = None,
            deceleration_mps2: Optional[float] = None, self_nav=None,
        ):
            def finite_number(name: str, value) -> float:
                if isinstance(value, bool):
                    raise ValueError(f"{name} must be a finite number")
                try:
                    number = float(value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"{name} must be a finite number") from exc
                if not math.isfinite(number):
                    raise ValueError(f"{name} must be a finite number")
                return number

            try:
                requested_speed = max(
                    0.0, finite_number("speed_kmh", speed_kmh))
                requested_acceleration = (
                    None if acceleration_mps2 is None
                    else finite_number(
                        "acceleration_mps2", acceleration_mps2))
                requested_deceleration = (
                    None if deceleration_mps2 is None
                    else finite_number(
                        "deceleration_mps2", deceleration_mps2))
                if (requested_acceleration is not None
                        and requested_acceleration <= 0.0):
                    raise ValueError(
                        "acceleration_mps2 must be greater than zero")
                if (requested_deceleration is not None
                        and requested_deceleration <= 0.0):
                    raise ValueError(
                        "deceleration_mps2 must be greater than zero")
            except ValueError as exc:
                return {
                    "success": False,
                    "reason": "invalid_control_argument",
                    "error": str(exc),
                }
            if vs.route_failed:
                return {"success": False, "reason": "vehicle_route_failed"}
            if vs.is_crashed:
                return {"success": False, "reason": "vehicle_crashed"}

            def commit_speed():
                committed = mgr.set_vehicle_speed(
                    vehicle_id, requested_speed, reason=str(reason or ""),
                    acceleration_mps2=requested_acceleration,
                    deceleration_mps2=requested_deceleration)
                if committed.get("success", True):
                    nav._desired_speed = requested_speed
                    nav._is_stopped = requested_speed <= 0.0
                return committed

            result = engine._submit_agent_command(
                vehicle_id, "navigation_set_speed",
                commit_speed,
                {
                    "success": True,
                    "accepted_target_speed_kmh": requested_speed,
                },
                control_slots=["longitudinal"])
            if not result.get("success", True):
                return result
            return {
                **result,
                "target_speed_kmh": round(
                    float(result.get(
                        "accepted_target_speed_kmh", requested_speed)), 1),
                "current_speed_kmh": round(vs.current_speed_kmh, 1),
                "acceleration_mps2": round(vs.acceleration_mps2, 2),
                "physical_note": (
                    "Target accepted; SUMO applies acceleration/braking "
                    "continuously on the 0.1 s world clock."),
            }
        nav.navigation_set_speed = _set_speed

        # ── navigation_emergency_stop ────────────────────────────
        def _emergency_stop(reason="", self_nav=None):
            if vs.route_failed:
                return {"success": False, "reason": "vehicle_route_failed"}
            if vs.is_crashed:
                return {"success": False, "reason": "vehicle_crashed"}
            result = engine._submit_agent_command(
                vehicle_id, "navigation_emergency_stop",
                lambda: mgr.emergency_stop_vehicle(
                    vehicle_id, reason=reason),
                {
                    "success": True,
                    "target_speed_kmh": 0.0,
                    "emergency_brake": True,
                },
                control_slots=["longitudinal"])
            result["current_speed_kmh"] = round(
                vs.current_speed_kmh, 1)
            result["physical_note"] = (
                "Emergency braking uses the chassis braking envelope through "
                "SUMO; speed is not teleported.")
            return result
        nav.navigation_emergency_stop = _emergency_stop

        # ── navigation_change_lane ───────────────────────────────
        def _change_lane(direction, self_nav=None):
            """Change lane: 'left' or 'right' relative to current lane."""
            seg = rn.get_segment(vs.current_segment) if vs.current_segment else None
            total_lanes = seg.lanes if seg else 2
            if isinstance(direction, str):
                if direction.lower() == "left":
                    target = vs.current_lane + 1
                elif direction.lower() == "right":
                    target = vs.current_lane - 1
                else:
                    return {"success": False, "reason": "direction must be 'left' or 'right'"}
            else:
                target = int(direction)
            if target < 0 or target >= total_lanes:
                return {"success": False, "reason": "no_such_lane"}
            validation = mgr.validate_lane_change(vehicle_id, target)
            if not validation.success:
                return {"success": False, "status": "rejected",
                        "reason": validation.reason}
            def commit_lane_change():
                committed = vars(mgr.change_lane(vehicle_id, target))
                committed.pop("new_lane", None)
                if committed.get("success"):
                    committed["target_lane"] = target
                return committed
            result = engine._submit_agent_command(
                vehicle_id, "navigation_change_lane",
                commit_lane_change,
                {"success": True, "reason": "queued",
                 "target_lane": target},
                control_slots=["lateral"])
            result.pop("new_lane", None)
            return {**result, "target_lane": target,
                    "physical_note": "Request only; observe the road to verify the maneuver."}
        nav.navigation_change_lane = _change_lane

        # ── navigation_select_maneuver ──────────────────────────
        def _select_maneuver(direction, self_nav=None):
            """Commit one driver-selected connector at the next junction."""
            normalized = str(direction or "").strip().lower().replace(
                "-", "_")
            aliases = {
                "left": "left", "straight": "straight",
                "right": "right", "uturn": "uturn", "u_turn": "uturn",
            }
            turn = aliases.get(normalized)
            if turn is None:
                return {
                    "success": False,
                    "reason": (
                        "direction_must_be_left_straight_right_or_u_turn"),
                }
            if vs.route_failed:
                return {"success": False, "reason": "vehicle_route_failed"}
            if vs.is_crashed:
                return {"success": False, "reason": "vehicle_crashed"}
            if mgr._current_lane_terminates_at_destination(vs):
                return {
                    "success": False,
                    "reason": "destination_is_end_of_current_lane",
                }
            if vs.active_connector_id or (vs.planned_connector_id
                    and vs.planned_maneuver_source != "default_straight"):
                return {
                    "success": False,
                    "reason": "maneuver_already_selected",
                }
            if vs.is_changing_lane:
                return {
                    "success": False,
                    "reason": "lane_change_in_progress",
                }
            if not any(
                    str(item.get("turn", "")) == turn
                    for item in mgr._lane_geometry._connectors_from.get(
                        vs.current_lane_id, [])):
                return {
                    "success": False,
                    "reason": "maneuver_unavailable_from_current_lane",
                    "requested_direction": turn,
                }
            return engine._submit_agent_command(
                vehicle_id, "navigation_select_maneuver",
                lambda: mgr.select_vehicle_maneuver(vehicle_id, turn),
                {
                    "success": True,
                    "maneuver": turn,
                    "execution": "next_0.1s_world_step",
                },
                control_slots=["route_maneuver"])
        nav.navigation_select_maneuver = _select_maneuver

        # ── navigation_u_turn ────────────────────────────────────
        def _u_turn(self_nav=None):
            """Schedule a continuous map-defined U-turn at the next junction."""
            if vs.route_failed:
                return {"success": False, "reason": "vehicle_route_failed"}
            if vs.is_crashed:
                return {"success": False, "reason": "vehicle_crashed"}
            return engine._submit_agent_command(
                vehicle_id, "navigation_u_turn",
                lambda: mgr.u_turn_vehicle(vehicle_id),
                {
                    "success": True,
                    "status": "uturn_intent_queued",
                },
                control_slots=["route_maneuver", "lateral"])
        nav.navigation_u_turn = _u_turn

        # ── navigation_route_plan ────────────────────────────────
        _orig_route_plan = nav.navigation_route_plan
        def _route_plan(address, placeOfDeparture="Current location", self_nav=None):
            # Resolve display guidance against the lane graph.  Planning must
            # not grant physical authority for any intersection movement.
            dest = address
            if dest not in rn.nodes:
                # The assigned task label is vehicle-local: resolve it before
                # generic names, which may be shared by unrelated map nodes.
                if str(address) == vs.destination_name:
                    dest = vs.destination_node
                else:
                    matches = [node_id for node_id, node in rn.nodes.items()
                               if str(address) == node.full_name]
                    dest = matches[0] if len(matches) == 1 else ""
            if not dest or dest not in rn.nodes:
                return {
                    "success": False,
                    "reason": "destination_not_resolved_to_map_node",
                    "destination": str(address),
                }

            route = rn.plan_route(
                vs.current_node, dest,
                time_sec=round(engine._sim_time))
            if not route:
                nav._lane_route_preview = None
                return {
                    "success": False,
                    "reason": "no_road_route_from_current_position",
                    "destination": str(address),
                }
            lane_preview = mgr.navigation_route_preview(vehicle_id, dest)
            if not lane_preview:
                nav._lane_route_preview = None
                return {
                    "success": False,
                    "reason": "no_lane_level_route_from_current_pose",
                    "destination": str(address),
                }
            orig_result = _orig_route_plan(address, placeOfDeparture)
            if not orig_result.get("success"):
                return orig_result
            nav._preview_destination_node = dest
            nav._lane_route_preview = copy.deepcopy(lane_preview)
            return {
                "success": True,
                "minimap_available": True,
                "physical_route_changed": False,
            }
        nav.navigation_route_plan = _route_plan

        def _reroute():
            if not nav.is_active:
                return {"success": False, "reason": "navigation_inactive"}
            return _route_plan(nav._preview_destination_node)

        def _destination_change(address):
            if not nav.is_active:
                return {"success": False, "reason": "navigation_inactive"}
            # Change display guidance only, as with route_plan. Never change
            # the scenario's assigned destination or grant steering authority.
            return _route_plan(address)

        def _unsupported_waypoints(*args, **kwargs):
            return {"success": False, "reason": "waypoint_routing_not_supported"}

        original_exit = nav.navigation_exit

        def _exit_navigation():
            result = original_exit()
            nav._lane_route_preview = None
            return result

        nav.navigation_reroute = _reroute
        nav.navigation_destination_change = _destination_change
        nav.navigation_midWay_add = _unsupported_waypoints
        nav.navigation_midWay_delete = _unsupported_waypoints
        nav.navigation_exit = _exit_navigation
