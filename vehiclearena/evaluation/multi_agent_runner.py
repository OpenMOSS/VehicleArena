"""
Multi-Agent Runner — Unified runner for multi-vehicle simulation with per-vehicle model config.

Each vehicle has one explicit control authority:
  - LLM agent (any model/api_base/api_key)
  - SUMO background traffic

Configuration is resolved from two layers:
  1. Scenario JSON: vehicle.agent_config field
  2. Runtime overrides: agent_overrides dict passed to run_multi_scenario()
  Priority: runtime overrides > scenario JSON > defaults

Usage:
    from evaluation.multi_agent_runner import run_multi_scenario, print_report

    # All config in scenario JSON
    result, elapsed, stats = run_multi_scenario(scenario_dict)

    # Override ego's model at runtime
    result, elapsed, stats = run_multi_scenario(
        scenario_dict,
        agent_overrides={"ego": {"type": "llm", "model": "qwen3-235b"}}
    )
"""

import sys
import os
import json
import time
import copy
import hashlib
import threading
import math
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from jsonschema import Draft202012Validator

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

from agent_client import AgentClient
from driving_eval import (
    generate_driving_instruction,
    _assistant_message_has_payload,
    _message_to_dict,
    _make_memory_search_tool,
)
from context_runtime import (
    ContextWindowExceeded,
    DATA_DELIVERY_REGISTRY,
    RollingWakeContext,
    TodoStore,
    build_loaded_capabilities_message,
    build_current_wake_message,
    build_vehicle_active_control,
    build_vehicle_self_now,
    load_skill_content,
    normalize_new_events,
)
from tool_utils import (
    generate_lazy_discovery_tools,
    generate_skill_tool,
    dispatch_lazy_discovery,
)
from skills.skill_loader import get_skill_loader
from simulation.multi_sim_engine import (
    MultiSimEngine, MultiScenario, MultiSimResult,
)
from simulation.memory import SessionHistory, ActionLogEntry
from simulation.perception_tools import (
    VEHICLE_PERCEPTION_TOOL_NAMES,
)
from visualization.agent_visual_renderer import multimodal_image_message
from personal_agent import (
    _CABIN_OBSERVABLE_MODULES,
    _compact_cabin_modules,
    PersonalAgentRuntime,
    PassengerJudgeRuntime,
    PersonalAgentVehicleCallback,
    aggregate_passenger_judgements,
)
from evaluation.snapshot_utils import snapshot_modules
from utils import execute

lock = threading.Lock()

# ── Defaults ── (read from environment; never hard-code credentials)
DEFAULT_API_BASE = os.environ.get("OPENAI_API_BASE", "")
DEFAULT_API_KEY = os.environ.get("OPENAI_API_KEY", "")
DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.4")
DEFAULT_CONTEXT_WINDOW_TOKENS = int(os.environ.get(
    "OPENAI_CONTEXT_WINDOW_TOKENS", "32768"))
DEFAULT_MAX_OUTPUT_TOKENS = int(os.environ.get(
    "OPENAI_MAX_OUTPUT_TOKENS", "4096"))
# Passenger judging is a separate completion task from vehicle control.  It
# needs enough room for the evidence-backed tool submission, especially when
# reasoning-capable providers are used.  Keep this independent from the
# driver's output cap.
DEFAULT_PASSENGER_JUDGE_MAX_OUTPUT_TOKENS = 32768
_AGENT_OVERRIDE_FIELDS = {
    "type", "model", "api_base", "api_key", "max_turns", "temperature",
    "max_tokens", "context_window_tokens", "todo_max_ttl_s",
    "thinking_mode", "reasoning_effort", "chat_template_enable_thinking",
    "heartbeat_interval_s", "driver_prompt",
    "personal_agent_enabled", "passenger_judge_enabled",
}

MIN_LLM_HEARTBEAT_INTERVAL_S = 0.1
MAX_LLM_HEARTBEAT_INTERVAL_S = 30.0
DEFAULT_LLM_HEARTBEAT_INTERVAL_S = 3.0


# ══════════════════════════════════════════════════════════════════════
#  AgentSpec — resolved configuration for one controlled entity
# ══════════════════════════════════════════════════════════════════════

@dataclass
class AgentSpec:
    """Fully resolved specification for one independently controlled entity."""
    entity_id: str
    entity_type: str = "vehicle"   # "vehicle" | "pedestrian"
    agent_type: str = "llm"       # "llm" | "sumo"
    model: str = ""               # LLM model name (only for type="llm")
    api_base: str = ""            # API endpoint
    api_key: str = ""             # API key
    max_turns: int = 10           # max tool-call turns per tick
    temperature: float = 0.7      # LLM temperature
    max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    thinking_mode: str = "default"  # provider default | enabled | disabled
    reasoning_effort: Optional[str] = None
    chat_template_enable_thinking: Optional[bool] = None
    context_window_tokens: int = DEFAULT_CONTEXT_WINDOW_TOKENS
    todo_max_ttl_s: float = 3600.0
    heartbeat_interval_s: float = DEFAULT_LLM_HEARTBEAT_INTERVAL_S
    is_evaluated: bool = True     # from VehicleConfig
    driver_prompt: str = ""       # optional per-vehicle driving role addendum
    # In-cabin PA/Judge are enabled per LLM vehicle by default.  Keeping these
    # switches on the resolved entity spec lets ablation manifests disable the
    # scored (focal) vehicle without changing the background agents.
    personal_agent_enabled: bool = True
    passenger_judge_enabled: bool = True

    @property
    def display_name(self) -> str:
        if self.agent_type == "llm":
            return f"LLM({self.model})"
        return self.agent_type.upper()

def resolve_agent_specs(
    scenario: MultiScenario,
    overrides: Optional[Dict[str, dict]] = None,
    llm_runtime_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, AgentSpec]:
    """Merge vehicle/pedestrian agent_config with runtime overrides.

    Priority for LLM entities: per-entity override > run-level config >
    scenario agent_config > defaults. This lets Multi-LLM experiments keep
    peer models fixed while changing only the focal entity.
    """
    overrides = overrides or {}
    llm_runtime_config = {
        key: value for key, value in (llm_runtime_config or {}).items()
        if value is not None
    }
    if not isinstance(overrides, dict):
        raise ValueError("agent overrides must be an object")
    unknown_runtime_fields = sorted(
        set(llm_runtime_config) - _AGENT_OVERRIDE_FIELDS)
    if unknown_runtime_fields:
        raise ValueError(
            f"Unknown LLM runtime fields: {unknown_runtime_fields!r}")
    for entity_id, values in overrides.items():
        if not isinstance(values, dict):
            raise ValueError(
                f"agent override for {entity_id!r} must be an object")
        unknown = sorted(set(values) - _AGENT_OVERRIDE_FIELDS)
        if unknown:
            raise ValueError(
                f"Unknown agent override fields for {entity_id!r}: "
                f"{unknown!r}")
    specs = {}

    entity_configs = [
        (vcfg.vehicle_id, "vehicle", vcfg, vcfg.is_evaluated,
         vcfg.agent_type)
        for vcfg in scenario.vehicles
    ]
    entity_configs.extend(
        (pcfg.ped_id, "pedestrian", pcfg, pcfg.is_evaluated,
         pcfg.agent_type)
        for pcfg in scenario.pedestrians
        if pcfg.agent_type == "llm"
    )
    configured_entity_ids = {item[0] for item in entity_configs}
    unknown_override_entities = sorted(set(overrides) - configured_entity_ids)
    if unknown_override_entities:
        raise ValueError(
            "Agent overrides reference unknown or non-LLM pedestrian entities: "
            f"{unknown_override_entities!r}")

    for vid, entity_type, config, is_evaluated, default_type in entity_configs:
        if vid in specs:
            raise ValueError(f"Duplicate vehicle/pedestrian entity id: {vid}")

        # Start with defaults
        merged = {
            "type": default_type,
            "model": DEFAULT_MODEL,
            "api_base": DEFAULT_API_BASE,
            "api_key": DEFAULT_API_KEY,
            "max_turns": 10,
            "temperature": 0.7,
            "max_tokens": DEFAULT_MAX_OUTPUT_TOKENS,
            "thinking_mode": "default",
            "reasoning_effort": None,
            "chat_template_enable_thinking": None,
            "context_window_tokens": DEFAULT_CONTEXT_WINDOW_TOKENS,
            "todo_max_ttl_s": 3600.0,
            "heartbeat_interval_s": DEFAULT_LLM_HEARTBEAT_INTERVAL_S,
        }

        # Layer 1: scenario JSON agent_config
        if config.agent_config:
            merged.update(config.agent_config)

        # Resolve authority before applying common LLM runtime defaults.
        if vid in overrides:
            merged.update(overrides[vid])

        # Evaluation-script settings are common defaults for every LLM.
        if merged.get("type", default_type) == "llm":
            merged.update(llm_runtime_config)

        # Entity-specific assignments are the final layer. In particular,
        # peer model assignments in Multi-LLM runs must not be overwritten by
        # the focal model supplied at the command line.
        if vid in overrides:
            merged.update(overrides[vid])

        context_window_tokens = int(merged.get(
            "context_window_tokens", DEFAULT_CONTEXT_WINDOW_TOKENS))
        max_tokens = int(merged.get(
            "max_tokens", DEFAULT_MAX_OUTPUT_TOKENS))
        if context_window_tokens <= max_tokens:
            raise ValueError(
                f"context_window_tokens ({context_window_tokens}) must exceed "
                f"max_tokens ({max_tokens}) for {vid}")
        todo_max_ttl_s = float(merged.get("todo_max_ttl_s", 3600.0))
        # Todo storage is scoped to one episode and discarded at shutdown, so
        # its per-item TTL limit need not be shortened to the episode length.
        # Exposing a tiny limit in short diagnostic scenes made agents spend
        # most wakes renewing a goal that the harness would discard anyway.
        heartbeat_interval_s = float(merged.get(
            "heartbeat_interval_s", DEFAULT_LLM_HEARTBEAT_INTERVAL_S))
        if not (MIN_LLM_HEARTBEAT_INTERVAL_S
                <= heartbeat_interval_s
                <= MAX_LLM_HEARTBEAT_INTERVAL_S):
            raise ValueError(
                "heartbeat_interval_s must be between "
                f"{MIN_LLM_HEARTBEAT_INTERVAL_S} and "
                f"{MAX_LLM_HEARTBEAT_INTERVAL_S} seconds for {vid}")
        if todo_max_ttl_s <= 0:
            raise ValueError(f"todo_max_ttl_s must be positive for {vid}")
        thinking_mode = str(
            merged.get("thinking_mode", "default") or "default").lower()
        if thinking_mode not in {"default", "enabled", "disabled"}:
            raise ValueError(
                f"thinking_mode must be default, enabled, or disabled for "
                f"{vid}; got {thinking_mode!r}")
        reasoning_effort = merged.get("reasoning_effort")
        if reasoning_effort is not None:
            reasoning_effort = str(reasoning_effort).lower()
            if reasoning_effort not in {
                    "max", "xhigh", "high", "medium", "low",
                    "minimal", "none"}:
                raise ValueError(
                    f"invalid reasoning_effort for {vid}: "
                    f"{reasoning_effort!r}")
        chat_template_enable_thinking = merged.get(
            "chat_template_enable_thinking")
        if (chat_template_enable_thinking is not None
                and not isinstance(chat_template_enable_thinking, bool)):
            raise ValueError(
                "chat_template_enable_thinking must be a boolean for "
                f"{vid}")

        driver_prompt = merged.get("driver_prompt", "")
        if not isinstance(driver_prompt, str):
            raise ValueError(f"driver_prompt must be a string for {vid}")
        if driver_prompt and entity_type != "vehicle":
            raise ValueError(f"driver_prompt is only supported for vehicles: {vid}")
        personal_agent_enabled = merged.get("personal_agent_enabled", True)
        passenger_judge_enabled = merged.get("passenger_judge_enabled", True)
        for field_name, field_value in (
                ("personal_agent_enabled", personal_agent_enabled),
                ("passenger_judge_enabled", passenger_judge_enabled)):
            if not isinstance(field_value, bool):
                raise ValueError(f"{field_name} must be a boolean for {vid}")

        specs[vid] = AgentSpec(
            entity_id=vid,
            entity_type=entity_type,
            agent_type=merged.get("type", default_type),
            model=merged.get("model", DEFAULT_MODEL),
            api_base=merged.get("api_base", DEFAULT_API_BASE),
            api_key=merged.get("api_key", DEFAULT_API_KEY),
            max_turns=merged.get("max_turns", 10),
            temperature=merged.get("temperature", 0.7),
            max_tokens=max_tokens,
            thinking_mode=thinking_mode,
            reasoning_effort=reasoning_effort,
            chat_template_enable_thinking=chat_template_enable_thinking,
            context_window_tokens=context_window_tokens,
            todo_max_ttl_s=todo_max_ttl_s,
            heartbeat_interval_s=heartbeat_interval_s,
            is_evaluated=is_evaluated,
            driver_prompt=driver_prompt,
            personal_agent_enabled=personal_agent_enabled,
            passenger_judge_enabled=passenger_judge_enabled,
        )

    return specs


def apply_resolved_agent_authorities(
    scenario: MultiScenario, specs: Dict[str, AgentSpec],
) -> None:
    """Commit resolved episode control roles before physics construction."""
    configs = {config.vehicle_id: config for config in scenario.vehicles}
    configs.update({
        config.ped_id: config for config in scenario.pedestrians
    })
    for entity_id, spec in specs.items():
        config = configs[entity_id]
        config.agent_config = dict(config.agent_config)
        config.agent_config["type"] = spec.agent_type


# ══════════════════════════════════════════════════════════════════════
#  Agent callback factories
# ══════════════════════════════════════════════════════════════════════

_DISCOVERY_TOOL_NAMES = {"get_module_api", "load_tools", "load_skill"}
_HARNESS_TOOL_NAMES = {
    "finish", "todo_manage", "set_heartbeat_interval", "schedule_next_wake"}
_DEVICE_STATE_TOOL_NAME = "get_device_state"
_HARD_MAX_TURNS_PER_WAKE = 16
_CORE_VEHICLE_TOOL_NAMES = (
    "navigation__navigation_route_plan",
    "navigation__navigation_minimap",
    "navigation__navigation_set_speed",
    "navigation__navigation_emergency_stop",
    "navigation__navigation_change_lane",
    "navigation__navigation_select_maneuver",
    "frontRadar__scan",
    "rearRadar__scan",
    "speedLimit__speed_limit_get",
    "turnSignal__switch",
)
_NAVIGATION_QUERY_NAMES = {
    "navigation_get_destination",
    "navigation_minimap",
    "navigation_route_plan",
}
_EXPLICIT_DEVICE_QUERY_NAMES = {
    "music__music_history_view",
    "music__music_collection_view",
    "music__music_local_view",
    "music__music_download_view",
    "music__music_currentDetail_view",
    "radio__radio_history_view",
    "radio__radio_collection_view",
    "video__video_common_history_view",
    "video__video_download_view",
    "video__video_profile_view",
    "video__video_collection_view",
    "video__video_local_view",
    "video__video_currentDetail_view",
    "conversation__conversation_message_view",
    "conversation__conversation_contact_view",
    "conversation__conversation_call_miss_view",
    "conversation__conversation_call_record_view",
    "conversation__conversation_contact_hag_view",
    "instrumentPanel__carcontrol_instrumentPanel_vehicleMileage_view",
}
_PASSENGER_BROADCAST_ACTION_NAMES = {
    "broadcast__broadcast_traffic_light",
    "broadcast__broadcast_road_event",
    "broadcast__broadcast_speed_camera",
    "broadcast__broadcast_congestion",
    "broadcast__broadcast_warning",
    "broadcast__broadcast_safety_refusal",
}

# These APIs previously converted camera-visible facts into exact text. In
# multimodal driving mode they are intentionally unavailable: the automatic
# driving image is the sole source for bodies, lane occupancy, crossings and
# traffic-light state. Non-visual sensors remain discoverable separately.
_IMAGE_ONLY_TEXT_TOOLS = {
    "road_perception__look_ahead",
    "road_perception__check_signal",
    "road_perception__scan_crosswalk",
    "road_perception__look_around",
    "road_perception__check_road",
    "map__map_update_location",
}
_VEHICLE_DENIED_TOOL_NAMES = _IMAGE_ONLY_TEXT_TOOLS | {
    # Vehicles may observe a road-network limit, never rewrite it.
    "speedLimit__speed_limit_set",
    "speedLimit__speed_limit_clear",
}

_NONVISUAL_WAKE_EVENTS = {
    "weather_initialized", "weather_changed",
    "daynight_initialized", "daynight_changed",
    "simulation_start", "driving_task_assigned", "heartbeat", "scheduled_wake",
    "passenger_request", "personal_agent_update", "command_result", "acoustic_cue",
    "front_collision_warning", "rear_collision_warning",
    "arrived", "route_failed", "crashed", "collision", "simulation_ended",
    "agent_callback_failed", "control_constraint_overrun",
}


def _events_visible_to_multimodal_driver(events: List[dict]) -> List[dict]:
    """Hide semantic visual answers while preserving non-visual events."""
    return [
        copy.deepcopy(event) for event in events
        if str(event.get("event_type", "")) in _NONVISUAL_WAKE_EVENTS
        or str(event.get("event_type", "")).endswith("_request_due")
    ]


def _messages_for_log(messages: List[dict]) -> List[dict]:
    """Keep multimodal audit structure without embedding base64 in JSON."""
    value = copy.deepcopy(messages)
    for message in value:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "image_url":
                continue
            image_url = part.get("image_url")
            if not isinstance(image_url, dict):
                continue
            url = str(image_url.get("url", ""))
            if url.startswith("data:image/"):
                image_url["url"] = "[embedded image omitted from log]"
    return value


def _save_visual_observation(
    state: dict, rendered, *, entity_id: str, tick_index: int,
    turn_index: int, label: str,
) -> dict:
    """Persist the exact PNG supplied to the model and return audit metadata."""
    metadata = rendered.metadata()
    output_dir = state.get("visual_observation_output_dir")
    relative_root = state.get("visual_observation_reference_root")
    suffix = label.lower().replace(" ", "-")
    observation_index = len(state["visual_observation_log"]) + 1
    filename = (
        f"wake-{int(tick_index):06d}-turn-{int(turn_index):02d}-"
        f"{observation_index:04d}-{suffix}.png")
    relative_path = None
    if output_dir:
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / filename
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_bytes(rendered.png_bytes)
        os.replace(temporary, destination)
        relative_path = (
            str(Path(relative_root) / filename)
            if relative_root else str(destination))
    record = {
        "entity_id": entity_id,
        "tick_index": int(tick_index),
        "turn_index": int(turn_index),
        "observation_index": observation_index,
        "label": label,
        "path": relative_path,
        **metadata,
    }
    state["visual_observation_log"].append(record)
    return record


def _session_history_catalog() -> str:
    return (
        "Module: session_history — 当前主体的结构化历史索引，不自动注入上下文。\n"
        "1 API available:\n"
        "  - memory_search: 按关键词、范围和时间查询历史事件与动作。")


def _dispatch_memory_search(memory, arguments):
    """Validate this harness tool just as strictly as ordinary module tools.

    Do not coerce strings to seconds: return a repairable receipt to the model.
    Unexpected implementation errors still propagate to the audited callback
    boundary, rather than silently turning a broken experiment into success.
    """
    schema = _make_memory_search_tool()["function"]["parameters"]
    errors = list(Draft202012Validator(schema).iter_errors(arguments))
    if errors:
        error = errors[0]
        return {
            "success": False, "error": "invalid_tool_arguments",
            "tool": "memory_search", "message": error.message,
            "parameter": next(iter(error.path), None),
            "expected": error.schema,
        }
    for name, value in arguments.items():
        if isinstance(value, float) and not math.isfinite(value):
            return {
                "success": False, "error": "invalid_tool_arguments",
                "tool": "memory_search", "parameter": name,
                "message": "Expected a finite number, not NaN or Infinity.",
            }
    if ("time_from" in arguments and "time_to" in arguments
            and arguments["time_from"] > arguments["time_to"]):
        return {
            "success": False, "error": "invalid_tool_arguments",
            "tool": "memory_search", "parameter": "time_to",
            "message": "time_to must be greater than or equal to time_from.",
        }
    # JSON Schema considers 1.0 an integer. Normalize valid numeric counts
    # for Python range(), but never coerce invalid strings such as "1".
    normalized = dict(arguments)
    for name in ("context", "limit"):
        if name in normalized:
            normalized[name] = int(normalized[name])
    return memory.search(**normalized)


def _classify_vehicle_tool(func_name: str) -> str:
    """Return the explicit audit category for one vehicle tool call."""
    if func_name in _DISCOVERY_TOOL_NAMES:
        return "discovery"
    if func_name in _HARNESS_TOOL_NAMES:
        return "harness"
    if func_name == "memory_search":
        return "memory"
    if func_name in VEHICLE_PERCEPTION_TOOL_NAMES:
        return "perception"
    if func_name in _EXPLICIT_DEVICE_QUERY_NAMES:
        return "query"
    method = func_name.split("__", 1)[-1]
    if (method in _NAVIGATION_QUERY_NAMES or method.startswith("get_")
            or method.endswith("_get")
            or "_get_" in method or method.startswith("query_")
            or method.startswith("check_") or method.startswith("look_")
            or method.startswith("scan")):
        return "query"
    return "action"


def _make_device_state_tool(vw) -> dict:
    available = [
        name for name in _CABIN_OBSERVABLE_MODULES
        if not hasattr(vw, "has_module") or vw.has_module(name)
    ]
    return {
        "type": "function",
        "function": {
            "name": _DEVICE_STATE_TOOL_NAME,
            "description": (
                "Read current physical state for selected installed vehicle "
                "devices. Use this to verify delayed or ongoing passenger "
                "requests; earlier successful commands do not prove that a "
                "state still holds."),
            "parameters": {
                "type": "object",
                "properties": {
                    "modules": {
                        "type": "array",
                        "items": {"type": "string", "enum": available},
                        "minItems": 1,
                        "maxItems": 6,
                        "uniqueItems": True,
                    },
                },
                "required": ["modules"],
                "additionalProperties": False,
            },
        },
    }


def _dispatch_device_state(vw, arguments: dict) -> dict:
    modules = list(arguments.get("modules") or [])
    available = {
        name for name in _CABIN_OBSERVABLE_MODULES
        if not hasattr(vw, "has_module") or vw.has_module(name)
    }
    unavailable = sorted(set(modules) - available)
    if unavailable:
        return {
            "success": False,
            "error": "capability_not_available",
            "modules": unavailable,
        }
    return {
        "success": True,
        "device_state": _compact_cabin_modules(
            snapshot_modules(vw, modules)),
    }


def _passenger_response_authorized(
    passenger_messages, active_request_ids,
) -> bool:
    """Keep passenger authority alive for the full request-sheet lifetime."""
    return bool(list(passenger_messages or ()) or tuple(
        request_id for request_id in (active_request_ids or ())
        if request_id))


def _make_finish_tool() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "finish",
            "description": (
                "Finish this wake after commands already submitted in this "
                "turn. It does not stop the vehicle or simulation."),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Why no more work is needed this wake.",
                    },
                },
                "required": [],
            },
        },
    }


def _make_heartbeat_interval_tool() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "set_heartbeat_interval",
            "description": (
                "Set the persistent quiet-world heartbeat interval in "
                "simulation seconds. Default is 3 s. Safety, task and "
                "lifecycle events can still wake you earlier. A longer "
                "interval reduces calls but delays visual reassessment when "
                "no pushed event occurs. The world does not rewrite the "
                "selected interval based on driving state."),
            "parameters": {
                "type": "object",
                "properties": {
                    "interval_s": {
                        "type": "number",
                        "minimum": MIN_LLM_HEARTBEAT_INTERVAL_S,
                        "maximum": MAX_LLM_HEARTBEAT_INTERVAL_S,
                        "default": DEFAULT_LLM_HEARTBEAT_INTERVAL_S,
                        "description": (
                            "Seconds until the next periodic wake if no "
                            "event wakes the vehicle first."),
                    },
                    "reason": {
                        "type": "string",
                        "maxLength": 120,
                        "description": "Why this observation cadence fits.",
                    },
                },
                "required": ["interval_s"],
                "additionalProperties": False,
            },
        },
    }


def _make_schedule_next_wake_tool() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "schedule_next_wake",
            "description": (
                "Schedule one future self-observation in simulation seconds. "
                "Use this to revisit a delayed or ordered passenger task near "
                "the time you predict action will be appropriate. This replaces "
                "any earlier one-shot wake you scheduled, but does not change "
                "or postpone the persistent heartbeat: ordinary heartbeats and "
                "the one-shot observation coexist. Safety, task and lifecycle "
                "events may also wake you earlier; an early event does not "
                "cancel the scheduled observation unless you replace it. Use "
                "set_heartbeat_interval separately if you want fewer periodic "
                "observations."),
            "parameters": {
                "type": "object",
                "properties": {
                    "delay_s": {
                        "type": "number",
                        "minimum": MIN_LLM_HEARTBEAT_INTERVAL_S,
                        "maximum": MAX_LLM_HEARTBEAT_INTERVAL_S,
                        "description": (
                            "Delay from the current simulation time to the "
                            "requested one-shot observation."),
                    },
                    "reason": {
                        "type": "string", "maxLength": 120,
                        "description": "What future condition will be rechecked.",
                    },
                },
                "required": ["delay_s", "reason"],
                "additionalProperties": False,
            },
        },
    }

def make_llm_agent_callback(
    vehicle_id: str,
    agent_client: AgentClient,
    max_turns: int = 10,
    context_window_tokens: int = DEFAULT_CONTEXT_WINDOW_TOKENS,
    todo_max_ttl_s: float = 3600.0,
    heartbeat_interval_s: float = DEFAULT_LLM_HEARTBEAT_INTERVAL_S,
    driver_prompt: str = "",
):
    """Create one bounded, direct-control LLM vehicle callback."""
    skill_loader = get_skill_loader()
    traffic_addendum = (
        "\n\n## Multi-Vehicle Context\n\n"
        f"You are driving vehicle **{vehicle_id}**. Other vehicles and "
        "pedestrians move independently. A collision disables the involved "
        "vehicle, and a wreck remains an obstacle only in its physical "
        "lane/connector domain. `navigation__navigation_u_turn` is "
        "loadable through `load_tools` on two-way roads.\n"
    )
    if driver_prompt:
        traffic_addendum += "\n## Driving Role\n\n" + driver_prompt + "\n"
    effective_max_turns = min(
        _HARD_MAX_TURNS_PER_WAKE, max(1, int(max_turns)))
    todo_store = TodoStore(max_ttl_s=float(todo_max_ttl_s))
    heartbeat_interval_s = float(heartbeat_interval_s)
    if not (MIN_LLM_HEARTBEAT_INTERVAL_S
            <= heartbeat_interval_s
            <= MAX_LLM_HEARTBEAT_INTERVAL_S):
        raise ValueError(
            "heartbeat_interval_s must be between "
            f"{MIN_LLM_HEARTBEAT_INTERVAL_S} and "
            f"{MAX_LLM_HEARTBEAT_INTERVAL_S} seconds")
    state = {
        "messages": None,
        "previous_wake_messages": [],
        "tools": None,
        "instruction": None,
        "token_stats": {"input": [], "output": []},
        "total_turns": 0,
        "total_tool_calls": 0,
        "tool_calls_discovery": 0,
        "tool_calls_query": 0,
        "tool_calls_perception": 0,
        "tool_calls_action": 0,
        "tool_calls_memory": 0,
        "tool_calls_harness": 0,
        "loaded_skills": [],
        "loaded_skill_contents": {},
        "capability_epoch": 0,
        "last_announced_capability_signature": None,
        "context_budget_log": [],
        "context_data_registry": copy.deepcopy(DATA_DELIVERY_REGISTRY),
        "todo_audit_log": [],
        "todo_state": todo_store.audit_dict(),
        "next_todo_deadline_s": None,
        "heartbeat_interval_s": heartbeat_interval_s,
        "heartbeat_interval_revision": 0,
        "heartbeat_interval_audit_log": [],
        "scheduled_wake_at_s": None,
        "scheduled_wake_reason": "",
        "scheduled_wake_revision": 0,
        "scheduled_wake_audit_log": [],
        "context_snapshot_log": [],
        "visual_observation_log": [],
        "visual_observation_output_dir": None,
        "visual_observation_reference_root": None,
        "wake_runtime_log": [],
        "all_messages": [],
        "tool_call_log": [],
        "model_call_log": [],
        "infrastructure_errors": [],
        "protocol_events": [],
        "model": agent_client.model,
        "prompt_profile_id": "vehicle-bounded-context-v1",
        "effective_config": {
            "entity_type": "vehicle",
            "driver_prompt": driver_prompt,
            "configured_max_turns": int(max_turns),
            "max_turns": effective_max_turns,
            "hard_max_turns": _HARD_MAX_TURNS_PER_WAKE,
            "max_tokens": int(getattr(
                agent_client, "max_tokens", DEFAULT_MAX_OUTPUT_TOKENS)),
            "context_window_tokens": int(context_window_tokens),
            "todo_max_ttl_s": float(todo_max_ttl_s),
            "heartbeat_interval_s": heartbeat_interval_s,
            "temperature": float(getattr(agent_client, "temperature", 0.7)),
            "thinking_mode": str(getattr(
                agent_client, "thinking_mode", "default")),
            "reasoning_effort": getattr(
                agent_client, "reasoning_effort", None),
            "chat_template_enable_thinking": getattr(
                agent_client, "chat_template_enable_thinking", None),
            "api_base": getattr(agent_client, "api_base", ""),
        },
        "initial_tools": None,
        "core_tool_names": [],
    }
    rolling_context = None

    def run_wake(vw, t, passenger_messages, memory, tick_index, **kwargs):
        nonlocal rolling_context
        wake_wall_started = time.perf_counter()
        model_calls_before = len(state["model_call_log"])
        tool_calls_before = len(state["tool_call_log"])
        truncations_before = sum(
            item.get("type") == "response_truncated"
            for item in state["protocol_events"])
        actions_taken = []
        action_log = []
        wake_events = list(kwargs.get("_wake_events", []))
        wake_event_ids = [
            event.get("event_id") for event in wake_events
            if event.get("event_id")]

        if state["tools"] is None:
            state["instruction"] = (
                generate_driving_instruction(vw) + traffic_addendum)
            state["tools"] = (
                list(generate_lazy_discovery_tools())
                + [generate_skill_tool(skill_loader),
                   _make_memory_search_tool(), todo_store.tool_schema(),
                   _make_heartbeat_interval_tool(),
                   _make_schedule_next_wake_tool(),
                   _make_device_state_tool(vw),
                   _make_finish_tool()]
            )
            available_core_tools = [
                name for name in _CORE_VEHICLE_TOOL_NAMES
                if (not hasattr(vw, "has_module")
                    or vw.has_module(name.split("__", 1)[0]))
            ]
            preload_result = dispatch_lazy_discovery(
                vw, "load_tools", {"tools": available_core_tools},
                state["tools"], denied_tool_names=_VEHICLE_DENIED_TOOL_NAMES)
            if not preload_result.get("success", False):
                raise RuntimeError(
                    f"core driving tool preload failed: {preload_result}")
            state["initial_tools"] = copy.deepcopy(state["tools"])
            state["core_tool_names"] = sorted(
                _DISCOVERY_TOOL_NAMES | _HARNESS_TOOL_NAMES
                | {_DEVICE_STATE_TOOL_NAME} | set(available_core_tools))
            rolling_context = RollingWakeContext(
                state["instruction"],
                context_window_tokens=int(context_window_tokens),
                max_output_tokens=int(getattr(
                    agent_client, "max_tokens", DEFAULT_MAX_OUTPUT_TOKENS)),
            )
            state["context_budget_log"] = rolling_context.budget_log

        tools = state["tools"]
        loaded_skill_contents = state["loaded_skill_contents"]
        visible_wake_events = _events_visible_to_multimodal_driver(
            wake_events)
        new_events = normalize_new_events(
            passenger_messages, visible_wake_events)
        new_events.extend(todo_store.expiration_events(t))
        wake_policy = {
            "heartbeat_interval_s": state["heartbeat_interval_s"],
            "event_wake_can_preempt": True,
        }
        if state["scheduled_wake_at_s"] is not None:
            wake_policy.update(
                scheduled_wake_at_s=state["scheduled_wake_at_s"],
                scheduled_wake_reason=(
                    state["scheduled_wake_reason"] or None),
            )
        current_wake = build_current_wake_message(
            sim_time_s=t,
            tick_index=tick_index,
            agent_id=vehicle_id,
            entity_type="vehicle",
            self_now=build_vehicle_self_now(kwargs.get("_vehicle_state")),
            active_control=build_vehicle_active_control(
                kwargs.get("_vehicle_state")),
            wake_policy=wake_policy,
            todo=todo_store.to_context(t),
            new_events=new_events,
        )
        wake_messages = [current_wake]
        # Keep live references until this wake is persisted. If execution
        # fails, the wrapper below saves the partial conversation as well.
        state["_pending_wake"] = {
            "tick_index": tick_index, "time_s": t,
            "messages": wake_messages,
        }
        camera_visual = kwargs.get("_camera_visual")
        if camera_visual is not None:
            _save_visual_observation(
                state, camera_visual, entity_id=vehicle_id,
                tick_index=tick_index, turn_index=0,
                label="CameraVisual")
            wake_messages.append(multimodal_image_message(
                camera_visual, label="CameraVisual"))
        lidar_visual = kwargs.get("_lidar_visual")
        if lidar_visual is not None:
            _save_visual_observation(
                state, lidar_visual, entity_id=vehicle_id,
                tick_index=tick_index, turn_index=0,
                label="LidarBEV")
            wake_messages.append(multimodal_image_message(
                lidar_visual, label="LidarBEV"))
        capability_signature = (
            rolling_context.capability_epoch,
            tuple(sorted(
                item.get("function", {}).get("name", "")
                for item in tools)),
            tuple(sorted(loaded_skill_contents)),
        )
        if (capability_signature
                != state["last_announced_capability_signature"]):
            wake_messages.append(build_loaded_capabilities_message(
                tools, loaded_skill_contents,
                rolling_context.capability_epoch))
            state["last_announced_capability_signature"] = \
                capability_signature

        real_msgs = list(passenger_messages)
        active_passenger_request_ids = tuple(
            str(request_id) for request_id in
            kwargs.get("_active_passenger_request_ids", ())
            if request_id)
        passenger_response_authorized = _passenger_response_authorized(
            real_msgs, active_passenger_request_ids)
        with lock:
            print(
                f"    [{vehicle_id}|{agent_client.model}] Tick {tick_index}, "
                f"t={t:.0f}s, {len(tools)} tools"
                f"{', passenger: ' + real_msgs[0][:40] if real_msgs else ''}")

        wake_finished = False
        completed_naturally = False
        for turn in range(1, effective_max_turns + 1):
            budget_record = None
            try:
                messages, budget_record = rolling_context.prepare_request(
                    wake_messages=wake_messages,
                    tools=tools,
                    core_tool_names=set(state["core_tool_names"]),
                    loaded_skills=loaded_skill_contents,
                    entity_id=vehicle_id,
                    sim_time_s=t,
                    tick_index=tick_index,
                    turn_index=turn,
                )
                # Add the exact remaining budget to CurrentWake without a
                # fourth top-level context section or cross-wake retention.
                for message in reversed(messages):
                    if message.get("role") != "user" \
                            or not isinstance(message.get("content"), str):
                        continue
                    try:
                        payload = json.loads(message["content"])
                    except (TypeError, ValueError, json.JSONDecodeError):
                        continue
                    if payload.get("type") != "wake":
                        continue
                    payload["decision_budget"] = {
                        "current_turn": int(turn),
                        "max_turns": int(effective_max_turns),
                        "turns_remaining_including_current": int(
                            effective_max_turns - turn + 1),
                        "final_turn_rule": (
                            "Prioritize a necessary action or finish; a "
                            "query cannot be followed by another decision."
                        ),
                    }
                    message["content"] = json.dumps(
                        payload, ensure_ascii=False, separators=(",", ":"))
                    break
                if (budget_record["unloaded_tools"]
                        or budget_record["unloaded_skills"]):
                    state["protocol_events"].append({
                        **copy.deepcopy(budget_record),
                        "type": "capabilities_unloaded",
                    })
                state["loaded_skills"] = list(loaded_skill_contents)
                state["capability_epoch"] = rolling_context.capability_epoch
                state["messages"] = copy.deepcopy(messages)
                state["context_snapshot_log"].append({
                    "entity_id": vehicle_id,
                    "time_s": round(float(t), 6),
                    "tick_index": int(tick_index),
                    "turn_index": int(turn),
                    "messages": _messages_for_log(messages),
                    "tools": copy.deepcopy(tools),
                    "capability_epoch": rolling_context.capability_epoch,
                })
                response_msg, _, pt, ct = agent_client.chat_with_tools(
                    messages, tools=tools)
            except ContextWindowExceeded as exc:
                # If this wake has already made progress, exhausting its
                # online transcript is an agent/harness turn boundary, not a
                # reason to invalidate and retry the entire physical episode.
                # Submitted commands remain active, while Todo and the next
                # CurrentWake carry the unfinished work forward.
                if len(state["model_call_log"]) > model_calls_before:
                    last_budget = (
                        copy.deepcopy(rolling_context.budget_log[-1])
                        if rolling_context.budget_log else None)
                    state["protocol_events"].append({
                        "entity_id": vehicle_id,
                        "time_s": round(float(t), 6),
                        "tick_index": int(tick_index),
                        "turn_index": int(turn),
                        "type": "wake_context_limit_reached",
                        "error": str(exc),
                        "context_budget": last_budget,
                        "commands_already_submitted_remain_queued": True,
                        "unfinished_work_remains_in_todo": True,
                    })
                    with lock:
                        print(
                            f"    [{vehicle_id}|{agent_client.model}] Turn "
                            f"{turn}: wake context limit reached; continuing "
                            "the episode on the next wake")
                    break
                error = {
                    "entity_id": vehicle_id,
                    "time_s": round(float(t), 6),
                    "tick_index": int(tick_index),
                    "turn_index": int(turn),
                    "error": f"{type(exc).__name__}: {exc}",
                }
                state["infrastructure_errors"].append(error)
                raise
            except Exception as exc:
                error = {
                    "entity_id": vehicle_id,
                    "time_s": round(float(t), 6),
                    "tick_index": int(tick_index),
                    "turn_index": int(turn),
                    "error": f"{type(exc).__name__}: {exc}",
                }
                state["infrastructure_errors"].append(error)
                if getattr(agent_client, "last_call_metadata", None):
                    state["model_call_log"].append({
                        **copy.deepcopy(agent_client.last_call_metadata),
                        **error,
                        "wake_event_ids": wake_event_ids,
                        "context_budget": copy.deepcopy(budget_record),
                    })
                with lock:
                    print(
                        f"    [{vehicle_id}|{agent_client.model}] Turn "
                        f"{turn}: infrastructure error: {exc}")
                raise

            state["model_call_log"].append({
                **copy.deepcopy(getattr(
                    agent_client, "last_call_metadata", {}) or {}),
                "entity_id": vehicle_id,
                "time_s": round(float(t), 6),
                "tick_index": int(tick_index),
                "turn_index": int(turn),
                "wake_event_ids": wake_event_ids,
                "context_budget": copy.deepcopy(budget_record),
            })
            state["token_stats"]["input"].append(pt)
            state["token_stats"]["output"].append(ct)
            state["total_turns"] += 1
            if response_msg is None:
                error = "AgentClient returned no response message"
                state["infrastructure_errors"].append({
                    "entity_id": vehicle_id,
                    "time_s": round(float(t), 6),
                    "tick_index": int(tick_index),
                    "turn_index": int(turn),
                    "error": error,
                })
                raise RuntimeError(error)

            serialized_response = _message_to_dict(response_msg)
            state["_last_response"] = copy.deepcopy(serialized_response)
            if not response_msg.tool_calls:
                finish_reason = str(
                    (getattr(agent_client, "last_call_metadata", {}) or {})
                    .get("finish_reason", "") or "")
                if finish_reason == "length":
                    # Never replay an empty assistant turn.  Strict hosted
                    # endpoints reject it before the model can see the legal
                    # recovery user message.  Provider reasoning content is
                    # preserved by _message_to_dict when available.
                    if _assistant_message_has_payload(serialized_response):
                        wake_messages.append(serialized_response)
                    state["protocol_events"].append({
                        "entity_id": vehicle_id,
                        "time_s": round(float(t), 6),
                        "tick_index": int(tick_index),
                        "turn_index": int(turn),
                        "type": "response_truncated",
                        "max_tokens": int(getattr(
                            agent_client, "max_tokens", 0) or 0),
                    })
                    wake_messages.append({
                        "role": "user",
                        "content": (
                            "[Harness] Your previous response reached the "
                            "output limit before completing this wake. "
                            "Continue concisely from the available facts: "
                            "call the necessary action/query tool now, or "
                            "call finish if no action is needed."),
                    })
                    continue
                if _assistant_message_has_payload(serialized_response):
                    wake_messages.append(serialized_response)
                completed_naturally = True
                state["protocol_events"].append({
                    "entity_id": vehicle_id,
                    "time_s": round(float(t), 6),
                    "tick_index": int(tick_index),
                    "turn_index": int(turn),
                    "type": "implicit_finish",
                })
                break

            wake_messages.append(serialized_response)

            for tc in response_msg.tool_calls:
                func_name = tc.function.name
                minimap_image = None
                capabilities_before = (
                    tuple(sorted(
                        item.get("function", {}).get("name", "")
                        for item in tools)),
                    tuple(sorted(loaded_skill_contents)),
                )
                try:
                    func_args = json.loads(
                        tc.function.arguments) if tc.function.arguments else {}
                    argument_error = None
                    if not isinstance(func_args, dict):
                        argument_error = {
                            "success": False, "error": "invalid_tool_arguments",
                            "message": "arguments must be a JSON object",
                        }
                except (ValueError, TypeError) as exc:
                    func_args = None
                    argument_error = {
                        "success": False, "error": "invalid_tool_arguments",
                        "message": f"Invalid JSON arguments: {exc}",
                    }

                state["total_tool_calls"] += 1
                tool_kind = _classify_vehicle_tool(func_name)
                counter_key = {
                    "discovery": "tool_calls_discovery",
                    "memory": "tool_calls_memory",
                    "harness": "tool_calls_harness",
                    "perception": "tool_calls_perception",
                    "query": "tool_calls_query",
                    "action": "tool_calls_action",
                }.get(tool_kind, "tool_calls_action")
                state[counter_key] += 1

                # Append BEFORE executing. A failed dispatch must not erase
                # the only evidence of which tool and arguments triggered it.
                call_record = {
                    "vehicle_id": vehicle_id,
                    "time_s": round(float(t), 6),
                    "tick_index": int(tick_index),
                    "turn_index": int(turn),
                    "tool_call_index": len(state["tool_call_log"]),
                    "tool_call_id": tc.id,
                    "function": func_name,
                    "arguments": copy.deepcopy(func_args),
                    "raw_arguments": tc.function.arguments,
                    "result": None, "execution_status": "started",
                    "kind": tool_kind, "wake_event_ids": wake_event_ids,
                }
                state["tool_call_log"].append(call_record)
                state["_active_tool_call"] = call_record

                if wake_finished:
                    result = {
                        "success": False, "error": "wake_already_finished"}
                elif argument_error is not None:
                    result = argument_error
                elif func_name == "finish":
                    wake_finished = True
                    result = {
                        "success": True,
                        "wake_finished": True,
                        "commands_already_submitted_remain_queued": True,
                    }
                    state["protocol_events"].append({
                        "entity_id": vehicle_id,
                        "time_s": round(float(t), 6),
                        "tick_index": int(tick_index),
                        "turn_index": int(turn),
                        "type": "explicit_finish",
                        "reason": str(func_args.get("reason", "")),
                    })
                elif func_name == "todo_manage":
                    before = todo_store.audit_dict()
                    result = todo_store.apply(
                        func_args.get("operations", []), now_s=t,
                        expected_revision=func_args.get("expected_revision"))
                    state["todo_audit_log"].append({
                        "entity_id": vehicle_id,
                        "time_s": round(float(t), 6),
                        "tick_index": int(tick_index),
                        "before": before,
                        "result": copy.deepcopy(result),
                        "after": todo_store.audit_dict(),
                    })
                elif func_name == "set_heartbeat_interval":
                    previous = float(state["heartbeat_interval_s"])
                    try:
                        requested = float(func_args.get("interval_s"))
                    except (TypeError, ValueError):
                        requested = float("nan")
                    if not (MIN_LLM_HEARTBEAT_INTERVAL_S
                            <= requested
                            <= MAX_LLM_HEARTBEAT_INTERVAL_S):
                        result = {
                            "success": False,
                            "error": "heartbeat_interval_out_of_range",
                            "minimum_s": MIN_LLM_HEARTBEAT_INTERVAL_S,
                            "maximum_s": MAX_LLM_HEARTBEAT_INTERVAL_S,
                            "current_interval_s": previous,
                        }
                    else:
                        state["heartbeat_interval_s"] = requested
                        state["heartbeat_interval_revision"] += 1
                        result = {
                            "success": True,
                            "previous_interval_s": previous,
                            "heartbeat_interval_s": requested,
                            "applies_after_current_wake": True,
                            "event_wake_can_preempt": True,
                        }
                    state["heartbeat_interval_audit_log"].append({
                        "entity_id": vehicle_id,
                        "time_s": round(float(t), 6),
                        "tick_index": int(tick_index),
                        "requested_interval_s": func_args.get("interval_s"),
                        "reason": str(func_args.get("reason", "")),
                        "result": copy.deepcopy(result),
                    })
                elif func_name == "schedule_next_wake":
                    try:
                        delay = float(func_args.get("delay_s"))
                    except (TypeError, ValueError):
                        delay = float("nan")
                    reason = str(func_args.get("reason", "")).strip()
                    if (not math.isfinite(delay)
                            or not MIN_LLM_HEARTBEAT_INTERVAL_S <= delay
                            <= MAX_LLM_HEARTBEAT_INTERVAL_S):
                        result = {
                            "success": False,
                            "error": "scheduled_wake_delay_out_of_range",
                            "minimum_s": MIN_LLM_HEARTBEAT_INTERVAL_S,
                            "maximum_s": MAX_LLM_HEARTBEAT_INTERVAL_S,
                        }
                    elif not reason:
                        result = {
                            "success": False,
                            "error": "scheduled_wake_reason_required",
                        }
                    else:
                        target = round(float(t) + delay, 6)
                        previous = state["scheduled_wake_at_s"]
                        state["scheduled_wake_at_s"] = target
                        state["scheduled_wake_reason"] = reason[:120]
                        state["scheduled_wake_revision"] += 1
                        result = {
                            "success": True,
                            "scheduled_wake_at_s": target,
                            "delay_s": delay,
                            "replaced_scheduled_wake_at_s": previous,
                            "event_wake_can_preempt": True,
                        }
                    state["scheduled_wake_audit_log"].append({
                        "entity_id": vehicle_id,
                        "time_s": round(float(t), 6),
                        "tick_index": int(tick_index),
                        "requested_delay_s": func_args.get("delay_s"),
                        "reason": reason[:120],
                        "result": copy.deepcopy(result),
                    })
                elif func_name == "memory_search":
                    result = _dispatch_memory_search(memory, func_args)
                elif func_name == _DEVICE_STATE_TOOL_NAME:
                    result = _dispatch_device_state(vw, func_args)
                elif (func_name in _PASSENGER_BROADCAST_ACTION_NAMES
                      and not passenger_response_authorized):
                    result = {
                        "success": False,
                        "error": "explicit_passenger_request_required",
                        "message": (
                            "Cabin safety broadcasts are passenger-response "
                            "features; they cannot be used as automatic road "
                            "condition announcements."),
                    }
                elif func_name == "navigation__navigation_minimap":
                    render_minimap = kwargs.get(
                        "_render_navigation_minimap")
                    if render_minimap is None:
                        result = {
                            "success": False,
                            "error": "navigation_minimap_unavailable",
                        }
                    else:
                        try:
                            # Viewing changes the camera/ego pose, not guidance.
                            # Pass the saved preview explicitly, including None:
                            # only an explicit planning call may replace it.
                            minimap_image = render_minimap(
                                vehicle_id, t,
                                scope=str(func_args.get("scope", "route")),
                                route_preview=getattr(
                                    vw.navigation,
                                    "_lane_route_preview", None),
                            )
                            result = {
                                "success": True,
                                "image_attached": True,
                                **minimap_image.metadata(),
                            }
                        except Exception as exc:
                            minimap_image = None
                            result = {
                                "success": False,
                                "error": (
                                    f"{type(exc).__name__}: {exc}"),
                            }
                elif func_name in VEHICLE_PERCEPTION_TOOL_NAMES:
                    world_state = kwargs.get("world_state")
                    result = (
                        {"error": "agent world view unavailable"}
                        if world_state is None else
                        world_state.dispatch_tool(
                            vehicle_id, func_name, func_args))
                elif func_name == "load_skill":
                    skill_name = str(func_args.get("skill_name", ""))
                    available = (
                        vw.available_module_names()
                        if hasattr(vw, "available_module_names") else None)
                    content, result = load_skill_content(
                        skill_loader, skill_name,
                        available_modules=available)
                    if content is not None:
                        loaded_skill_contents[skill_name] = content
                        state["loaded_skills"] = list(
                            loaded_skill_contents)
                else:
                    result = dispatch_lazy_discovery(
                        vw, func_name, func_args, tools,
                        skill_loader=None,
                        extra_schemas=[_make_memory_search_tool()],
                        virtual_module_catalogs={
                            "session_history": _session_history_catalog(),
                        },
                        denied_tool_names=_VEHICLE_DENIED_TOOL_NAMES)

                capabilities_after = (
                    tuple(sorted(
                        item.get("function", {}).get("name", "")
                        for item in tools)),
                    tuple(sorted(loaded_skill_contents)),
                )
                if capabilities_after != capabilities_before:
                    rolling_context.capability_epoch += 1
                    state["capability_epoch"] = \
                        rolling_context.capability_epoch

                result_str = json.dumps(
                    result, ensure_ascii=False, default=str,
                    separators=(",", ":"))
                if tool_kind not in ("discovery", "memory", "harness"):
                    timestamp = SessionHistory.make_timestamp(t)
                    snippet = (
                        result_str[:150] + "..."
                        if len(result_str) > 150 else result_str)
                    action_log.append(ActionLogEntry(
                        timestamp=timestamp,
                        func_name=func_name,
                        arguments=json.dumps(
                            func_args, ensure_ascii=False,
                            separators=(",", ":")),
                        result_snippet=snippet,
                    ))
                if tool_kind not in (
                        "discovery", "memory", "query", "perception",
                        "harness"):
                    actions_taken.append(
                        f"{func_name}({json.dumps(func_args, ensure_ascii=False)})")

                call_record.update({
                    "result": result, "execution_status": "returned",
                })
                state["_active_tool_call"] = None
                wake_messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_str,
                })
                if (func_name == "navigation__navigation_minimap"
                        and result.get("success")
                        and minimap_image is not None):
                    _save_visual_observation(
                        state, minimap_image, entity_id=vehicle_id,
                        tick_index=tick_index, turn_index=turn,
                        label="NavigationMinimap")
                    wake_messages.append(multimodal_image_message(
                        minimap_image, label="NavigationMinimap"))
            if wake_finished:
                break

        if (not wake_finished and not completed_naturally
                and turn >= effective_max_turns):
            state["protocol_events"].append({
                "entity_id": vehicle_id,
                "time_s": round(float(t), 6),
                "tick_index": int(tick_index),
                "turn_index": int(turn),
                "type": "turn_limit_reached",
                "max_turns": effective_max_turns,
                "commands_already_submitted_remain_queued": True,
            })

        event_id = f"evt_{tick_index + 1:03d}"
        event_desc = (
            f"Wake at t={t:.1f}s with {len(new_events)} new events")
        memory.record_wake(
            time_s=t,
            event_description=event_desc,
            actions_taken=actions_taken,
            action_log=action_log,
            event_id=event_id,
            passenger_messages=real_msgs,
        )
        state["all_messages"].append({
            "tick_index": tick_index,
            "time_s": t,
            "messages": _messages_for_log(wake_messages),
        })
        state["_pending_wake"] = None
        rolling_context.finish_wake(wake_messages)
        state["previous_wake_messages"] = copy.deepcopy(
            rolling_context.previous_wake_messages)
        state["messages"] = rolling_context.compose(
            [], loaded_skill_contents)
        state["loaded_skills"] = list(loaded_skill_contents)
        state["capability_epoch"] = rolling_context.capability_epoch
        state["todo_state"] = todo_store.audit_dict()
        state["next_todo_deadline_s"] = todo_store.next_deadline_s(t)
        state["wake_runtime_log"].append({
            "entity_id": vehicle_id,
            "time_s": round(float(t), 6),
            "tick_index": int(tick_index),
            "wall_time_s": round(
                time.perf_counter() - wake_wall_started, 6),
            "model_calls": (
                len(state["model_call_log"]) - model_calls_before),
            "tool_calls": (
                len(state["tool_call_log"]) - tool_calls_before),
            "response_truncations": (
                sum(item.get("type") == "response_truncated"
                    for item in state["protocol_events"])
                - truncations_before),
        })
        return actions_taken

    def callback(vw, t, passenger_messages, memory, tick_index, **kwargs):
        state["_pending_wake"] = None
        state["_active_tool_call"] = None
        state["_last_response"] = None
        try:
            return run_wake(vw, t, passenger_messages, memory, tick_index, **kwargs)
        except Exception as exc:
            error = {
                "entity_id": vehicle_id, "time_s": round(float(t), 6),
                "tick_index": int(tick_index), "phase": "callback",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
            active = state.get("_active_tool_call")
            if active is not None:
                active.update({
                    "execution_status": "raised",
                    "result": {"success": False, "error": "tool_execution_failed"},
                    "exception": error["error"], "traceback": error["traceback"],
                })
                error["tool_call"] = copy.deepcopy(active)
            if state.get("_last_response") is not None:
                key = "model_response" if active is not None else "last_model_response"
                error[key] = state["_last_response"]
            state["infrastructure_errors"].append(error)
            pending = state.get("_pending_wake")
            if pending is not None:
                state["all_messages"].append({
                    **pending, "messages": _messages_for_log(pending["messages"]),
                    "incomplete": True, "error": error["error"],
                })
            # Do not swallow real code errors or imply partial actions were
            # rolled back. The engine retains its fatal/invalidate semantics.
            raise
        finally:
            state["_pending_wake"] = None
            state["_active_tool_call"] = None
            state["_last_response"] = None

    callback._state = state
    callback._vehicle_id = vehicle_id
    return callback


def make_noop_agent(vehicle_id: str):
    """Empty harness callback for a SUMO-owned background actor."""
    def callback(vw, t, passenger_messages, memory, tick_index, **kwargs):
        return []
    return callback


# ══════════════════════════════════════════════════════════════════════
#  Agent factory registry
# ══════════════════════════════════════════════════════════════════════

_AGENT_FACTORIES: Dict[Tuple[str, str], Callable] = {}


def register_agent_factory(agent_type: str, *, entity_type: str = "vehicle"):
    """Decorator: register a factory function for *agent_type*.

    The decorated function signature must be ``(vid: str, spec: AgentSpec) -> callback``.
    """
    def decorator(fn):
        _AGENT_FACTORIES[(entity_type, agent_type)] = fn
        return fn
    return decorator


@register_agent_factory("llm")
def _create_llm_agent(vid: str, spec: AgentSpec):
    client = AgentClient(
        api_base=spec.api_base,
        api_key=spec.api_key,
        model=spec.model,
        temperature=spec.temperature,
        max_tokens=spec.max_tokens,
        thinking_mode=spec.thinking_mode,
        reasoning_effort=spec.reasoning_effort,
        chat_template_enable_thinking=(
            spec.chat_template_enable_thinking),
    )
    return make_llm_agent_callback(
        vid, client, max_turns=spec.max_turns,
        context_window_tokens=spec.context_window_tokens,
        todo_max_ttl_s=spec.todo_max_ttl_s,
        heartbeat_interval_s=spec.heartbeat_interval_s,
        driver_prompt=spec.driver_prompt)


@register_agent_factory("sumo")
def _create_sumo_background_agent(vid: str, spec: AgentSpec):
    """No policy callback: SUMO owns this background actor."""
    return make_noop_agent(vid)


@register_agent_factory("llm", entity_type="pedestrian")
def _create_llm_pedestrian_agent(ped_id: str, spec: AgentSpec):
    from pedestrian_agent import make_llm_pedestrian_callback
    client = AgentClient(
        api_base=spec.api_base,
        api_key=spec.api_key,
        model=spec.model,
        temperature=spec.temperature,
        max_tokens=spec.max_tokens,
        thinking_mode=spec.thinking_mode,
        reasoning_effort=spec.reasoning_effort,
        chat_template_enable_thinking=(
            spec.chat_template_enable_thinking),
    )
    return make_llm_pedestrian_callback(
        ped_id, client, max_turns=spec.max_turns,
        context_window_tokens=spec.context_window_tokens,
        todo_max_ttl_s=spec.todo_max_ttl_s)


# ══════════════════════════════════════════════════════════════════════
#  Build callbacks from AgentSpecs
# ══════════════════════════════════════════════════════════════════════

def build_callbacks(
    specs: Dict[str, AgentSpec],
    personal_agent_runtime_config: Optional[Dict[str, Any]] = None,
    passenger_judge_runtime_config: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Callable], Dict[str, dict]]:
    """Create agent callbacks from resolved specs.

    Returns:
        (callbacks, vehicle_meta) where vehicle_meta has per-vehicle info
        like agent_type and model for reporting.
    """
    personal_config = {
        key: value
        for key, value in (personal_agent_runtime_config or {}).items()
        if value is not None
    }
    personal_enabled = bool(personal_config.pop("enabled", False))
    judge_config = {
        key: value
        for key, value in (passenger_judge_runtime_config or {}).items()
        if value is not None
    }
    judge_config.pop("enabled", None)
    trigger_mode = personal_config.pop("trigger_mode", "event_random")
    if trigger_mode not in {"event_random", "legacy"}:
        raise ValueError("personal trigger_mode must be event_random or legacy")
    schedule_fields = {"seed", "random_min_s", "random_max_s", "cooldown_s",
        "stopped_after_s", "hard_brake_mps2", "hard_brake_duration_s", "event_types"}
    schedule_config = {key: personal_config.pop(key) for key in schedule_fields
                       if key in personal_config}
    judge_window_s = judge_config.pop("window_s", 1.0 if trigger_mode == "legacy" else None)
    judge_max_checks = judge_config.pop("max_checks", 3)
    request_ttl_s = judge_config.pop("request_ttl_s", None)
    judge_check_offsets_s = judge_config.pop("check_offsets_s", None)
    acceptance_timeout_s = judge_config.pop("acceptance_timeout_s", None)
    if judge_window_s is not None and float(judge_window_s) <= 0.0:
        raise ValueError("passenger-judge window_s must be positive")
    role_client_fields = {
        "api_base", "api_key", "model", "temperature", "max_tokens",
        "thinking_mode", "reasoning_effort",
        "chat_template_enable_thinking",
    }
    # The context window is an experiment-side prompt budget/provenance field,
    # not an OpenAI chat-completions request parameter.  Accept and validate it
    # here while keeping it out of AgentClient(**kwargs).
    role_runtime_fields = role_client_fields | {"context_window_tokens"}
    for name, config in (
            ("personal-agent", personal_config),
            ("passenger-judge", judge_config)):
        if ("context_window_tokens" in config
                and int(config["context_window_tokens"])
                <= int(config.get("max_tokens", 0))):
            raise ValueError(
                f"{name} context_window_tokens must exceed max_tokens")
    unknown_personal = set(personal_config) - role_runtime_fields - {
        "persona", "long_term_goal",
    }
    unknown_judge = set(judge_config) - role_runtime_fields
    if unknown_personal:
        raise ValueError(
            "Unknown personal-agent runtime fields: "
            f"{sorted(unknown_personal)!r}")
    if unknown_judge:
        raise ValueError(
            "Unknown passenger-judge runtime fields: "
            f"{sorted(unknown_judge)!r}")

    def role_client(spec: AgentSpec, config: Dict[str, Any], *, judge=False):
        base = {
            "api_base": spec.api_base,
            "api_key": spec.api_key,
            "model": spec.model,
            # Inherit the provider-compatible vehicle setting. Some hosted
            # models (including Kimi-K3 endpoints) accept only temperature=1.
            # Deterministic judges may still override this explicitly when
            # their provider permits it.
            "temperature": spec.temperature,
            "max_tokens": (
                DEFAULT_PASSENGER_JUDGE_MAX_OUTPUT_TOKENS
                if judge else min(spec.max_tokens, 2048)),
            "thinking_mode": spec.thinking_mode,
            "reasoning_effort": spec.reasoning_effort,
            "chat_template_enable_thinking": (
                spec.chat_template_enable_thinking),
        }
        base.update({
            key: value for key, value in config.items()
            if key in role_client_fields
        })
        return AgentClient(**base)

    callbacks = {}
    vehicle_meta = {}

    for vid, spec in specs.items():
        vehicle_meta[vid] = {
            "entity_type": spec.entity_type,
            "agent_type": spec.agent_type,
            "model": spec.model if spec.agent_type == "llm" else "",
            "is_evaluated": spec.is_evaluated,
            "temperature": spec.temperature if spec.agent_type == "llm" else None,
            "max_turns": spec.max_turns if spec.agent_type == "llm" else None,
            "max_tokens": spec.max_tokens if spec.agent_type == "llm" else None,
            "thinking_mode": (
                spec.thinking_mode if spec.agent_type == "llm" else None),
            "reasoning_effort": (
                spec.reasoning_effort if spec.agent_type == "llm" else None),
            "chat_template_enable_thinking": (
                spec.chat_template_enable_thinking
                if spec.agent_type == "llm" else None),
            "context_window_tokens": (
                spec.context_window_tokens
                if spec.agent_type == "llm" else None),
            "todo_max_ttl_s": (
                spec.todo_max_ttl_s if spec.agent_type == "llm" else None),
            "heartbeat_interval_s": (
                spec.heartbeat_interval_s
                if spec.agent_type == "llm" else None),
            "api_base": spec.api_base if spec.agent_type == "llm" else "",
            "driver_prompt": spec.driver_prompt if spec.agent_type == "llm" else "",
            "personal_agent_enabled": (
                spec.personal_agent_enabled if spec.agent_type == "llm" else None),
            "passenger_judge_enabled": (
                spec.passenger_judge_enabled if spec.agent_type == "llm" else None),
        }

        factory = _AGENT_FACTORIES.get((spec.entity_type, spec.agent_type))
        if factory is None:
            raise ValueError(
                f"Unknown {spec.entity_type} agent type: "
                f"{spec.agent_type!r} for {vid}. Available: "
                f"{sorted(_AGENT_FACTORIES)}"
            )
        callbacks[vid] = factory(vid, spec)
        if (personal_enabled and spec.personal_agent_enabled
                and spec.passenger_judge_enabled
                and spec.entity_type == "vehicle"
                and spec.agent_type == "llm"):
            # Every LLM-driven vehicle gets its own PA/Judge, including fixed
            # background peers. A SUMO-driven focal never receives either
            # in-cabin model during reference calibration.
            personal = PersonalAgentRuntime(
                vid, role_client(spec, personal_config),
                persona=str(personal_config.get("persona", "")),
                long_term_goal=str(
                    personal_config.get("long_term_goal", "")),
            )
            judge_source = dict(personal_config)
            judge_source.update(judge_config)
            # Judging should be reproducible by default and must not inherit a
            # sampling temperature chosen for the driver or passenger persona.
            # An explicit judge-only setting can still override this default.
            judge_source["temperature"] = judge_config.get("temperature", 0.0)
            judge = PassengerJudgeRuntime(
                vid, role_client(spec, judge_source, judge=True))
            if trigger_mode == "legacy":
                callbacks[vid] = PersonalAgentVehicleCallback(
                    vid, callbacks[vid], personal, judge,
                    judge_window_s=judge_window_s)
            else:
                from passenger_orchestration import EventDrivenPassengerCallback
                callbacks[vid] = EventDrivenPassengerCallback(
                    vid, callbacks[vid], personal, judge,
                    judge_window_s=judge_window_s, schedule_config=schedule_config,
                    max_checks=judge_max_checks, request_ttl_s=request_ttl_s,
                    check_offsets_s=judge_check_offsets_s,
                    acceptance_timeout_s=acceptance_timeout_s)
        callbacks[vid]._agent_type = spec.agent_type
        callbacks[vid]._entity_type = spec.entity_type
        callbacks[vid]._effective_agent_config = copy.deepcopy(
            vehicle_meta[vid])

    return callbacks, vehicle_meta


# ══════════════════════════════════════════════════════════════════════
#  MultiSimEngine with traffic tracking
# ══════════════════════════════════════════════════════════════════════

class TrackedMultiSimEngine(MultiSimEngine):
    """MultiSimEngine + driving-API intercepts + traffic/encounter/position logs.

    The base engine already runs the physics loop; this subclass only:
      1. Installs driving-API intercepts on evaluated vehicles' VWs so LLM
         agents can control navigation.
      2. Records per-substep encounter logs and per-heartbeat position logs
         used by visualization / trajectory inspection tools.
    """

    def __init__(self, scenario):
        super().__init__(scenario)
        self._traffic_log: List[dict] = []
        self._encounter_log: List[dict] = []
        self._position_log: List[dict] = []
        self._next_heartbeat: float = 0.0  # tracked for position_log gating

    # ── Hooks ─────────────────────────────────────────────────────

    def _post_vehicle_init(self, vid: str, vcfg) -> None:
        """Install driving-API intercepts on evaluated vehicles."""
        super()._post_vehicle_init(vid, vcfg)
        vehicle = self.traffic_mgr.get_state(vid)
        if (vcfg.is_evaluated
                and vehicle is not None
                and not vehicle.is_llm):
            self._install_driving_intercepts(vid)

    def _log_per_substep(self, physics_time, tick_index, trigger_events):
        # Fine-grained encounter log: every sub-step, for every evaluated
        # vehicle, record any vehicles_ahead entries within perception range.
        for vcfg in self.scenario.vehicles:
            if not vcfg.is_evaluated:
                continue
            vid = vcfg.vehicle_id
            vs = self.traffic_mgr.get_state(vid)
            if not vs or vs.arrived or vs.route_failed or vs.is_crashed:
                continue
            directions = self.traffic_mgr.get_directions_at(vid)
            for v in directions.get("vehicles_ahead", []):
                self._encounter_log.append({
                    "tick": tick_index, "time": round(physics_time, 4),
                    "vehicle": vid,
                    "vehicle_node": vs.current_node,
                    "other_id": v["vehicle_id"],
                    "distance_m": v["distance_m"],
                    "speed_kmh": v["speed_kmh"],
                    "lane": v["lane"],
                })
        # Traffic log: collect crashed/arrived/intersection/collision events.
        for te in trigger_events:
            if te.type in ("crashed", "collision_warning", "arrived", "route_failed",
                           "intersection_arrival"):
                self._traffic_log.append({
                    "tick": tick_index, "time": round(physics_time, 4),
                    "vehicle": te.vehicle_id,
                    "flags": [te.type],
                })

    def _log_per_wake(self, physics_time, tick_index, is_heartbeat,
                      trigger_events):
        # Position log only at heartbeat (constant frame rate for GIF).
        if not is_heartbeat:
            return
        tick_positions = {}
        for vcfg in self.scenario.vehicles:
            vs = self.traffic_mgr.get_state(vcfg.vehicle_id)
            if vs:
                tick_positions[vcfg.vehicle_id] = {
                    "node": vs.current_node,
                    "segment": vs.current_segment,
                    "lane": vs.current_lane,
                    "lane_route_action_index": vs.lane_route_action_index,
                    "arrived": vs.arrived,
                    "route_failed": vs.route_failed,
                    "route_failure_reason": vs.route_failure_reason,
                    "route_failure_time_s": vs.route_failure_time_s,
                    "present_in_physics_world": (
                        vs.present_in_physics_world),
                    "terminal_crossing_speed_kmh": (
                        vs.terminal_crossing_speed_kmh),
                    "pose_x_m": round(float(vs.pose_x_m), 4),
                    "pose_y_m": round(float(vs.pose_y_m), 4),
                    "yaw_rad": round(float(vs.yaw_rad), 6),
                    "speed_kmh": round(float(vs.current_speed_kmh), 3),
                    "signal_state": vs.signal_state.as_dict(),
                    "perception_profile": vs.perception_profile_name,
                }
        self._position_log.append({
            "tick": tick_index,
            "time": round(physics_time, 4),
            "positions": tick_positions,
        })

    def _finalize_result(self, result):
        super()._finalize_result(result)
        result._traffic_log = self._traffic_log
        result._encounter_log = self._encounter_log
        result._position_log = self._position_log

# ══════════════════════════════════════════════════════════════════════
#  run_multi_scenario — one-call entry point
# ══════════════════════════════════════════════════════════════════════

def run_multi_scenario(
    scenario_dict: dict,
    agent_overrides: Optional[Dict[str, dict]] = None,
    llm_runtime_config: Optional[Dict[str, Any]] = None,
    engine_class=None,
    extensions: Optional[List[str]] = None,
    personal_agent_runtime_config: Optional[Dict[str, Any]] = None,
    passenger_judge_runtime_config: Optional[Dict[str, Any]] = None,
) -> Tuple[MultiSimResult, float, Dict[str, dict], Dict[str, Callable]]:
    """End-to-end: parse scenario, resolve agents, run, return results.

    Args:
        scenario_dict: Scenario definition dict.
        agent_overrides: Optional per-vehicle overrides, e.g.
            {"ego": {"type": "llm", "model": "qwen3-235b"}}
        llm_runtime_config: Run-level model endpoint, model, output budget,
            context window and Todo TTL applied to every LLM entity.
        engine_class: Engine class to use (default: TrackedMultiSimEngine).
        extensions: Trusted Python import paths loaded before scenario parsing.
        personal_agent_runtime_config: Set ``enabled=true`` to run a passenger
            PA with event subscriptions and an independent seeded random clock.
            ``trigger_mode=legacy`` retains the old driver-wake coupling.
        passenger_judge_runtime_config: Optional provider/model, check interval
            ``check_offsets_s`` (default [0.1, 1, 3]), ``max_checks`` and
            ``acceptance_timeout_s``; old ``window_s``/``request_ttl_s`` inputs
            remain supported for fixed-interval schedules. Legacy mode
            retains one-shot judging.

    Returns:
        (result, elapsed_seconds, vehicle_stats, callbacks)
    """
    if engine_class is None:
        engine_class = TrackedMultiSimEngine

    if extensions:
        from extensions import load_extensions
        load_extensions(extensions)

    scenario = MultiScenario.from_dict(scenario_dict)
    specs = resolve_agent_specs(
        scenario, agent_overrides,
        llm_runtime_config=llm_runtime_config)
    # Prevent a run-level LLM override from leaving the physical entity
    # registered as a SUMO-controlled background actor.
    apply_resolved_agent_authorities(scenario, specs)
    engine = engine_class(scenario)

    callbacks, vehicle_meta = build_callbacks(
        specs,
        personal_agent_runtime_config=personal_agent_runtime_config,
        passenger_judge_runtime_config=passenger_judge_runtime_config,
    )

    # Print header
    vehicle_specs = [s for s in specs.values()
                     if s.entity_type == "vehicle"]
    pedestrian_specs = [s for s in specs.values()
                        if s.entity_type == "pedestrian"]
    n_llm = sum(1 for s in vehicle_specs if s.agent_type == "llm")
    n_sumo = len(vehicle_specs) - n_llm

    print(f"\n{'='*70}")
    print(f"Scenario: {scenario_dict.get('name', scenario_dict['scenario_id'])}")
    print(f"  Vehicles: {n_llm} LLM + {n_sumo} SUMO background")
    if pedestrian_specs:
        print(f"  LLM pedestrians: {len(pedestrian_specs)}")
    for vid, spec in specs.items():
        print(f"    {vid}: {spec.display_name} "
              f"({'evaluated' if spec.is_evaluated else 'background'})")
    print(f"  Duration: {scenario_dict.get('total_time_s', 3600)}s, "
          f"interval: {scenario_dict.get('tick_interval_s', 300)}s")
    print(f"{'='*70}")

    t0 = time.time()
    result = engine.run(callbacks)
    elapsed = time.time() - t0

    for vid, callback in callbacks.items():
        state = getattr(callback, "_state", {})
        evaluations = state.get("passenger_evaluations")
        if evaluations is None or vid not in result.vehicle_results:
            continue
        result.vehicle_results[vid].passenger_evaluation = (
            aggregate_passenger_judgements(
                evaluations,
                request_count=len(state.get("personal_requests", []))))

    # Collect stats
    vehicle_stats = {}
    for vid, cb in callbacks.items():
        meta = vehicle_meta.get(vid, {})
        if hasattr(cb, '_state'):
            st = cb._state
            vehicle_stats[vid] = {
                "agent_type": meta.get("agent_type", "llm"),
                "model": st.get("model", meta.get("model", "")),
                "total_turns": st["total_turns"],
                "total_tool_calls": st["total_tool_calls"],
                "tool_calls_discovery": st["tool_calls_discovery"],
                "tool_calls_query": st["tool_calls_query"],
                "tool_calls_perception": st.get(
                    "tool_calls_perception", st.get("perception_calls", 0)),
                "tool_calls_action": st["tool_calls_action"],
                "tool_calls_memory": st["tool_calls_memory"],
                "tool_calls_harness": st.get("tool_calls_harness", 0),
                "protocol_events": copy.deepcopy(
                    st.get("protocol_events", [])),
                "context_window_tokens": st.get(
                    "effective_config", {}).get("context_window_tokens"),
                "capability_epoch": st.get("capability_epoch", 0),
                "capability_unloads": sum(
                    1 for item in st.get("protocol_events", [])
                    if item.get("type") == "capabilities_unloaded"),
                "todo_state": copy.deepcopy(st.get("todo_state", {})),
                "effective_config": copy.deepcopy(
                    st.get("effective_config", {})),
                "total_input_tokens": sum(st["token_stats"]["input"]),
                "total_output_tokens": sum(st["token_stats"]["output"]),
                "personal_agent": copy.deepcopy(st.get("personal_agent")),
                "passenger_judge": copy.deepcopy(st.get("passenger_judge")),
                "passenger_evaluation": copy.deepcopy(
                    st.get("passenger_evaluations", [])),
            }
        else:
            vehicle_stats[vid] = {
                "agent_type": meta.get("agent_type", "sumo"),
                "model": "",
            }

    return result, elapsed, vehicle_stats, callbacks


# ══════════════════════════════════════════════════════════════════════
#  Report printing
# ══════════════════════════════════════════════════════════════════════

def print_report(scenario_id: str, result: MultiSimResult, elapsed: float,
                 vehicle_stats: Dict[str, dict]):
    """Print detailed report with per-vehicle results and traffic analysis."""
    print(f"\n{'─'*70}")
    print(f"Results: {scenario_id} ({elapsed:.0f}s)")
    print(f"{'─'*70}")

    for vid, vr in result.vehicle_results.items():
        vs = vehicle_stats.get(vid, {})
        agent_type = vs.get("agent_type", "?")
        model = vs.get("model", "")
        label = f"{agent_type}({model})" if model else agent_type

        if not vr.is_evaluated:
            print(f"\n  {vid} [{label}]: not evaluated (background)")
            print(f"    Arrived: {'tick ' + str(vr.arrival_tick) if vr.arrived else 'no'}")
            continue

        total = sum(c.get("total_fields", 0) for c in vr.checkpoints)
        correct = sum(c.get("correct_fields", 0) for c in vr.checkpoints)
        acc = correct / total if total > 0 else None
        n_cp = len(vr.checkpoints)
        n_pass = sum(1 for c in vr.checkpoints if c.get("accuracy", 0) == 1.0)

        print(f"\n  {vid} [{label}]:")
        print(
            f"    Cabin YAML:  {acc:.1%} ({correct}/{total})"
            if acc is not None else
            "    Cabin YAML:  N/A (no applicable YAML rule)")
        print(f"    Rule events: {n_pass}/{n_cp} passed")
        driving = vr.driving_evaluation
        if driving:
            scores = driving.get("dimension_scores", {})
            print(
                f"    Driving:     "
                f"{driving.get('trajectory_quality_score', 0):.1%} "
                f"(safety={scores.get('safety', 0):.1%}, "
                f"rules={scores.get('compliance', 0):.1%}, "
                f"comfort={scores.get('comfort', 0):.1%}, "
                f"interaction={scores.get('interaction', 0):.1%}, "
                f"efficiency={scores.get('efficiency', 0):.1%})")
            print(
                "    Hard safety: "
                + ("PASS" if driving.get("hard_safety_passed") else "FAIL"))
        print(f"    Arrived:     {'tick ' + str(vr.arrival_tick) if vr.arrived else 'no'}")

        if agent_type == "llm":
            print(f"    LLM Turns:   {vs.get('total_turns', 0)}")
            print(f"    Tool Calls:  {vs.get('total_tool_calls', 0)} "
                  f"(disc={vs.get('tool_calls_discovery', 0)}, "
                  f"query={vs.get('tool_calls_query', 0)}, "
                  f"action={vs.get('tool_calls_action', 0)}, "
                  f"mem={vs.get('tool_calls_memory', 0)})")
            print(f"    Tokens:      in={vs.get('total_input_tokens', 0):,}, "
                  f"out={vs.get('total_output_tokens', 0):,}")

        for cp in vr.checkpoints:
            if cp.get("accuracy", 1.0) < 1.0:
                print(f"    [FAIL] t={cp['tick']}: "
                      f"{cp.get('correct_fields',0)}/{cp.get('total_fields',0)}")
                for d in cp.get("details", []):
                    if not d.get("passed"):
                        print(f"      {d['field']}: "
                              f"expected={d.get('expected')} "
                              f"actual={d.get('actual')}")

    # Traffic analysis
    traffic_log = getattr(result, '_traffic_log', [])
    encounter_log = getattr(result, '_encounter_log', [])
    position_log = getattr(result, '_position_log', [])

    if traffic_log or encounter_log:
        print(f"\n{'─'*70}")
        print("Traffic Interaction Analysis")
        print(f"{'─'*70}")

        blocked = [e for e in traffic_log if "blocked_by_vehicle" in e.get("flags", [])]
        print(f"\n  Blocked events: {len(blocked)}")
        for be in blocked:
            print(f"    tick {be['tick']} (t={be['time']:.0f}s): {be['vehicle']} blocked")

        if encounter_log:
            by_pair = {}
            for enc in encounter_log:
                key = (enc["vehicle"], enc["other_id"])
                by_pair.setdefault(key, []).append(enc)

            print(f"\n  Vehicle encounters: {len(encounter_log)}")
            for (ego_id, npc_id), encs in by_pair.items():
                min_dist = min(e["distance_m"] for e in encs)
                print(f"    {ego_id} vs {npc_id}: {len(encs)} ticks nearby, "
                      f"min dist={min_dist:.0f}m")

    # Position timeline (compact)
    if position_log:
        print(f"\n  Position timeline:")
        for pl in position_log:
            parts = []
            for vid, pos in pl["positions"].items():
                status = "ARRIVED" if pos["arrived"] else pos["node"]
                parts.append(f"{vid}={status}")
            print(f"    tick {pl['tick']:2d} (t={pl['time']:.0f}s): {' | '.join(parts)}")

    cabin_text = (
        f"{result.overall_cabin_score:.1%}"
        if result.overall_cabin_score is not None else "N/A")
    driving_text = (
        f"{result.overall_trajectory_quality_score:.1%}"
        if result.overall_trajectory_quality_score is not None else "N/A")
    print(
        f"\n  Cabin YAML: {cabin_text} | Driving: {driving_text} | "
        f"Hard safety: {'PASS' if result.hard_safety_passed else 'FAIL'} | "
        f"Ticks: {result.total_ticks} | Time: {elapsed:.0f}s")


def save_results(scenario_id: str, result: MultiSimResult, elapsed: float,
                 vehicle_stats: Dict, callbacks: Dict,
                 output_dir: str = "evaluation/outputs"):
    """Save results to JSON files."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%m%d_%H%M")
    scenario_config = copy.deepcopy(
        getattr(result, "_scenario_config", {}))
    scenario_config_sha256 = (
        hashlib.sha256(json.dumps(
            scenario_config, sort_keys=True, ensure_ascii=False,
            separators=(",", ":")).encode("utf-8")).hexdigest()
        if scenario_config else None)

    # Summary
    summary = {
        "scenario_id": scenario_id,
        "scenario_config_sha256": scenario_config_sha256,
        "evaluation": {
            "cabin": {
                "score": (
                    round(result.overall_cabin_score, 4)
                    if result.overall_cabin_score is not None else None),
            },
            "driving": {
                "trajectory_quality_score": (
                    round(result.overall_trajectory_quality_score, 4)
                    if result.overall_trajectory_quality_score is not None else None),
                "hard_safety_passed": result.hard_safety_passed,
            },
            "passenger_interaction": {
                **result.passenger_request_summary,
                "vehicles": {
                    vid: copy.deepcopy(vr.passenger_evaluation)
                    for vid, vr in result.vehicle_results.items()
                    if vr.passenger_evaluation is not None
                },
            },
        },
        "total_ticks": result.total_ticks,
        "elapsed_s": round(elapsed, 1),
        "vehicles": {},
    }
    for vid, vr in result.vehicle_results.items():
        vs = vehicle_stats.get(vid, {})
        entry = {
            "agent_type": vs.get("agent_type", "?"),
            "model": vs.get("model", ""),
            "is_evaluated": vr.is_evaluated,
            "arrived": vr.arrived,
            "route_failed": vr.route_failed,
            "route_failure_reason": vr.route_failure_reason,
            "route_failure_time_s": vr.route_failure_time_s,
        }
        if vr.is_evaluated:
            total = sum(c.get("total_fields", 0) for c in vr.checkpoints)
            correct = sum(c.get("correct_fields", 0) for c in vr.checkpoints)
            entry["cabin_score"] = (
                round(correct / total, 4) if total > 0 else None)
            entry["fields"] = f"{correct}/{total}"
            entry["driving_evaluation"] = vr.driving_evaluation
            entry["passenger_evaluation"] = vr.passenger_evaluation
        if vs.get("agent_type") == "llm":
            entry["llm_runtime"] = copy.deepcopy(
                vs.get("effective_config", {}))
        summary["vehicles"][vid] = entry

    outfile = os.path.join(output_dir, f"multi_run_{timestamp}_{scenario_id}.json")
    with open(outfile, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n  Summary saved to {outfile}")

    # Detailed output with messages
    detail = {
        "scenario_id": scenario_id,
        "scenario_config_sha256": scenario_config_sha256,
        "scenario_config": scenario_config,
        "overall_cabin_score": result.overall_cabin_score,
        "trajectory_quality_score": result.overall_trajectory_quality_score,
        "hard_safety_passed": result.hard_safety_passed,
        "traffic_log": getattr(result, '_traffic_log', []),
        "encounter_log": getattr(result, '_encounter_log', []),
        "vehicle_results": {},
    }
    for vid, vr in result.vehicle_results.items():
        cb = callbacks.get(vid)
        callback_state = (
            cb._state if hasattr(cb, "_state") else {})
        detail["vehicle_results"][vid] = {
            "is_evaluated": vr.is_evaluated,
            "cabin_score": vr.cabin_score,
            "driving_evaluation": vr.driving_evaluation,
            "passenger_evaluation": vr.passenger_evaluation,
            "arrived": vr.arrived,
            "route_failed": vr.route_failed,
            "route_failure_reason": vr.route_failure_reason,
            "route_failure_time_s": vr.route_failure_time_s,
            "arrival_tick": vr.arrival_tick,
            "checkpoints": vr.checkpoints,
            "tick_interactions": vr.tick_interactions,
            "messages": callback_state.get("all_messages", []),
            "tool_call_log": callback_state.get("tool_call_log", []),
            "model_call_log": callback_state.get("model_call_log", []),
            "effective_config": callback_state.get("effective_config", {}),
            "protocol_events": callback_state.get("protocol_events", []),
            "wake_runtime_log": callback_state.get("wake_runtime_log", []),
            "context_budget_log": callback_state.get(
                "context_budget_log", []),
            "context_snapshot_log": callback_state.get(
                "context_snapshot_log", []),
            "todo_audit_log": callback_state.get("todo_audit_log", []),
            "heartbeat_interval_audit_log": callback_state.get(
                "heartbeat_interval_audit_log", []),
            "scheduled_wake_audit_log": callback_state.get(
                "scheduled_wake_audit_log", []),
            "visual_observation_log": callback_state.get(
                "visual_observation_log", []),
            "personal_agent": callback_state.get("personal_agent"),
            "passenger_judge": callback_state.get("passenger_judge"),
            "personal_requests": callback_state.get(
                "personal_requests", []),
        }
    detail_file = os.path.join(output_dir, f"multi_run_{timestamp}_{scenario_id}_detail.json")
    with open(detail_file, "w") as f:
        json.dump(detail, f, indent=2, ensure_ascii=False, default=str)
    print(f"  Detail saved to {detail_file}")
