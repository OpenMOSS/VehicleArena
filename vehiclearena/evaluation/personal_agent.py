"""Passenger runtimes, judge evidence, and the legacy one-shot wrapper.

The Personal Agent is the passenger, not an in-car assistant. New event/random
scheduling and repeated bounded judging live in passenger_orchestration.py.
The wrapper in this module remains available for legacy experiment replay.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional

from evaluation.snapshot_utils import snapshot_modules


_CABIN_OBSERVABLE_MODULES = [
    "airConditioner", "window", "seat", "sunroof", "sunshade", "door",
    "music", "radio", "video", "conversation", "bluetooth",
    "readingLight", "overheadScreen", "centerInformationDisplay",
    "wiper", "fogLight", "lowBeamHeadlight", "highBeamHeadlight",
    "positionLight", "hazardLight", "steeringWheel", "rearviewMirror",
    "HUD", "instrumentPanel", "broadcast",
]

_MOTION_CONTROL_TOOLS = {
    "navigation__navigation_set_speed",
    "navigation__navigation_emergency_stop",
    "navigation__navigation_change_lane",
}


_DRIVING_OUTCOME_CONTRACT = (
    "SUMO advances vehicle physics. For the LLM-controlled Vehicle Agent, "
    "previously issued controls can remain active across wakes. Do not assume "
    "SUMO supplied automatic safety braking or car-following just because "
    "the current wake contains no new motion command. Judge the requested "
    "outcome from observed state and motion within the relevant observation "
    "window. A new successful action or a previously active command can "
    "explain behavior, but a matching command in the current response is not "
    "a prerequisite for completion or scoring. Do not invent an intervention "
    "when its attribution is not established by evidence. Todo edits and "
    "sensor/query calls alone do not establish physical execution. An action "
    "that only masks an effect does not prove that the requested underlying "
    "state changed, and a substitute operation does not satisfy a different "
    "requested outcome."
)


def _client_audit_config(client: Any) -> dict:
    """Persist reproducibility fields without ever storing credentials."""
    return {
        key: copy.deepcopy(getattr(client, key))
        for key in (
            "model", "api_base", "temperature", "max_tokens",
            "thinking_mode", "reasoning_effort",
            "chat_template_enable_thinking",
        )
        if hasattr(client, key)
    }


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "as_dict"):
        return _json_safe(value.as_dict())
    if hasattr(value, "value"):
        return _json_safe(value.value)
    if hasattr(value, "__dict__"):
        return {
            key.lstrip("_"): _json_safe(item)
            for key, item in vars(value).items()
            if not key.startswith("__") and not callable(item)
        }
    return str(value)


def _canonical_json_sha256(value: Any) -> str:
    """Hash a JSON-safe value independently of mapping insertion order."""
    payload = json.dumps(
        _json_safe(value), ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _replayable_tool_request(
    client: Any, messages: List[Dict], tools: List[Dict],
) -> dict:
    """Return the exact provider kwargs, with a mock-client fallback."""
    builder = getattr(client, "build_tool_chat_request", None)
    if callable(builder):
        return _json_safe(builder(messages, tools))
    # Scripted test clients and legacy integrations receive only messages and
    # tools. Preserve those exact arguments plus any declared model identity.
    request = {
        "messages": copy.deepcopy(messages),
        "tools": copy.deepcopy(tools),
    }
    if hasattr(client, "model"):
        request["model"] = copy.deepcopy(client.model)
    return _json_safe(request)


def _assistant_message_dict(message: Any) -> dict:
    tool_calls = []
    for call in list(getattr(message, "tool_calls", None) or []):
        tool_calls.append({
            "id": str(getattr(call, "id", "")),
            "type": "function",
            "function": {
                "name": str(getattr(call.function, "name", "")),
                "arguments": str(getattr(call.function, "arguments", "{}")),
            },
        })
    result = {
        "role": "assistant",
        "content": getattr(message, "content", None),
    }
    if tool_calls:
        result["tool_calls"] = tool_calls
    return result


def _tool_arguments(call: Any) -> dict:
    try:
        raw = getattr(call.function, "arguments", "{}") or "{}"
        value = json.loads(raw)
        return value if isinstance(value, dict) else {}
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}


def _prepare_judge_evidence(evidence: Mapping[str, Any]) -> dict:
    """Make current-state semantics explicit and summarize numeric traces."""
    augmented = copy.deepcopy(evidence)
    for observation_key in ("observation_when_requested", "observation_at_next_wake"):
        observation = augmented.get(observation_key)
        cabin = observation.get("cabin") if isinstance(observation, dict) else None
        if isinstance(cabin, dict) and isinstance(cabin.get("shared_settings"), dict):
            cabin.setdefault("current_state", copy.deepcopy(cabin["shared_settings"]))
    execution = augmented.get("physical_execution")
    if not isinstance(execution, dict):
        return augmented
    trace = execution.get("ego_motion_trace")
    if not isinstance(trace, list) or not trace:
        return augmented
    fields = sorted({
        key for row in trace if isinstance(row, dict)
        for key, value in row.items()
        if type(value) in (int, float) and math.isfinite(value)
    })
    summaries = {}
    for key in fields:
        samples = [
            (index, float(row[key]), row.get("time_s"))
            for index, row in enumerate(trace)
            if isinstance(row, dict)
            and type(row.get(key)) in (int, float)
            and math.isfinite(row[key])
        ]
        if not samples:
            continue
        minimum = min(value for _, value, _ in samples)
        maximum = max(value for _, value, _ in samples)
        summaries[key] = {
            "sample_count": len(samples),
            "min": minimum,
            "max": maximum,
            "min_occurrences": [
                {"index": index, "time_s": time_s}
                for index, value, time_s in samples if value == minimum
            ][:8],
            "max_occurrences": [
                {"index": index, "time_s": time_s}
                for index, value, time_s in samples if value == maximum
            ][:8],
        }
    if summaries:
        execution["numeric_trace_summary"] = summaries
    return augmented


def _selected(mapping: Any, keys: Iterable[str]) -> dict:
    if not isinstance(mapping, Mapping):
        return {}
    return {
        key: _json_safe(mapping[key]) for key in keys if key in mapping}


def _compact_cabin_modules(modules: Mapping[str, Any]) -> dict:
    """Build a small current-state projection, never a model-written summary.

    Module snapshots contain catalogs, UI lookup tables and six copies of
    identical defaults.  A passenger needs only present cabin conditions and
    active/non-default equipment state.  The Driving Agent can inspect a
    specific module if a generated request later requires API detail.
    """
    result: Dict[str, Any] = {}

    air = modules.get("airConditioner")
    if isinstance(air, Mapping):
        zones = {}
        for name, state in (air.get("ac_states") or {}).items():
            summary = _selected(
                state, ("is_on", "temperature", "wind_speed"))
            if isinstance(state, Mapping):
                outlets = state.get("outlet_modes", {})
                summary["active_outlets"] = [
                    str(key) for key, enabled in outlets.items() if enabled
                ] if isinstance(outlets, Mapping) else []
            zones[str(name)] = summary
        air_summary: Dict[str, Any] = {}
        if zones:
            values = list(zones.values())
            if all(value == values[0] for value in values[1:]):
                air_summary["all_zones"] = values[0]
            else:
                air_summary["zones"] = zones
        active_modes = [
            key for key in (
                "auto_mode", "ac_mode", "heat_mode", "cool_mode",
                "defrost_mode", "auto_defog_mode", "energy_saving_mode",
                "parking_ventilation_mode")
            if air.get(key) is True
        ]
        air_summary["active_modes"] = active_modes
        air_summary.update(_selected(
            air, ("circulation_mode", "purification_mode")))
        result["airConditioner"] = air_summary

    windows = modules.get("window")
    if isinstance(windows, Mapping):
        window_states = windows.get("windows") or {}
        result["window"] = {
            "open": {
                str(name): _json_safe(state.get("open_degree", 0))
                for name, state in window_states.items()
                if isinstance(state, Mapping) and state.get("is_open")
            },
            "child_locks": [
                str(name) for name, state in window_states.items()
                if isinstance(state, Mapping)
                and state.get("child_safety_lock")
            ],
            "auto_close_on_lock": bool(
                windows.get("auto_close_on_lock", False)),
        }

    seats = modules.get("seat")
    if isinstance(seats, Mapping):
        active_seats = {}
        for name, state in (seats.get("seats") or {}).items():
            if not isinstance(state, Mapping):
                continue
            active = {}
            for feature, level_key in (
                    ("heater", "temperature_level"),
                    ("massager", "intensity_level"),
                    ("ventilation", "airflow_level")):
                feature_state = state.get(feature, {})
                if (isinstance(feature_state, Mapping)
                        and feature_state.get("is_on")):
                    active[feature] = _json_safe(
                        feature_state.get(level_key))
            position = state.get("position", {})
            if isinstance(position, Mapping):
                neutral_position = {
                    "horizontal_position": 50,
                    "vertical_position": 50,
                    "cushion_length": 50,
                    "cushion_angle": 50,
                    "backrest_angle": 50,
                    "leg_rest_height": 0,
                    "feet_rest_height": 0,
                    "headrest_height": 50,
                    "is_folded": False,
                    "guest_welcome_mode": False,
                }
                changed_position = {
                    key: _json_safe(value)
                    for key, value in position.items()
                    if (key not in neutral_position
                        or value != neutral_position[key])
                }
                if changed_position:
                    active["position"] = changed_position
            if active:
                active_seats[str(name)] = active
        result["seat"] = {"active_or_adjusted": active_seats}

    sunroof = modules.get("sunroof")
    if isinstance(sunroof, Mapping):
        result["sunroof"] = _selected(
            sunroof, ("state", "open_degree_percentage"))

    sunshade = modules.get("sunshade")
    if isinstance(sunshade, Mapping):
        result["sunshade"] = {
            "front": _selected(sunshade.get("front_row_status"), (
                "is_open", "is_paused", "open_degree_value")),
            "rear": _selected(sunshade.get("rear_row_status"), (
                "is_open", "is_paused", "open_degree_value")),
        }

    doors = modules.get("door")
    if isinstance(doors, Mapping):
        door_states = doors.get("doors") or {}
        result["door"] = {
            "not_closed": {
                str(name): _selected(state, ("status", "angle"))
                for name, state in door_states.items()
                if isinstance(state, Mapping)
                and state.get("status", "closed") != "closed"
            },
            "locked": [
                str(name) for name, state in door_states.items()
                if isinstance(state, Mapping) and state.get("is_locked")
            ],
            "child_locks": [
                str(name) for name, state in door_states.items()
                if isinstance(state, Mapping)
                and state.get("child_safety_lock_enabled")
            ],
        }

    music = modules.get("music")
    if isinstance(music, Mapping):
        result["music"] = _selected(
            music, ("is_playing", "playback_mode"))
        tracks = music.get("tracks")
        index = music.get("current_track_index")
        if isinstance(tracks, list) and isinstance(index, int) \
                and 0 <= index < len(tracks) \
                and isinstance(tracks[index], Mapping):
            result.setdefault("music", {})["current_track"] = {
                key: _json_safe(tracks[index][key])
                for key in ("id", "title", "artist", "album", "duration")
                if key in tracks[index]
            }

    radio = modules.get("radio")
    if isinstance(radio, Mapping):
        result["radio"] = _selected(
            radio, ("is_playing", "current_station"))

    video = modules.get("video")
    if isinstance(video, Mapping):
        result["video"] = _selected(
            video, ("is_playing", "quality", "is_fullscreen"))
        current_video = (
            _selected(video.get("current_video"), ("video_id", "title"))
            if video.get("is_playing") else {})
        if current_video:
            result["video"]["current_video"] = current_video

    conversation = modules.get("conversation")
    if isinstance(conversation, Mapping):
        result["conversation"] = _selected(
            conversation, ("call_state", "hands_free"))
        if conversation.get("call_state") != "idle":
            result["conversation"]["current_contact"] = _json_safe(
                conversation.get("current_contact"))

    bluetooth = modules.get("bluetooth")
    if isinstance(bluetooth, Mapping):
        result["bluetooth"] = _selected(
            bluetooth, ("is_enabled", "connection_state"))

    reading_light = modules.get("readingLight")
    if isinstance(reading_light, Mapping):
        lights = reading_light.get("lights") or {}
        result["readingLight"] = {
            "on": {
                str(name): _json_safe(state.get("brightness_value"))
                for name, state in lights.items()
                if isinstance(state, Mapping) and state.get("is_on")
            },
            "auto_mode": bool(reading_light.get("auto_mode", False)),
        }

    overhead = modules.get("overheadScreen")
    if isinstance(overhead, Mapping):
        result["overheadScreen"] = _selected(
            overhead, ("state", "brightness_percentage"))

    display = modules.get("centerInformationDisplay")
    if isinstance(display, Mapping):
        brightness = display.get("brightness_settings", {})
        result["centerInformationDisplay"] = {
            "color_theme": _json_safe(display.get("color_theme")),
            **_selected(brightness, (
                "brightness_level", "auto_brightness")),
        }

    wiper = modules.get("wiper")
    if isinstance(wiper, Mapping):
        result["wiper"] = {
            str(name): {
                "is_on": bool(state.get("is_on", False)),
                "speed": _selected(
                    state.get("speed_setting"), ("value", "unit")),
            }
            for name, state in (wiper.get("wipers") or {}).items()
            if isinstance(state, Mapping)
        }

    fog = modules.get("fogLight")
    if isinstance(fog, Mapping):
        result["fogLight"] = {
            "front_on": bool((fog.get("front_light") or {}).get("is_on")),
            "rear_on": bool((fog.get("rear_light") or {}).get("is_on")),
        }

    low_beam = modules.get("lowBeamHeadlight")
    if isinstance(low_beam, Mapping):
        result["lowBeamHeadlight"] = _selected(
            low_beam, ("mode", "height_level", "height_percentage"))

    high_beam = modules.get("highBeamHeadlight")
    if isinstance(high_beam, Mapping):
        result["highBeamHeadlight"] = _selected(
            high_beam, (
                "high_beam_on", "delay_off_enabled",
                "delay_off_duration_seconds"))

    position_light = modules.get("positionLight")
    if isinstance(position_light, Mapping):
        result["positionLight"] = _selected(
            position_light, ("is_on", "status"))

    hazard = modules.get("hazardLight")
    if isinstance(hazard, Mapping):
        result["hazardLight"] = _selected(
            hazard, ("is_active", "status"))

    steering = modules.get("steeringWheel")
    if isinstance(steering, Mapping):
        result["steeringWheel"] = _selected(
            steering, (
                "is_heater_on", "heater_level", "heater_celsius",
                "heater_percentage", "current_unit"))

    mirror = modules.get("rearviewMirror")
    if isinstance(mirror, Mapping):
        result["rearviewMirror"] = {
            "left": _selected(
                mirror.get("left_mirror"),
                ("is_open", "height", "horizontal_position")),
            "right": _selected(
                mirror.get("right_mirror"),
                ("is_open", "height", "horizontal_position")),
            **_selected(mirror, (
                "auto_flip_enabled", "auto_fold_enabled",
                "auto_adjust_enabled", "heating_enabled",
                "auxiliary_view_enabled")),
        }

    hud = modules.get("HUD")
    if isinstance(hud, Mapping):
        result["HUD"] = _selected(
            hud, (
                "is_on", "brightness_level", "brightness_percentage",
                "height_level", "height_percentage"))

    panel = modules.get("instrumentPanel")
    if isinstance(panel, Mapping):
        result["instrumentPanel"] = _selected(
            panel, (
                "theme", "brightness", "brightness_unit",
                "auto_brightness", "distance_unit"))

    broadcast = modules.get("broadcast")
    if isinstance(broadcast, Mapping):
        announcements = broadcast.get("announcements") or []
        result["broadcast"] = {
            "recent_announcements": _json_safe(list(announcements)[-8:]),
            "count": len(announcements),
        }
    return result


def _terminal_event(observation: Mapping[str, Any]) -> bool:
    return any(
        str(event.get("event_type", "")) in {
            "arrived", "route_failed", "crashed", "collision", "simulation_ended",
        }
        for event in observation.get("current_events", [])
        if isinstance(event, Mapping)
    )


def _tool_succeeded(record: Mapping[str, Any]) -> bool:
    result = record.get("result")
    if not isinstance(result, Mapping):
        return False
    if result.get("success") is True:
        return True
    if result.get("success") is False or "error" in result:
        return False
    return str(result.get("status", "")).strip().lower() in {
        "success", "info",
    }


def _assistant_responses(wakes: Iterable[dict]) -> List[dict]:
    """Keep passenger-visible model text with its simulation timestamp."""
    responses = []
    for wake in wakes:
        time_s = wake.get("time_s")
        for message in wake.get("messages", []):
            if message.get("role") != "assistant":
                continue
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                entry = {"text": content.strip()[:600]}
                if type(time_s) in (int, float) and math.isfinite(time_s):
                    entry["time_s"] = round(float(time_s), 6)
                responses.append(entry)
    return responses


def _assistant_texts(wakes: Iterable[dict]) -> List[str]:
    return [item["text"] for item in _assistant_responses(wakes)]


def _compact_driver_response(
    *, actions: Iterable[Any], tool_calls: Iterable[dict],
    wakes: Iterable[dict], protocol_events: Iterable[dict],
) -> dict:
    """Build bounded Judge evidence with explicit action attribution."""
    compact_calls = []
    for record in tool_calls:
        compact_calls.append({
            key: copy.deepcopy(record[key])
            for key in ("time_s", "turn_index", "function", "kind",
                        "arguments", "result")
            if key in record
        })
    successful_actions = [
        item for item in compact_calls
        if item.get("kind") == "action" and _tool_succeeded(item)
    ]
    motion_commands = [
        item for item in successful_actions
        if item.get("function") in _MOTION_CONTROL_TOOLS
    ]
    assistant_responses = _assistant_responses(wakes)
    texts = [item["text"] for item in assistant_responses]
    return {
        "returned_actions": _json_safe(list(actions or [])),
        "assistant_texts": texts,
        "assistant_responses": assistant_responses,
        "tool_calls": compact_calls,
        "successful_motion_commands": motion_commands,
        "successful_non_motion_actions": [
            item for item in successful_actions
            if item.get("function") not in _MOTION_CONTROL_TOOLS
        ],
        "query_and_perception_calls": [
            item for item in compact_calls
            if item.get("kind") in {"query", "perception"}
        ],
        "todo_updates": [
            item for item in compact_calls
            if item.get("function") == "todo_manage"
        ],
        "protocol_events": _json_safe(list(protocol_events)),
        "has_passenger_visible_response": bool(texts or successful_actions),
        "attribution_contract": _DRIVING_OUTCOME_CONTRACT,
    }


def _passenger_visible_driver_response(response: Mapping[str, Any]) -> dict:
    """Expose only what the passenger could hear or observe later."""
    actions = []
    for item in (
            list(response.get("successful_motion_commands", []))
            + list(response.get("successful_non_motion_actions", []))):
        result = item.get("result")
        actions.append({
            "function": item.get("function"),
            "arguments": copy.deepcopy(item.get("arguments", {})),
            "result": {
                key: copy.deepcopy(result[key])
                for key in ("success", "error", "command_id")
                if isinstance(result, Mapping) and key in result
            },
        })
    return {
        "assistant_texts": copy.deepcopy(
            response.get("assistant_texts", [])),
        "assistant_responses": copy.deepcopy(
            response.get("assistant_responses", [])),
        "submitted_actions": actions,
    }


def _nearby_changes(before: Mapping[str, Any], after: Mapping[str, Any]) -> list:
    """Compare like-named evaluator metrics; never mix radar and pose range."""
    before_by_id = {
        str(item.get("entity_id")): item
        for item in before.get("nearby_entities", [])
        if isinstance(item, Mapping) and item.get("entity_id") is not None
    }
    after_by_id = {
        str(item.get("entity_id")): item
        for item in after.get("nearby_entities", [])
        if isinstance(item, Mapping) and item.get("entity_id") is not None
    }
    changes = []
    comparable_fields = (
        "center_distance_m", "same_lane_bumper_clearance_m", "speed_kmh",
        "speed_mps", "acceleration_mps2", "is_crashed",
    )
    for entity_id in sorted(set(before_by_id) | set(after_by_id)):
        old = before_by_id.get(entity_id)
        new = after_by_id.get(entity_id)
        entry = {
            "entity_id": entity_id,
            "entity_type": (new or old).get("entity_type"),
            "present_before": old is not None,
            "present_after": new is not None,
        }
        if old is not None and new is not None:
            for field in comparable_fields:
                if field in old or field in new:
                    entry[field] = {
                        "before": copy.deepcopy(old.get(field)),
                        "after": copy.deepcopy(new.get(field)),
                    }
        changes.append(entry)
    return changes


def build_personal_observation(
    vw: Any,
    *,
    vehicle_id: str,
    sim_time_s: float,
    tick_index: int,
    vehicle_state: Any,
    wake_events: Iterable[dict],
    recent_motion: Iterable[dict],
    episode_total_time_s: Optional[float] = None,
    navigation_status: Optional[Mapping[str, Any]] = None,
    trigger_context: Optional[Mapping[str, Any]] = None,
    judge_observation_window_s: float = 3.0,
) -> dict:
    """Build compact, rule-generated text state observable by a passenger."""
    raw_state = (
        vehicle_state.as_dict()
        if hasattr(vehicle_state, "as_dict") else vehicle_state)
    raw_state = raw_state if isinstance(raw_state, Mapping) else {}
    self_state = {
        key: _json_safe(raw_state[key])
        for key in (
            "current_speed_kmh", "acceleration_mps2", "current_lane",
            "current_segment", "is_crashed", "arrived", "signal_state",
        )
        if key in raw_state
    }
    active_connector = str(raw_state.get("active_connector_id", "") or "")
    planned_connector = str(raw_state.get("planned_connector_id", "") or "")
    settings = getattr(vw, "settings", None)
    cabin_settings = {
        key: _json_safe(getattr(settings, key))
        for key in (
            "temperature", "volume", "sound_channel", "speaker", "language")
        if settings is not None and hasattr(settings, key)
    }
    available_cabin_modules = [
        name for name in _CABIN_OBSERVABLE_MODULES
        if not hasattr(vw, "has_module") or vw.has_module(name)
    ]
    try:
        cabin_modules = _compact_cabin_modules(
            snapshot_modules(vw, available_cabin_modules))
    except Exception:
        cabin_modules = {}
    try:
        outside = snapshot_modules(vw, ["weather", "dayNight"])
    except Exception:
        outside = {}
    curve = [
        _json_safe(sample) for sample in recent_motion
        if float(sample.get("time_s", -1.0)) >= float(sim_time_s) - 10.0
    ]
    temporal_task_context = {
        "inside_intersection": bool(active_connector),
        "intersection_ahead_known": bool(
            active_connector or planned_connector),
        "vehicle_stopped": float(
            raw_state.get("current_speed_kmh", 0.0) or 0.0) < 0.5,
        # Production passes this explicitly, so state predicates are offered
        # only when their truth is already observable rather than merely
        # syntactically registerable.
        "guaranteed_trigger_conditions": {},
    }
    if isinstance(trigger_context, Mapping):
        temporal_task_context.update(_json_safe(trigger_context))
    route_remaining_m = (
        navigation_status.get("remaining_distance_m")
        if isinstance(navigation_status, Mapping) else None)
    if (type(route_remaining_m) in (int, float)
            and math.isfinite(route_remaining_m) and route_remaining_m >= 0):
        temporal_task_context["remaining_distance_m"] = round(
            float(route_remaining_m), 3)
    if (type(episode_total_time_s) in (int, float)
            and math.isfinite(episode_total_time_s)):
        judge_window = (
            float(judge_observation_window_s)
            if type(judge_observation_window_s) in (int, float)
            and math.isfinite(judge_observation_window_s)
            else 3.0)
        judge_window = max(0.1, judge_window)
        total = max(0.0, float(episode_total_time_s))
        remaining = max(0.0, total - float(sim_time_s))
        max_after_delay = max(0.0, remaining - judge_window - 0.5)
        if (type(route_remaining_m) in (int, float)
                and math.isfinite(route_remaining_m)
                and route_remaining_m >= 0.0):
            # Bound delayed passenger work by a conservative free-flow ETA,
            # not only by the whole-scene deadline. Otherwise a short focal
            # trip can terminate long before a syntactically valid trigger.
            speed_candidates = [15.0]
            for key in (
                    "desired_speed_kmh", "target_speed_kmh",
                    "current_speed_kmh"):
                value = raw_state.get(key)
                if (type(value) in (int, float) and math.isfinite(value)
                        and value > 0.0):
                    speed_candidates.append(float(value))
            cruise_kmh = max(speed_candidates)
            free_flow_remaining_s = (
                float(route_remaining_m) / (cruise_kmh / 3.6))
            # A delayed task is eligible only when its trigger can still be
            # followed by the complete Judge observation window. This uses a
            # conservative free-flow ETA, not merely the scene deadline.
            max_after_delay = min(
                max_after_delay,
                max(0.0, free_flow_remaining_s - judge_window - 0.5),
            )
            temporal_task_context.update({
                "estimated_free_flow_remaining_s": round(
                    free_flow_remaining_s, 6),
                "estimated_free_flow_speed_mps": round(
                    cruise_kmh / 3.6, 6),
            })
        temporal_task_context.update({
            "episode_total_time_s": round(total, 6),
            "remaining_episode_s": round(remaining, 6),
            "max_after_delay_s": round(max_after_delay, 6),
            "judge_observation_window_s": round(judge_window, 6),
        })
    if temporal_task_context["vehicle_stopped"]:
        stopped_since = float(sim_time_s)
        for sample in reversed(curve):
            speed = sample.get("speed_kmh")
            if (type(speed) not in (int, float)
                    or not math.isfinite(speed) or speed >= 0.5):
                break
            stopped_since = min(stopped_since, float(sample["time_s"]))
        stopped_for_s = max(0.0, float(sim_time_s) - stopped_since)
        temporal_task_context["stopped_for_s"] = round(stopped_for_s, 6)
        if stopped_for_s >= 0.5:
            temporal_task_context["guaranteed_trigger_conditions"] = {
                "vehicle_stopped": {
                    "max_hold_for_s": round(stopped_for_s, 6),
                },
            }
    return {
        "sim_time_s": round(float(sim_time_s), 6),
        "wake_id": f"wake-{int(tick_index):06d}",
        "vehicle_id": vehicle_id,
        "current_events": _json_safe(list(wake_events)),
        "recent_motion_10s": curve,
        "vehicle_motion": self_state,
        "temporal_task_context": temporal_task_context,
        "outside": _json_safe(outside),
        "cabin": {
            "current_state": copy.deepcopy(cabin_settings),
            "shared_settings": cabin_settings,
            "available_modules": available_cabin_modules,
            "current_module_state": _json_safe(cabin_modules),
        },
    }


from evaluation.request_satisfaction import (
    CONTRACT_SCHEMA, CRITERION_STATUSES, JUDGE_TRIGGER_SCHEMA, GRADE_SCORES,
    RESOLVED_CRITERION_STATUSES, REVISION, RUBRIC, validate_contract,
    validate_judge_trigger, score_submission, available_judge_triggers,
    validate_available_judge_trigger,
)


_PASSENGER_SHEET_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "cancel_passenger_request",
            "description": (
                "Cancel the one currently active passenger request sheet. "
                "Use only when the passenger no longer wants any of its "
                "still-pending outcomes. To change the sheet while retaining "
                "some outcomes, submit a complete replacement with "
                "a send request tool instead."),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string", "minLength": 1, "maxLength": 160,
                    },
                },
                "required": ["reason"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": (
                "End this passenger turn without creating, changing, or "
                "cancelling the current request sheet. If a sheet is active, "
                "it remains active unchanged."),
            "parameters": {
                "type": "object", "properties": {},
                "additionalProperties": False,
            },
        },
    },
]


_OUTCOME_PHASE_SCHEMA = {
    "type": "object",
    "description": (
        "Requested observable outcomes only. The runtime writes the passenger "
        "utterance and uses these same outcomes as judge criteria."),
    "properties": {
        "core": {
            "type": "array", "minItems": 1, "maxItems": 5,
            "items": {"type": "string", "minLength": 1, "maxLength": 100},
        },
        "secondary": {
            "type": "array", "maxItems": 5,
            "items": {"type": "string", "minLength": 1, "maxLength": 100},
        },
    },
    "required": ["core", "secondary"],
    "additionalProperties": False,
}


def _structured_request_parameters(*, triggered: bool) -> dict:
    properties = {
        "message": {
            "type": "string", "minLength": 1, "maxLength": 600,
            "description": (
                "The complete natural passenger utterance delivered unchanged "
                "to the Driving Agent. It must explicitly express every listed "
                "outcome and, for a two-stage request, the chosen timing."),
        },
        "immediate_request": copy.deepcopy(_OUTCOME_PHASE_SCHEMA),
        "request_kind": {"type": "string", "enum": ["one_shot", "ongoing"]},
        "expected_response_s": {
            "type": "number", "exclusiveMinimum": 0, "maximum": 30},
        "valid_for_s": {
            "type": "number", "exclusiveMinimum": 0, "maximum": 60},
    }
    required = [
        "message", "immediate_request", "request_kind", "expected_response_s",
        "valid_for_s",
    ]
    if triggered:
        properties.update({
            "triggered_request": copy.deepcopy(_OUTCOME_PHASE_SCHEMA),
            "judge_trigger": copy.deepcopy(JUDGE_TRIGGER_SCHEMA),
        })
        required.extend(["triggered_request", "judge_trigger"])
    return {
        "type": "object", "properties": properties,
        "required": required, "additionalProperties": False,
    }


_STRUCTURED_PERSONAL_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "send_immediate_request",
            "description": (
                "Create or replace a request whose outcomes should begin now. "
                "Write the natural passenger utterance and list its observable "
                "outcomes for evaluation."),
            "parameters": _structured_request_parameters(triggered=False),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_triggered_request",
            "description": (
                "Create or replace one two-stage request: immediate outcomes "
                "now and delayed outcomes after one offered structured trigger. "
                "Write the complete natural utterance, including its timing, "
                "and separately list observable outcomes for both phases."),
            "parameters": _structured_request_parameters(triggered=True),
        },
    },
    copy.deepcopy(_PASSENGER_SHEET_TOOLS[0]),
    copy.deepcopy(_PASSENGER_SHEET_TOOLS[1]),
]


def _validated_outcome_phase(value: Any, *, name: str) -> dict:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    if set(value) != {"core", "secondary"}:
        raise ValueError(f"{name} must contain exactly core and secondary")
    result = {}
    for field in ("core", "secondary"):
        items = value.get(field)
        if not isinstance(items, list):
            raise ValueError(f"{name}.{field} must be an array")
        if field == "core" and not items:
            raise ValueError(f"{name}.core must contain at least one outcome")
        if len(items) > 5:
            raise ValueError(f"{name}.{field} may contain at most five outcomes")
        cleaned = []
        for index, item in enumerate(items):
            if not isinstance(item, str) or not item.strip():
                raise ValueError(
                    f"{name}.{field}[{index}] must be a non-empty string")
            if len(item.strip()) > 100:
                raise ValueError(
                    f"{name}.{field}[{index}] exceeds 100 characters")
            cleaned.append(item.strip())
        result[field] = cleaned
    return result


def _build_outcome_contract(
    arguments: Mapping[str, Any], *, triggered: bool,
) -> tuple:
    expected = {
        "message", "immediate_request", "request_kind", "expected_response_s",
        "valid_for_s",
    }
    if triggered:
        expected |= {"triggered_request", "judge_trigger"}
    missing = sorted(expected - set(arguments))
    unexpected = sorted(set(arguments) - expected)
    if missing:
        raise ValueError(f"missing required fields: {', '.join(missing)}")
    if unexpected:
        raise ValueError(f"unexpected fields: {', '.join(unexpected)}")
    message = arguments.get("message")
    if not isinstance(message, str) or not message.strip():
        raise ValueError("message must be a non-empty string")
    if len(message.strip()) > 600:
        raise ValueError("message exceeds 600 characters")
    immediate = _validated_outcome_phase(
        arguments["immediate_request"], name="immediate_request")
    delayed = (
        _validated_outcome_phase(
            arguments["triggered_request"], name="triggered_request")
        if triggered else {"core": [], "secondary": []})
    criteria = {
        "core": immediate["core"] + delayed["core"],
        "secondary": immediate["secondary"] + delayed["secondary"],
        "request_kind": arguments["request_kind"],
        "expected_response_s": arguments["expected_response_s"],
        "valid_for_s": arguments["valid_for_s"],
    }
    validate_contract(criteria)
    immediate_core = len(immediate["core"])
    immediate_secondary = len(immediate["secondary"])
    phases = {
        "immediate": {
            "core_indices": list(range(immediate_core)),
            "secondary_indices": list(range(immediate_secondary)),
        },
        "triggered": {
            "core_indices": list(range(immediate_core, len(criteria["core"]))),
            "secondary_indices": list(range(
                immediate_secondary, len(criteria["secondary"]))),
        },
    }
    return immediate, delayed, criteria, phases


def _validate_trigger_horizon(trigger: dict, observation: dict) -> None:
    context = observation.get("temporal_task_context") or {}
    condition = trigger["condition"]
    if (condition in {"exit_next_intersection", "enter_next_intersection",
                      "approach_next_intersection"}
            and context.get("intersection_ahead_known") is False):
        raise ValueError("no observable next intersection for map trigger")
    remaining = context.get("remaining_episode_s")
    if type(remaining) in (int, float) and math.isfinite(remaining):
        if remaining < 0.5:
            raise ValueError("insufficient remaining episode time")
        if (trigger["condition"] != "after_delay"
                and float(trigger["timeout_s"]) > float(remaining) - 0.1 + 1e-9):
            raise ValueError("trigger timeout exceeds remaining episode time")
        if trigger["condition"] == "after_delay":
            maximum = context.get("max_after_delay_s", float(remaining) - 0.5)
            if (type(maximum) not in (int, float)
                    or not math.isfinite(maximum)
                    or float(trigger["after_s"]) > float(maximum) + 1e-9):
                raise ValueError(
                    "time trigger exceeds remaining episode or trip horizon")
    if "available_judge_triggers" in observation:
        validate_available_judge_trigger(
            trigger, observation["available_judge_triggers"])


def _validate_request_design(
    criteria: dict, trigger: Optional[dict], observation: dict,
    criterion_phases: Optional[dict] = None,
) -> None:
    """Make the requested difficulty contract enforceable, not advisory."""
    design = observation.get("request_design") or {}
    minimum = design.get("min_explicit_outcomes", 1)
    if type(minimum) is not int or minimum < 1:
        minimum = 1
    outcome_count = len(criteria["core"]) + len(criteria["secondary"])
    if outcome_count < minimum:
        raise ValueError(
            f"request_design requires at least {minimum} explicit outcomes")
    min_core = design.get("min_core_outcomes", 1)
    if type(min_core) is not int or min_core < 1:
        min_core = 1
    if len(criteria["core"]) < min_core:
        raise ValueError(
            f"request_design requires at least {min_core} core outcomes")
    if design.get("judge_trigger") == "required" and trigger is None:
        raise ValueError("request_design requires judge_trigger")
    if design.get("judge_trigger") == "forbidden" and trigger is not None:
        raise ValueError("request_design forbids judge_trigger")
    allowed = design.get("allowed_trigger_conditions")
    if trigger is not None and isinstance(allowed, list):
        if trigger.get("condition") not in allowed:
            raise ValueError(
                "request_design does not allow this judge_trigger condition")
    if design.get("phase_structure") == "immediate_and_triggered":
        phases = criterion_phases or {}
        immediate = phases.get("immediate") or {}
        triggered = phases.get("triggered") or {}
        immediate_count = len(immediate.get("core_indices", [])) + len(
            immediate.get("secondary_indices", []))
        triggered_count = len(triggered.get("core_indices", [])) + len(
            triggered.get("secondary_indices", []))
        if trigger is None or not immediate_count or not triggered_count:
            raise ValueError(
                "request_design requires outcomes in both request phases")
    if (design.get("pattern") == "sustained_behavior"
            and criteria["request_kind"] != "ongoing"):
        raise ValueError(
            "sustained_behavior requires request_kind=ongoing")

_JUDGE_TOOLS = [{
    "type": "function",
    "function": {
        "name": "submit_passenger_judgement",
        "description": "Select one evidence-backed request satisfaction grade.",
        "parameters": {
            "type": "object",
            "properties": {
                "grade": {
                    "type": "string", "enum": [*GRADE_SCORES, "NA"],
                    "description": (
                        "A/B require every core and secondary status to be "
                        "resolved (met or unsupported_refused); C requires "
                        "all core statuses resolved and at least one "
                        "secondary status unmet or unverified; D/E/F require "
                        "at least one core status unmet or unverified."),
                },
                "reason": {"type": "string", "minLength": 1, "maxLength": 600},
                "na_reason": {"type": "string", "enum": ["invalid_request", "insufficient_evidence"]},
                "core_statuses": {
                    "type": "array", "minItems": 1,
                    "items": {
                        "type": "string", "enum": list(CRITERION_STATUSES),
                    },
                    "description": (
                        "Statuses covering every frozen core criterion in order; "
                        "a compound criterion may be split into multiple statuses."),
                },
                "secondary_statuses": {
                    "type": "array",
                    "items": {
                        "type": "string", "enum": list(CRITERION_STATUSES),
                    },
                    "description": (
                        "Statuses covering every frozen secondary criterion in order; "
                        "a compound criterion may be split into multiple statuses."),
                },
                "fulfilled_at_s": {
                    "type": "number", "minimum": 0,
                    "description": (
                        "For A/B, evidence-backed first full resolution time "
                        "in simulation seconds; an unsupported_refused item "
                        "resolves at its explicit refusal time."),
                },
                "immediate_fulfilled_at_s": {
                    "type": "number", "minimum": 0,
                    "description": (
                        "First time every immediate-phase criterion was "
                        "resolved. For unsupported_refused, use the explicit "
                        "refusal time. Required when that phase is resolved."),
                },
                "triggered_fulfilled_at_s": {
                    "type": "number", "minimum": 0,
                    "description": (
                        "First time every triggered-phase criterion was "
                        "resolved. For unsupported_refused, use the explicit "
                        "refusal time. Required when that phase is resolved."),
                },
            },
            "required": ["grade", "reason"],
            "additionalProperties": False,
        },
    },
}]

_JUDGE_CHECK_TOOLS = [{
    "type": "function",
    "function": {
        "name": "submit_passenger_check",
        "description": (
            "Record criterion status at an intermediate observation window. "
            "This is not an A-F passenger-request grade."),
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string", "minLength": 1, "maxLength": 600,
                    "description": (
                        "Compact evidence-backed status of every criterion."),
                },
                "core_statuses": {
                    "type": "array", "minItems": 1,
                    "items": {
                        "type": "string",
                        "enum": list(CRITERION_STATUSES),
                    },
                    "description": (
                        "Current statuses for every frozen core criterion in "
                        "order; a compound criterion may be split."),
                },
                "secondary_statuses": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": list(CRITERION_STATUSES),
                    },
                    "description": (
                        "Current statuses for every frozen secondary criterion "
                        "in order; use an empty list when there are none."),
                },
                "fulfilled_at_s": {
                    "type": "number", "minimum": 0,
                    "description": (
                        "Evidence-backed first time all criteria were resolved, "
                        "when that can already be established; an "
                        "unsupported_refused item resolves at its explicit "
                        "refusal time."),
                },
                "immediate_fulfilled_at_s": {
                    "type": "number", "minimum": 0,
                    "description": (
                        "First time every immediate-phase criterion was "
                        "resolved; use the explicit refusal time for "
                        "unsupported_refused."),
                },
                "triggered_fulfilled_at_s": {
                    "type": "number", "minimum": 0,
                    "description": (
                        "First time every triggered-phase criterion was "
                        "resolved; use the explicit refusal time for "
                        "unsupported_refused."),
                },
            },
            "required": ["reason", "core_statuses", "secondary_statuses"],
            "additionalProperties": False,
        },
    },
}]


def _criterion_statuses(arguments: Mapping[str, Any], criteria: dict):
    """Validate one ordered status list against the frozen contract."""
    validate_contract(criteria)
    core_statuses = arguments.get("core_statuses")
    secondary_statuses = arguments.get("secondary_statuses")
    allowed = set(CRITERION_STATUSES)
    if (not isinstance(core_statuses, list)
            or len(core_statuses) < len(criteria["core"])
            or any(status not in allowed for status in core_statuses)
            or not isinstance(secondary_statuses, list)
            or len(secondary_statuses) < len(criteria["secondary"])
            or (not criteria["secondary"] and secondary_statuses)
            or any(status not in allowed for status in secondary_statuses)):
        raise ValueError("criterion statuses must match frozen criteria")
    return list(core_statuses), list(secondary_statuses)


def _criterion_phase_indices(request: Mapping[str, Any], criteria: dict) -> dict:
    """Return validated phase membership, with legacy whole-request fallback."""
    phases = request.get("criterion_phases")
    if not isinstance(phases, Mapping):
        phase = "triggered" if request.get("judge_trigger") else "immediate"
        other = "immediate" if phase == "triggered" else "triggered"
        return {
            phase: {
                "core_indices": list(range(len(criteria["core"]))),
                "secondary_indices": list(range(len(criteria["secondary"]))),
            },
            other: {"core_indices": [], "secondary_indices": []},
        }
    normalized = {}
    for phase in ("immediate", "triggered"):
        value = phases.get(phase)
        if not isinstance(value, Mapping):
            raise ValueError("criterion phases are incomplete")
        normalized[phase] = {}
        for key, size in (
                ("core_indices", len(criteria["core"])),
                ("secondary_indices", len(criteria["secondary"]))):
            indices = value.get(key)
            if (not isinstance(indices, list)
                    or any(type(index) is not int or not 0 <= index < size
                           for index in indices)
                    or len(set(indices)) != len(indices)):
                raise ValueError("criterion phase indices are invalid")
            normalized[phase][key] = list(indices)
    for key, size in (
            ("core_indices", len(criteria["core"])),
            ("secondary_indices", len(criteria["secondary"]))):
        combined = (normalized["immediate"][key]
                    + normalized["triggered"][key])
        if sorted(combined) != list(range(size)):
            raise ValueError("criterion phases must partition the contract")
    return normalized


def _replace_phase_statuses(
    core_statuses: List[str], secondary_statuses: List[str],
    phase_indices: Mapping[str, Any], phase: str, status: str,
) -> tuple[List[str], List[str]]:
    """Return status copies with one frozen criterion phase overwritten."""
    core = list(core_statuses)
    secondary = list(secondary_statuses)
    mapping = phase_indices.get(phase, {})
    for index in mapping.get("core_indices", []):
        core[index] = status
    for index in mapping.get("secondary_indices", []):
        secondary[index] = status
    return core, secondary


def _phase_timing(
    arguments: Mapping[str, Any], evidence: Mapping[str, Any], criteria: dict,
    core_statuses: List[str], secondary_statuses: List[str],
) -> dict:
    """Validate per-phase fulfillment clocks without conflating phases."""
    request = evidence["request"]
    phases = _criterion_phase_indices(request, criteria)
    created = float(request["created_at_s"])
    trigger_origin = float(evidence.get("response_origin_s", created))
    judged_at = float(evidence["judged_at_s"])
    explicit_phases = isinstance(request.get("criterion_phases"), Mapping)
    if not explicit_phases and evidence.get("judge_trigger"):
        phases = {
            "immediate": {"core_indices": [], "secondary_indices": []},
            "triggered": {
                "core_indices": list(range(len(criteria["core"]))),
                "secondary_indices": list(range(
                    len(criteria["secondary"]))),
            },
        }
    result = {
        "phase_indices": phases,
        "values": {},
        "trigger_timing_unmet": False,
        "all_phase_times_known": True,
        "all_within_validity": True,
        "all_timely": True,
        "overall_fulfilled_at_s": None,
    }
    for phase, origin, field in (
            ("immediate", created, "immediate_fulfilled_at_s"),
            ("triggered", trigger_origin, "triggered_fulfilled_at_s")):
        mapping = phases[phase]
        statuses = (
            [core_statuses[index] for index in mapping["core_indices"]]
            + [secondary_statuses[index]
               for index in mapping["secondary_indices"]])
        if not statuses:
            continue
        phase_all_resolved = all(
            status in RESOLVED_CRITERION_STATUSES for status in statuses)
        value = arguments.get(field)
        if not explicit_phases and value is None:
            value = arguments.get("fulfilled_at_s")
        if value is not None:
            if (type(value) not in (int, float) or not math.isfinite(value)
                    or value < created - 1e-9 or value > judged_at + 1e-9):
                raise ValueError(f"{field} is outside observed time")
            value = float(value)
            result["values"][field] = value
            if phase == "triggered" and value < origin - 1e-9:
                result["trigger_timing_unmet"] = True
            if value > origin + criteria["valid_for_s"] + 1e-9:
                result["all_within_validity"] = False
            if value > origin + criteria["expected_response_s"] + 1e-9:
                result["all_timely"] = False
        elif phase_all_resolved:
            result["all_phase_times_known"] = False
    if result["values"]:
        result["overall_fulfilled_at_s"] = max(result["values"].values())
    return result


@dataclass(frozen=True)
class GeneratedPassengerRequest:
    request_id: str
    message: str
    created_at_s: float
    acceptance_criteria: Optional[dict] = None
    judge_trigger: Optional[dict] = None
    criterion_phases: Optional[dict] = None
    immediate_request: Optional[dict] = None
    triggered_request: Optional[dict] = None
    sheet_id: Optional[str] = None
    sheet_revision: int = 1
    sheet_action: str = "create"
    replaces_request_id: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "message": self.message,
            "created_at_s": self.created_at_s,
            "source": "personal_agent",
            "acceptance_criteria": copy.deepcopy(self.acceptance_criteria),
            "judge_trigger": copy.deepcopy(self.judge_trigger),
            "criterion_phases": copy.deepcopy(self.criterion_phases),
            "immediate_request": copy.deepcopy(self.immediate_request),
            "triggered_request": copy.deepcopy(self.triggered_request),
            "sheet_id": self.sheet_id or self.request_id,
            "sheet_revision": self.sheet_revision,
            "sheet_action": self.sheet_action,
            "replaces_request_id": self.replaces_request_id,
        }


class PersonalAgentRuntime:
    """One-call passenger LLM with exactly one previous PA turn retained."""

    def __init__(
        self,
        vehicle_id: str,
        client: Any,
        *,
        persona: str = "",
        long_term_goal: str = "",
    ):
        self.vehicle_id = vehicle_id
        self.client = client
        self.persona = str(persona or "").strip()
        self.long_term_goal = str(long_term_goal or "").strip()
        self.previous_turn: List[dict] = []
        self.request_counter = 0
        self.last_sheet_decision = {
            "action": "none", "active_request_id": None,
            "request": None, "reason": None,
        }
        self.state = {
            "model": getattr(client, "model", ""),
            "config": {
                **_client_audit_config(client),
                "persona": self.persona,
                "long_term_goal": self.long_term_goal,
                "history_policy": "one_previous_exchange_without_observation",
            },
            "model_call_log": [],
            "turn_log": [],
            "token_stats": {"input": [], "output": []},
            "errors": [],
        }
        self.system_prompt = (
            "You are the human passenger riding in this vehicle. At each "
            "vehicle-agent wake, decide whether you naturally want to ask the "
            "in-vehicle agent for something. You may react to cabin state, "
            "weather, motion, comfort, traffic behaviour, or an apparent "
            "safety concern. You are not limited to cabin requests. The trip "
            "destination is assigned by the scenario: never request a new "
            "destination, route, or destination change. You have no direct "
            "control over the vehicle or its equipment. Call at most one "
            "offered send request tool, or call "
            "finish when no request is warranted. Do not invent facts absent "
            "from the supplied passenger-observable state."
            " The observation may contain active_request_sheet. There is at "
            "most one active sheet per vehicle. If no sheet is active, "
            "a send request tool creates it. If a sheet is active, calling "
            "finish keeps it unchanged; calling cancel_passenger_request "
            "cancels it; and a send request tool replaces it with a complete "
            "new version. A replacement must restate every still-desired old "
            "outcome and may remove, change or add outcomes. Never submit a "
            "replacement merely to reset a deadline, and never silently omit "
            "an old outcome that the passenger still wants. At most one "
            "replacement is allowed for a sheet."
            " Use send_immediate_request for outcomes that begin now. Use "
            "send_triggered_request only for a two-stage request with outcomes "
            "both now and after one offered judge trigger. Write one complete, "
            "natural passenger message yourself. Each phase also contains "
            "observable core and secondary outcomes used by the evaluator. "
            "The message must explicitly express those same outcomes, and a "
            "two-stage message must naturally state the chosen timing. The "
            "runtime preserves the message verbatim and never parses it to "
            "infer a trigger. "
            "Also provide one_shot/ongoing request_kind, expected_response_s "
            "and valid_for_s in simulation seconds. Every criterion must be "
            "explicit and self-contained; never add hidden requirements. Every "
            "direct passenger requirement is core. Secondary is only for a "
            "genuinely optional outcome; otherwise use an empty secondary list. "
            "Every criterion must be textually traceable to a requested outcome "
            "in its phase and must be judgeable from evidence available to the "
            "evaluator. Do not turn an "
            "action request into a requirement for a reply or an assumed later "
            "effect unless the passenger explicitly requested that outcome. "
            "Generate demanding but natural requests. Prefer two to four "
            "explicit, independently observable outcomes spanning at least "
            "two available controls or one control plus observable driving "
            "behaviour. Do not reduce a requested medium or hard design to an "
            "atomic device toggle. Never exceed six total core plus secondary "
            "criteria across both phases. Never manufacture difficulty with "
            "an unsafe request: in particular, do not ask to open a door, "
            "trunk or hood while the vehicle is moving. "
            "The observation may include request_design. When a request is "
            "warranted, follow its difficulty, pattern, minimum number of "
            "explicit and core outcomes, phase_structure, judge_trigger "
            "requirement, and allowed_trigger_conditions. A two-stage design "
            "must contain at least one requested outcome in each phase. If "
            "that design "
            "cannot be expressed naturally using observable and controllable "
            "outcomes in the current situation, call finish instead of silently "
            "downgrading it to an easier request. "
            "When the observable situation supports it, prefer a compound, "
            "ordered, delayed, or sustained request with two or three explicit "
            "outcomes instead of an atomic one-step request. Never add "
            "complexity that the available evidence cannot verify. A one-shot "
            "request must not ask for a state that is already fully satisfied. "
            "You may attach one immutable judge_trigger so evaluation starts "
            "from an independently observed condition rather than immediately. "
            "Use only the structured trigger choices and parameter ranges in "
            "observation.available_judge_triggers. A choice may wait for a "
            "future traffic condition; it is not guaranteed to occur. Every "
            "trigger needs timeout_s. Set timeout_s no shorter than after_s "
            "or hold_for_s when either is present. "
            "When using a trigger, put immediate outcomes in "
            "immediate_request and only delayed outcomes in triggered_request. "
            "Do not copy request_design.pattern into request_kind: request_kind "
            "is always exactly one_shot or ongoing. The structured judge_trigger "
            "alone determines when the triggered phase starts. When there is no "
            "judge_trigger, use send_immediate_request. Respect "
            "temporal_task_context.remaining_episode_s and "
            "max_after_delay_s. Never create a request whose time trigger cannot "
            "activate before the episode ends. expected_response_s and "
            "valid_for_s are measured independently for each phase: immediate "
            "from request creation and triggered from trigger activation. "
            "timeout_s bounds how long the evaluator waits for activation. The trigger is evaluator-"
            "only and does not wake the Driving Agent; the Driving Agent must "
            "schedule its own observations. "
            "For ongoing requests valid_for_s is the required observation period. "
            " On arrival, collision, or simulation end, call finish rather "
            "than opening a new request."
        )

    def generate(
        self, observation: dict, passenger_messages: Iterable[str] = (),
    ) -> Optional[GeneratedPassengerRequest]:
        # The Personal Agent is the only passenger-request source.
        del passenger_messages
        observation = copy.deepcopy(observation)
        if "available_judge_triggers" not in observation:
            observation["available_judge_triggers"] = available_judge_triggers(
                observation.get("temporal_task_context"),
                observation.get("request_design"))
        design = observation.get("request_design")
        if isinstance(design, dict):
            design["allowed_trigger_conditions"] = [
                option["condition"] for option in
                observation["available_judge_triggers"]]
        context = observation.get("temporal_task_context")
        if isinstance(context, dict):
            context.pop("future_weather_conditions", None)
            context.pop("future_dark_transition", None)
            context.pop("future_weather_events", None)
            context.pop("future_dark_after_s", None)
            context.pop("guaranteed_trigger_conditions", None)
        active_sheet = observation.get("active_request_sheet")
        active_request_id = (
            str(active_sheet.get("request_id"))
            if isinstance(active_sheet, Mapping)
            and active_sheet.get("request_id") else None)
        default_action = "keep" if active_request_id else "none"
        self.last_sheet_decision = {
            "action": default_action,
            "active_request_id": active_request_id,
            "request": None,
            "reason": None,
        }
        payload = {
            "type": "personal_agent_wake",
            "persona": self.persona or None,
            "todo": {
                "long_term_goal": self.long_term_goal or None,
            },
            "observation": observation,
        }
        current = {
            "role": "user",
            "content": json.dumps(
                payload, ensure_ascii=False, separators=(",", ":")),
        }
        messages = (
            [{"role": "system", "content": self.system_prompt}]
            + copy.deepcopy(self.previous_turn) + [current])
        offered = observation.get("available_judge_triggers")
        tools = copy.deepcopy(_STRUCTURED_PERSONAL_TOOLS)
        triggered_tool = next((tool for tool in tools
            if tool["function"]["name"] == "send_triggered_request"), None)
        if offered and triggered_tool is not None:
            trigger_schema = triggered_tool["function"]["parameters"][
                "properties"]["judge_trigger"]
            parameter_names = {key for option in offered for key in option}
            trigger_schema["properties"] = {
                key: value for key, value in trigger_schema["properties"].items()
                if key in parameter_names}
            trigger_schema["properties"]["condition"]["enum"] = sorted({
                option["condition"] for option in offered})
            trigger_schema["properties"]["trigger_type"]["enum"] = sorted({
                option["trigger_type"] for option in offered})
            weather_targets = sorted({weather for option in offered
                for weather in option.get("weather_condition", [])})
            if weather_targets:
                trigger_schema["properties"]["weather_condition"]["enum"] = (
                    weather_targets)
        elif offered is not None and triggered_tool is not None:
            tools.remove(triggered_tool)
        request = None
        sheet_action = default_action
        cancellation_reason = None
        assistant = {"role": "assistant", "content": None}
        assistant_attempts = []
        tool_receipts = []
        exchange_messages = []
        for attempt_index in range(2):
            try:
                response, _, prompt_tokens, completion_tokens = (
                    self.client.chat_with_tools(messages, tools=tools))
            except Exception as exc:
                self.state["errors"].append({
                    "time_s": observation["sim_time_s"],
                    "attempt": attempt_index + 1,
                    "error": f"{type(exc).__name__}: {exc}",
                })
                break
            assistant = _assistant_message_dict(response)
            assistant_attempts.append(copy.deepcopy(assistant))
            self.state["token_stats"]["input"].append(int(prompt_tokens))
            self.state["token_stats"]["output"].append(int(completion_tokens))
            self.state["model_call_log"].append({
                **copy.deepcopy(
                    getattr(self.client, "last_call_metadata", {}) or {}),
                "role": "personal_agent",
                "vehicle_id": self.vehicle_id,
                "time_s": observation["sim_time_s"],
                "validation_attempt": attempt_index + 1,
            })
            attempt_receipts = []
            retryable_rejection = False
            for call in list(getattr(response, "tool_calls", None) or []):
                function_name = str(getattr(call.function, "name", ""))
                receipt = {"success": True}
                if function_name in {
                        "send_immediate_request", "send_triggered_request"}:
                    arguments = _tool_arguments(call)
                    if request is not None:
                        receipt = {
                            "success": False,
                            "error": "only_one_request_allowed_per_wake",
                        }
                    else:
                        try:
                            has_trigger = function_name == "send_triggered_request"
                            judge_trigger = arguments.get("judge_trigger")
                            criterion_phases = None
                            immediate_request = None
                            triggered_request = None
                            (immediate_request, triggered_request, criteria,
                             criterion_phases) = _build_outcome_contract(
                                arguments, triggered=has_trigger)
                            message = str(arguments["message"]).strip()
                            if has_trigger:
                                validate_judge_trigger(judge_trigger)
                                _validate_trigger_horizon(
                                    judge_trigger, observation)
                            else:
                                judge_trigger = None
                            if not message.strip():
                                raise ValueError("passenger message is empty")
                            _validate_request_design(
                                criteria, judge_trigger, observation,
                                criterion_phases)
                            if active_request_id:
                                revision = int(
                                    active_sheet.get("sheet_revision", 1))
                                if revision >= 2:
                                    raise ValueError(
                                        "active request sheet already used its "
                                        "single allowed replacement")
                                old_message = str(
                                    active_sheet.get("message", "")).strip()
                                if message.strip() == old_message:
                                    raise ValueError(
                                        "replacement request must materially "
                                        "update the active sheet")
                                sheet_action = "update"
                            else:
                                revision = 0
                                sheet_action = "create"
                            self.request_counter += 1
                            request_id = (
                                f"pa-{self.vehicle_id}-"
                                f"{self.request_counter:04d}")
                            request = GeneratedPassengerRequest(
                                request_id=request_id,
                                message=message,
                                created_at_s=float(
                                    observation["sim_time_s"]),
                                acceptance_criteria=copy.deepcopy(criteria),
                                judge_trigger=copy.deepcopy(judge_trigger),
                                criterion_phases=copy.deepcopy(
                                    criterion_phases),
                                immediate_request=copy.deepcopy(
                                    immediate_request),
                                triggered_request=copy.deepcopy(
                                    triggered_request),
                                sheet_id=(
                                    str(active_sheet.get("sheet_id"))
                                    if active_request_id else request_id),
                                sheet_revision=revision + 1,
                                sheet_action=sheet_action,
                                replaces_request_id=active_request_id,
                            )
                            receipt.update(
                                request_id=request.request_id,
                                sheet_action=sheet_action,
                                replaces_request_id=active_request_id)
                        except ValueError as exc:
                            retryable_rejection = True
                            receipt = {
                                "success": False,
                                "error": "invalid_passenger_request",
                                "detail": str(exc),
                                "retry_allowed": attempt_index == 0,
                            }
                elif function_name == "cancel_passenger_request":
                    if not active_request_id:
                        receipt = {
                            "success": False,
                            "error": "no_active_request_sheet",
                        }
                    elif request is not None:
                        receipt = {
                            "success": False,
                            "error": "only_one_sheet_decision_allowed_per_wake",
                        }
                    else:
                        cancellation_reason = str(
                            _tool_arguments(call).get("reason", "")).strip()
                        if not cancellation_reason:
                            receipt = {
                                "success": False,
                                "error": "cancellation_reason_required",
                            }
                        else:
                            sheet_action = "cancel"
                            receipt.update(
                                sheet_action="cancel",
                                request_id=active_request_id)
                elif function_name == "finish":
                    receipt["wake_finished"] = True
                    receipt["sheet_action"] = default_action
                else:
                    receipt = {"success": False, "error": "unknown_tool"}
                attempt_receipts.append({
                    "role": "tool",
                    "tool_call_id": str(getattr(call, "id", "")),
                    "content": json.dumps(
                        receipt, ensure_ascii=False,
                        separators=(",", ":")),
                })
            tool_receipts.extend(copy.deepcopy(attempt_receipts))
            exchange_messages.extend([
                copy.deepcopy(assistant), *copy.deepcopy(attempt_receipts)])
            if (request is not None or sheet_action == "cancel"
                    or not retryable_rejection) \
                    or attempt_index == 1:
                break
            # Give the PA exactly one opportunity to repair its tool arguments
            # using the concrete validation receipt from this same wake.
            messages.extend([
                copy.deepcopy(assistant), *copy.deepcopy(attempt_receipts)])
        # Retain the previous exchange verbatim enough for tool-call protocol
        # validity, but do not duplicate its 10-second motion/cabin snapshot.
        # The current observation already contains the rolling speed curve.
        previous_marker = {
            "role": "user",
            "content": json.dumps({
                "type": "previous_personal_wake",
                "sim_time_s": observation["sim_time_s"],
                "request": request.as_dict() if request else None,
            }, ensure_ascii=False, separators=(",", ":")),
        }
        self.previous_turn = [previous_marker, *exchange_messages]
        self.last_sheet_decision = {
            "action": sheet_action,
            "active_request_id": active_request_id,
            "request": request.as_dict() if request else None,
            "reason": cancellation_reason,
        }
        self.state["turn_log"].append({
            "time_s": observation["sim_time_s"],
            "observation": copy.deepcopy(observation),
            "assistant": assistant,
            "assistant_attempts": assistant_attempts,
            "tool_receipts": copy.deepcopy(tool_receipts),
            "request": request.as_dict() if request else None,
            "sheet_decision": copy.deepcopy(self.last_sheet_decision),
        })
        return request

    def record_terminal_skip(self, observation: dict) -> None:
        """Audit a deterministic terminal no-op without spending an LLM call."""
        self.state["turn_log"].append({
            "time_s": observation["sim_time_s"],
            "observation": copy.deepcopy(observation),
            "assistant": None,
            "tool_receipts": [],
            "request": None,
            "skipped": True,
            "skip_reason": "simulation_ended",
        })

    def record_pending_skip(self, observation: dict) -> None:
        """Audit a wake suppressed while an earlier request is in flight."""
        self.state["turn_log"].append({
            "time_s": observation["sim_time_s"],
            "observation": copy.deepcopy(observation),
            "assistant": None,
            "tool_receipts": [],
            "request": None,
            "skipped": True,
            "skip_reason": "previous_request_awaiting_judgement",
        })


class PassengerJudgeRuntime:
    """Independent LLM that scores one request after a fixed time window."""

    def __init__(self, vehicle_id: str, client: Any):
        self.vehicle_id = vehicle_id
        self.client = client
        self.state = {
            "model": getattr(client, "model", ""),
            "config": _client_audit_config(client),
            "model_call_log": [],
            "judgements": [],
            "response_log": [],
            "evidence_log": [],
            "input_log": [],
            "token_stats": {"input": [], "output": []},
            "errors": [],
        }
        self.system_prompt = (
            "You are an independent VehicleArena passenger-request judge. "
            "Judge one passenger request exactly once. " + RUBRIC
            + _DRIVING_OUTCOME_CONTRACT
            + " You must call submit_passenger_judgement exactly once as a tool call. "
            "Do not write the judgement JSON in ordinary assistant text; "
            "only the tool call submits a judgement."
        )

    def judge(self, evidence: dict, *, finalize: bool = True) -> dict:
        """Judge frozen evidence, with one auditable end-to-end retry."""
        evidence_log_index = len(self.state["evidence_log"])
        self.state["evidence_log"].append(copy.deepcopy(evidence))
        first = self._judge_once(
            evidence, evidence_log_index=evidence_log_index)
        attempt_ids = [first["judge_attempt_id"]]
        if first.get("judged"):
            result = first
            result.setdefault("judge_status", "succeeded")
            result.setdefault("judge_retry_count", 0)
        else:
            repair_context = None
            if first.get("judge_failure_stage") != "provider":
                repair_context = {
                    "type": "judge_submission_repair",
                    "previous_error": first.get(
                        "error", "invalid_judge_result"),
                    "instruction": (
                        "The previous Judge submission was rejected. Re-read "
                        "the same frozen evidence and submit exactly one valid "
                        "tool call whose fields satisfy the supplied schema. "
                        "Plain-text JSON in the assistant message is not a tool call. "
                        "Do not change or invent evidence."),
                }
            second = self._judge_once(
                evidence, evidence_log_index=evidence_log_index,
                repair_context=repair_context,
                retry_of_attempt_id=first["judge_attempt_id"])
            attempt_ids.append(second["judge_attempt_id"])
            result = second
            result["judge_retry_count"] = 1
            result["initial_judge_error"] = first.get("error")
            if result.get("judged"):
                result["judge_status"] = "succeeded_after_retry"
                for error in reversed(self.state["errors"]):
                    if error.get("judge_attempt_id") == first.get(
                            "judge_attempt_id"):
                        error.update(
                            recovered=True,
                            recovered_by_attempt_id=result.get(
                                "judge_attempt_id"))
                        break
                constraints = result.get("score_constraints")
                if isinstance(constraints, list):
                    constraints.append("judge_retry_recovered")
            else:
                # Judge/runtime failures are evaluator failures, not evidence
                # that the vehicle failed the passenger request.
                result.update({
                    "judged": False,
                    "judge_status": "failed",
                    "judge_retry_exhausted": True,
                    "judge_replay_pending": True,
                    "grade": None,
                    "applicable": None,
                    "dimension_scores_100": {},
                    "overall_score_100": None,
                })
        result["judge_attempt_ids"] = attempt_ids
        if finalize:
            self.state["judgements"].append(copy.deepcopy(result))
        return result

    def _judge_once(
        self, evidence: dict, *, evidence_log_index: int,
        repair_context: Optional[dict] = None,
        retry_of_attempt_id: Optional[str] = None,
    ) -> dict:
        bounded = evidence.get("schema_version") == "passenger-judge-evidence-v3"
        prompt = self.system_prompt
        tools = _JUDGE_TOOLS
        if bounded:
            prompt = prompt.replace("Judge one passenger request exactly once", "Check one passenger request at this scheduled window")
            prompt += (
                " This request may be checked at multiple bounded windows. "
                "Report completion_status=completed only when observable evidence "
                "establishes the requested outcome, not merely an acknowledgement, "
                "Todo edit or accepted queued command. Use pending when unmet and "
                "uncertain when evidence is insufficient. Classify request_kind as "
                "one_shot or ongoing. Ongoing requests cannot complete before the "
                "final observation window. For ongoing requests, judge compliance "
                "only within the observed window, never claim future fulfillment. "
                "When final_window=true, completed means the supplied evidence "
                "demonstrates compliance within this observed window, NOT that "
                "the ongoing request has been fulfilled forever. Return completed "
                "when that window-scoped requirement is met. Never return pending "
                "or uncertain solely because the request is ongoing or future "
                "behavior cannot be guaranteed. Return pending for observed "
                "noncompliance, or uncertain for insufficient evidence of the "
                "requested behavior within the window. A queued command or a "
                "stationary vehicle alone does not prove smooth driving. "
                "Example: the passenger asks for smoother driving; the agent "
                "reduces acceleration and the subsequent motion trace demonstrates "
                "smooth driving with no contradictory behavior in the window. "
                "At the final check, return completed with request_kind=ongoing "
                "and cite that trace; do not demand proof about the rest of the trip. "
                "Use frozen acceptance_criteria timing when provided; do not invent a new expiry. "
                "When judge_trigger is present, use its authoritative timeline. "
                "criterion_phases is authoritative: immediate criteria use "
                "request.created_at_s as their response origin, while triggered "
                "criteria use response_origin_s (the trigger activation time). "
                "The trigger never delays or invalidates correct immediate-phase "
                "work. Only a triggered-phase outcome produced before activation "
                "violates trigger timing. Report immediate_fulfilled_at_s and "
                "triggered_fulfilled_at_s separately whenever the corresponding "
                "phase is fully resolved. The trigger never wakes the Driving Agent. "
                "If the trigger timed out without "
                "activating, do not invent an activation or award completion. "
                "At the final window, pending means not completed within that "
                "window; uncertain means unverified. Invalid or unsafe requests must be marked NA, without "
                "fulfilling the original request. Cite evidence in completion_reason. "
                "Previous conclusions are context, not authoritative proof.")
            prompt += (
                " This is the FINAL observation window: make the window-scoped "
                "decision now; no further check is scheduled for this request."
                if evidence.get("final_window") else
                " This is an INTERMEDIATE observation window: ongoing requests "
                "must remain pending or uncertain even when currently compliant.")
            tools = copy.deepcopy(_JUDGE_TOOLS)
            parameters = tools[0]["function"]["parameters"]
            parameters["properties"].update({
                "completion_status": {
                    "type": "string", "enum": ["completed", "pending", "uncertain"],
                    "description": (
                        "For ongoing requests at final_window=true, completed "
                        "means evidence supports compliance within the observed "
                        "window, not forever. Pending means unmet; uncertain "
                        "means insufficient evidence. Before the final window, "
                        "ongoing requests cannot be completed."),
                },
                "request_kind": {"type": "string", "enum": ["one_shot", "ongoing"]},
                "completion_reason": {"type": "string", "minLength": 1, "maxLength": 600},
            })
            parameters["required"].extend(["completion_status", "request_kind", "completion_reason"])
        prompt += (
            "\nFINAL GRADE DECISION PROCEDURE (mandatory): "
            "First classify each core and each secondary requirement as met, "
            "unsupported_refused, unmet or unverified. Use "
            "unsupported_refused only when a timestamped passenger-facing "
            "response clearly declines that specific outcome AND an "
            "authoritative capability query or failed action proves it is "
            "unavailable on this vehicle. Merely missing from the currently "
            "loaded tool list is not proof. A vague refusal, silent omission, "
            "substitute operation, or false completion claim must not receive "
            "unsupported_refused. Saying that the requested operation is "
            "unavailable or cannot be performed is an explicit refusal even "
            "without the word 'refuse'. Internal Todo closure is bookkeeping, "
            "not a physical completion claim, unless the response also says "
            "the unavailable outcome occurred. Include a compact checklist "
            "in reason. "
            "Apply these branches IN ORDER: "
            "(1) If any core outcome is unmet, A/B/C are FORBIDDEN; choose "
            "D for useful partial fulfillment, E for an actual ineffective "
            "attempt, F for no useful action or opposite action. "
            "(2) If core fulfillment cannot be established, do not award A/B/C; "
            "use NA when evidence cannot distinguish fulfillment from failure. "
            "(3) If all core outcomes are resolved but ANY secondary requirement "
            "is unmet OR unverified, choose C, NEVER A or B. A minor omission "
            "is precisely what distinguishes C from A/B; 'not core' is NOT "
            "an exemption. (4) ONLY if every core AND secondary is resolved "
            "may you choose A (timely) or B (late within validity). "
            "Do not infer an unobserved outcome beyond what the supplied "
            "action result or observable state directly establishes. "
            "For an ongoing physical request, a promised action alone is "
            "not the requested outcome; later contradictory physical evidence "
            "must affect the grade. Check the whole supplied window. For any "
            "criterion that applies over an interval, a single clear observed "
            "counterexample makes it unmet unless exceptions were explicitly "
            "allowed. Before claiming that no counterexample exists, inspect "
            "the complete chronology and its extrema; never cherry-pick only "
            "the favorable samples. When the passenger identifies an earlier "
            "observed event as the unwanted behavior to avoid, use that event "
            "as the evidence-grounded reference. A materially similar event "
            "after the request is a counterexample; do not relabel it acceptable "
            "by inventing a different threshold. "
            "If reason contains 'secondary unmet', 'secondary unverified', "
            "'not evidenced' or an equivalent finding, selecting A/B is a "
            "contradiction. Correct the grade before submitting. Do not "
            "fabricate supporting facts to justify an initially chosen grade. "
            "SUBMISSION PREFLIGHT: derive elapsed_s separately for every "
            "non-empty criterion phase. Immediate uses request.created_at_s; "
            "triggered uses authoritative trigger activation. If every phase "
            "is within expected_response_s and all criteria are resolved, choose A. "
            "If every phase is within valid_for_s but at least one exceeds "
            "expected_response_s, choose B. Independently scan your final "
            "reason: if it "
            "labels even one secondary criterion unmet, unverified, unknown, "
            "missing, or not evidenced, the maximum grade is C. All core "
            "criteria resolved plus any unverified secondary criterion must be C, "
            "not A. Evidence is positive, not merely non-contradictory: absence "
            "of a conflicting action does not prove a requested attribute or "
            "effect. For every criterion, identify the specific action result, "
            "observable state, motion trace, or passenger-facing response that "
            "supports it. If no supplied evidence directly supports that "
            "criterion, mark it unverified regardless of the domain or device, "
            "and apply the corresponding grade branch. Respect the meanings "
            "given by the evidence itself; do not redefine a recorded state. "
            "Simulator observations are authoritative for the state they name. "
            "Do not reinterpret an observed value using assumptions about a "
            "real-world implementation or demand an extra sensor absent from "
            "this environment. A direct before/after change in the requested "
            "state is positive evidence. Do not downgrade an observed state "
            "to merely a setting, intention, or proxy unless the supplied "
            "evidence explicitly labels it that way. "
            "Perform all checks immediately before calling the tool."
        )
        criteria = evidence["request"].get("acceptance_criteria")
        if criteria:
            if tools is _JUDGE_TOOLS:
                tools = copy.deepcopy(tools)
            required = tools[0]["function"]["parameters"]["required"]
            required.extend(["core_statuses", "secondary_statuses"])
        intermediate = bool(
            bounded and criteria and not evidence.get("final_window"))
        if intermediate:
            prompt = (
                "You are an independent VehicleArena passenger-request "
                "checker. This is an INTERMEDIATE observation window, not a "
                "final scoring window. Do not assign an A-F or NA grade. Call "
                "submit_passenger_check exactly once as a tool call. Do not "
                "write the check JSON in ordinary assistant text; only the "
                "tool call submits a check. Classify every frozen "
                "core and secondary criterion in order as met, "
                "unsupported_refused, unmet or unverified and cite compact "
                "concrete evidence in reason. unsupported_refused requires "
                "both authoritative proof that the operation is unavailable "
                "on this vehicle and a timestamped, explicit, truthful refusal "
                "to the passenger. A clear statement that the operation is "
                "unavailable or cannot be performed is an explicit refusal; "
                "internal Todo closure alone does not negate it. Substitution "
                "or a claim that the unavailable physical outcome occurred is "
                "unmet. "
                "For an ongoing request, every result remains provisional even "
                "when all criteria are currently resolved; later behavior may "
                "contradict it. For a one-shot request, supply fulfilled_at_s "
                "only when all core and secondary criteria are resolved, so "
                "the harness can close it early. For a phased request also "
                "supply each resolved phase's fulfillment time; for "
                "unsupported_refused use the explicit refusal time. "
                "never use an immediate action time for a triggered phase. A successful "
                "queued command, Todo edit, promise, or lack of contradictory "
                "evidence is not proof of physical fulfillment. Inspect the "
                "whole supplied chronology and its extrema. Previous checks "
                "are provisional context, not authoritative evidence. "
                + _DRIVING_OUTCOME_CONTRACT
            )
            tools = copy.deepcopy(_JUDGE_CHECK_TOOLS)
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": json.dumps(
                _prepare_judge_evidence(evidence),
                ensure_ascii=False, separators=(",", ":"))},
        ]
        if repair_context is not None:
            messages.append({
                "role": "user",
                "content": json.dumps(
                    repair_context, ensure_ascii=False, separators=(",", ":")),
            })
        request_kwargs = _replayable_tool_request(
            self.client, messages, tools)
        call_index = len(self.state["input_log"]) + 1
        request_id = evidence["request"]["request_id"]
        check_index = evidence.get("check_index")
        attempt_id = (
            f"{self.vehicle_id}:{request_id}:"
            f"check-{check_index if check_index is not None else 'legacy'}:"
            f"call-{call_index}")
        input_record = {
            "schema": "vehiclearena-passenger-judge-input-v1",
            "attempt_id": attempt_id,
            "retry_of_attempt_id": retry_of_attempt_id,
            "call_index": call_index,
            "vehicle_id": self.vehicle_id,
            "request_id": request_id,
            "check_index": check_index,
            "time_s": evidence["judged_at_s"],
            "window_kind": "intermediate" if intermediate else "final",
            "evidence_log_index": evidence_log_index,
            "api_base": copy.deepcopy(getattr(
                self.client, "api_base", None)),
            "request_kwargs": request_kwargs,
            "request_sha256": _canonical_json_sha256(request_kwargs),
        }
        self.state["input_log"].append(input_record)
        try:
            response, _, prompt_tokens, completion_tokens = (
                self.client.chat_with_tools(messages, tools=tools))
        except Exception as exc:
            input_record["call_metadata"] = copy.deepcopy(
                getattr(self.client, "last_call_metadata", {}) or {})
            result = {
                "request_id": request_id,
                "judged": False,
                "error": f"{type(exc).__name__}: {exc}",
                "judge_failure_stage": "provider",
                "judge_attempt_id": attempt_id,
            }
            self.state["errors"].append(copy.deepcopy(result))
            return result
        input_record["call_metadata"] = copy.deepcopy(
            getattr(self.client, "last_call_metadata", {}) or {})
        self.state["token_stats"]["input"].append(int(prompt_tokens))
        self.state["token_stats"]["output"].append(int(completion_tokens))
        self.state["model_call_log"].append({
            **copy.deepcopy(getattr(self.client, "last_call_metadata", {}) or {}),
            "role": "passenger_judge",
            "vehicle_id": self.vehicle_id,
            "time_s": evidence["judged_at_s"],
            "request_id": request_id,
            "judge_attempt_id": attempt_id,
            "request_sha256": input_record["request_sha256"],
        })
        assistant = _assistant_message_dict(response)
        self.state["response_log"].append({
            "request_id": request_id,
            "time_s": evidence["judged_at_s"],
            "judge_attempt_id": attempt_id,
            "request_sha256": input_record["request_sha256"],
            "assistant": assistant,
        })
        if intermediate:
            arguments = None
            for call in list(getattr(response, "tool_calls", None) or []):
                if getattr(call.function, "name", "") == \
                        "submit_passenger_check":
                    arguments = _tool_arguments(call)
                    break
            if arguments is None:
                result = {
                    "request_id": evidence["request"]["request_id"],
                    "judged": False,
                    "error": "judge_did_not_submit_check",
                    "judge_failure_stage": "protocol",
                }
            else:
                try:
                    core_statuses, secondary_statuses = _criterion_statuses(
                        arguments, criteria)
                    reason = str(arguments.get("reason", "")).strip()
                    if not reason:
                        raise ValueError("intermediate check requires evidence")
                    created = float(evidence["request"]["created_at_s"])
                    all_resolved = all(
                        status in RESOLVED_CRITERION_STATUSES
                        for status in core_statuses + secondary_statuses)
                    timing = _phase_timing(
                        arguments, evidence, criteria,
                        core_statuses, secondary_statuses)
                    fulfilled = timing["overall_fulfilled_at_s"]
                    trigger_timing_unmet = timing[
                        "trigger_timing_unmet"]
                    if trigger_timing_unmet:
                        core_statuses, secondary_statuses = \
                            _replace_phase_statuses(
                                core_statuses, secondary_statuses,
                                timing["phase_indices"], "triggered", "unmet")
                    result = {
                        "request_id": evidence["request"]["request_id"],
                        "judged": True,
                        "requested_at_s": created,
                        "judged_at_s": evidence["judged_at_s"],
                        "reason": reason[:600],
                        "criterion_statuses": {
                            "core": core_statuses,
                            "secondary": secondary_statuses,
                        },
                        "request_kind": criteria["request_kind"],
                        "completion_reason": reason[:600],
                        "check_only": True,
                        **timing["values"],
                    }
                    if trigger_timing_unmet:
                        triggered_at = timing["values"].get(
                            "triggered_fulfilled_at_s")
                        origin = float(evidence.get(
                            "response_origin_s", created))
                        result.update(
                            completion_status="pending",
                            trigger_timing_status="unmet",
                            reason=(
                                "Temporal constraint unmet: triggered-phase "
                                f"fulfillment at {triggered_at}s preceded "
                                "trigger activation at "
                                f"{origin}s. {reason}")[:600],
                            completion_reason=(
                                "A triggered-phase physical outcome occurred "
                                "before the frozen trigger activated."),
                        )
                        if fulfilled is not None:
                            result["fulfilled_at_s"] = fulfilled
                    elif (criteria["request_kind"] == "one_shot"
                            and all_resolved
                            and timing["all_phase_times_known"]
                            and timing["all_within_validity"]):
                        grade = "A" if timing["all_timely"] else "B"
                        result.update(score_submission({
                            "grade": grade, "reason": reason}))
                        result.update(
                            fulfilled_at_s=fulfilled,
                            completion_status="completed",
                            check_only=False,
                            promoted_to_final=True,
                        )
                    else:
                        statuses = core_statuses + secondary_statuses
                        result["completion_status"] = (
                            "pending" if criteria["request_kind"] == "ongoing"
                            or "unmet" in statuses else "uncertain")
                        if fulfilled is not None:
                            result["fulfilled_at_s"] = fulfilled
                except ValueError as exc:
                    result = {
                        "request_id": evidence["request"]["request_id"],
                        "judged": False,
                        "error": str(exc),
                        "judge_failure_stage": "validation",
                    }
            result.setdefault("judge_attempt_id", attempt_id)
            if not result.get("judged"):
                self.state["errors"].append(copy.deepcopy(result))
            return result
        arguments = None
        for call in list(getattr(response, "tool_calls", None) or []):
            if getattr(call.function, "name", "") == \
                    "submit_passenger_judgement":
                arguments = _tool_arguments(call)
                break
        if arguments is None:
            result = {"request_id": evidence["request"]["request_id"],
                      "judged": False,
                      "error": "judge_did_not_submit_judgement",
                      "judge_failure_stage": "protocol"}
        else:
            trigger_timing_message = None
            temporal_completion_reason = None
            try:
                raw_judge_reason = str(arguments.get("reason", ""))[:600]
                scored = score_submission(arguments)
                if criteria:
                    core_statuses, secondary_statuses = _criterion_statuses(
                        arguments, criteria)
                    all_core_resolved = all(
                        status in RESOLVED_CRITERION_STATUSES
                        for status in core_statuses)
                    all_secondary_resolved = all(
                        status in RESOLVED_CRITERION_STATUSES
                        for status in secondary_statuses)
                    submitted_grade = scored["grade"]
                    trigger = evidence.get("judge_trigger")
                    origin = float(evidence.get(
                        "response_origin_s",
                        evidence["request"]["created_at_s"]))
                    timing = _phase_timing(
                        arguments, evidence, criteria,
                        core_statuses, secondary_statuses)
                    fulfilled_for_timing = timing["values"].get(
                        "triggered_fulfilled_at_s")
                    trigger_timing_unmet = bool(
                        isinstance(trigger, dict)
                        and trigger.get("status") == "activated"
                        and timing["trigger_timing_unmet"])
                    trigger_not_activated = bool(
                        isinstance(trigger, dict)
                        and trigger.get("status") != "activated")
                    all_requirements_resolved = bool(
                        all_core_resolved and all_secondary_resolved)
                    timing_out_of_validity = bool(
                        timing["all_phase_times_known"]
                        and not timing["all_within_validity"])
                    if trigger_not_activated:
                        core_statuses, secondary_statuses = \
                            _replace_phase_statuses(
                                core_statuses, secondary_statuses,
                                timing["phase_indices"], "triggered", "unmet")
                        temporal_completion_reason = (
                            "Temporal condition unmet: the frozen judge "
                            "trigger never activated before the final "
                            "observation window closed.")
                        if submitted_grade in ("A", "B", "C"):
                            corrected_arguments = {
                                **arguments, "grade": "D",
                                "reason": (
                                    temporal_completion_reason + " "
                                    + str(arguments.get("reason", "")))[:600],
                            }
                            scored = score_submission(corrected_arguments)
                            scored["score_constraints"].append(
                                "grade_normalized_from_unactivated_trigger")
                        else:
                            scored["reason"] = (
                                temporal_completion_reason + " "
                                + scored["reason"])[:600]
                    elif trigger_timing_unmet:
                        trigger_timing_message = (
                            "Triggered-phase outcome occurred at "
                            f"{float(fulfilled_for_timing):.6g}s before the "
                            f"trigger activated at {origin:.6g}s.")
                        temporal_completion_reason = (
                            "Temporal constraint unmet: "
                            + trigger_timing_message)
                        core_statuses, secondary_statuses = \
                            _replace_phase_statuses(
                                core_statuses, secondary_statuses,
                                timing["phase_indices"], "triggered", "unmet")
                        if submitted_grade in ("A", "B", "C"):
                            corrected_arguments = {
                                **arguments, "grade": "D",
                                "reason": (
                                    temporal_completion_reason + " "
                                    + str(arguments.get("reason", "")))[:600],
                            }
                            scored = score_submission(corrected_arguments)
                            scored["score_constraints"].append(
                                "grade_normalized_from_trigger_timing")
                        else:
                            scored["reason"] = (
                                temporal_completion_reason + " "
                                + scored["reason"])[:600]
                    elif (all_requirements_resolved
                            and timing["all_phase_times_known"]
                            and timing["all_within_validity"]
                            and not trigger_timing_unmet
                            and (not isinstance(trigger, dict)
                                 or trigger.get("status") == "activated")):
                        required_grade = (
                            "A" if timing["all_timely"] else "B")
                        corrected_arguments = {
                            **arguments, "grade": required_grade}
                        scored = score_submission(corrected_arguments)
                        if submitted_grade != required_grade:
                            scored["score_constraints"].append(
                                "grade_normalized_from_phase_timing")
                    elif (all_core_resolved and secondary_statuses
                            and not all_secondary_resolved):
                        corrected_arguments = {**arguments, "grade": "C"}
                        scored = score_submission(corrected_arguments)
                        if submitted_grade != "C":
                            scored["score_constraints"].append(
                                "grade_normalized_from_criterion_statuses")
                    elif all_requirements_resolved and timing_out_of_validity:
                        corrected_arguments = {**arguments, "grade": "D"}
                        scored = score_submission(corrected_arguments)
                        if submitted_grade != "D":
                            scored["score_constraints"].append(
                                "grade_normalized_from_expired_fulfillment")
                    elif (submitted_grade in ("A", "B", "C")
                            and "unmet" in core_statuses):
                        corrected_arguments = {**arguments, "grade": "D"}
                        scored = score_submission(corrected_arguments)
                        scored["score_constraints"].append(
                            "grade_normalized_from_criterion_statuses")
                    elif submitted_grade in ("A", "B", "C") \
                            and not all_core_resolved:
                        corrected_arguments = {
                            **arguments, "grade": "NA",
                            "na_reason": "insufficient_evidence"}
                        scored = score_submission(corrected_arguments)
                        scored["score_constraints"].append(
                            "grade_normalized_from_unverified_core")
                    all_core_resolved = all(
                        status in RESOLVED_CRITERION_STATUSES
                        for status in core_statuses)
                    all_secondary_resolved = all(
                        status in RESOLVED_CRITERION_STATUSES
                        for status in secondary_statuses)
                    if scored["grade"] == "C" and not (
                            all_core_resolved and secondary_statuses
                            and not all_secondary_resolved):
                        raise ValueError("grade C contradicts criterion statuses")
                    if (scored["grade"] in ("D", "E", "F")
                            and all_core_resolved
                            and not trigger_timing_unmet
                            and not trigger_not_activated
                            and not timing_out_of_validity):
                        raise ValueError("partial/failure grade contradicts criterion statuses")
                    scored["criterion_statuses"] = {
                        "core": list(core_statuses),
                        "secondary": list(secondary_statuses),
                    }
                    scored["judge_reason_raw"] = raw_judge_reason
                    scored.update(timing["values"])
                    if scored["grade"] == "C" and not criteria["secondary"]:
                        raise ValueError("grade C requires frozen secondary requirements")
                    if scored["grade"] in ("A", "B"):
                        trigger = evidence.get("judge_trigger")
                        if (isinstance(trigger, dict)
                                and trigger.get("status") != "activated"):
                            raise ValueError(
                                "A/B requires judge trigger activation")
                        fulfilled = timing["overall_fulfilled_at_s"]
                        if (not timing["all_phase_times_known"]
                                or not timing["all_within_validity"]
                                or fulfilled is None):
                            raise ValueError(
                                "A/B requires evidence-backed fulfillment "
                                "times for every criterion phase")
                        # A/B is derived above from immutable phase origins and
                        # fulfillment times. The model supplies evidence, not
                        # the authoritative timing bucket.
                        created = float(evidence["request"]["created_at_s"])
                        origin = float(evidence.get("response_origin_s", created))
                        if (criteria["request_kind"] == "ongoing"
                                and evidence.get("final_window")
                                and not evidence.get("terminal")
                                and evidence["judged_at_s"] < origin + criteria["valid_for_s"] - 1e-9):
                            raise ValueError("ongoing request observation period incomplete")
                        scored["fulfilled_at_s"] = fulfilled
                result = {
                    "request_id": evidence["request"]["request_id"], "judged": True,
                    "requested_at_s": evidence["request"]["created_at_s"],
                    "judged_at_s": evidence["judged_at_s"], **scored,
                }
            except ValueError as exc:
                result = {"request_id": evidence["request"]["request_id"],
                          "judged": False, "error": str(exc),
                          "judge_failure_stage": "validation"}
        if bounded and result.get("judged"):
            status = arguments.get("completion_status")
            kind = (evidence["request"].get("acceptance_criteria") or {}).get("request_kind", arguments.get("request_kind"))
            completion_reason = str(arguments.get("completion_reason", "")).strip()
            if (status not in {"completed", "pending", "uncertain"}
                    or kind not in {"one_shot", "ongoing"} or not completion_reason):
                result.update(judged=False, error="judge_submitted_invalid_completion",
                              completion_status="uncertain",
                              judge_failure_stage="validation")
            else:
                if status == "completed" and kind == "ongoing" and not evidence.get("final_window"):
                    status = "pending"
                    result["score_constraints"].append("ongoing_request_requires_final_window")
                if result.get("grade") == "NA":
                    status = "uncertain"
                elif result.get("grade") not in ("A", "B"):
                    status = "pending"
                elif status != "completed" and (kind != "ongoing" or evidence.get("final_window")):
                    result.update(
                        judged=False,
                        error="grade_completion_contradiction",
                        judge_failure_stage="validation")
                if not evidence.get("final_window") and result.get("grade") == "B":
                    status = "pending"
                if temporal_completion_reason:
                    completion_reason = temporal_completion_reason
                    result["trigger_timing_status"] = "unmet"
                result.update(completion_status=status, request_kind=kind,
                              completion_reason=completion_reason[:600])
        result.setdefault("judge_attempt_id", attempt_id)
        if not result.get("judged"):
            self.state["errors"].append(copy.deepcopy(result))
        return result


class PersonalAgentVehicleCallback:
    """Wrap a vehicle callback with PA generation and timed judging."""

    def __init__(
        self,
        vehicle_id: str,
        vehicle_callback: Any,
        personal_agent: PersonalAgentRuntime,
        judge: PassengerJudgeRuntime,
        judge_window_s: float = 1.0,
    ):
        self.vehicle_id = vehicle_id
        self.vehicle_callback = vehicle_callback
        self.personal_agent = personal_agent
        self.judge = judge
        self.judge_window_s = float(judge_window_s)
        if self.judge_window_s <= 0.0:
            raise ValueError("judge_window_s must be positive")
        self.pending_evidence: Optional[dict] = None
        # Preserve the driver's established audit/state surface.
        self._state = vehicle_callback._state
        self._state["personal_agent"] = personal_agent.state
        self._state["passenger_judge"] = judge.state
        self._state["passenger_evaluations"] = judge.state["judgements"]
        self._state["personal_requests"] = []
        self._state["passenger_judge_window_s"] = self.judge_window_s

    @property
    def pending_judge_due_at_s(self) -> Optional[float]:
        if self.pending_evidence is None:
            return None
        return float(self.pending_evidence["judge_due_at_s"])

    def __call__(self, vw, t, passenger_messages, memory, tick_index, **kwargs):
        wake_events = [
            event for event in kwargs.get("_wake_events", [])
            if isinstance(event, Mapping)
        ]
        judge_event = any(
            str(event.get("event_type", "")) == "passenger_judge_due"
            for event in wake_events)
        non_judge_events = [
            event for event in wake_events
            if str(event.get("event_type", "")) != "passenger_judge_due"
        ]
        judge_only = bool(judge_event and not non_judge_events)
        observation = build_personal_observation(
            vw,
            vehicle_id=self.vehicle_id,
            sim_time_s=t,
            tick_index=tick_index,
            vehicle_state=kwargs.get("_vehicle_state"),
            wake_events=non_judge_events,
            recent_motion=kwargs.get("_recent_motion", []),
            episode_total_time_s=kwargs.get("_episode_total_time_s"),
            navigation_status=kwargs.get("_navigation_status"),
        )

        if self.pending_evidence is not None:
            observation["previous_vehicle_response"] = (
                _passenger_visible_driver_response(
                    self.pending_evidence.get("vehicle_agent_response", {})))

        # Score an in-flight request exactly at its timer deadline (or at a
        # terminal boundary). This happens before a coincident ordinary wake
        # so the Personal Agent may open a new request after the old one closes.
        terminal = _terminal_event(observation)
        if self.pending_evidence is not None and (judge_event or terminal):
            evidence = self.pending_evidence
            evidence["judged_at_s"] = round(float(t), 6)
            if evidence["request"].get("acceptance_criteria"):
                evidence.update(schema_version="passenger-judge-evidence-v3",
                                final_window=True, terminal=terminal)
            evidence["observation_at_next_wake"] = copy.deepcopy(observation)
            evidence["evaluator_world_at_next_wake"] = copy.deepcopy(
                kwargs.get("_passenger_judge_world", {}))
            evidence["physical_execution"] = {
                "window_s": round(
                    float(t) - float(evidence["request"]["created_at_s"]),
                    6),
                "ego_motion_trace": [
                    copy.deepcopy(item)
                    for item in kwargs.get("_recent_motion", [])
                    if float(item.get("time_s", -1.0))
                    >= float(evidence["request"]["created_at_s"])
                ],
                "nearby_entity_changes": _nearby_changes(
                    evidence.get("evaluator_world_when_requested", {}),
                    evidence.get("evaluator_world_at_next_wake", {})),
                "source_contract": _DRIVING_OUTCOME_CONTRACT,
            }
            self.judge.judge(evidence)
            self.pending_evidence = None

        # A time-limit notification and an evaluator-only timer are not new
        # control opportunities. While a request is in flight, suppress only
        # another PA generation; the Driver still receives ordinary events.
        if terminal:
            self.personal_agent.record_terminal_skip(observation)
            generated = None
        elif judge_only:
            generated = None
        elif self.pending_evidence is not None:
            self.personal_agent.record_pending_skip(observation)
            generated = None
        else:
            # Required order for active wakes: passenger first, then driver.
            generated = self.personal_agent.generate(
                observation, passenger_messages=passenger_messages)
        if generated is not None:
            self._state["personal_requests"].append(generated.as_dict())

        if terminal or judge_only:
            return []

        driver_state = getattr(self.vehicle_callback, "_state", {})
        tool_start = len(driver_state.get("tool_call_log", []))
        message_start = len(driver_state.get("all_messages", []))
        protocol_start = len(driver_state.get("protocol_events", []))
        generated_messages = [generated.message] if generated else []
        actions = self.vehicle_callback(
            vw, t, generated_messages, memory, tick_index,
            **{
                **kwargs,
                "_active_passenger_request_ids": tuple(
                    [self.pending_evidence["request"]["request_id"]]
                    if self.pending_evidence is not None else []),
            })

        if generated is not None:
            driver_response = _compact_driver_response(
                actions=actions or [],
                tool_calls=driver_state.get(
                    "tool_call_log", [])[tool_start:],
                wakes=driver_state.get("all_messages", [])[message_start:],
                protocol_events=driver_state.get(
                    "protocol_events", [])[protocol_start:],
            )
            self.pending_evidence = {
                "schema_version": "passenger-judge-evidence-v2",
                "request": generated.as_dict(),
                "judge_due_at_s": round(
                    float(t) + (generated.acceptance_criteria["valid_for_s"]
                                if generated.acceptance_criteria else self.judge_window_s), 6),
                "judge_window_s": self.judge_window_s,
                "observation_when_requested": copy.deepcopy(observation),
                "evaluator_world_when_requested": copy.deepcopy(
                    kwargs.get("_passenger_judge_world", {})),
                "vehicle_agent_response": driver_response,
                "distance_semantics": {
                    "radar.distance_m": "measured bumper clearance",
                    "center_distance_m": "Euclidean pose-center range",
                    "same_lane_bumper_clearance_m": (
                        "dimension-adjusted longitudinal clearance; compare "
                        "only with the same field at another time"),
                },
            }
        return actions


def aggregate_passenger_judgements(
    judgements: Iterable[dict], *, request_count: Optional[int] = None,
) -> dict:
    # Check histories are separate; one latest final record per request.
    latest = {}
    for index, item in enumerate(judgements):
        latest[item.get("request_id", f"legacy-{index}")] = item
    all_judgements = list(latest.values())
    excluded = [item for item in all_judgements if item.get("excluded")]
    evaluable = [item for item in all_judgements if not item.get("excluded")]
    scored = [item for item in evaluable if item.get("judged")
              and item.get("overall_score_100") is not None
              and item.get("metric_revision") == REVISION]
    mean = round(sum(item["overall_score_100"] for item in scored) / len(scored), 2) if scored else None
    total = max(
        len(all_judgements),
        int(request_count) if request_count is not None else 0)
    evaluable_count = max(0, total - len(excluded))
    score_coverage = len(scored) / total if total else None
    evaluable_rate = evaluable_count / total if total else None
    excluded_rate = len(excluded) / total if total else None
    coverage_adjusted = (
        round(mean * score_coverage, 2)
        if mean is not None and score_coverage is not None else None)
    return {
        "evaluation_type": "request_satisfaction", "metric_revision": REVISION,
        "request_count": total,
        "evaluable_request_count": evaluable_count,
        "excluded_request_count": len(excluded),
        "evaluable_rate": (
            round(evaluable_rate, 4) if evaluable_rate is not None else None),
        "excluded_rate": (
            round(excluded_rate, 4) if excluded_rate is not None else None),
        "judged_count": sum(bool(item.get("judged")) for item in evaluable),
        "scored_count": len(scored),
        "score_coverage_rate": (
            round(score_coverage, 4) if score_coverage is not None else None),
        "coverage_adjusted_score_100": coverage_adjusted,
        "unresolved_request_count": max(
            0, total - len(excluded) - len(scored)),
        "judge_failed_count": sum(not item.get("judged") for item in evaluable),
        "na_count": sum(item.get("grade") == "NA" for item in evaluable),
        "legacy_unscored_count": sum(item.get("judged") and item.get("metric_revision") != REVISION for item in evaluable),
        "grade_counts": {g: sum(item.get("grade") == g for item in scored) for g in GRADE_SCORES},
        "completion_counts": {status: sum(item.get("completion_status") == status for item in all_judgements)
                              for status in ("completed", "uncompleted", "unverified",
                                             "superseded", "cancelled")},
        "unjudged_count": max(0,
            total - len(all_judgements)),
        "dimension_scores_100": {"request_response": mean} if mean is not None else {},
        "overall_score_100": mean,
        "judgements": copy.deepcopy(all_judgements),
    }
