"""Bounded cross-wake context and Todo state for LLM entities.

This module contains no driving policy.  It only constructs observable
current-state messages, retains exactly one previous wake, manages the
model-owned Todo list, and enforces the explicitly configured context budget.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any, Dict, Iterable, List, MutableMapping, Optional


DATA_DELIVERY_REGISTRY = {
    "time_identity": "always",
    "self_now": "always",
    "own_signals": "self_now_always_vehicle_wake",
    "active_control": "always_vehicle_wake",
    "wake_policy": "always_vehicle_wake",
    "loaded_capabilities": "initial_and_on_change",
    "todo": "always",
    "camera_visual": "always_vehicle_wake",
    "lidar_bev": "installed_lidar_vehicle_wake",
    "previous_wake_messages": (
        "previous_wake_without_images_or_capability_list_when_budget_allows"),
    "initial_task": "one_shot",
    "passenger_request": "one_shot",
    "external_message": "one_shot",
    "risk_lifecycle": "visual_only",
    "signal_change": "visual_only",
    "collision": "event_push",
    "todo_expired": "event_push",
    "surrounding_entities": "camera_or_installed_sensor_image_only",
    "navigation_minimap": "tool_image_only",
    "radar_warning": "event_push",
    "radar_frame": "tool_only",
    "control_status": "active_control_only",
    "earlier_history": "tool_only",
    "other_agent_private_state": "never",
    "future_events": "never",
    "evaluator_labels": "never",
}


class ContextWindowExceeded(RuntimeError):
    """Raised when even the minimal request exceeds the configured window."""


class TodoValidationError(ValueError):
    """Raised for an invalid atomic Todo mutation."""


def _tool_name(schema: dict) -> str:
    return str(schema.get("function", {}).get("name", ""))


def _bounded_text(value: Any, *, field: str, maximum: int) -> str:
    text = str(value or "").strip()
    if not text:
        raise TodoValidationError(f"{field} must not be empty")
    if len(text) > maximum:
        raise TodoValidationError(
            f"{field} exceeds {maximum} characters")
    return text


def _positive_seconds(value: Any, *, field: str, maximum: float) -> float:
    try:
        seconds = float(value)
    except (TypeError, ValueError) as exc:
        raise TodoValidationError(f"{field} must be a number") from exc
    if not math.isfinite(seconds) or seconds <= 0.0:
        raise TodoValidationError(f"{field} must be finite and positive")
    if seconds > maximum + 1e-9:
        raise TodoValidationError(
            f"{field} exceeds configured maximum {maximum:g}s")
    return seconds


class TodoStore:
    """One bounded long-term goal plus bounded model-managed subgoals."""

    def __init__(
        self,
        *,
        max_subgoals: int = 6,
        max_long_text_chars: int = 80,
        max_subgoal_text_chars: int = 96,
        max_ttl_s: float = 3600.0,
    ):
        if max_subgoals < 0:
            raise ValueError("max_subgoals must be non-negative")
        if max_ttl_s <= 0:
            raise ValueError("max_ttl_s must be positive")
        self.max_subgoals = int(max_subgoals)
        self.max_long_text_chars = int(max_long_text_chars)
        self.max_subgoal_text_chars = int(max_subgoal_text_chars)
        self.max_ttl_s = float(max_ttl_s)
        self.revision = 0
        self.long_term_goal: Optional[dict] = None
        self.subgoals: List[dict] = []
        self.archive: List[dict] = []
        self._goal_counter = 0
        self._subgoal_counter = 0
        self._notified_expired_ids = set()

    def tool_schema(self) -> dict:
        """Return the sole model interface for mutating the Todo list."""
        return {
            "type": "function",
            "function": {
                "name": "todo_manage",
                "description": (
                    "Atomically modify your automatically visible Todo list. "
                    "It does not inspect the world or execute actions. Goals "
                    "require a finite ttl_s and may later be extended. The "
                    "whole list is discarded when the episode ends. After "
                    "the action tools for an item return authoritative "
                    "success, call todo_manage in the same wake to complete "
                    "that item; never mark it complete before success."),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expected_revision": {
                            "type": "integer",
                            "description": (
                                "Optional Todo revision currently visible in "
                                "the wake context."),
                        },
                        "operations": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": 8,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "op": {
                                        "type": "string",
                                        "enum": [
                                            "set_long_term", "add_subgoal",
                                            "update", "extend", "complete",
                                            "cancel",
                                        ],
                                    },
                                    "id": {
                                        "type": "string",
                                        "minLength": 1,
                                    },
                                    "text": {
                                        "type": "string",
                                        "minLength": 1,
                                        "maxLength": (
                                            self.max_long_text_chars),
                                        "description": (
                                            "Required by set_long_term and "
                                            "add_subgoal. Long-term text is "
                                            f"limited to {self.max_long_text_chars} "
                                            "characters; subgoal text is limited "
                                            f"to {self.max_subgoal_text_chars}."),
                                    },
                                    "ttl_s": {
                                        "type": "number",
                                        "exclusiveMinimum": 0,
                                        "maximum": self.max_ttl_s,
                                        "description": (
                                            "Finite lifetime in simulation "
                                            "seconds, required when creating a "
                                            "goal and bounded by the configured "
                                            f"maximum of {self.max_ttl_s:g}s."),
                                    },
                                    "extra_s": {
                                        "type": "number",
                                        "exclusiveMinimum": 0,
                                        "maximum": self.max_ttl_s,
                                    },
                                },
                                "required": ["op"],
                                "anyOf": [
                                    {
                                        "properties": {"op": {"enum": ["set_long_term"]}},
                                        "required": ["op", "text", "ttl_s"],
                                    },
                                    {
                                        "properties": {
                                            "op": {"enum": ["add_subgoal"]},
                                            "text": {"maxLength": self.max_subgoal_text_chars},
                                        },
                                        "required": ["op", "text", "ttl_s"],
                                    },
                                    {
                                        "properties": {"op": {"enum": ["update"]}},
                                        "required": ["op", "id"],
                                        "anyOf": [{"required": ["text"]}, {"required": ["ttl_s"]}],
                                    },
                                    {
                                        "properties": {"op": {"enum": ["extend"]}},
                                        "required": ["op", "id", "extra_s"],
                                    },
                                    {
                                        "properties": {"op": {"enum": ["complete", "cancel"]}},
                                        "required": ["op", "id"],
                                    },
                                ],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": ["operations"],
                    "additionalProperties": False,
                },
            },
        }

    def _find(self, item_id: str, long_goal: Optional[dict],
              subgoals: List[dict]):
        if long_goal and long_goal["id"] == item_id:
            return "long_term_goal", long_goal
        for item in subgoals:
            if item["id"] == item_id:
                return "subgoal", item
        open_ids = []
        if long_goal:
            open_ids.append(str(long_goal["id"]))
        open_ids.extend(str(item["id"]) for item in subgoals)
        visible = ", ".join(open_ids) if open_ids else "(none)"
        raise TodoValidationError(
            f"unknown open Todo id: {item_id}; open ids: {visible}")

    def _receipt_open_todos(self, now_s: float) -> dict:
        """Return authoritative recovery state with every mutation receipt."""
        return self.to_context(now_s)

    def apply(
        self,
        operations: Iterable[dict],
        *,
        now_s: float,
        expected_revision: Optional[int] = None,
    ) -> dict:
        """Apply a batch atomically and return a compact mutation receipt."""
        if expected_revision is not None and int(expected_revision) != \
                self.revision:
            return {
                "success": False,
                "error": "todo_revision_conflict",
                "expected_revision": int(expected_revision),
                "current_revision": self.revision,
                "open_todos": self._receipt_open_todos(now_s),
            }
        if not isinstance(operations, (list, tuple)) or not operations:
            return {
                "success": False,
                "error": "operations must be a non-empty list",
                "current_revision": self.revision,
                "open_todos": self._receipt_open_todos(now_s),
            }
        if len(operations) > 8:
            return {
                "success": False,
                "error": "at most 8 operations are allowed per call",
                "current_revision": self.revision,
                "open_todos": self._receipt_open_todos(now_s),
            }

        long_goal = copy.deepcopy(self.long_term_goal)
        subgoals = copy.deepcopy(self.subgoals)
        archive = copy.deepcopy(self.archive)
        goal_counter = self._goal_counter
        subgoal_counter = self._subgoal_counter
        touched: List[str] = []
        already_closed: List[str] = []

        try:
            for raw_operation in operations:
                if not isinstance(raw_operation, dict):
                    raise TodoValidationError("each operation must be an object")
                operation = dict(raw_operation)
                op = str(operation.get("op", ""))
                if op == "set_long_term":
                    text = _bounded_text(
                        operation.get("text"), field="text",
                        maximum=self.max_long_text_chars)
                    ttl = _positive_seconds(
                        operation.get("ttl_s"), field="ttl_s",
                        maximum=self.max_ttl_s)
                    if long_goal is None:
                        goal_counter += 1
                        goal_id = f"goal-{goal_counter:04d}"
                        created_at = float(now_s)
                    else:
                        goal_id = long_goal["id"]
                        created_at = long_goal["created_at_s"]
                    long_goal = {
                        "id": goal_id,
                        "text": text,
                        "created_at_s": created_at,
                        "deadline_s": float(now_s) + ttl,
                        "status": "open",
                    }
                    touched.append(goal_id)
                elif op == "add_subgoal":
                    if len(subgoals) >= self.max_subgoals:
                        raise TodoValidationError(
                            f"at most {self.max_subgoals} open subgoals")
                    text = _bounded_text(
                        operation.get("text"), field="text",
                        maximum=self.max_subgoal_text_chars)
                    ttl = _positive_seconds(
                        operation.get("ttl_s"), field="ttl_s",
                        maximum=self.max_ttl_s)
                    subgoal_counter += 1
                    item_id = f"todo-{subgoal_counter:04d}"
                    subgoals.append({
                        "id": item_id,
                        "parent_id": (
                            long_goal["id"] if long_goal else None),
                        "text": text,
                        "created_at_s": float(now_s),
                        "deadline_s": float(now_s) + ttl,
                        "status": "open",
                    })
                    touched.append(item_id)
                elif op == "update":
                    item_id = str(operation.get("id", ""))
                    kind, item = self._find(item_id, long_goal, subgoals)
                    if "text" in operation:
                        item["text"] = _bounded_text(
                            operation.get("text"), field="text",
                            maximum=(
                                self.max_long_text_chars
                                if kind == "long_term_goal"
                                else self.max_subgoal_text_chars))
                    if "ttl_s" in operation:
                        ttl = _positive_seconds(
                            operation.get("ttl_s"), field="ttl_s",
                            maximum=self.max_ttl_s)
                        item["deadline_s"] = float(now_s) + ttl
                    if "text" not in operation and "ttl_s" not in operation:
                        raise TodoValidationError(
                            "update requires text and/or ttl_s")
                    touched.append(item_id)
                elif op == "extend":
                    item_id = str(operation.get("id", ""))
                    _, item = self._find(item_id, long_goal, subgoals)
                    extra = _positive_seconds(
                        operation.get("extra_s"), field="extra_s",
                        maximum=self.max_ttl_s)
                    new_deadline = max(
                        float(now_s), float(item["deadline_s"])) + extra
                    if new_deadline - float(now_s) > self.max_ttl_s + 1e-9:
                        raise TodoValidationError(
                            "extended deadline exceeds configured TTL horizon")
                    item["deadline_s"] = new_deadline
                    touched.append(item_id)
                elif op in ("complete", "cancel"):
                    item_id = str(operation.get("id", ""))
                    closed_status = "completed" if op == "complete" else "cancelled"
                    archived = next((
                        item for item in reversed(archive)
                        if item.get("id") == item_id), None)
                    if archived is not None:
                        if archived.get("status") != closed_status:
                            raise TodoValidationError(
                                f"Todo id {item_id} is already "
                                f"{archived.get('status')}")
                        already_closed.append(item_id)
                        continue
                    kind, item = self._find(item_id, long_goal, subgoals)
                    if kind == "long_term_goal":
                        for child in subgoals:
                            child["status"] = closed_status
                            child["closed_at_s"] = float(now_s)
                            archive.append(child)
                            touched.append(child["id"])
                        subgoals = []
                        item["status"] = closed_status
                        item["closed_at_s"] = float(now_s)
                        archive.append(item)
                        long_goal = None
                    else:
                        item["status"] = closed_status
                        item["closed_at_s"] = float(now_s)
                        archive.append(item)
                        subgoals = [
                            candidate for candidate in subgoals
                            if candidate["id"] != item_id]
                    touched.append(item_id)
                else:
                    raise TodoValidationError(f"unsupported operation: {op}")
        except TodoValidationError as exc:
            return {
                "success": False,
                "error": str(exc),
                "current_revision": self.revision,
                "open_todos": self._receipt_open_todos(now_s),
            }

        if not touched:
            return {
                "success": True,
                "revision": self.revision,
                "modified_ids": [],
                "already_closed_ids": list(dict.fromkeys(already_closed)),
                "open_todos": self._receipt_open_todos(now_s),
            }

        self.long_term_goal = long_goal
        self.subgoals = subgoals
        self.archive = archive
        self._goal_counter = goal_counter
        self._subgoal_counter = subgoal_counter
        self.revision += 1
        for item_id in touched:
            self._notified_expired_ids.discard(item_id)
        return {
            "success": True,
            "revision": self.revision,
            "modified_ids": list(dict.fromkeys(touched)),
            "already_closed_ids": list(dict.fromkeys(already_closed)),
            "open_todos": self._receipt_open_todos(now_s),
        }

    @staticmethod
    def _visible_item(item: dict, now_s: float) -> dict:
        deadline = float(item["deadline_s"])
        value = {
            "id": item["id"],
            "text": item["text"],
            "deadline_s": round(deadline, 6),
            "remaining_s": round(max(0.0, deadline - float(now_s)), 6),
        }
        if item.get("parent_id"):
            value["parent_id"] = item["parent_id"]
        if deadline <= float(now_s) + 1e-9:
            value["overdue"] = True
        return value

    def to_context(self, now_s: float) -> dict:
        """Return the bounded automatically injected open list."""
        return {
            "revision": self.revision,
            "long_term_goal": (
                self._visible_item(self.long_term_goal, now_s)
                if self.long_term_goal else None),
            "subgoals": [
                self._visible_item(item, now_s) for item in self.subgoals],
        }

    def expiration_events(self, now_s: float) -> List[dict]:
        """Return one event per newly observed overdue item."""
        candidates = ([self.long_term_goal] if self.long_term_goal else []) \
            + list(self.subgoals)
        events = []
        for item in candidates:
            if (float(item["deadline_s"]) <= float(now_s) + 1e-9
                    and item["id"] not in self._notified_expired_ids):
                self._notified_expired_ids.add(item["id"])
                events.append({
                    "event_type": "todo_expired",
                    "occurred_at_s": round(float(item["deadline_s"]), 6),
                    "detected_at_s": round(float(now_s), 6),
                    "details": {"todo_id": item["id"]},
                })
        return events

    def next_deadline_s(self, now_s: float) -> Optional[float]:
        """Return the nearest future deadline that still needs a wake.

        The engine uses this value only for scheduling. Expiry semantics and
        the visible ``todo_expired`` event remain owned by this store, so the
        physics scheduler never interprets or alters model goals.
        """
        candidates = ([self.long_term_goal] if self.long_term_goal else []) \
            + list(self.subgoals)
        deadlines = [
            float(item["deadline_s"])
            for item in candidates
            if float(item["deadline_s"]) > float(now_s) + 1e-9
        ]
        return min(deadlines) if deadlines else None

    def audit_dict(self) -> dict:
        return {
            "revision": self.revision,
            "long_term_goal": copy.deepcopy(self.long_term_goal),
            "subgoals": copy.deepcopy(self.subgoals),
            "archive": copy.deepcopy(self.archive),
        }


def build_vehicle_self_now(state: Any) -> dict:
    """Expose actual ego-body and dashboard state.

    ``own_signals`` is actuator self-knowledge, not a duplicate observation of
    the road scene.  Keeping it visible lets a driver verify persistent lamp
    and indicator state after the wake that changed it has rolled out.
    """
    raw = state.as_dict() if hasattr(state, "as_dict") else state
    raw = raw if isinstance(raw, dict) else {}
    mapping = {
        "current_speed_kmh": "speed_kmh",
        "acceleration_mps2": "acceleration_mps2",
        "current_lane": "current_lane",
        "is_crashed": "is_crashed",
        "arrived": "has_arrived",
        "route_failed": "route_failed",
        "route_failure_reason": "route_failure_reason",
        "present_in_physics_world": "present_in_physics_world",
        "terminal_crossing_speed_kmh": "terminal_crossing_speed_kmh",
    }
    result = {
        exposed: raw[source]
        for source, exposed in mapping.items() if source in raw
    }
    signal_state = raw.get("signal_state")
    if hasattr(signal_state, "as_dict"):
        signal_state = signal_state.as_dict()
    if isinstance(signal_state, Mapping):
        signal_names = (
            "left_indicator", "right_indicator", "hazard", "brake_light",
            "low_beam", "high_beam", "front_fog_light", "rear_fog_light",
            "position_light", "tail_light",
        )
        result["own_signals"] = {
            name: bool(signal_state.get(name, False))
            for name in signal_names
        }
    result["is_disabled"] = bool(
        result.get("is_crashed") or result.get("has_arrived"))
    return result


def build_vehicle_active_control(state: Any) -> dict:
    """Expose only commands that the ego vehicle has actually committed.

    This is actuator self-knowledge, not environmental perception.  It keeps
    a persistent command authoritative after the wake that submitted it has
    rolled out of the one-wake message window.
    """
    raw = state.as_dict() if hasattr(state, "as_dict") else state
    raw = raw if isinstance(raw, dict) else {}
    if raw.get("is_crashed") or raw.get("arrived"):
        return {}
    committed = raw.get("active_control_commands", {})
    if not isinstance(committed, Mapping):
        return {}
    result = {}
    for slot in ("longitudinal", "lateral", "route_maneuver"):
        value = committed.get(slot)
        if not isinstance(value, Mapping):
            continue
        if (slot == "lateral" and not raw.get("is_changing_lane")
                and int(raw.get("target_lane", -1)) < 0):
            continue
        if (slot == "route_maneuver"
                and not raw.get("planned_connector_id")
                and not raw.get("active_connector_id")):
            continue
        result[slot] = {
            key: copy.deepcopy(value[key]) for key in (
                "command_id", "command", "committed_at_s",
                "target_speed_kmh", "target_lane", "emergency_brake",
                "maneuver",
            ) if key in value
        }
    return result


def build_pedestrian_self_now(state: Any) -> dict:
    """Expose actual pedestrian-body state without route/destination truth."""
    raw = state.as_dict() if hasattr(state, "as_dict") else state
    raw = raw if isinstance(raw, dict) else {}
    mapping = {
        "speed": "speed_mps",
        "is_waiting": "is_waiting",
        "is_walking": "is_walking",
        "is_on_crosswalk": "is_on_crosswalk",
        "crossing_progress": "crossing_progress",
        "walking_progress": "walking_progress",
        "is_crashed": "is_crashed",
        "has_arrived": "has_arrived",
    }
    return {
        exposed: raw[source]
        for source, exposed in mapping.items() if source in raw
    }


def normalize_new_events(
    passenger_messages: Iterable[str],
    wake_events: Optional[Iterable[dict]] = None,
) -> List[dict]:
    """Combine authoritative world events with passenger requests."""
    events = [copy.deepcopy(item) for item in (wake_events or [])]
    events.extend({
        "event_type": "passenger_request",
        "message": str(message),
    } for message in (passenger_messages or []))
    return events


def build_current_wake_message(
    *,
    sim_time_s: float,
    tick_index: int,
    agent_id: str,
    entity_type: str,
    self_now: dict,
    todo: dict,
    new_events: Iterable[dict],
    active_control: Optional[dict] = None,
    wake_policy: Optional[dict] = None,
) -> dict:
    payload = {
        "type": "wake",
        "sim_time_s": round(float(sim_time_s), 6),
        "wake_id": f"wake-{int(tick_index):06d}",
        "entity": {
            "agent_id": agent_id,
            "entity_type": entity_type,
        },
        "self_now": self_now,
        "active_control": copy.deepcopy(active_control or {}),
        "todo": todo,
        "new_events": list(new_events),
    }
    if wake_policy is not None:
        payload["wake_policy"] = copy.deepcopy(wake_policy)
    return {
        "role": "user",
        "content": json.dumps(
            payload, ensure_ascii=False, default=str,
            separators=(",", ":")),
    }


def build_loaded_capabilities_message(
    tools: Iterable[dict], loaded_skills: Iterable[str], capability_epoch: int,
) -> dict:
    """Tell the model which supplied schemas are already directly callable."""
    payload = {
        "type": "loaded_capabilities",
        "capability_epoch": int(capability_epoch),
        "tools": sorted(filter(None, (_tool_name(item) for item in tools))),
        "skills": sorted(str(item) for item in loaded_skills),
        "instruction": (
            "Every listed tool is already loaded. Call it directly; do not "
            "request it again with load_tools."),
    }
    return {
        "role": "user",
        "content": json.dumps(
            payload, ensure_ascii=False, separators=(",", ":")),
    }


def estimate_request_tokens(messages: List[dict], tools: List[dict]) -> int:
    """Conservative tokenizer-independent upper estimate for preflight.

    VehicleArena supports arbitrary OpenAI-compatible providers, so a local
    exact tokenizer is not always available.  UTF-8 bytes / 2 is deliberately
    conservative for the Chinese/English prompts used here and includes a
    small per-message/schema framing allowance.  Provider-reported usage is
    still recorded after each real call.
    """
    sanitized, image_count = _sanitize_media_for_estimate({
        "messages": messages, "tools": tools})
    serialized = json.dumps(
        sanitized,
        ensure_ascii=False, default=str, separators=(",", ":"))
    framing = 8 * len(messages) + 12 * len(tools) + 16
    # Provider image accounting varies by model.  A conservative fixed charge
    # avoids treating base64 as text while still reserving meaningful context
    # for each high-detail 768px observation.
    image_tokens = image_count * 1800
    return (int(math.ceil(len(serialized.encode("utf-8")) / 2.0))
            + framing + image_tokens)


def _sanitize_media_for_estimate(value: Any) -> tuple[Any, int]:
    """Replace embedded image bytes with fixed markers for token preflight."""
    if isinstance(value, list):
        values = []
        count = 0
        for item in value:
            sanitized, nested = _sanitize_media_for_estimate(item)
            values.append(sanitized)
            count += nested
        return values, count
    if isinstance(value, dict):
        if value.get("type") == "image_url":
            image_url = value.get("image_url")
            if isinstance(image_url, dict):
                url = str(image_url.get("url", ""))
                if url.startswith("data:image/"):
                    replacement = copy.deepcopy(value)
                    replacement["image_url"]["url"] = "[embedded-image]"
                    return replacement, 1
        result = {}
        count = 0
        for key, item in value.items():
            sanitized, nested = _sanitize_media_for_estimate(item)
            result[key] = sanitized
            count += nested
        return result, count
    return value, 0


def _estimate_value_tokens(value: Any) -> int:
    sanitized, image_count = _sanitize_media_for_estimate(value)
    serialized = json.dumps(
        sanitized, ensure_ascii=False, default=str, separators=(",", ":"))
    return (int(math.ceil(len(serialized.encode("utf-8")) / 2.0))
            + image_count * 1800)


def _wake_id(message: dict) -> Optional[str]:
    if message.get("role") != "user":
        return None
    try:
        payload = json.loads(message.get("content", ""))
    except (TypeError, json.JSONDecodeError):
        return None
    if isinstance(payload, dict) and payload.get("type") == "wake":
        return str(payload.get("wake_id", "")) or None
    return None


def _request_component_estimates(
    *, system_contract: str, previous_wake_messages: List[dict],
    wake_messages: List[dict], tools: List[dict],
    loaded_skills: MutableMapping[str, str],
) -> dict:
    current_wake_payload = {}
    if wake_messages:
        try:
            parsed = json.loads(wake_messages[0].get("content", ""))
        except (TypeError, json.JSONDecodeError):
            parsed = {}
        if isinstance(parsed, dict) and parsed.get("type") == "wake":
            current_wake_payload = parsed
    wake_field_tokens = {
        field: _estimate_value_tokens(current_wake_payload.get(field))
        for field in (
            "sim_time_s", "wake_id", "entity", "self_now",
            "active_control", "todo", "new_events")
        if field in current_wake_payload
    }
    return {
        "system_contract": _estimate_value_tokens(system_contract),
        "previous_wake_messages": _estimate_value_tokens(
            previous_wake_messages),
        "current_wake": _estimate_value_tokens(
            wake_messages[0] if wake_messages else {}),
        "current_wake_fields": wake_field_tokens,
        "current_wake_followup_messages": _estimate_value_tokens(
            wake_messages[1:]),
        "loaded_skills": _estimate_value_tokens(dict(loaded_skills)),
        "tool_schemas": _estimate_value_tokens(tools),
    }


class RollingWakeContext:
    """Retain one previous wake without replaying stale image payloads."""

    def __init__(
        self,
        system_contract: str,
        *,
        context_window_tokens: int,
        max_output_tokens: int,
    ):
        if context_window_tokens <= max_output_tokens:
            raise ValueError(
                "context_window_tokens must exceed max_output_tokens")
        self.system_contract = str(system_contract)
        self.context_window_tokens = int(context_window_tokens)
        self.max_output_tokens = int(max_output_tokens)
        self.previous_wake_messages: List[dict] = []
        self.capability_epoch = 0
        self.budget_log: List[dict] = []

    @property
    def input_budget(self) -> int:
        return self.context_window_tokens - self.max_output_tokens

    @staticmethod
    def skill_messages(loaded_skills: MutableMapping[str, str]) -> List[dict]:
        return [{
            "role": "system",
            "content": f"[LoadedSkill:{name}]\n{content}",
        } for name, content in loaded_skills.items()]

    def compose(
        self,
        wake_messages: List[dict],
        loaded_skills: MutableMapping[str, str],
    ) -> List[dict]:
        # Some OpenAI-compatible chat templates accept exactly one system
        # message, at index zero. Keep skills at the same authority level by
        # appending their bodies to the contract rather than adding messages.
        system_content = "\n\n".join(
            [self.system_contract]
            + [message["content"] for message in self.skill_messages(loaded_skills)])
        return ([{"role": "system", "content": system_content}]
                + copy.deepcopy(self.previous_wake_messages)
                + copy.deepcopy(wake_messages))

    def prepare_request(
        self,
        *,
        wake_messages: List[dict],
        tools: List[dict],
        core_tool_names: set,
        loaded_skills: MutableMapping[str, str],
        entity_id: str,
        sim_time_s: float,
        tick_index: int,
        turn_index: int,
    ) -> tuple[List[dict], dict]:
        """Compose a bounded request, discarding only recoverable context."""
        messages = self.compose(wake_messages, loaded_skills)
        estimate_before = estimate_request_tokens(messages, tools)
        unloaded_tools: List[str] = []
        unloaded_skills: List[str] = []
        dropped_previous_wake_ids: List[str] = []
        dropped_previous_wake_message_count = 0
        if estimate_before > self.input_budget:
            unloaded_tools = [
                _tool_name(item) for item in tools
                if _tool_name(item) not in core_tool_names]
            unloaded_skills = list(loaded_skills)
            if unloaded_tools or unloaded_skills:
                tools[:] = [
                    item for item in tools
                    if _tool_name(item) in core_tool_names]
                loaded_skills.clear()
                self.capability_epoch += 1
                notice = {
                    "type": "capabilities_unloaded",
                    "reason": "configured_context_limit",
                    "capability_epoch": self.capability_epoch,
                    "unloaded_tools": unloaded_tools,
                    "unloaded_skills": unloaded_skills,
                }
                wake_messages.append({
                    "role": "user",
                    "content": json.dumps(
                        notice, ensure_ascii=False, separators=(",", ":")),
                })
                messages = self.compose(wake_messages, loaded_skills)
        if (estimate_request_tokens(messages, tools) > self.input_budget
                and self.previous_wake_messages):
            dropped_previous_wake_ids = [
                wake_id for wake_id in (
                    _wake_id(item) for item in self.previous_wake_messages)
                if wake_id]
            dropped_previous_wake_message_count = len(
                self.previous_wake_messages)
            self.previous_wake_messages.clear()
            wake_messages.append({
                "role": "user",
                "content": json.dumps({
                    "type": "previous_wake_context_dropped",
                    "reason": "configured_context_limit",
                    "message": (
                        "The prior wake transcript no longer fits. Rely on "
                        "CurrentWake, active_control, Todo, and new events; "
                        "query current state again if needed."),
                }, ensure_ascii=False, separators=(",", ":")),
            })
            messages = self.compose(wake_messages, loaded_skills)
        estimate_after = estimate_request_tokens(messages, tools)
        record = {
            "entity_id": entity_id,
            "time_s": round(float(sim_time_s), 6),
            "tick_index": int(tick_index),
            "turn_index": int(turn_index),
            "context_window_tokens": self.context_window_tokens,
            "max_output_tokens": self.max_output_tokens,
            "input_budget_tokens": self.input_budget,
            "estimated_input_tokens_before": estimate_before,
            "estimated_input_tokens_after": estimate_after,
            "estimated_component_tokens_after": (
                _request_component_estimates(
                    system_contract=self.system_contract,
                    previous_wake_messages=self.previous_wake_messages,
                    wake_messages=wake_messages,
                    tools=tools,
                    loaded_skills=loaded_skills)),
            "previous_wake_ids": [
                wake_id for wake_id in (
                    _wake_id(item)
                    for item in self.previous_wake_messages)
                if wake_id],
            "dropped_previous_wake_ids": dropped_previous_wake_ids,
            "dropped_previous_wake_message_count": (
                dropped_previous_wake_message_count),
            "estimator": "utf8_bytes_div_2_v1",
            "capability_epoch": self.capability_epoch,
            "unloaded_tools": unloaded_tools,
            "unloaded_skills": unloaded_skills,
        }
        self.budget_log.append(record)
        if estimate_after > self.input_budget:
            raise ContextWindowExceeded(
                f"minimal request estimate {estimate_after} exceeds input "
                f"budget {self.input_budget} for {entity_id}")
        return messages, record

    @staticmethod
    def _retain_across_wakes(message: dict) -> bool:
        """Keep decisions/receipts while dropping redundant observations."""
        content = message.get("content")
        if isinstance(content, list) and any(
                isinstance(item, dict) and item.get("type") == "image_url"
                for item in content):
            return False
        if message.get("role") == "user" and isinstance(content, str):
            try:
                payload = json.loads(content)
            except (TypeError, json.JSONDecodeError):
                payload = None
            if (isinstance(payload, dict)
                    and payload.get("type") == "loaded_capabilities"):
                return False
        return True

    def finish_wake(self, wake_messages: List[dict]) -> None:
        self.previous_wake_messages = copy.deepcopy([
            message for message in wake_messages
            if self._retain_across_wakes(message)
        ])


def load_skill_content(skill_loader, skill_name: str, *, available_modules=None):
    """Load a Skill body while returning only a compact tool receipt."""
    content = skill_loader.load_skill(
        skill_name, available_modules=available_modules)
    if isinstance(content, str) and content.startswith("[ERROR]"):
        return None, {"success": False, "error": content}
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return content, {
        "success": True,
        "loaded_skill": skill_name,
        "content_sha256": digest,
    }
