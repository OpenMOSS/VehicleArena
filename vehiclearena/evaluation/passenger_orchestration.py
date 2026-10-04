"""Event/random passenger scheduling and bounded, non-notifying judging.

All deadlines use simulation time. This module never advances the world and
never publishes judge conclusions as passenger or driver wake events.
"""
from __future__ import annotations

import copy
import hashlib
import math
import random
from evaluation.judge_schedule import resolve_judge_schedule

from personal_agent import (
    PersonalAgentVehicleCallback, build_personal_observation,
    _compact_driver_response, _passenger_visible_driver_response, _terminal_event,
)
from evaluation.request_satisfaction import (
    available_judge_triggers, score_submission,
    validate_available_judge_trigger,
)


PA_SHARED_EVENTS = frozenset({
    "simulation_start", "acoustic_cue", "weather_changed", "daynight_changed",
})

_REQUEST_DESIGN_CYCLE = (
    {"difficulty": "medium", "pattern": "compound_immediate",
     "min_explicit_outcomes": 3, "min_core_outcomes": 2,
     "judge_trigger": "forbidden", "allowed_trigger_conditions": []},
    {"difficulty": "hard", "pattern": "two_stage_temporal",
     "min_explicit_outcomes": 4, "min_core_outcomes": 3,
     "judge_trigger": "required", "phase_structure": "immediate_and_triggered",
     "allowed_trigger_conditions": [
         "exit_next_intersection", "enter_next_intersection",
         "vehicle_stopped", "vehicle_resumed_moving"]},
    {"difficulty": "hard", "pattern": "delayed_compound",
     "min_explicit_outcomes": 4, "min_core_outcomes": 3,
     "judge_trigger": "required", "phase_structure": "immediate_and_triggered",
     "allowed_trigger_conditions": [
         "exit_next_intersection", "approach_next_intersection",
         "distance_to_destination_below"]},
    {"difficulty": "medium", "pattern": "sustained_behavior",
     "min_explicit_outcomes": 3, "min_core_outcomes": 2,
     "judge_trigger": "required", "phase_structure": "immediate_and_triggered",
     "allowed_trigger_conditions": ["vehicle_stopped", "speed_threshold_held"]},
    {"difficulty": "hard", "pattern": "cross_domain_temporal",
     "min_explicit_outcomes": 4, "min_core_outcomes": 3,
     "judge_trigger": "required", "phase_structure": "immediate_and_triggered",
     "allowed_trigger_conditions": [
         "weather_changed_to", "daynight_became_dark", "vehicle_stopped"]},
    {"difficulty": "medium", "pattern": "compound_immediate",
     "min_explicit_outcomes": 3, "min_core_outcomes": 2,
     "judge_trigger": "forbidden", "allowed_trigger_conditions": []},
    {"difficulty": "hard", "pattern": "ordered_compound",
     "min_explicit_outcomes": 4, "min_core_outcomes": 3,
     "judge_trigger": "required", "phase_structure": "immediate_and_triggered",
     "allowed_trigger_conditions": [
         "enter_next_intersection", "exit_next_intersection",
         "vehicle_resumed_moving"]},
    {"difficulty": "medium", "pattern": "delayed_compound",
     "min_explicit_outcomes": 3, "min_core_outcomes": 2,
     "judge_trigger": "required", "phase_structure": "immediate_and_triggered",
     "allowed_trigger_conditions": ["after_delay"]},
    {"difficulty": "medium", "pattern": "two_stage_temporal",
     "min_explicit_outcomes": 3, "min_core_outcomes": 2,
     "judge_trigger": "required", "phase_structure": "immediate_and_triggered",
     "allowed_trigger_conditions": [
         "approach_next_intersection", "exit_next_intersection",
         "vehicle_stopped"]},
    {"difficulty": "hard", "pattern": "cross_domain_temporal",
     "min_explicit_outcomes": 4, "min_core_outcomes": 3,
     "judge_trigger": "required", "phase_structure": "immediate_and_triggered",
     "allowed_trigger_conditions": [
         "distance_to_destination_below", "weather_changed_to",
         "daynight_became_dark", "vehicle_stopped"]},
)


class PassengerWakeScheduler:
    """Passenger-observable edge detectors plus an independent random clock."""

    def __init__(self, vehicle_id, *, seed=0, random_min_s=15.0,
                 random_max_s=45.0, cooldown_s=5.0, stopped_after_s=15.0,
                 hard_brake_mps2=3.0, hard_brake_duration_s=0.3,
                 event_types=None):
        self.vehicle_id = vehicle_id
        self.config = dict(seed=seed, random_min_s=random_min_s,
            random_max_s=random_max_s, cooldown_s=cooldown_s,
            stopped_after_s=stopped_after_s, hard_brake_mps2=hard_brake_mps2,
            hard_brake_duration_s=hard_brake_duration_s,
            event_types=sorted(PA_SHARED_EVENTS if event_types is None else event_types))
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("passenger seed must be an integer")
        for key in ("random_min_s", "random_max_s", "cooldown_s", "stopped_after_s",
                    "hard_brake_mps2", "hard_brake_duration_s"):
            value = float(self.config[key])
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be finite and positive")
            self.config[key] = value
        if self.config["random_max_s"] < self.config["random_min_s"]:
            raise ValueError("random_max_s must be >= random_min_s")
        if not set(self.config["event_types"]).issubset(PA_SHARED_EVENTS):
            raise ValueError("PA subscriptions must be passenger-observable shared events")
        self.rng = None
        self.next_random_at_s = None
        self.last_wake_s = -math.inf
        self.stopped_since = self.braking_since = None
        self.stop_reported = self.brake_reported = False
        self.pending = {}
        self.log = []

    def poll(self, time_s, state, events, *, episode_id=""):
        if state.get("arrived") or state.get("route_failed") or state.get("is_crashed"):
            return None
        if self.rng is None:
            material = f"{self.config['seed']}:{episode_id}:{self.vehicle_id}"
            self.rng = random.Random(int.from_bytes(
                hashlib.sha256(material.encode()).digest(), "big"))
            self._next_random(time_s)
        for event in events:
            kind = event.get("event_type")
            if kind not in self.config["event_types"]:
                continue
            # Deliberately do not forward the original event's evaluator or
            # sensor-internal payload. Current PA observations supply facts.
            self.pending[kind] = {"event_type": kind}
        speed = float(state.get("current_speed_kmh", 0.0))
        acceleration = float(state.get("acceleration_mps2", 0.0))
        if speed < 0.5:
            if self.stopped_since is None:
                self.stopped_since = time_s
            if (not self.stop_reported and
                    time_s - self.stopped_since >= self.config["stopped_after_s"] - 1e-9):
                self.pending["passenger_long_stop"] = {
                    "event_type": "passenger_long_stop"}
                self.stop_reported = True
        elif speed > 2.0:
            self.stopped_since = None
            self.stop_reported = False
            self.pending.pop("passenger_long_stop", None)
        if acceleration <= -self.config["hard_brake_mps2"]:
            if self.braking_since is None:
                self.braking_since = time_s
            if (not self.brake_reported and time_s - self.braking_since
                    >= self.config["hard_brake_duration_s"] - 1e-9):
                self.pending["passenger_hard_brake"] = {
                    "event_type": "passenger_hard_brake"}
                self.brake_reported = True
        elif acceleration > -self.config["hard_brake_mps2"] / 2:
            self.braking_since = None
            self.brake_reported = False
        if time_s >= self.next_random_at_s - 1e-9:
            self.pending["passenger_random"] = {"event_type": "passenger_random"}
            # Random deadlines advance independently of event/driver wakes.
            self._next_random(time_s)
        if not self.pending or time_s - self.last_wake_s < self.config["cooldown_s"] - 1e-9:
            return None
        triggers = list(self.pending.values())
        self.pending.clear()
        self.last_wake_s = time_s
        row = {"time_s": round(time_s, 6), "triggers": triggers}
        self.log.append(copy.deepcopy(row))
        return row

    def _next_random(self, time_s):
        self.next_random_at_s = time_s + self.rng.uniform(
            self.config["random_min_s"], self.config["random_max_s"])


class EventDrivenPassengerCallback(PersonalAgentVehicleCallback):
    """PA wakes also wake the driver; judge-only checks never wake either."""

    def __init__(self, *args, schedule_config=None, judge_window_s=None,
                 max_checks=3, request_ttl_s=None, check_offsets_s=None,
                 acceptance_timeout_s=None, **kwargs):
        self.check_offsets_s, self.acceptance_timeout_s = resolve_judge_schedule(
            window_s=judge_window_s, max_checks=max_checks,
            check_offsets_s=check_offsets_s, request_ttl_s=request_ttl_s,
            acceptance_timeout_s=acceptance_timeout_s)
        super().__init__(*args, judge_window_s=self.check_offsets_s[0], **kwargs)
        self.max_checks = len(self.check_offsets_s)
        self.request_ttl_s = self.acceptance_timeout_s  # legacy Python alias only
        self.scheduler = PassengerWakeScheduler(self.vehicle_id, **(schedule_config or {}))
        self.pending_requests = {}
        self.request_records = []
        self.request_design_counter = 0
        # Bound to the episode on the first scheduler poll. Keeping this lazy
        # ensures different scenes do not all receive the same first template
        # merely because their ego vehicle shares the same id.
        self.request_design_offset = None
        self.request_design_episode_id = None
        self.judge.state["check_history"] = []
        self.judge.state["config"].update({
            "mode": "scheduled_acceptance", "check_offsets_s": self.check_offsets_s,
            "max_checks": self.max_checks,
            "acceptance_timeout_s": self.acceptance_timeout_s})
        self.personal_agent.state["config"].update({
            "trigger_mode": "event_random",
            "request_sheet_policy": "single_active_full_replacement_v1",
            "max_revisions_per_sheet": 1,
            **self.scheduler.config})
        self.personal_agent.state["trigger_log"] = self.scheduler.log
        self.personal_agent.state["request_records"] = self.request_records
        self._state["passenger_request_records"] = self.request_records

    def _next_request_design(self, temporal_context=None):
        """Choose the next viable request shape for this wake.

        A required-trigger template is not a valid task if this particular
        vehicle cannot activate any of its offered predicates and still leave
        an observation window.  Skip such templates at generation time rather
        than asking the PA to invent an impossible trigger.
        """
        if self.request_design_offset is None:
            self._bind_request_design_episode("")
        has_production_horizon = isinstance(temporal_context, dict) and any(
            key in temporal_context for key in (
                "remaining_episode_s", "estimated_free_flow_remaining_s"))
        attempts = len(_REQUEST_DESIGN_CYCLE) if has_production_horizon else 1
        for _ in range(attempts):
            index = ((self.request_design_offset + self.request_design_counter)
                     % len(_REQUEST_DESIGN_CYCLE))
            self.request_design_counter += 1
            design = copy.deepcopy(_REQUEST_DESIGN_CYCLE[index])
            if (not has_production_horizon
                    or design.get("judge_trigger") != "required"
                    or available_judge_triggers(temporal_context, design)):
                return design
        # The cycle always contains immediate templates. This fallback keeps
        # the callback total if a future edit accidentally removes them.
        return copy.deepcopy(_REQUEST_DESIGN_CYCLE[0])

    def _bind_request_design_episode(self, episode_id):
        if self.request_design_offset is not None:
            return
        seed = int(self.scheduler.config["seed"])
        material = f"{seed}:{episode_id}:{self.vehicle_id}:request-design"
        digest = hashlib.sha256(material.encode()).digest()
        self.request_design_offset = int.from_bytes(
            digest[:4], "big") % len(_REQUEST_DESIGN_CYCLE)
        self.request_design_episode_id = str(episode_id)

    @property
    def pending_judge_due_at_s(self):
        return min((p["next_check_at_s"] for p in self.pending_requests.values()), default=None)

    def _active_sheet_view(self):
        if not self.pending_requests:
            return None
        # The single-sheet invariant is enforced when a replacement is
        # accepted. Keep this deterministic if loading an older checkpoint.
        pending = list(self.pending_requests.values())[-1]
        request = pending["request"]
        record = pending["record"]
        return {
            "request_id": request["request_id"],
            "sheet_id": request.get("sheet_id", request["request_id"]),
            "sheet_revision": int(request.get("sheet_revision", 1)),
            "message": request["message"],
            "created_at_s": request["created_at_s"],
            "immediate_request": copy.deepcopy(
                request.get("immediate_request")),
            "triggered_request": copy.deepcopy(
                request.get("triggered_request")),
            "acceptance_criteria": copy.deepcopy(
                request.get("acceptance_criteria")),
            "judge_trigger": copy.deepcopy(request.get("judge_trigger")),
            "status": record.get("status", "pending"),
            "trigger_status": record.get("trigger_status"),
            "visible_response": copy.deepcopy(
                record.get("visible_response", {})),
            "allowed_actions": (
                ["keep", "cancel"] if int(
                    request.get("sheet_revision", 1)) >= 2
                else ["keep", "update", "cancel"]),
        }

    def _retire_active_sheet(
        self, *, action, time_s, replacement_request_id=None, reason=None,
    ):
        """Archive the active version without treating revision as failure."""
        retired_ids = []
        for request_id, pending in list(self.pending_requests.items()):
            record = pending["record"]
            status = "superseded" if action == "update" else "cancelled"
            close_reason = (
                "passenger_updated" if action == "update"
                else "passenger_cancelled")
            record.update(
                status=status, closed_at_s=float(time_s),
                close_reason=close_reason,
                superseded_by_request_id=replacement_request_id,
                passenger_revision_reason=reason)
            runtime = pending.get("trigger_runtime")
            if runtime and runtime.get("status") == "waiting":
                runtime["status"] = status
                runtime["timeline"].append({
                    "event": close_reason,
                    "time_s": round(float(time_s), 6),
                    "replacement_request_id": replacement_request_id,
                })
            self.judge.state["judgements"].append({
                "request_id": request_id,
                "judged": False,
                "excluded": True,
                "completion_status": status,
                "reason": reason or close_reason,
                "closed_at_s": float(time_s),
                "close_reason": close_reason,
                "superseded_by_request_id": replacement_request_id,
            })
            retired_ids.append(request_id)
            del self.pending_requests[request_id]
        return retired_ids

    @staticmethod
    def _state_dict(state):
        value = state.as_dict() if hasattr(state, "as_dict") else state
        return value if isinstance(value, dict) else {}

    def _new_trigger_runtime(self, trigger, request, state, *, stopped_for_s=0.0):
        if not trigger:
            return None
        now = float(request["created_at_s"])
        raw = self._state_dict(state)
        active = str(raw.get("active_connector_id", "") or "")
        stopped = float(raw.get("current_speed_kmh", 0.0) or 0.0) < 0.5
        credited_stop_s = 0.0
        if (trigger["condition"] == "vehicle_stopped" and stopped
                and type(stopped_for_s) in (int, float)
                and math.isfinite(stopped_for_s)):
            # The offered state trigger was already true at PA wake. Preserve
            # that observed dwell time so a 0.5 s hold can activate on the
            # next engine poll instead of becoming impossible near arrival.
            credited_stop_s = max(0.0, float(stopped_for_s))
        runtime = {
            "spec": copy.deepcopy(trigger),
            "status": "waiting",
            "registered_at_s": now,
            "trigger_deadline_s": now + float(trigger["timeout_s"]),
            "activated_at_s": None,
            "tracked_connector_id": (
                active if trigger["condition"] == "exit_next_intersection"
                else None),
            "stopped_since_s": (
                now - credited_stop_s
                if trigger["condition"] == "vehicle_stopped" and stopped
                else now if trigger["condition"] == "vehicle_resumed_moving" and stopped
                else None),
            "qualified_stop": False,
            "condition_since_s": None,
            "timeline": [{
                "event": "judge_trigger_registered", "time_s": now,
                "condition": trigger["condition"],
            }],
        }
        if active and trigger["condition"] == "exit_next_intersection":
            runtime["timeline"].append({
                "event": "intersection_tracking_started", "time_s": now,
                "connector_id": active,
            })
        return runtime

    def _activate_trigger(self, pending, time_s, event, **details):
        runtime = pending["trigger_runtime"]
        if not runtime or runtime["status"] != "waiting":
            return
        activated = round(float(time_s), 6)
        runtime.update(status="activated", activated_at_s=activated)
        runtime["timeline"].append({
            "event": event, "time_s": activated, **details,
        })
        criteria = pending["request"].get("acceptance_criteria") or {}
        validity = float(criteria.get(
            "valid_for_s", self.acceptance_timeout_s))
        pending["response_origin_s"] = activated
        pending["record"].update(
            trigger_status="activated", trigger_activated_at_s=activated,
            acceptance_deadline_s=activated + validity)
        pending["next_check_at_s"] = (
            activated + pending["check_offsets_s"][0])

    def _poll_judge_triggers(self, time_s, state, events):
        """Advance evaluator-only predicates without waking either agent."""
        raw = self._state_dict(state)
        now = float(time_s)
        active = str(raw.get("active_connector_id", "") or "")
        speed = float(raw.get("current_speed_kmh", 0.0) or 0.0)
        intersection_distance = raw.get("next_intersection_distance_m")
        remaining_distance = raw.get("remaining_distance_m")
        for pending in self.pending_requests.values():
            runtime = pending.get("trigger_runtime")
            if not runtime or runtime["status"] != "waiting":
                continue
            if now > runtime["trigger_deadline_s"] + 1e-9:
                runtime["status"] = "timed_out"
                runtime["timeline"].append({
                    "event": "judge_trigger_timed_out",
                    "time_s": round(runtime["trigger_deadline_s"], 6),
                })
                pending["record"]["trigger_status"] = "timed_out"
                pending["next_check_at_s"] = runtime["trigger_deadline_s"]
                continue
            spec = runtime["spec"]
            condition = spec["condition"]
            if condition == "after_delay":
                target = (runtime["registered_at_s"]
                          + float(spec["after_s"]))
                if now >= target - 1e-9:
                    self._activate_trigger(
                        pending, target, "time_delay_elapsed",
                        requested_after_s=float(spec["after_s"]))
            elif condition == "exit_next_intersection":
                tracked = runtime.get("tracked_connector_id")
                if not tracked and active:
                    runtime["tracked_connector_id"] = active
                    runtime["timeline"].append({
                        "event": "intersection_tracking_started",
                        "time_s": round(now, 6), "connector_id": active,
                    })
                elif tracked and active != tracked:
                    self._activate_trigger(
                        pending, now, "intersection_exited",
                        connector_id=tracked)
            elif condition == "enter_next_intersection":
                if active:
                    self._activate_trigger(
                        pending, now, "intersection_entered",
                        connector_id=active)
            elif condition == "approach_next_intersection":
                if (type(intersection_distance) in (int, float)
                        and intersection_distance <= float(spec["distance_m"])):
                    self._activate_trigger(
                        pending, now, "intersection_approached",
                        distance_m=intersection_distance)
            elif condition == "distance_to_destination_below":
                if (type(remaining_distance) in (int, float)
                        and remaining_distance < float(spec["distance_m"])):
                    self._activate_trigger(
                        pending, now, "destination_distance_reached",
                        remaining_distance_m=remaining_distance)
            elif condition in {"vehicle_stopped", "vehicle_resumed_moving"}:
                if speed < 0.5:
                    if runtime["stopped_since_s"] is None:
                        runtime["stopped_since_s"] = now
                    hold = float(spec.get("hold_for_s", 0.5))
                    if now - runtime["stopped_since_s"] >= hold - 1e-9:
                        runtime["qualified_stop"] = True
                        if condition == "vehicle_stopped":
                            self._activate_trigger(
                                pending, max(
                                    runtime["registered_at_s"],
                                    runtime["stopped_since_s"] + hold),
                                "vehicle_stopped", hold_for_s=hold)
                elif speed > 2.0:
                    if (condition == "vehicle_resumed_moving"
                            and runtime["qualified_stop"]):
                        self._activate_trigger(
                            pending, now, "vehicle_resumed_moving")
                    runtime["stopped_since_s"] = None
                    runtime["qualified_stop"] = False
                elif not runtime["qualified_stop"]:
                    runtime["stopped_since_s"] = None
            elif condition == "speed_threshold_held":
                threshold = float(spec["speed_kmh"])
                matches = (speed > threshold if spec["comparison"] == "above"
                           else speed < threshold)
                if matches:
                    if runtime["condition_since_s"] is None:
                        runtime["condition_since_s"] = now
                    hold = float(spec["hold_for_s"])
                    if now - runtime["condition_since_s"] >= hold - 1e-9:
                        self._activate_trigger(
                            pending, runtime["condition_since_s"] + hold,
                            "speed_threshold_held", speed_kmh=speed)
                else:
                    runtime["condition_since_s"] = None
            elif condition == "weather_changed_to":
                if any(event.get("event_type") == "weather_changed"
                       and (event.get("details") or {}).get("condition")
                       == spec["weather_condition"] for event in events):
                    self._activate_trigger(
                        pending, now, "weather_changed_to",
                        weather_condition=spec["weather_condition"])
            elif condition == "daynight_became_dark":
                if any(event.get("event_type") == "daynight_changed"
                       and (event.get("details") or {}).get("is_dark") is True
                       and (event.get("details") or {}).get("previous_period")
                       not in {"dawn", "dusk", "night"} for event in events):
                    self._activate_trigger(
                        pending, now, "daynight_became_dark")
            if (runtime["status"] == "waiting"
                    and now >= runtime["trigger_deadline_s"] - 1e-9):
                runtime["status"] = "timed_out"
                runtime["timeline"].append({
                    "event": "judge_trigger_timed_out",
                    "time_s": round(runtime["trigger_deadline_s"], 6),
                })
                pending["record"]["trigger_status"] = "timed_out"
                pending["next_check_at_s"] = runtime["trigger_deadline_s"]

    def poll_personal_events(self, time_s, state, events, *, episode_id=""):
        self._bind_request_design_episode(episode_id)
        self._poll_judge_triggers(time_s, state, events)
        return self.scheduler.poll(time_s, state, events, episode_id=episode_id)

    def finalize_passenger(self, vw, t, memory, tick_index, **kwargs):
        return self(vw, t, [], memory, tick_index, **kwargs)

    def _driver_response(self, pending):
        state = self.vehicle_callback._state
        return _compact_driver_response(
            actions=pending["actions"],
            tool_calls=state.get("tool_call_log", [])[pending["tool_start"]:],
            wakes=state.get("all_messages", [])[pending["message_start"]:],
            protocol_events=state.get("protocol_events", [])[pending["protocol_start"]:])

    @staticmethod
    def _arrival_before_trigger_result(request, t):
        result = score_submission({
            "grade": "NA",
            "na_reason": "invalid_request",
            "reason": (
                "The vehicle reached its destination before the frozen "
                "trigger activated, so the delayed phase was no longer "
                "physically executable."),
        })
        return {
            "request_id": request["request_id"],
            "judged": True,
            "requested_at_s": request["created_at_s"],
            "judged_at_s": t,
            **result,
            "completion_status": "uncertain",
            "request_kind": (request.get("acceptance_criteria") or {}).get(
                "request_kind", "one_shot"),
            "completion_reason": (
                "Normal trip completion preceded trigger activation."),
            "score_constraints": ["normal_arrival_before_trigger"],
        }

    @staticmethod
    def _judge_failure_record(pending, evidence, failed):
        """Close an exhausted Judge retry without inventing a vehicle score."""
        reason = (
            "The Judge did not return a valid evidence-backed outcome after "
            "one retry. The frozen input is retained for offline replay.")
        return {
            **copy.deepcopy(failed),
            "request_id": pending["request"]["request_id"],
            "judged": False,
            "requested_at_s": pending["request"]["created_at_s"],
            "judged_at_s": evidence["judged_at_s"],
            "grade": None,
            "applicable": None,
            "dimension_scores_100": {},
            "overall_score_100": None,
            "completion_status": "unverified",
            "request_kind": (
                pending["request"].get("acceptance_criteria") or {}).get(
                    "request_kind", "one_shot"),
            "completion_reason": reason,
            "judge_error": failed.get("error", "invalid_judge_result"),
            "judge_status": "failed",
            "judge_retry_exhausted": True,
            "judge_replay_pending": True,
            "evaluation_status": "judge_failed_unscored",
        }

    def _check_requests(self, observation, t, terminal, arrived, kwargs):
        for request_id, pending in list(self.pending_requests.items()):
            if not terminal and t < pending["next_check_at_s"] - 1e-9:
                continue
            record = pending["record"]
            check_index = record["checks"] + 1
            trigger_runtime = pending.get("trigger_runtime")
            if (trigger_runtime
                    and trigger_runtime.get("status") == "waiting"
                    and t >= trigger_runtime["trigger_deadline_s"] - 1e-9):
                trigger_runtime["status"] = "timed_out"
                trigger_runtime["timeline"].append({
                    "event": "judge_trigger_timed_out",
                    "time_s": round(
                        trigger_runtime["trigger_deadline_s"], 6),
                })
                record["trigger_status"] = "timed_out"
            trigger_timed_out = bool(
                trigger_runtime
                and trigger_runtime.get("status") == "timed_out")
            final_window = (terminal or trigger_timed_out
                            or check_index >= len(pending["check_offsets_s"])
                            or t >= record["acceptance_deadline_s"] - 1e-9)
            evidence = {
                "schema_version": "passenger-judge-evidence-v3",
                "request": pending["request"], "judged_at_s": t,
                "check_index": check_index, "max_checks": self.max_checks,
                "final_window": final_window, "terminal": terminal,
                "acceptance_deadline_s": record["acceptance_deadline_s"],
                "response_origin_s": pending.get(
                    "response_origin_s", pending["request"]["created_at_s"]),
                "response_origins_s": {
                    "immediate": pending["request"]["created_at_s"],
                    "triggered": pending.get(
                        "response_origin_s",
                        pending["request"]["created_at_s"]),
                },
                "judge_trigger": copy.deepcopy(trigger_runtime),
                "evaluation_scope": "observed_window_only",
                "previous_check": pending.get("previous_check"),
                "prior_checks": copy.deepcopy(pending["check_history"]),
                "observation_when_requested": pending["observation"],
                "observation_at_next_wake": observation,
                "vehicle_agent_response": self._driver_response(pending),
                "evaluator_world_when_requested": pending["world"],
                "evaluator_world_at_next_wake": kwargs.get("_passenger_judge_world", {}),
                "physical_execution": {
                    "window_s": round(t - pending["request"]["created_at_s"], 6),
                    "ego_motion_trace": copy.deepcopy(pending["motion"]),
                },
            }
            arrival_before_trigger = bool(
                arrived and trigger_runtime
                and trigger_runtime.get("status") == "waiting")
            if arrival_before_trigger:
                result = self._arrival_before_trigger_result(
                    pending["request"], t)
            else:
                result = self.judge.judge(evidence, finalize=False)
                if final_window and not result.get("judged"):
                    result = self._judge_failure_record(
                        pending, evidence, result)
            record["checks"] = check_index
            status = result.get("completion_status", "uncertain")
            self.judge.state["check_history"].append(copy.deepcopy({
                **result, "request_id": request_id, "check_index": check_index,
                "judged_at_s": t}))
            check_summary = {
                key: copy.deepcopy(result.get(key)) for key in
                ("completion_status", "completion_reason", "reason", "judged",
                 "grade", "fulfilled_at_s", "criterion_statuses", "check_only",
                 "promoted_to_final")}
            pending["previous_check"] = check_summary
            pending["check_history"].append({
                **check_summary, "check_index": check_index,
                "judged_at_s": t,
            })
            if status == "completed" or final_window:
                outcome = ("completed" if status == "completed" else
                           "uncompleted" if status == "pending" else "unverified")
                judge_failed = not result.get("judged")
                record.update(status=outcome, closed_at_s=t,
                    close_reason=("judge_failure" if judge_failed else
                                  "completed" if status == "completed" else
                                  "arrived_before_judge_trigger"
                                  if arrival_before_trigger else
                                  "episode_ended" if terminal else
                                  "judge_trigger_timeout" if trigger_timed_out else
                                  "acceptance_deadline"))
                self.judge.state["judgements"].append(copy.deepcopy({
                    **result, "request_id": request_id, "completion_status": outcome,
                    "evaluation_scope": "observed_window_only",
                    "checks": check_index, "closed_at_s": t,
                    "close_reason": record["close_reason"]}))
                del self.pending_requests[request_id]
            else:
                pending["next_check_at_s"] = (
                    pending.get(
                        "response_origin_s",
                        pending["request"]["created_at_s"])
                    + pending["check_offsets_s"][check_index])

    def __call__(self, vw, t, passenger_messages, memory, tick_index, **kwargs):
        events = kwargs.get("_wake_events", [])
        terminal = _terminal_event({"current_events": events})
        arrived = any(
            str(event.get("event_type", "")) == "arrived"
            for event in events if isinstance(event, dict))
        pa_events = [e for e in events if e.get("event_type") == "personal_agent_due"]
        driver_events = [e for e in events if e.get("event_type") not in
                         {"personal_agent_due", "passenger_judge_due"}]
        observation = build_personal_observation(vw, vehicle_id=self.vehicle_id,
            sim_time_s=t, tick_index=tick_index, vehicle_state=kwargs.get("_vehicle_state"),
            wake_events=[], recent_motion=kwargs.get("_recent_motion", []),
            episode_total_time_s=kwargs.get("_episode_total_time_s"),
            navigation_status=kwargs.get("_navigation_status"),
            trigger_context=kwargs.get("_trigger_context"),
            judge_observation_window_s=max(self.check_offsets_s))
        for pending in self.pending_requests.values():
            last = pending["motion"][-1]["time_s"] if pending["motion"] else -math.inf
            pending["motion"].extend(copy.deepcopy([
                row for row in kwargs.get("_recent_motion", [])
                if row.get("time_s", -1) > last
                and row.get("time_s", -1) >= pending["request"]["created_at_s"]]))
        # Evaluation receives full evidence, but none of its conclusions are
        # injected into PA observations or driver events.
        self._check_requests(observation, t, terminal, arrived, kwargs)
        if terminal:
            self.personal_agent.record_terminal_skip(observation)
            return []
        generated = None
        if pa_events:
            observation["current_events"] = [trigger for e in pa_events
                for trigger in e.get("details", {}).get("triggers", [])]
            observation["recent_requests"] = [{
                "request_id": r["request_id"], "message": r["message"],
                "created_at_s": r["created_at_s"],
                "status": r.get("status"),
                "response": r.get("visible_response", {}),
            } for r in self.request_records[-6:]]
            observation["active_request_sheet"] = self._active_sheet_view()
            registration_context = copy.deepcopy(
                observation["temporal_task_context"])
            observation["request_design"] = self._next_request_design(
                registration_context)
            observation["available_judge_triggers"] = available_judge_triggers(
                registration_context,
                observation["request_design"])
            observation["request_design"]["allowed_trigger_conditions"] = [
                option["condition"] for option in
                observation["available_judge_triggers"]]
            observation["temporal_task_context"].pop(
                "future_weather_conditions", None)
            observation["temporal_task_context"].pop(
                "future_dark_transition", None)
            observation["temporal_task_context"].pop(
                "future_weather_events", None)
            observation["temporal_task_context"].pop(
                "future_dark_after_s", None)
            observation["temporal_task_context"].pop(
                "guaranteed_trigger_conditions", None)
            generated = self.personal_agent.generate(observation)
            decision = copy.deepcopy(
                self.personal_agent.last_sheet_decision)
            sheet_action = decision.get("action", "none")
            previous_request_id = decision.get("active_request_id")
            if sheet_action in {"update", "cancel"}:
                self._retire_active_sheet(
                    action=sheet_action, time_s=t,
                    replacement_request_id=(
                        generated.request_id if generated else None),
                    reason=decision.get("reason"))
            details = {
                "has_request": generated is not None,
                "sheet_action": sheet_action,
                "previous_request_id": previous_request_id,
                "active_request_id": (
                    generated.request_id if generated else
                    previous_request_id if sheet_action == "keep" else None),
            }
            if generated:
                details["request"] = generated.as_dict()
            emit = kwargs.get("_emit_personal_update")
            update = emit(details) if emit else {
                "event_type": "personal_agent_update", "details": details}
            driver_events.append(update)
        if not driver_events:
            return []
        state = self.vehicle_callback._state
        starts = {"tool_start": len(state.get("tool_call_log", [])),
                  "message_start": len(state.get("all_messages", [])),
                  "protocol_start": len(state.get("protocol_events", []))}
        actions = self.vehicle_callback(vw, t,
            [generated.message] if generated else [], memory, tick_index,
            **{
                **kwargs,
                "_wake_events": driver_events,
                "_active_passenger_request_ids": tuple(
                    self.pending_requests),
            }) or []
        for pending in self.pending_requests.values():
            pending["actions"].extend(actions)
        if generated:
            request = generated.as_dict()
            # Validate once at PA tool use and once when the request joins the
            # evaluator.  This makes a stale or fabricated trigger impossible
            # even if the generation/registration path is later refactored.
            if request.get("judge_trigger"):
                validate_available_judge_trigger(
                    request["judge_trigger"],
                    available_judge_triggers(
                        registration_context, observation["request_design"]))
            self._state["personal_requests"].append(request)
            criteria = request.get("acceptance_criteria")
            offsets = list(self.check_offsets_s)
            timeout = self.acceptance_timeout_s
            if criteria:
                timeout = float(criteria["valid_for_s"])
                # Preserve the check budget, but never finalize a request before
                # its own expiry just because the old three-second budget ends.
                offsets = sorted(set(min(x, timeout) for x in offsets[:-1]) | {timeout})
            trigger_runtime = self._new_trigger_runtime(
                request.get("judge_trigger"), request,
                kwargs.get("_vehicle_state"),
                stopped_for_s=registration_context.get("stopped_for_s", 0.0))
            initial_deadline = (
                trigger_runtime["trigger_deadline_s"]
                if trigger_runtime else t + timeout)
            record = {**request, "acceptance_deadline_s": initial_deadline,
                      "evaluation_scope": "observed_window_only",
                      "status": "pending", "checks": 0,
                      "sheet_status": "active",
                      "trigger_status": (
                          "waiting" if trigger_runtime else "immediate")}
            pending = {**starts, "request": request, "record": record,
                "observation": copy.deepcopy(observation),
                "world": copy.deepcopy(kwargs.get("_passenger_judge_world", {})),
                "actions": list(actions), "motion": [], "check_history": [],
                "check_offsets_s": offsets,
                "response_origin_s": t,
                "trigger_runtime": trigger_runtime,
                "next_check_at_s": (
                    trigger_runtime["trigger_deadline_s"]
                    if trigger_runtime else t + offsets[0])}
            self.pending_requests[generated.request_id] = pending
            self.request_records.append(record)
            if len(self.pending_requests) != 1:
                raise RuntimeError(
                    "single active passenger request sheet invariant violated")
        for pending in self.pending_requests.values():
            pending["record"]["visible_response"] = _passenger_visible_driver_response(
                self._driver_response(pending))
        return actions
