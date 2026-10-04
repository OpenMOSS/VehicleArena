"""Regression tests for unified, stateful LLM wake-up delivery."""

from __future__ import annotations

import json
import os
import sys
from types import SimpleNamespace

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from simulation.multi_sim_engine import MultiScenario, MultiSimEngine  # noqa: E402
from simulation.wake_events import (  # noqa: E402
    WakeEventBroker, WakePriority)
from evaluation.experiments.telemetry import build_event_audit  # noqa: E402
from evaluation.multi_agent_runner import (  # noqa: E402
    _events_visible_to_multimodal_driver)


def test_model_selected_heartbeat_is_not_shortened_by_driving_state():
    engine = object.__new__(MultiSimEngine)
    callback = lambda *args, **kwargs: []
    callback._state = {"heartbeat_interval_s": 2.0}

    assert engine._heartbeat_interval_for(
        "ego", 1.0, {"ego": callback}) == 2.0


def test_model_scheduled_one_shot_wake_fires_with_long_heartbeat():
    scenario = MultiScenario.from_dict({
        "scenario_id": "model_scheduled_one_shot", "road_network_id": "beijing_guomao",
        "total_time_s": 1.5,
        "vehicles": [{
            "vehicle_id": "ego", "initial_node": "n33399858",
            "destination_node": "n35722739",
            "agent_config": {"type": "llm", "heartbeat_interval_s": 10.0},
        }],
    })
    wakes = []

    def callback(_vw, time_s, *_args, **kwargs):
        wakes.append((round(time_s, 6), [
            event["event_type"] for event in kwargs["_wake_events"]]))
        if time_s == 0.0:
            callback._state.update(
                scheduled_wake_at_s=1.3,
                scheduled_wake_reason="recheck delayed passenger task")
        return []

    callback._agent_type = "llm"
    callback._state = {
        "heartbeat_interval_s": 10.0,
        "scheduled_wake_at_s": None,
        "scheduled_wake_reason": "",
    }
    MultiSimEngine(scenario).run({"ego": callback})

    assert [time_s for time_s, _ in wakes] == [0.0, 1.3, 1.5]
    assert "scheduled_wake" in wakes[1][1]
    assert callback._state["scheduled_wake_at_s"] is None


def test_model_scheduled_one_shot_is_additive_to_periodic_heartbeat():
    scenario = MultiScenario.from_dict({
        "scenario_id": "additive_model_scheduled_one_shot",
        "road_network_id": "beijing_guomao",
        "total_time_s": 2.1,
        "vehicles": [{
            "vehicle_id": "ego", "initial_node": "n33399858",
            "destination_node": "n35722739",
            "agent_config": {"type": "llm", "heartbeat_interval_s": 1.0},
        }],
    })
    wakes = []

    def callback(_vw, time_s, *_args, **kwargs):
        wakes.append((round(time_s, 6), [
            event["event_type"] for event in kwargs["_wake_events"]]))
        if time_s == 0.0:
            callback._state.update(
                scheduled_wake_at_s=1.3,
                scheduled_wake_reason="precise delayed task check")
        return []

    callback._agent_type = "llm"
    callback._state = {
        "heartbeat_interval_s": 1.0,
        "scheduled_wake_at_s": None,
        "scheduled_wake_reason": "",
    }
    MultiSimEngine(scenario).run({"ego": callback})

    assert [time_s for time_s, _ in wakes] == [
        0.0, 1.0, 1.3, 2.0, 2.1]
    assert "heartbeat" in wakes[1][1]
    assert "scheduled_wake" in wakes[2][1]
    assert "heartbeat" in wakes[3][1]


def test_evaluator_truth_does_not_wake_llm_but_radar_warning_does():
    engine = object.__new__(MultiSimEngine)
    callback = lambda *args, **kwargs: []
    callback._agent_type = "llm"
    callbacks = {"ego": callback}
    broker = WakeEventBroker()
    hidden_truth = broker.transition(
        "vehicle_proximity_risk", "ego", "vehicle", 0.4, "critical",
        dedupe_key="risk:ego:lead", priority=WakePriority.CRITICAL,
        source="ground_truth_evaluator")
    radar_warning = broker.discrete(
        "front_collision_warning", "ego", "vehicle", 0.4,
        priority=WakePriority.CRITICAL, source="front_radar")

    assert hidden_truth in broker.log
    assert engine._event_can_wake_agent(
        hidden_truth, callbacks) is False
    assert engine._event_can_wake_agent(
        radar_warning, callbacks) is True
    decision = broker.discrete(
        "navigation_decision_required", "ego", "vehicle", 0.4,
        source="traffic_manager", details={"reason": "no_unique_straight_continuation"})
    # Even a legacy/injected topology event must not purchase a driver wake.
    assert engine._event_can_wake_agent(decision, callbacks) is False


def test_visible_pedestrian_hazard_wakes_driver_without_leaking_map_truth():
    engine = object.__new__(MultiSimEngine)
    callback = lambda *args, **kwargs: []
    callback._agent_type = "llm"
    callbacks = {"ego": callback}
    broker = WakeEventBroker()

    def hazard(level, *, source="connector_crosswalk_monitor"):
        return broker.transition(
            "pedestrian_crosswalk_conflict", "ego", "vehicle", 2.6,
            level, dedupe_key=f"crosswalk:{level}:{source}",
            priority=WakePriority.CRITICAL, source=source,
            details={"pedestrian_id": "pedestrian", "vehicle_ttc_s": 1.4})

    critical = hazard("critical")
    caution = hazard("caution")
    observed = hazard("observed")
    observed_key = "crosswalk:observed:connector_crosswalk_monitor"
    escalated = broker.transition(
        "pedestrian_crosswalk_conflict", "ego", "vehicle", 2.7,
        "critical", dedupe_key=observed_key,
        source="connector_crosswalk_monitor")
    cleared = broker.transition(
        "pedestrian_crosswalk_conflict", "ego", "vehicle", 2.8,
        "clear", dedupe_key=observed_key,
        source="connector_crosswalk_monitor")
    evaluator_truth = hazard("critical", source="ground_truth_evaluator")

    assert engine._event_can_wake_agent(critical, callbacks)
    assert engine._event_can_wake_agent(caution, callbacks)
    assert not engine._event_can_wake_agent(observed, callbacks)
    assert engine._event_can_wake_agent(escalated, callbacks)
    assert not engine._event_can_wake_agent(cleared, callbacks)
    assert not engine._event_can_wake_agent(evaluator_truth, callbacks)
    # The wake prompts a fresh camera observation, not an exact TTC or ID.
    assert _events_visible_to_multimodal_driver([critical.as_dict()]) == []


def test_event_audit_separates_agent_visible_and_evaluator_only_events():
    broker = WakeEventBroker()
    hidden = broker.discrete(
        "vehicle_proximity_risk", "ego", "vehicle", 0.0,
        source="evaluator")
    visible = broker.discrete(
        "simulation_start", "ego", "vehicle", 0.0,
        source="engine")
    internal = broker.discrete(
        "passenger_judge_due", "ego", "vehicle", 1.0,
        source="passenger_judge_timer")
    broker.record_delivery(
        "ego", [visible], 0.0, status="delivered", actions=[])
    broker.record_delivery(
        "ego", [internal], 1.0, status="delivered", actions=[])

    audit = build_event_audit(broker.log, broker.delivery_log)

    assert [item["event_id"] for item in audit["agent_visible_events"]] == [
        visible.event_id]
    assert audit["agent_visible_events"][0]["delivered_at_s"] == 0.0
    assert [item["event_id"] for item in audit["evaluator_only_events"]] == [
        hidden.event_id, internal.event_id]
    assert audit["summary"] == {
        "agent_visible_event_count": 1,
        "evaluator_only_event_count": 2,
        "delivery_batch_count": 2,
    }


def test_navigation_status_reports_remaining_route_geometry():
    scenario = MultiScenario.from_dict({
        "scenario_id": "navigation_instrument",
        "road_network_id": "beijing_guomao",
        "total_time_s": 0.0,
        "vehicles": [{
            "vehicle_id": "ego",
            "initial_node": "n317995521",
            "destination_node": "n1634509692",
            "initial_lane": 3,
            "initial_physical_state": {
                "lane_id": "n1634509692_n317995521::lane_3",
                "progress": 0.54,
            },
        }],
    })
    engine = MultiSimEngine(scenario)
    engine.run({})
    vehicle = engine.traffic_mgr.vehicles["ego"]
    initial = engine.traffic_mgr.get_navigation_status("ego")
    vehicle.edge_progress = 0.99
    near_end = engine.traffic_mgr.get_navigation_status("ego")

    assert initial["status"] == "active"
    assert initial["remaining_distance_m"] > near_end[
        "remaining_distance_m"] > 0.0
    assert near_end["next_maneuver"] == {
        "type": "arrive",
        "state": "upcoming",
        "distance_m": near_end["remaining_distance_m"],
        "completion": "cross_route_endpoint",
    }


def test_terminal_body_state_is_independent_of_wake_bookkeeping():
    engine = object.__new__(MultiSimEngine)
    engine.traffic_mgr = SimpleNamespace(
        vehicles={
            "arrived": SimpleNamespace(arrived=True, is_crashed=False),
            "active": SimpleNamespace(arrived=False, is_crashed=False),
            "route-failed": SimpleNamespace(
                arrived=False, is_crashed=False, route_failed=True),
        },
        pedestrians={
            "crashed-ped": SimpleNamespace(
                has_arrived=False, is_crashed=True),
        },
    )
    assert engine._entity_is_physically_terminal("arrived") is True
    assert engine._entity_is_physically_terminal("crashed-ped") is True
    assert engine._entity_is_physically_terminal("active") is False
    assert engine._entity_is_physically_terminal("route-failed") is True


def test_vehicle_speed_limit_sensor_reads_authoritative_road_and_is_read_only():
    scenario = MultiScenario.from_dict({
        "scenario_id": "authoritative_speed_limit",
        "road_network_id": "beijing_guomao",
        "total_time_s": 0.0,
        "vehicles": [{
            "vehicle_id": "ego",
            "initial_node": "n317995521",
            "destination_node": "n1634509692",
            "initial_lane": 3,
            "agent_config": {"type": "llm"},
            "initial_physical_state": {
                "lane_id": "n1634509692_n317995521::lane_3",
                "progress": 0.54,
            },
        }],
    })
    engine = MultiSimEngine(scenario)
    observed = {}

    def callback(vw, *args, **kwargs):
        observed["get"] = vw.speedLimit.speed_limit_get()
        observed["set"] = vw.speedLimit.speed_limit_set(5, "school")
        observed["clear"] = vw.speedLimit.speed_limit_clear()
        return []

    engine.run({"ego": callback})

    vehicle = engine.traffic_mgr.vehicles["ego"]
    expected = engine.road_network.get_segment(
        vehicle.current_segment).speed_limit
    assert observed["get"]["current_limit"] == expected
    assert observed["get"]["source"] == "authoritative_road_network"
    assert observed["set"] == {
        "success": False, "reason": "road_speed_limit_is_read_only"}
    assert observed["clear"] == observed["set"]


def test_committed_lane_change_does_not_override_model_heartbeat():
    scenario = MultiScenario.from_dict({
        "scenario_id": "post_commit_heartbeat",
        "road_network_id": "beijing_guomao",
        "total_time_s": 0.6,
        "vehicles": [{
            "vehicle_id": "ego",
            "initial_node": "n317995521",
            "destination_node": "n1634509692",
            "initial_lane": 3,
            "agent_config": {
                "type": "llm", "heartbeat_interval_s": 2.0},
            "initial_physical_state": {
                "lane_id": "n1634509692_n317995521::lane_3",
                "progress": 0.54,
            },
        }],
    })
    engine = MultiSimEngine(scenario)
    wake_times = []

    def callback(vw, time_s, *args, **kwargs):
        wake_times.append(time_s)
        if time_s == 0.0:
            vw.turnSignal.switch("right")
            result = vw.navigation.navigation_change_lane("right")
            assert result["success"] is True
        return []

    engine.run({"ego": callback})

    # The 0.6 s simulation-end event preempts the model's 2 s heartbeat.
    # Starting a lane change must not invent a 0.5 s periodic wake.
    assert wake_times == [0.0, 0.6]


def test_simulation_end_replaces_same_boundary_heartbeat():
    scenario = MultiScenario.from_dict({
        "scenario_id": "terminal_heartbeat_dedup",
        "road_network_id": "beijing_guomao",
        "total_time_s": 1.0,
        "vehicles": [{
            "vehicle_id": "ego",
            "initial_node": "n33399858",
            "destination_node": "n35722739",
            "agent_config": {
                "type": "llm", "heartbeat_interval_s": 1.0},
        }],
    })
    wakes = []

    def callback(_vw, time_s, *_args, **kwargs):
        wakes.append((round(time_s, 6), [
            event["event_type"] for event in kwargs.get("_wake_events", [])]))
        return []

    MultiSimEngine(scenario).run({"ego": callback})
    terminal = next(events for time_s, events in wakes if time_s == 1.0)
    assert "simulation_ended" in terminal
    assert "heartbeat" not in terminal


def wake_payloads(messages):
    return [message for message in messages if isinstance(message, dict)]


def test_deterministic_identity_and_time_semantics():
    brokers = [WakeEventBroker(), WakeEventBroker()]
    emitted = []
    for broker in brokers:
        emitted.append([
            broker.discrete(
                "exact_boundary", "ego", "vehicle", 0.63,
                detected_at_s=0.7),
            broker.transition(
                "risk", "ego", "vehicle", 0.7, "critical",
                dedupe_key="risk:ego"),
        ])
    first_serialized = [event.as_dict(0.8) for event in emitted[0]]
    second_serialized = [event.as_dict(0.8) for event in emitted[1]]
    assert first_serialized == second_serialized
    assert first_serialized[0]["event_id"] == "wake-000000000001"
    assert first_serialized[0]["occurred_at_s"] == 0.63
    assert first_serialized[0]["detected_at_s"] == 0.7
    assert first_serialized[0]["delivered_at_s"] == 0.8


def test_transition_reentry_cooldown_uses_simulation_time():
    broker = WakeEventBroker()
    entered = broker.transition(
        "front_collision_warning", "ego", "vehicle", 1.0, "caution",
        dedupe_key="radar:ego:front", reenter_cooldown_s=1.0)
    cleared = broker.transition(
        "front_collision_warning", "ego", "vehicle", 1.1, "clear",
        dedupe_key="radar:ego:front", reenter_cooldown_s=1.0)
    suppressed = broker.transition(
        "front_collision_warning", "ego", "vehicle", 1.2, "caution",
        dedupe_key="radar:ego:front", reenter_cooldown_s=1.0)
    reentered = broker.transition(
        "front_collision_warning", "ego", "vehicle", 2.1, "caution",
        dedupe_key="radar:ego:front", reenter_cooldown_s=1.0)

    assert entered.state == "entered"
    assert cleared.state == "cleared"
    assert suppressed is None
    assert reentered.state == "entered"


def test_transition_can_bypass_reentry_cooldown_for_critical_risk():
    broker = WakeEventBroker()
    broker.transition(
        "front_collision_warning", "ego", "vehicle", 1.0, "caution",
        dedupe_key="radar:ego:front", reenter_cooldown_s=1.0)
    broker.transition(
        "front_collision_warning", "ego", "vehicle", 1.1, "clear",
        dedupe_key="radar:ego:front", reenter_cooldown_s=1.0)
    critical = broker.transition(
        "front_collision_warning", "ego", "vehicle", 1.2, "critical",
        dedupe_key="radar:ego:front", reenter_cooldown_s=0.0)

    assert critical is not None
    assert critical.state == "entered"


def test_delayed_pedestrian_lifecycle():
    scenario = MultiScenario.from_dict({
        "scenario_id": "delayed_pedestrian_wake",
        "name": "delayed pedestrian wake",
        "road_network_id": "beijing_guomao",
        "total_time_s": 1.0,
        "tick_interval_s": 100.0,
        "weather_keyframes": [
            {"t": 0.0, "condition": "sunny"},
            {"t": 0.01, "condition": "rainy"},
        ],
        "vehicles": [],
        "pedestrians": [{
            "ped_id": "future_ped",
            "initial_node": "n33399858",
            "destination_node": "n35553582",
            "agent_config": {"type": "llm"},
            # Scenario pedestrian times are authored in minutes.
            "start_time": 0.01,
        }],
    })
    calls = []

    def callback(state, t, messages, memory, tick_index, world_state):
        calls.append((t, wake_payloads(messages)))
        return []

    engine = MultiSimEngine(scenario)
    engine.run({"future_ped": callback})
    assert calls
    assert all(time_s >= 0.6 - 1e-9 for time_s, _ in calls)
    start = next(
        payload
        for _, payloads in calls
        for payload in payloads
        if payload["event_type"] == "simulation_start")
    assert start["occurred_at_s"] == 0.6
    assert start["detected_at_s"] == 0.6
    assert start["delivered_at_s"] == 0.6
    end_events = [
        payload
        for _, payloads in calls
        for payload in payloads
        if payload["event_type"] == "simulation_ended"
    ]
    assert len(end_events) == 1


def test_callback_todo_deadline_advances_next_wake():
    scenario = MultiScenario.from_dict({
        "scenario_id": "todo_deadline_wake",
        "name": "todo deadline wake",
        "road_network_id": "beijing_guomao",
        "physics_step_s": 0.1,
        "total_time_s": 0.4,
        "tick_interval_s": 100.0,
        "vehicles": [{
            "vehicle_id": "ego",
            "initial_node": "n33399858",
            "destination_node": "n35722739",
            "agent_config": {
                "type": "llm",
                "heartbeat_interval_s": 100.0,
            },
        }],
    })
    calls = []

    def callback(vw, t, messages, memory, tick_index, **kwargs):
        calls.append(round(t, 6))
        callback._state["next_todo_deadline_s"] = (
            0.2 if t < 0.2 - 1e-9 else None)
        return []

    callback._state = {"next_todo_deadline_s": None}
    MultiSimEngine(scenario).run({"ego": callback})
    assert 0.2 in calls


def test_signal_and_horn_modules_publish_at_batch_boundary():
    scenario = MultiScenario.from_dict({
        "scenario_id": "signal_horn_boundary",
        "road_network_id": "beijing_guomao",
        "total_time_s": 0.1,
        "tick_interval_s": 100.0,
        "vehicles": [{
            "vehicle_id": "ego",
            "initial_node": "n33399858",
            "destination_node": "n35722739",
            "agent_config": {"type": "llm"},
        }],
    })
    called = False

    def callback(vw, t, messages, memory, tick_index, **kwargs):
        nonlocal called
        if not called:
            called = True
            vw.turnSignal.switch("left")
            vw.horn.honk(duration_s=0.2, intensity="urgent")
        return []

    result = MultiSimEngine(scenario).run({"ego": callback})
    assert any(
        event["state"]["left_indicator"]
        for event in result.signal_events
        if event["vehicle_id"] == "ego")
    assert len(result.horn_events) == 1
    assert result.horn_events[0]["intensity"] == "urgent"
    communication = result.vehicle_results[
        "ego"].driving_evaluation
    assert "communication" in communication["dimension_scores"]
    assert communication["metrics"]["horn_events"] == 1


def test_callback_and_decision_batch_isolation():
    scenario = MultiScenario.from_dict({
        "scenario_id": "callback_batch_isolation",
        "name": "callback batch isolation",
        "road_network_id": "beijing_guomao",
        "total_time_s": 0.1,
        "tick_interval_s": 100.0,
        "vehicles": [
            {
                "vehicle_id": "first",
                "initial_node": "n33399858",
                "destination_node": "n35722739",
                "agent_config": {"type": "llm"},
            },
            {
                "vehicle_id": "second",
                "initial_node": "n35553582",
                "destination_node": "n35722739",
                "agent_config": {"type": "llm"},
            },
        ],
    })
    observations = {
        "second_saw_first_reason": None,
        "state_is_immutable": False,
        "world_is_narrow": False,
    }
    engine = MultiSimEngine(scenario)

    def first_callback(
        vw, t, messages, memory, tick_index, **kwargs,
    ):
        if t == 0.0:
            state = kwargs["_vehicle_state"]
            try:
                state.current_speed_kmh = 999.0
            except AttributeError:
                observations["state_is_immutable"] = True
            observations["world_is_narrow"] = not hasattr(
                kwargs["world_state"], "_tm")
            vw.navigation.navigation_set_speed(
                35.0, reason="must_be_rolled_back")
            raise RuntimeError("intentional callback failure")
        return []

    def second_callback(
        vw, t, messages, memory, tick_index, **kwargs,
    ):
        if t == 0.0:
            observations["second_saw_first_reason"] = (
                engine.traffic_mgr.get_state("first")
                .llm_control_command.get("reason", ""))
            vw.navigation.navigation_set_speed(
                25.0, reason="second_committed")
        return []

    engine.run({
        "first": first_callback,
        "second": second_callback,
    })
    assert observations["state_is_immutable"]
    assert observations["world_is_narrow"]
    assert observations["second_saw_first_reason"] != (
        "must_be_rolled_back")
    assert engine.traffic_mgr.get_state(
        "first").llm_control_command.get("reason") != (
            "must_be_rolled_back")
    assert engine.traffic_mgr.get_state(
        "second").llm_control_command.get("reason") == (
            "second_committed")
    assert any(
        item["entity_id"] == "first"
        and item["phase"] == "callback"
        for item in engine.agent_callback_errors)
    assert any(
        item["entity_id"] == "second"
        and item["status"] == "delivered"
        for item in engine.wake_broker.delivery_log)


def test_successful_command_receipt_piggybacks_on_next_natural_wake():
    scenario = MultiScenario.from_dict({
        "scenario_id": "command_receipt",
        "name": "command receipt",
        "road_network_id": "beijing_guomao",
        "total_time_s": 0.2,
        "tick_interval_s": 100.0,
        "vehicles": [{
            "vehicle_id": "ego",
            "initial_node": "n33399858",
            "destination_node": "n35722739",
            "agent_config": {"type": "llm"},
        }],
    })
    observed = {"ack": None, "receipts": []}

    def callback(vw, t, messages, memory, tick_index, **kwargs):
        observed["receipts"].extend(
            payload for payload in kwargs.get("_wake_events", [])
            if payload["event_type"] == "command_result")
        if t == 0.0:
            observed["ack"] = vw.navigation.navigation_set_speed(
                20.0, reason="receipt_test")
        return []

    engine = MultiSimEngine(scenario)
    engine.run({"ego": callback})
    assert observed["ack"]["queued_for_world_commit"] is True
    command_id = observed["ack"]["command_id"]
    receipt = next(
        item for item in observed["receipts"]
        if item["details"]["command_id"] == command_id)
    assert receipt["occurred_at_s"] == 0.0
    # No callback is created solely for the successful receipt. In this short
    # scenario it is attached to the simulation-ended wake at t=0.2.
    assert receipt["detected_at_s"] == 0.2
    assert receipt["details"]["status"] == "committed"


def test_numeric_string_speed_arguments_are_normalized_before_queueing():
    scenario = MultiScenario.from_dict({
        "scenario_id": "numeric_string_speed",
        "road_network_id": "beijing_guomao",
        "total_time_s": 0.1,
        "vehicles": [{
            "vehicle_id": "ego",
            "initial_node": "n33399858",
            "destination_node": "n35722739",
            "agent_config": {"type": "llm"},
        }],
    })
    observed = {}

    def callback(vw, t, messages, memory, tick_index, **kwargs):
        if "ack" in observed:
            return []
        observed["nav_before"] = vw.navigation._desired_speed
        observed["ack"] = vw.navigation.navigation_set_speed(
            "18", reason="provider encoded numbers as strings",
            acceleration_mps2="1.5", deceleration_mps2="2.5")
        # A queued action must not mutate even the cabin-side command mirror
        # before the frozen world batch commits.
        observed["nav_after_call"] = vw.navigation._desired_speed
        return []

    engine = MultiSimEngine(scenario)
    engine.run({"ego": callback})

    assert observed["ack"]["success"] is True
    assert observed["ack"]["queued_for_world_commit"] is True
    assert observed["nav_after_call"] == observed["nav_before"]
    vehicle = engine.traffic_mgr.get_state("ego")
    assert vehicle.desired_speed_kmh == 18.0
    assert vehicle.control_acceleration_limit_mps2 == 1.5
    assert vehicle.control_deceleration_limit_mps2 == 2.5
    assert engine._vw["ego"].navigation._desired_speed == 18.0
    assert not engine.agent_callback_errors


@pytest.mark.parametrize("arguments", [
    {"speed_kmh": "fast"},
    {"speed_kmh": "nan"},
    {"speed_kmh": "18", "acceleration_mps2": "fast"},
    {"speed_kmh": "18", "deceleration_mps2": "0"},
])
def test_invalid_speed_arguments_have_no_side_effects(arguments):
    scenario = MultiScenario.from_dict({
        "scenario_id": "invalid_speed_is_atomic",
        "road_network_id": "beijing_guomao",
        "total_time_s": 0.0,
        "vehicles": [{
            "vehicle_id": "ego",
            "initial_node": "n33399858",
            "destination_node": "n35722739",
            "agent_config": {"type": "llm"},
        }],
    })
    observed = {}
    engine = MultiSimEngine(scenario)

    def callback(vw, t, messages, memory, tick_index, **kwargs):
        vehicle = engine.traffic_mgr.get_state("ego")
        observed["before"] = (
            vehicle.desired_speed_kmh,
            vehicle.control_acceleration_limit_mps2,
            vehicle.control_deceleration_limit_mps2,
            dict(vehicle.llm_control_command),
            vw.navigation._desired_speed,
        )
        observed["ack"] = vw.navigation.navigation_set_speed(**arguments)
        observed["after_call"] = (
            vehicle.desired_speed_kmh,
            vehicle.control_acceleration_limit_mps2,
            vehicle.control_deceleration_limit_mps2,
            dict(vehicle.llm_control_command),
            vw.navigation._desired_speed,
        )
        return []

    engine.run({"ego": callback})

    assert observed["ack"]["success"] is False
    assert observed["ack"]["reason"] == "invalid_control_argument"
    assert "command_id" not in observed["ack"]
    assert observed["after_call"] == observed["before"]
    vehicle = engine.traffic_mgr.get_state("ego")
    assert (
        vehicle.desired_speed_kmh,
        vehicle.control_acceleration_limit_mps2,
        vehicle.control_deceleration_limit_mps2,
        dict(vehicle.llm_control_command),
        engine._vw["ego"].navigation._desired_speed,
    ) == observed["before"]
    assert not engine._pending_command_receipts


def test_active_driving_task_uses_one_second_visual_heartbeat():
    scenario = MultiScenario.from_dict({
        "scenario_id": "sparse_stopped_heartbeat",
        "road_network_id": "beijing_guomao",
        "total_time_s": 6.0,
        "tick_interval_s": 2.0,
        "vehicles": [{
            "vehicle_id": "ego",
            "initial_node": "n33399858",
            "destination_node": "n35722739",
            "agent_config": {"type": "llm"},
        }],
    })
    calls = []

    def callback(vw, t, messages, memory, tick_index, **kwargs):
        calls.append(t)
        return []

    engine = MultiSimEngine(scenario)
    engine.run({"ego": callback})
    assert calls == [float(value) for value in range(7)]


def test_navigation_motion_commands_are_queued_but_route_preview_is_not():
    scenario = MultiScenario.from_dict({
        "scenario_id": "navigation_command_identity",
        "road_network_id": "beijing_guomao",
        "total_time_s": 0.0,
        "vehicles": [{
            "vehicle_id": "ego",
            "initial_node": "n345589058",
            "destination_node": "n10676277830",
            "initial_lane": 3,
            "agent_config": {"type": "llm"},
            "initial_physical_state": {
                "lane_id": "n1317216454_n345589058::lane_3",
                "progress": 0.5,
            },
        }],
    })
    acknowledgements = {}

    def callback(vw, t, messages, memory, tick_index, **kwargs):
        acknowledgements["speed"] = (
            vw.navigation.navigation_set_speed(15.0))
        acknowledgements["lane"] = (
            vw.navigation.navigation_change_lane("right"))
        acknowledgements["route"] = (
            vw.navigation.navigation_route_plan("n10676277830"))
        return []

    engine = MultiSimEngine(scenario)
    engine.run({"ego": callback})
    for name in ("speed", "lane"):
        acknowledgement = acknowledgements[name]
        assert acknowledgement.get("queued_for_world_commit") is True, \
            acknowledgements
        assert acknowledgement["command_id"].startswith("command-")
        assert acknowledgement["control_slots"]
    assert acknowledgements["route"] == {
        "success": True,
        "minimap_available": True,
        "physical_route_changed": False,
    }
    vehicle = engine.traffic_mgr.vehicles["ego"]
    assert vehicle.destination_node == "n10676277830"
    assert vehicle.route_control_authority == "llm_maneuver"
    assert vehicle.suggested_lane_route_actions
    assert len(vehicle.lane_route_actions) == 1
    assert vehicle.planned_maneuver_source == "default_straight"
    assert vehicle.planned_turn == "straight"


def test_explicit_maneuver_command_selects_exactly_one_connector():
    scenario = MultiScenario.from_dict({
        "scenario_id": "explicit_junction_maneuver",
        "road_network_id": "beijing_guomao",
        "total_time_s": 0.1,
        "stop_when_all_vehicles_terminal": False,
        "vehicles": [{
            "vehicle_id": "ego",
            "initial_node": "n345589058",
            "destination_node": "n10676277830",
            "initial_lane": 3,
            "agent_config": {"type": "llm"},
            "initial_physical_state": {
                "lane_id": "n1317216454_n345589058::lane_3",
                "progress": 0.5,
            },
        }],
    })
    acknowledgement = {}

    def callback(vw, t, messages, memory, tick_index, **kwargs):
        acknowledgement.update(
            vw.navigation.navigation_select_maneuver("straight"))
        return []

    engine = MultiSimEngine(scenario)
    engine.run({"ego": callback})
    assert acknowledgement["queued_for_world_commit"] is True
    assert acknowledgement["control_slots"] == ["route_maneuver"]
    vehicle = engine.traffic_mgr.vehicles["ego"]
    assert vehicle.planned_turn == "straight"
    assert len(vehicle.lane_route_actions) == 1
    assert vehicle.lane_route_actions[0]["connector_id"] == \
        vehicle.planned_connector_id


def test_later_command_wins_each_control_slot_in_one_wake():
    engine = MultiSimEngine(MultiScenario.from_dict({
        "scenario_id": "command_slot_resolution",
        "road_network_id": "beijing_guomao",
        "vehicles": [],
    }))
    applied = []
    engine._collect_agent_commands = True

    def submit(name, slots):
        return engine._submit_agent_command(
            "ego", name,
            lambda: (applied.append(name) or {"success": True}),
            control_slots=slots,
        )

    acknowledgements = {
        "left": submit("left", ["lateral"]),
        "right": submit("right", ["lateral"]),
        "speed": submit("speed", ["longitudinal"]),
        "cruise_keep": submit(
            "cruise_keep", ["longitudinal", "lateral"]),
        "route_query": submit("route_query", []),
    }
    engine._commit_agent_commands(1.0)

    assert applied == ["cruise_keep", "route_query"]
    receipts = {
        item["command_id"]: item
        for item in engine._pending_command_receipts
    }
    for name in ("left", "right", "speed"):
        assert receipts[acknowledgements[name]["command_id"]][
            "status"] == "superseded"
    for name in ("cruise_keep", "route_query"):
        assert receipts[acknowledgements[name]["command_id"]][
            "status"] == "committed"


def test_old_authored_passenger_request_fields_are_rejected():
    for field in ("passenger_requests", "experiment_event_requests"):
        payload = {
            "scenario_id": "old_passenger_protocol",
            "road_network_id": "beijing_guomao",
            "vehicles": [{
                "vehicle_id": "ego",
                "initial_node": "n33399858",
                "agent_config": {"type": "llm"},
            }],
            field: {},
        }
        try:
            MultiScenario.from_dict(payload)
        except ValueError as exc:
            assert "Unknown scenario fields" in str(exc)
        else:
            raise AssertionError(f"legacy field {field} was accepted")


def main():
    test_deterministic_identity_and_time_semantics()
    test_delayed_pedestrian_lifecycle()
    test_callback_and_decision_batch_isolation()
    test_successful_command_receipt_piggybacks_on_next_natural_wake()
    test_later_command_wins_each_control_slot_in_one_wake()
    test_old_authored_passenger_request_fields_are_rejected()

    broker = WakeEventBroker()
    first = broker.transition(
        "vehicle_proximity_risk", "ego", "vehicle", 1.0, "observed",
        dedupe_key="proximity:ego:lead")
    duplicate = broker.transition(
        "vehicle_proximity_risk", "ego", "vehicle", 1.1, "observed",
        dedupe_key="proximity:ego:lead")
    escalated = broker.transition(
        "vehicle_proximity_risk", "ego", "vehicle", 1.2, "critical",
        dedupe_key="proximity:ego:lead",
        priority=WakePriority.CRITICAL)
    cleared = broker.transition(
        "vehicle_proximity_risk", "ego", "vehicle", 2.0, "clear",
        dedupe_key="proximity:ego:lead")
    assert first and first.state == "entered"
    assert duplicate is None
    assert escalated and escalated.state == "updated"
    assert cleared and cleared.state == "cleared"

    scenario = MultiScenario.from_dict({
        "scenario_id": "wake_event_integration",
        "name": "wake event integration",
        "road_network_id": "beijing_guomao",
        "total_time_s": 1.0,
        "tick_interval_s": 100.0,
        "vehicles": [{
            "vehicle_id": "ego",
            "initial_node": "n33399858",
            "destination_node": "n35722739",
            "agent_config": {"type": "llm"},
        }],
        "pedestrians": [{
            "ped_id": "ped_llm",
            "initial_node": "n33399858",
            "destination_node": "n35553582",
            "agent_config": {"type": "llm"},
            "start_time": 0.0,
        }],
    })
    calls = {"ego": [], "ped_llm": []}

    def vehicle_callback(vw, t, messages, memory, tick_index, **kwargs):
        calls["ego"].append((
            t, list(messages), list(kwargs.get("_wake_events", []))))
        return []

    def pedestrian_callback(
        state, t, messages, memory, tick_index, world_state,
    ):
        calls["ped_llm"].append((t, list(messages)))
        return []

    engine = MultiSimEngine(scenario)
    engine.run({
        "ego": vehicle_callback,
        "ped_llm": pedestrian_callback,
    })

    assert calls["ego"] and calls["ego"][0][0] == 0.0
    assert any(
        event["event_type"] == "simulation_start"
        for event in calls["ego"][0][2])
    weather_calls = [
        time_s for time_s, _, events in calls["ego"]
        if any(event["event_type"] == "weather_changed"
               for event in events)]
    # The 0.6 s world event must not wait for the 100 s heartbeat.
    assert weather_calls == [0.6]
    assert calls["ped_llm"] and calls["ped_llm"][0][0] == 0.0
    assert any(
        event["event_type"] == "simulation_start"
        for event in calls["ped_llm"][0][1])
    assert all(
        delivery["event_ids"]
        for delivery in engine.wake_broker.delivery_log)

    print(json.dumps({
        "status": "PASS",
        "dedupe": {
            "entered": first.state,
            "unchanged_suppressed": duplicate is None,
            "escalated": escalated.state,
            "cleared": cleared.state,
        },
        "vehicle_first_wake_s": calls["ego"][0][0],
        "passenger_wake_s": passenger_calls[0][0],
        "pedestrian_first_wake_s": calls["ped_llm"][0][0],
        "broker_events": len(engine.wake_broker.log),
        "deterministic_event_ids": True,
        "delayed_pedestrian_lifecycle": True,
        "callback_batch_isolation": True,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
