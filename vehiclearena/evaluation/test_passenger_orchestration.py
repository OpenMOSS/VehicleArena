"""Offline tests: no real model endpoint is called."""
import copy
import json
from types import SimpleNamespace

import pytest

from evaluation.passenger_orchestration import (
    EventDrivenPassengerCallback, PassengerWakeScheduler,
)
from personal_agent import PersonalAgentRuntime, PassengerJudgeRuntime
from vehiclearena import VehicleWorld


def response(name, arguments):
    if name == "send_passenger_request":
        legacy = arguments
        phase = arguments.get("immediate_request") or {}
        criteria = arguments.get("acceptance_criteria") or {}
        message = str(arguments.get("message") or phase.get("message") or "")
        immediate = {
            "core": list(criteria.get("core") or phase.get("core") or [message]),
            "secondary": list(criteria.get("secondary") or phase.get("secondary") or []),
        }
        arguments = {
            "message": message,
            "immediate_request": immediate,
            "request_kind": criteria.get("request_kind", arguments.get("request_kind", "one_shot")),
            "expected_response_s": criteria.get("expected_response_s", arguments.get("expected_response_s", 1.0)),
            "valid_for_s": criteria.get("valid_for_s", arguments.get("valid_for_s", 5.0)),
        }
        trigger = legacy.get("judge_trigger")
        if trigger is not None:
            delayed = list(criteria.get("core") or [
                str(legacy.get("triggered_request") or message)])
            arguments["immediate_request"] = {
                "core": ["Prepare the requested follow-up."], "secondary": []}
            arguments["triggered_request"] = {
                "core": delayed, "secondary": []}
            arguments["judge_trigger"] = trigger
            name = "send_triggered_request"
        else:
            name = "send_immediate_request"
    return SimpleNamespace(content=None, tool_calls=[SimpleNamespace(
        id="test-call", function=SimpleNamespace(name=name, arguments=json.dumps(arguments)))])


class Client:
    model = "test-qwen"

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []
        self.tool_schemas = []
        self.last_call_metadata = {}

    def chat_with_tools(self, messages, tools):
        self.calls.append(copy.deepcopy(messages))
        self.tool_schemas.append(copy.deepcopy(tools))
        self.last_call_metadata = {"ok": True, "model": self.model}
        return self.replies.pop(0), 12, 10, 2


class _SchedulingTestPersonalAgent(PersonalAgentRuntime):
    """Keep scheduler tests focused on timing rather than request difficulty."""
    def generate(self, observation, passenger_messages=()):
        observation = copy.deepcopy(observation)
        observation.pop("request_design", None)
        return super().generate(observation, passenger_messages)


def judgement(status, kind="one_shot"):
    return response("submit_passenger_judgement", {
        "grade": "A" if status == "completed" else "NA" if status == "uncertain" else "D",
        "na_reason": "insufficient_evidence", "reason": "test evidence",
        "core_statuses": (["met"] * 6 if status == "completed"
                          else ["unmet"] * 6),
        "secondary_statuses": [],
        "completion_status": status, "request_kind": kind,
        "completion_reason": "Current cabin state and command receipt inspected.",
    })


def make_callback(pa_replies, judge_replies, **config):
    # Long-window cases below explicitly exercise configurable intervals.
    config.setdefault("judge_window_s", 2.0)
    pa = Client(pa_replies)
    judge = Client(judge_replies)
    driver_calls = []

    def driver(_vw, t, messages, _memory, _tick, **kwargs):
        driver_calls.append((t, list(messages), kwargs["_wake_events"]))
        driver._state["all_messages"].append({"messages": [
            {"role": "assistant", "content": "I will handle the request."}]})
        driver._state["tool_call_log"].append({"time_s": t,
            "function": "airConditioner__temperature_set", "kind": "action",
            "arguments": {"temperature": 24}, "result": {"success": True}})
        return ["airConditioner__temperature_set"]

    driver._state = {"all_messages": [], "tool_call_log": [], "protocol_events": [],
                     "heartbeat_interval_s": 100.0}
    cb = EventDrivenPassengerCallback("ego", driver, _SchedulingTestPersonalAgent("ego", pa),
        PassengerJudgeRuntime("ego", judge), **config)
    return cb, pa, judge, driver_calls


def wake(cb, vw, time_s, *event_types):
    return cb(vw, time_s, [], None, int(time_s * 10),
        _wake_events=[{"event_type": kind, "details": {"triggers": [
            {"event_type": "simulation_start"}]}} for kind in event_types],
        _vehicle_state={"current_speed_kmh": 20},
        _recent_motion=[{"time_s": time_s, "speed_kmh": 20}],
        _passenger_judge_world={"private_judge_marker": True})


def test_event_and_random_clocks_do_not_follow_heartbeat():
    def run(heartbeat):
        s = PassengerWakeScheduler("ego", random_min_s=2, random_max_s=3,
                                   cooldown_s=0.5, stopped_after_s=100)
        for tick in range(101):
            t = tick / 10
            events = [{"event_type": "simulation_start"}] if tick == 0 else []
            if tick % heartbeat == 0:
                events.append({"event_type": "heartbeat"})
            s.poll(t, {"current_speed_kmh": 20}, events, episode_id="test")
        return s.log
    assert run(1) == run(20)
    assert len(run(1)) >= 4


def test_scheduler_cooldown_merge_rearm_and_payload_filter():
    s = PassengerWakeScheduler("ego", random_min_s=100, random_max_s=100,
        cooldown_s=2, stopped_after_s=3, hard_brake_duration_s=0.2)
    r = s.poll(0, {"current_speed_kmh": 20}, [{"event_type": "simulation_start"}])
    assert len(r["triggers"]) == 1
    assert s.poll(0.1, {"current_speed_kmh": 20, "acceleration_mps2": -4}, []) is None
    assert s.poll(0.3, {"current_speed_kmh": 20, "acceleration_mps2": -4}, []) is None
    r = s.poll(2, {"current_speed_kmh": 0}, [{
        "event_type": "weather_changed", "details": {"hidden_truth": 123}}])
    assert {t["event_type"] for t in r["triggers"]} == {"weather_changed", "passenger_hard_brake"}
    assert "hidden_truth" not in str(r)
    assert s.poll(5, {"current_speed_kmh": 0}, [])["triggers"] == [
        {"event_type": "passenger_long_stop"}]
    assert s.poll(10, {"current_speed_kmh": 0}, []) is None
    s.poll(11, {"current_speed_kmh": 10}, [])
    s.poll(12, {"current_speed_kmh": 0}, [])
    assert s.poll(15, {"current_speed_kmh": 0}, []) is not None


def test_random_streams_are_reproducible_and_vehicle_local():
    a, b, c = (PassengerWakeScheduler(name) for name in ("ego", "ego", "peer"))
    for s in (a, b, c):
        s.poll(0, {"current_speed_kmh": 20}, [], episode_id="scene")
    assert a.next_random_at_s == b.next_random_at_s
    assert a.next_random_at_s != c.next_random_at_s


def test_request_design_cycle_contains_only_enforced_medium_and_hard_work():
    cb, _, _, _ = make_callback([], [])
    designs = [cb._next_request_design() for _ in range(10)]
    assert {level: sum(item["difficulty"] == level for item in designs)
            for level in ("easy", "medium", "hard")} == {
                "easy": 0, "medium": 5, "hard": 5}
    assert all(item["min_explicit_outcomes"] >= 4
               for item in designs if item["difficulty"] == "hard")
    assert all(item["min_core_outcomes"] >= 2 for item in designs)
    assert sum("after_delay" in item["allowed_trigger_conditions"]
               for item in designs) == 1
    assert all(
        item.get("phase_structure") == "immediate_and_triggered"
        for item in designs if item["judge_trigger"] == "required")


def test_first_request_design_is_seeded_by_scenario_not_only_vehicle_id():
    callbacks = [make_callback([], [])[0] for _ in range(8)]
    for index, callback in enumerate(callbacks):
        callback._bind_request_design_episode(f"scenario-{index}")
    offsets = {callback.request_design_offset for callback in callbacks}
    assert len(offsets) >= 3
    assert all(callback.request_design_episode_id.startswith("scenario-")
               for callback in callbacks)


def test_required_trigger_template_is_skipped_when_trip_has_no_judge_horizon():
    cb, _, _, _ = make_callback([], [])
    cb.request_design_offset = 1  # a required-trigger template
    short_trip = {
        "remaining_episode_s": 2.0,
        "estimated_free_flow_remaining_s": 2.0,
        "estimated_free_flow_speed_mps": 10.0,
        "judge_observation_window_s": 3.0,
        "remaining_distance_m": 20.0,
        "intersection_ahead_known": False,
        "guaranteed_trigger_conditions": {},
    }
    design = cb._next_request_design(short_trip)
    assert design["judge_trigger"] == "forbidden"
    assert cb.request_design_counter > 1


def test_registered_stopped_trigger_credits_observed_stop_duration():
    cb, _, _, _ = make_callback([], [], judge_window_s=None)
    request = {"created_at_s": 10.0, "acceptance_criteria": {
        "core": ["Requested state is reached"], "secondary": [],
        "request_kind": "one_shot", "expected_response_s": 1,
        "valid_for_s": 3}}
    trigger = {"trigger_type": "vehicle_state", "condition": "vehicle_stopped",
               "hold_for_s": 0.5, "timeout_s": 5.0}
    runtime = cb._new_trigger_runtime(
        trigger, request, {"current_speed_kmh": 0.0}, stopped_for_s=1.0)
    pending = {"trigger_runtime": runtime, "request": request, "record": {},
               "check_offsets_s": [0.1], "next_check_at_s": 15.0}
    cb.pending_requests["test"] = pending
    cb.poll_personal_events(10.0, {"current_speed_kmh": 0.0}, [],
                            episode_id="test")
    assert runtime["status"] == "activated"
    assert runtime["activated_at_s"] == pytest.approx(10.0)
    assert pending["response_origin_s"] == pytest.approx(10.0)


def test_map_judge_trigger_waits_for_physical_intersection_exit():
    criteria = {
        "core": ["Music is playing after the intersection exit."],
        "secondary": [], "request_kind": "one_shot",
        "expected_response_s": 1.0, "valid_for_s": 3.0,
    }
    trigger = {
        "trigger_type": "map", "condition": "exit_next_intersection",
        "timeout_s": 10.0,
    }
    cb, pa, judge, drivers = make_callback([
        response("send_passenger_request", {
            "message": "Please handle the music for me.",
            "triggered_request": "play music",
            "acceptance_criteria": criteria, "judge_trigger": trigger,
            })], [], judge_window_s=None)
    cb.request_design_offset = 1
    vw = VehicleWorld()
    cb(vw, 0.0, [], None, 0,
       _wake_events=[{"event_type": "personal_agent_due"}],
       _vehicle_state={"current_speed_kmh": 20,
                           "active_connector_id": "connector-a"},
       _recent_motion=[], _passenger_judge_world={},
       _episode_total_time_s=30.0,
       _navigation_status={"remaining_distance_m": 100.0},
       _trigger_context={"next_intersection_exit_distance_m": 10.0})

    assert cb.pending_judge_due_at_s == 10.0
    cb.poll_personal_events(0.5, {
        "current_speed_kmh": 20, "active_connector_id": "connector-a",
    }, [], episode_id="test")
    assert cb.pending_judge_due_at_s == 10.0
    cb.poll_personal_events(1.0, {
        "current_speed_kmh": 20, "active_connector_id": "",
    }, [], episode_id="test")

    pending = next(iter(cb.pending_requests.values()))
    assert cb.pending_judge_due_at_s == pytest.approx(1.1)
    assert pending["response_origin_s"] == 1.0
    assert pending["record"]["acceptance_deadline_s"] == 4.0
    assert pending["trigger_runtime"]["timeline"][-1] == {
        "event": "intersection_exited", "time_s": 1.0,
        "connector_id": "connector-a",
    }
    assert len(pa.calls) == len(drivers) == 1
    assert not judge.calls


def test_time_and_stopped_judge_triggers_activate_on_physics_poll():
    criteria = {
        "core": ["Requested state is reached."], "secondary": [],
        "request_kind": "one_shot", "expected_response_s": 1.0,
        "valid_for_s": 2.0,
    }
    replies = [
        response("send_passenger_request", {
            "message": "The music is too loud.",
            "triggered_request": "lower the volume",
            "acceptance_criteria": criteria,
            "judge_trigger": {"trigger_type": "time",
                "condition": "after_delay", "after_s": 2.0,
                "timeout_s": 5.0},
        }),
        response("send_passenger_request", {
            "message": "The cabin is chilly.",
            "triggered_request": "set the temperature to 24 degrees",
            "acceptance_criteria": criteria,
            "judge_trigger": {"trigger_type": "vehicle_state",
                "condition": "vehicle_stopped", "hold_for_s": 0.5,
                "timeout_s": 5.0},
        }),
    ]
    cb, _, _, _ = make_callback(replies, [], judge_window_s=None,
        schedule_config={"cooldown_s": 0.1, "random_min_s": 100,
                         "random_max_s": 100, "stopped_after_s": 100})
    # Exercise the matching delayed and stopped templates, rather than
    # relying on the seeded production template order.
    cb.request_design_offset = 7
    vw = VehicleWorld()
    wake(cb, vw, 0, "personal_agent_due")
    cb.poll_personal_events(2.0, {"current_speed_kmh": 20}, [],
                            episode_id="test")
    first = cb.pending_requests["pa-ego-0001"]
    assert first["response_origin_s"] == 2.0
    assert first["next_check_at_s"] == pytest.approx(2.1)

    cb(vw, 2.1, [], None, 21,
       _wake_events=[{"event_type": "personal_agent_due", "details": {
           "triggers": [{"event_type": "passenger_long_stop"}]}}],
       _vehicle_state={"current_speed_kmh": 0},
       _recent_motion=[{"time_s": 1.5, "speed_kmh": 0},
                       {"time_s": 2.1, "speed_kmh": 0}],
       _passenger_judge_world={"private_judge_marker": True})
    cb.poll_personal_events(2.2, {"current_speed_kmh": 0}, [],
                            episode_id="test")
    cb.poll_personal_events(2.7, {"current_speed_kmh": 0}, [],
                            episode_id="test")
    second = cb.pending_requests["pa-ego-0002"]
    assert second["response_origin_s"] == pytest.approx(2.1)
    assert second["next_check_at_s"] == pytest.approx(2.2)


@pytest.mark.parametrize('trigger,observations,activated_at', [
    ({'trigger_type': 'map', 'condition': 'approach_next_intersection',
      'distance_m': 50, 'timeout_s': 5},
     [(1, {'current_speed_kmh': 20, 'next_intersection_distance_m': 60}, []),
      (2, {'current_speed_kmh': 20, 'next_intersection_distance_m': 48}, [])], 2),
    ({'trigger_type': 'map', 'condition': 'enter_next_intersection',
      'timeout_s': 5},
     [(1, {'current_speed_kmh': 20, 'active_connector_id': ''}, []),
      (2, {'current_speed_kmh': 20, 'active_connector_id': 'c1'}, [])], 2),
    ({'trigger_type': 'vehicle_state', 'condition': 'vehicle_stopped',
      'hold_for_s': 0.5, 'timeout_s': 5},
     [(1, {'current_speed_kmh': 0}, []),
      (1.5, {'current_speed_kmh': 0}, [])], 1.5),
    ({'trigger_type': 'vehicle_state', 'condition': 'vehicle_resumed_moving',
      'hold_for_s': 0.5, 'timeout_s': 5},
     [(1, {'current_speed_kmh': 0}, []),
      (1.5, {'current_speed_kmh': 0}, []),
      (2, {'current_speed_kmh': 10}, [])], 2),
    ({'trigger_type': 'vehicle_state', 'condition': 'speed_threshold_held',
      'comparison': 'above', 'speed_kmh': 40, 'hold_for_s': 0.5,
      'timeout_s': 5},
     [(1, {'current_speed_kmh': 45}, []),
      (1.5, {'current_speed_kmh': 45}, [])], 1.5),
    ({'trigger_type': 'map', 'condition': 'distance_to_destination_below',
      'distance_m': 100, 'timeout_s': 5},
     [(1, {'current_speed_kmh': 20, 'remaining_distance_m': 120}, []),
      (2, {'current_speed_kmh': 20, 'remaining_distance_m': 90}, [])], 2),
    ({'trigger_type': 'environment', 'condition': 'weather_changed_to',
      'weather_condition': 'rainy', 'timeout_s': 5},
     [(1, {'current_speed_kmh': 20}, [{'event_type': 'weather_initialized',
         'details': {'condition': 'rainy'}}]),
      (2, {'current_speed_kmh': 20}, [{'event_type': 'weather_changed',
         'details': {'condition': 'rainy', 'previous_condition': 'sunny'}}])], 2),
    ({'trigger_type': 'environment', 'condition': 'daynight_became_dark',
      'timeout_s': 5},
     [(1, {'current_speed_kmh': 20}, [{'event_type': 'daynight_changed',
         'details': {'is_dark': True, 'previous_period': 'dusk'}}]),
      (2, {'current_speed_kmh': 20}, [{'event_type': 'daynight_changed',
         'details': {'is_dark': True, 'previous_period': 'afternoon'}}])], 2),
])
def test_judge_trigger_predicates_follow_physics_and_event_edges(
        trigger, observations, activated_at):
    cb, _, _, _ = make_callback([], [], judge_window_s=None)
    request = {'created_at_s': 0, 'acceptance_criteria': {
        'core': ['Requested state is reached'], 'secondary': [],
        'request_kind': 'one_shot', 'expected_response_s': 1,
        'valid_for_s': 3}}
    runtime = cb._new_trigger_runtime(
        trigger, request, {'current_speed_kmh': 20})
    pending = {'trigger_runtime': runtime, 'request': request, 'record': {},
               'check_offsets_s': [0.1], 'next_check_at_s': 5}
    cb.pending_requests['test'] = pending
    for time_s, state, events in observations:
        cb.poll_personal_events(time_s, state, events, episode_id='test')
    assert runtime['status'] == 'activated'
    assert runtime['activated_at_s'] == pytest.approx(activated_at)
    assert pending['response_origin_s'] == pytest.approx(activated_at)


def test_late_environment_edge_cannot_activate_expired_trigger():
    cb, _, _, _ = make_callback([], [], judge_window_s=None)
    request = {'created_at_s': 0, 'acceptance_criteria': {}}
    runtime = cb._new_trigger_runtime({
        'trigger_type': 'environment', 'condition': 'weather_changed_to',
        'weather_condition': 'foggy', 'timeout_s': 1}, request,
        {'current_speed_kmh': 20})
    cb.pending_requests['test'] = {
        'trigger_runtime': runtime, 'request': request, 'record': {},
        'check_offsets_s': [0.1], 'next_check_at_s': 1}
    cb.poll_personal_events(1.1, {'current_speed_kmh': 20}, [{
        'event_type': 'weather_changed',
        'details': {'condition': 'foggy'}}], episode_id='test')
    assert runtime['status'] == 'timed_out'


def test_real_engine_hides_environment_edges_without_judge_horizon():
    from simulation.multi_sim_engine import MultiScenario, MultiSimEngine
    pa = Client([response('finish', {})])
    judge = Client([])

    def driver(_vw, _time_s, _messages, _memory, _tick, **_kwargs):
        return []

    driver._state = {'all_messages': [], 'tool_call_log': [],
                     'protocol_events': []}
    cb = EventDrivenPassengerCallback(
        'ego', driver, PersonalAgentRuntime('ego', pa),
        PassengerJudgeRuntime('ego', judge),
        schedule_config={'random_min_s': 100, 'random_max_s': 100,
                         'stopped_after_s': 100})
    cb._agent_type = 'llm'
    cb.request_design_offset = 4  # cross-domain temporal design
    scenario = MultiScenario.from_dict({
        'scenario_id': 'pa_available_environment_triggers',
        'road_network_id': 'beijing_guomao', 'total_time_s': 1.5,
        'weather_keyframes': [
            {'t': 0, 'condition': 'sunny'},
            {'t': 0.01, 'condition': 'rainy'}],
        'daynight_keyframes': [
            {'t': 0, 'period': 'afternoon'},
            {'t': 0.02, 'period': 'night'}],
        'vehicles': [{
            'vehicle_id': 'ego', 'initial_node': 'n317995521',
            'destination_node': 'n1634509692',
            'agent_config': {'type': 'llm', 'heartbeat_interval_s': 100},
            'initial_physical_state': {
                'lane_id': 'n1634509692_n317995521::lane_3',
                'progress': 0.54}}]})
    MultiSimEngine(scenario).run({'ego': cb})
    assert pa.calls
    observation = json.loads(pa.calls[0][-1]['content'])['observation']
    options = observation['available_judge_triggers']
    names = {option['condition'] for option in options}
    assert 'weather_changed_to' not in names
    assert 'daynight_became_dark' not in names
    assert 'future_weather_conditions' not in observation['temporal_task_context']
    assert set(observation['request_design']['allowed_trigger_conditions']) == names


def test_normal_arrival_before_trigger_is_na_without_calling_judge():
    criteria = {
        "core": ["Lower the volume after the delay"], "secondary": [],
        "request_kind": "one_shot", "expected_response_s": 2.0,
        "valid_for_s": 5.0,
    }
    cb, _, judge, _ = make_callback([
        response("send_passenger_request", {
            "message": "The music is loud.",
            "triggered_request": "lower the volume",
            "acceptance_criteria": criteria,
            "judge_trigger": {
                "trigger_type": "time", "condition": "after_delay",
                "after_s": 4.0, "timeout_s": 5.0,
            },
        })], [], judge_window_s=None)
    vw = VehicleWorld()
    wake(cb, vw, 0.0, "personal_agent_due")
    wake(cb, vw, 1.0, "arrived")

    result = cb.judge.state["judgements"][0]
    assert result["judged"] is True
    assert result["grade"] == "NA"
    assert result["overall_score_100"] is None
    assert result["close_reason"] == "arrived_before_judge_trigger"
    assert judge.calls == []


def test_invalid_final_judge_result_retries_then_remains_unscored():
    from personal_agent import aggregate_passenger_judgements

    cb, _, _, _ = make_callback([
        response("send_passenger_request", {
            "message": "Please set the temperature"}),
    ], [
        response("submit_passenger_judgement", {}),
        response("submit_passenger_judgement", {}),
    ])
    vw = VehicleWorld()
    wake(cb, vw, 0.0, "personal_agent_due")
    wake(cb, vw, 1.0, "collision")

    result = cb.judge.state["judgements"][0]
    assert result["judged"] is False
    assert result["grade"] is None
    assert result["overall_score_100"] is None
    assert result["completion_status"] == "unverified"
    assert result["evaluation_status"] == "judge_failed_unscored"
    assert result["judge_retry_count"] == 1
    assert result["judge_retry_exhausted"] is True
    assert result["judge_replay_pending"] is True
    assert result["close_reason"] == "judge_failure"
    assert len(result["judge_attempt_ids"]) == 2
    assert len(cb.judge.state["input_log"]) == 2
    assert result["judge_error"]
    summary = aggregate_passenger_judgements(
        cb.judge.state["judgements"], request_count=1)
    assert summary["overall_score_100"] is None
    assert summary["scored_count"] == 0
    assert summary["judge_failed_count"] == 1
    assert summary["score_coverage_rate"] == 0.0


def test_heartbeat_does_not_call_pa_and_pa_noop_still_wakes_driver_once():
    cb, pa, judge, drivers = make_callback([response("finish", {})], [])
    vw = VehicleWorld()
    wake(cb, vw, 0, "heartbeat")
    assert len(pa.calls) == 0
    wake(cb, vw, 1, "personal_agent_due", "heartbeat")
    assert len(pa.calls) == 1 and len(drivers) == 2
    assert drivers[-1][2][-1]["event_type"] == "personal_agent_update"
    assert drivers[-1][2][-1]["details"]["has_request"] is False
    assert cb.pending_judge_due_at_s is None and not judge.calls


def test_repeated_checks_early_stop_no_pa_feedback_and_no_double_scoring():
    cb, pa, judge, drivers = make_callback([
        response("send_passenger_request", {"message": "Set AC to 24"}),
        response("finish", {}),
    ], [judgement("pending"), judgement("completed")])
    vw = VehicleWorld()
    wake(cb, vw, 0, "personal_agent_due")
    assert cb.pending_judge_due_at_s == 2
    wake(cb, vw, 1, "heartbeat")
    wake(cb, vw, 2, "passenger_judge_due")
    assert cb.pending_judge_due_at_s == 4
    assert len(drivers) == 2 and len(pa.calls) == 1
    assert cb.judge.state["judgements"] == []
    wake(cb, vw, 4, "passenger_judge_due")
    wake(cb, vw, 6, "passenger_judge_due")
    assert len(judge.calls) == 2 and len(drivers) == 2
    assert len(cb.judge.state["judgements"]) == 1
    assert cb.request_records[0]["status"] == "completed"
    wake(cb, vw, 7, "personal_agent_due")
    payload = json.loads(pa.calls[-1][-1]["content"])
    assert payload["observation"]["recent_requests"][0]["message"] == "Set AC to 24"
    assert "private_judge_marker" not in str(pa.calls)
    assert "completion_status" not in str(pa.calls)
    evidence = cb.judge.state["evidence_log"][-1]
    assert len(evidence["vehicle_agent_response"]["tool_calls"]) == 2
    assert evidence["previous_check"]["completion_status"] == "pending"


def test_already_on_fog_lights_complete_without_driver_reply_or_new_action():
    cb, pa, judge, _ = make_callback([
        response("send_passenger_request", {"message": "请打开雾灯"})],
        [judgement("completed")], judge_window_s=None)
    driver_wakes = []

    def silent_driver(_vw, t, _messages, _memory, _tick, **_kwargs):
        driver_wakes.append(t)
        return []

    silent_driver._state = {
        "all_messages": [], "tool_call_log": [], "protocol_events": [],
        "heartbeat_interval_s": 100.0,
    }
    cb.vehicle_callback = silent_driver
    vw = VehicleWorld()
    vw.fogLight.carcontrol_fogLight_switch(True, 'front')
    for t, event in [(0, "personal_agent_due"), (0.1, "passenger_judge_due"),
                     (1, "passenger_judge_due"), (3, "passenger_judge_due")]:
        cb(vw, t, [], None, int(t * 10),
           _wake_events=[{"event_type": event}],
           _vehicle_state={"current_speed_kmh": 20,
                           "signal_state": {"front_fog_light": True}},
           _recent_motion=[{"time_s": t, "speed_kmh": 20}],
           _passenger_judge_world={"front_fog_light": True})
    assert driver_wakes == [0]
    assert len(pa.calls) == len(judge.calls) == 1
    assert cb.request_records[0]["status"] == "completed"
    assert cb.request_records[0]["closed_at_s"] == 0.1
    assert cb.request_records[0]["checks"] == 1
    assert cb.pending_judge_due_at_s is None
    evidence = cb.judge.state["evidence_log"][0]
    assert evidence["vehicle_agent_response"]["has_passenger_visible_response"] is False
    assert evidence["evaluator_world_when_requested"]["front_fog_light"] is True
    result = cb.judge.state["judgements"][0]
    assert result["dimension_scores_100"]["request_response"] == 100
    assert result["score_constraints"] == []


@pytest.mark.parametrize("status,outcome", [("pending", "uncompleted"), ("uncertain", "unverified")])
def test_deadline_stops_checking_without_notifications(status, outcome):
    cb, pa, judge, drivers = make_callback([
        response("send_passenger_request", {"message": "Please help"})],
        [judgement(status), judgement(status)], max_checks=5, request_ttl_s=3)
    vw = VehicleWorld()
    wake(cb, vw, 0, "personal_agent_due")
    wake(cb, vw, 2, "passenger_judge_due")
    assert cb.pending_judge_due_at_s == 3
    wake(cb, vw, 3, "passenger_judge_due")
    assert cb.pending_judge_due_at_s is None
    assert cb.request_records[0]["status"] == outcome
    assert len(pa.calls) == len(drivers) == 1 and len(judge.calls) == 2


def test_ongoing_request_cannot_complete_before_last_window():
    cb, pa, judge, drivers = make_callback([
        response("send_passenger_request", {"message": "Keep driving smoothly"})],
        [judgement("completed", "ongoing"), judgement("completed", "ongoing")], max_checks=2)
    vw = VehicleWorld()
    wake(cb, vw, 0, "personal_agent_due")
    wake(cb, vw, 2, "passenger_judge_due")
    assert cb.pending_judge_due_at_s == 4
    wake(cb, vw, 4, "passenger_judge_due")
    assert cb.request_records[0]["checks"] == 2
    assert cb.request_records[0]["status"] == "completed"
    assert "INTERMEDIATE observation window" in judge.calls[0][0]["content"]
    assert "FINAL observation window" in judge.calls[1][0]["content"]
    assert "Never return pending or uncertain solely because the request is ongoing" in judge.calls[1][0]["content"]
    assert "stationary vehicle alone does not prove smooth driving" in judge.calls[1][0]["content"]
    assert "do not demand proof about the rest of the trip" in judge.calls[1][0]["content"]
    status_schema = judge.tool_schemas[1][0]["function"]["parameters"]["properties"]["completion_status"]
    assert "within the observed window, not forever" in status_schema["description"]


@pytest.mark.parametrize("status,outcome", [
    ("completed", "completed"), ("pending", "uncompleted"),
    ("uncertain", "unverified"),
])
def test_final_ongoing_prompt_does_not_force_model_verdict(status, outcome):
    cb, pa, judge, drivers = make_callback([
        response("send_passenger_request", {"message": "Keep driving smoothly"})],
        [judgement("pending", "ongoing"), judgement("pending", "ongoing"),
         judgement(status, "ongoing")], judge_window_s=None)
    vw = VehicleWorld()
    wake(cb, vw, 0, "personal_agent_due")
    for t in (0.1, 1.0, 3.0):
        wake(cb, vw, t, "passenger_judge_due")
    assert cb.request_records[0]["status"] == outcome
    assert cb.request_records[0]["checks"] == 3
    assert cb.pending_judge_due_at_s is None
    assert len(pa.calls) == len(drivers) == 1
    assert json.loads(judge.calls[-1][1]["content"])["final_window"] is True
    assert "FINAL observation window" in judge.calls[-1][0]["content"]


def test_new_sheet_supersedes_old_before_terminal_flush():
    cb, pa, judge, drivers = make_callback([
        response("send_passenger_request", {"message": "Request A"}),
        response("send_passenger_request", {"message": "Request B"})],
        [judgement("pending"), judgement("uncertain")])
    vw = VehicleWorld()
    wake(cb, vw, 0, "personal_agent_due")
    wake(cb, vw, 1, "personal_agent_due")
    wake(cb, vw, 1.5, "collision")
    assert len(pa.calls) == len(drivers) == 2
    assert len(judge.calls) == 1
    assert not cb.pending_requests
    assert cb.request_records[0]["status"] == "superseded"
    assert cb.request_records[0]["close_reason"] == "passenger_updated"
    assert cb.request_records[0]["superseded_by_request_id"] == "pa-ego-0002"
    assert cb.request_records[1]["sheet_revision"] == 2
    assert cb.request_records[1]["sheet_id"] == cb.request_records[0]["sheet_id"]
    assert cb.request_records[1]["close_reason"] == "episode_ended"


def test_pa_sees_active_sheet_and_can_cancel_it_without_judge_failure():
    cb, pa, judge, drivers = make_callback([
        response("send_passenger_request", {"message": "Request A"}),
        response("cancel_passenger_request", {
            "reason": "I no longer need it"}),
    ], [])
    vw = VehicleWorld()
    wake(cb, vw, 0, "personal_agent_due")
    wake(cb, vw, 1, "personal_agent_due")

    second_payload = json.loads(pa.calls[1][-1]["content"])
    active = second_payload["observation"]["active_request_sheet"]
    assert active["request_id"] == "pa-ego-0001"
    assert active["allowed_actions"] == ["keep", "update", "cancel"]
    assert not cb.pending_requests
    assert cb.request_records[0]["status"] == "cancelled"
    assert cb.request_records[0]["close_reason"] == "passenger_cancelled"
    assert judge.calls == []
    update = drivers[-1][2][-1]
    assert update["details"]["sheet_action"] == "cancel"
    assert update["details"]["active_request_id"] is None


@pytest.mark.parametrize("config", [
    {"random_min_s": 0}, {"random_max_s": float("nan")},
    {"random_min_s": 50, "random_max_s": 10}, {"event_types": ["passenger_judge_due"]},
])
def test_invalid_schedule_rejected(config):
    with pytest.raises(ValueError):
        PassengerWakeScheduler("ego", **config)


@pytest.mark.parametrize("judge_interval,end_time,expected_checks", [
    (0.5, 2.0, 2), (5.0, 1.2, 1), (0.1, 0.4, 3),
    (None, 3.2, 3), (None, 0.4, 2)])
def test_real_engine_passenger_random_judge_and_terminal_scheduling(
        judge_interval, end_time, expected_checks):
    from simulation.multi_sim_engine import MultiScenario, MultiSimEngine
    cb, pa, judge, drivers = make_callback([
        response("send_passenger_request", {"message": "Please set the temperature"}),
        response("finish", {}),
        response("finish", {}),
    ], [judgement("pending") for _ in range(3)],
        judge_window_s=judge_interval, max_checks=3 if judge_interval in (None, 0.1) else 2,
        schedule_config={"random_min_s": 1.5, "random_max_s": 1.5,
                         "cooldown_s": 0.2, "stopped_after_s": 100})
    cb._agent_type = "llm"
    engine = MultiSimEngine(MultiScenario.from_dict({
        "scenario_id": "event_random_pa_integration", "road_network_id": "beijing_guomao",
        "total_time_s": end_time, "vehicles": [{
            "vehicle_id": "ego", "initial_node": "n317995521",
            "destination_node": "n1634509692",
            "agent_config": {"type": "llm", "heartbeat_interval_s": 100},
            "initial_physical_state": {
                "lane_id": "n1634509692_n317995521::lane_3", "progress": 0.54},
        }]}))
    engine.run({"ego": cb})
    assert not engine.agent_callback_errors
    expected_times = [t for t in [0.0, 1.5, 3.0] if t < end_time]
    assert [round(row[0], 6) for row in drivers] == expected_times
    assert len(pa.calls) == len(expected_times)
    assert len(judge.calls) == expected_checks
    if judge_interval == 0.1:
        assert [round(e["judged_at_s"], 6) for e in cb.judge.state["check_history"]] == [0.1, 0.2, 0.3]
    if judge_interval is None:
        assert [round(e["judged_at_s"], 6) for e in cb.judge.state["check_history"]] == (
            [0.1, 1, 3] if end_time > 3 else [0.1, end_time])
    assert len(cb.judge.state["judgements"]) == 1
    assert cb.pending_judge_due_at_s is None
    assert [round(e.occurred_at_s, 6) for e in engine.wake_broker.log
            if e.event_type == "personal_agent_update"] == expected_times
    assert not any(e.event_type in {"request_pending", "request_timeout"}
                   for e in engine.wake_broker.log)


def test_factory_selects_new_mode_and_preserves_legacy_option(monkeypatch):
    from evaluation import multi_agent_runner as runner
    from simulation.multi_sim_engine import MultiScenario
    monkeypatch.setattr(runner, "AgentClient", lambda **kw: SimpleNamespace(**kw))
    scenario = MultiScenario.from_dict({
        "scenario_id": "passenger_factory", "road_network_id": "beijing_guomao",
        "total_time_s": 1, "vehicles": [{"vehicle_id": "ego",
            "initial_node": "n317995521", "agent_config": {"type": "llm"}}]})
    specs = runner.resolve_agent_specs(scenario)
    # The driver factory need not contact an endpoint or render a scene here.
    def factory(_vid, _spec):
        def cb(*args, **kwargs):
            return []
        cb._state = {}
        return cb
    monkeypatch.setitem(runner._AGENT_FACTORIES, ("vehicle", "llm"), factory)
    callbacks, _ = runner.build_callbacks(specs,
        personal_agent_runtime_config={"enabled": True, "seed": 42},
        passenger_judge_runtime_config={"window_s": 3, "max_checks": 4})
    callback = callbacks["ego"]
    assert callback.scheduler.config["seed"] == 42
    assert callback.max_checks == 4 and callback.request_ttl_s == 12
    assert callback.judge.client.temperature == 0.0
    assert callback.judge.client.max_tokens == 32768
    assert callback.personal_agent.client.temperature == 0.7
    assert callback.personal_agent.client.max_tokens == 2048
    callbacks, _ = runner.build_callbacks(specs,
        personal_agent_runtime_config={"enabled": True, "temperature": 0.4},
        passenger_judge_runtime_config={"temperature": 0.2,
                                         "max_tokens": 8192})
    assert callbacks["ego"].personal_agent.client.temperature == 0.4
    assert callbacks["ego"].judge.client.temperature == 0.2
    assert callbacks["ego"].judge.client.max_tokens == 8192
    callbacks, _ = runner.build_callbacks(specs,
        personal_agent_runtime_config={"enabled": True})
    assert callbacks["ego"].judge_window_s == 0.1
    assert callbacks["ego"].max_checks == 3
    assert callbacks["ego"].check_offsets_s == [0.1, 1.0, 3.0]
    assert callbacks["ego"].acceptance_timeout_s == 3.0
    callbacks, _ = runner.build_callbacks(specs,
        personal_agent_runtime_config={"enabled": True, "trigger_mode": "legacy"})
    assert not hasattr(callbacks["ego"], "scheduler")


def test_pa_and_judge_attach_only_to_llm_drivers(monkeypatch):
    from evaluation import multi_agent_runner as runner

    monkeypatch.setattr(
        runner, "AgentClient", lambda **kwargs: SimpleNamespace(**kwargs))
    specs = {
        "ego": runner.AgentSpec(
            entity_id="ego", agent_type="sumo", is_evaluated=True),
        **{
            f"peer_{index}": runner.AgentSpec(
                entity_id=f"peer_{index}", agent_type="llm",
                is_evaluated=False, model="Qwen3.8-27B")
            for index in range(2, 6)
        },
        "npc": runner.AgentSpec(
            entity_id="npc", agent_type="sumo", is_evaluated=False),
    }
    callbacks, _ = runner.build_callbacks(
        specs,
        personal_agent_runtime_config={
            "enabled": True,
            "model": "Qwen3.8-27B",
            "api_base": "https://qwen.invalid/v1",
            "temperature": 0.7,
            "max_tokens": 32768,
            "context_window_tokens": 1000000,
            "thinking_mode": "enabled",
            "reasoning_effort": "xhigh",
            "chat_template_enable_thinking": True,
        },
        passenger_judge_runtime_config={
            "model": "Qwen3.8-27B",
            "api_base": "https://qwen.invalid/v1",
            "temperature": 0.0,
            "max_tokens": 32768,
            "context_window_tokens": 1000000,
            "thinking_mode": "enabled",
            "reasoning_effort": "xhigh",
            "chat_template_enable_thinking": True,
        },
    )

    for vehicle_id in (f"peer_{index}" for index in range(2, 6)):
        callback = callbacks[vehicle_id]
        assert hasattr(callback, "personal_agent")
        assert callback.personal_agent.client.model == "Qwen3.8-27B"
        assert callback.personal_agent.client.temperature == 0.7
        assert callback.personal_agent.client.reasoning_effort == "xhigh"
        assert callback.judge.client.temperature == 0.0
        assert callback.judge.client.reasoning_effort == "xhigh"
    assert not hasattr(callbacks["ego"], "personal_agent")
    assert not hasattr(callbacks["npc"], "personal_agent")


@pytest.mark.parametrize("mode,expected", [("event_random", 0.1), ("legacy", 1.0)])
def test_batch_runner_resolves_unspecified_cli_interval(tmp_path, mode, expected):
    from evaluation.experiments.batch_runner import ExperimentBatchRunner
    runner = ExperimentBatchRunner(SimpleNamespace(), tmp_path,
        personal_agent_runtime_config={"enabled": True, "trigger_mode": mode},
        passenger_judge_runtime_config={"window_s": None, "request_ttl_s": None})
    assert runner.passenger_judge_window_s == expected
    if mode == "legacy":
        assert runner.public_passenger_judge_runtime_config["window_s"] == expected
    else:
        assert runner.public_passenger_judge_runtime_config["check_offsets_s"] == [0.1, 1, 3]
    assert runner.public_passenger_judge_runtime_config["max_tokens"] == 32768


@pytest.mark.parametrize("statuses,outcome,times", [
    (["pending", "pending", "pending"], "uncompleted", [0.1, 1, 3]),
    (["pending", "completed"], "completed", [0.1, 1]),
    (["uncertain"] * 3, "unverified", [0.1, 1, 3]),
])
def test_default_offsets_and_non_notifying_final_state(statuses, outcome, times):
    cb, pa, judge, drivers = make_callback([
        response("send_passenger_request", {"message": "Slow down to 20"})],
        [judgement(s) for s in statuses], judge_window_s=None)
    vw = VehicleWorld()
    wake(cb, vw, 0, "personal_agent_due")
    for t in [0.1, 0.2, 0.3, 1.0, 2.0, 3.0, 4.0]:
        wake(cb, vw, t, "passenger_judge_due")
    assert [e["judged_at_s"] for e in cb.judge.state["check_history"]] == times
    assert cb.request_records[0]["status"] == outcome
    assert cb.request_records[0]["acceptance_deadline_s"] == 3
    assert "expires_at_s" not in cb.request_records[0]
    assert len(pa.calls) == len(drivers) == 1
    assert cb.pending_judge_due_at_s is None
    assert len(cb.judge.state["judgements"]) == 1
    evidence = cb.judge.state["evidence_log"][-1]
    assert len(evidence["physical_execution"]["ego_motion_trace"]) >= len(times)
    assert "request_expires_at_s" not in evidence


def test_replacement_sheet_resets_window_and_archives_old_version():
    cb, pa, judge, drivers = make_callback([
        response("send_passenger_request", {"message": "Keep smooth"}),
        response("send_passenger_request", {"message": "Keep quiet"})],
        [judgement("completed", "ongoing") for _ in range(6)], judge_window_s=None)
    vw = VehicleWorld()
    wake(cb, vw, 0, "personal_agent_due")
    wake(cb, vw, 0.1, "passenger_judge_due")
    wake(cb, vw, 0.5, "personal_agent_due")
    for t in [0.6, 1.0, 1.5, 3.0, 3.5]:
        wake(cb, vw, t, "passenger_judge_due")
    assert [r["closed_at_s"] for r in cb.request_records] == [0.5, 3.5]
    assert cb.request_records[0]["status"] == "superseded"
    assert cb.request_records[0]["checks"] == 1
    assert cb.request_records[1]["checks"] == 3
    assert len(pa.calls) == len(drivers) == 2
    assert "acceptance_deadline_s" not in str(pa.calls)
    assert "observed_window_only" in str(cb.judge.state["evidence_log"])


@pytest.mark.parametrize("config", [
    {"check_offsets_s": []}, {"check_offsets_s": [1, 0.1]},
    {"check_offsets_s": [0.1, 0.1]}, {"check_offsets_s": [float("nan")]},
    {"check_offsets_s": [0]}, {"window_s": 1, "check_offsets_s": [1]},
    {"acceptance_timeout_s": 0}, {"max_checks": 4},
    {"request_ttl_s": 1, "acceptance_timeout_s": 2},
])
def test_invalid_acceptance_schedule_rejected(config):
    from evaluation.judge_schedule import resolve_judge_schedule
    with pytest.raises(ValueError):
        resolve_judge_schedule(**config)


def test_suite_preparation_freezes_new_offsets(monkeypatch, tmp_path):
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    import run_llm_suite as suite
    import prepare_native_npc_catalog
    monkeypatch.setattr(prepare_native_npc_catalog, "check_native_calibration", lambda *args: None)
    manifest = SimpleNamespace(variants=[], metadata={}, write=lambda path: None)
    monkeypatch.setattr(suite, "prepare_scene_manifests",
                        lambda *a: {"MultiLLM": tmp_path / "manifest.json"})
    monkeypatch.setattr(suite, "load_manifest", lambda path: manifest)
    saved = {}
    monkeypatch.setattr(suite, "write_json", lambda path, value: saved.update(value))
    suite.prepare(SimpleNamespace(output=tmp_path, only=[], model="test-qwen",
        base_url="https://example.invalid/v1", context_window=1024,
        direct_api=False, workers=1))
    assert saved["passenger_judge_runtime"] == {
        "check_offsets_s": [0.1, 1.0, 3.0], "max_checks": 3,
        "acceptance_timeout_s": 3.0}


def test_completion_schema_rejects_missing_status_without_ending_request():
    malformed = response("submit_passenger_judgement", {
        "grade": "A", "reason": "No explicit completion proof"})
    cb, pa, judge, drivers = make_callback([
        response("send_passenger_request", {"message": "Please help"})],
        [malformed, judgement("pending")], max_checks=2)
    vw = VehicleWorld()
    wake(cb, vw, 0, "personal_agent_due")
    wake(cb, vw, 2, "passenger_judge_due")
    assert cb.pending_judge_due_at_s == 4
    assert cb.judge.state["check_history"][0]["judged"] is False
    wake(cb, vw, 4, "passenger_judge_due")
    assert cb.request_records[0]["status"] == "uncompleted"
