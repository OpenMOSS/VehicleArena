import json
import hashlib
from types import SimpleNamespace

import pytest

from vehiclearena import VehicleWorld
from simulation.multi_sim_engine import MultiScenario, MultiSimEngine
from evaluation.personal_agent import (
    PassengerJudgeRuntime,
    PersonalAgentRuntime,
    PersonalAgentVehicleCallback,
    aggregate_passenger_judgements,
    build_personal_observation,
    _compact_driver_response,
    _DRIVING_OUTCOME_CONTRACT,
    _tool_succeeded,
)


def _tool_response(name, arguments):
    if name == "send_passenger_request":
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
        name = "send_immediate_request"
    if (name == "submit_passenger_judgement" and "grade" in arguments
            and "core_statuses" not in arguments):
        arguments = {
            **arguments, "core_statuses": ["met"] * 6,
            "secondary_statuses": [],
        }
    if (name == "submit_passenger_judgement"
            and arguments.get("grade") in {"A", "B"}
            and "fulfilled_at_s" not in arguments):
        arguments = {
            **arguments,
            "fulfilled_at_s": 0.5,
            "immediate_fulfilled_at_s": 0.5,
        }
    return SimpleNamespace(
        content=None,
        tool_calls=[SimpleNamespace(
            id=f"call-{name}",
            function=SimpleNamespace(
                name=name,
                arguments=json.dumps(arguments, ensure_ascii=False),
            ),
        )],
    )


class _SequencedClient:
    def __init__(self, model, responses, order, role):
        self.model = model
        self.responses = list(responses)
        self.order = order
        self.role = role
        self.last_call_metadata = {}
        self.requests = []

    def chat_with_tools(self, messages, tools):
        self.requests.append({"messages": messages, "tools": tools})
        self.order.append(self.role)
        self.last_call_metadata = {
            "model": self.model, "prompt_tokens": 10,
            "completion_tokens": 2, "total_tokens": 12,
        }
        return self.responses.pop(0), 12, 10, 2


def test_personal_agent_judges_once_at_fixed_timer_not_early_heartbeat():
    order = []
    pa_client = _SequencedClient("pa-model", [
        _tool_response("send_passenger_request", {
            "message": "前面路况复杂，请开稳一点。"}),
        _tool_response("finish", {}),
    ], order, "personal")
    judge_client = _SequencedClient("judge-model", [
        _tool_response("submit_passenger_judgement", {
            "grade": "A",
            "reason": "已响应并平稳减速。",
            "completion_status": "completed",
            "request_kind": "one_shot",
            "completion_reason": "Observed smooth deceleration.",
        }),
    ], order, "judge")
    driver_calls = []

    def driver(vw, t, passenger_messages, memory, tick_index, **kwargs):
        order.append("vehicle")
        driver_calls.append(list(passenger_messages))
        driver._state["all_messages"].append({
            "time_s": t, "messages": [{"role": "assistant", "content": "好"}]})
        driver._state["tool_call_log"].append({
            "time_s": t, "function": "navigation__navigation_set_speed"})
        return ["navigation__navigation_set_speed"]

    driver._state = {"all_messages": [], "tool_call_log": []}
    callback = PersonalAgentVehicleCallback(
        "ego", driver,
        PersonalAgentRuntime("ego", pa_client),
        PassengerJudgeRuntime("ego", judge_client),
    )
    vw = VehicleWorld()
    common = {
        "_vehicle_state": {
            "current_speed_kmh": 30.0,
            "acceleration_mps2": 0.0,
        },
        "_navigation_status": {"status": "active"},
        "_wake_events": [{"event_type": "heartbeat"}],
    }
    callback(vw, 0.0, ["legacy request must not leak"], None, 0,
             _recent_motion=[{"time_s": 0.0, "speed_kmh": 30.0}], **common)
    assert order == ["personal", "vehicle"]
    assert driver_calls == [["前面路况复杂，请开稳一点。"]]
    assert callback._state["passenger_evaluations"] == []

    callback(vw, 0.5, [], None, 5,
             _recent_motion=[
                 {"time_s": 0.0, "speed_kmh": 30.0},
                 {"time_s": 0.5, "speed_kmh": 26.0},
             ], **common)
    assert order == ["personal", "vehicle", "vehicle"]
    assert callback._state["passenger_evaluations"] == []
    assert callback.personal_agent.state["turn_log"][-1]["skip_reason"] == (
        "previous_request_awaiting_judgement")

    callback(vw, 1.0, [], None, 10,
             _recent_motion=[
                 {"time_s": 0.0, "speed_kmh": 30.0},
                 {"time_s": 1.0, "speed_kmh": 22.0},
             ], **{**common, "_wake_events": [{
                 "event_type": "passenger_judge_due"}]})
    assert order == [
        "personal", "vehicle", "vehicle", "judge"]
    assert len(callback._state["passenger_evaluations"]) == 1
    judgement = callback._state["passenger_evaluations"][0]
    assert judgement["judged_at_s"] == 1.0
    assert judgement["overall_score_100"] == 100.0
    evidence = callback.judge.state["evidence_log"][-1]
    assert evidence["physical_execution"]["source_contract"] == _DRIVING_OUTCOME_CONTRACT
    assert evidence["vehicle_agent_response"]["attribution_contract"] == _DRIVING_OUTCOME_CONTRACT
    assert driver_calls == [["前面路况复杂，请开稳一点。"], []]


def test_passenger_judgement_aggregation_excludes_legacy_three_dimension_scores():
    summary = aggregate_passenger_judgements([{
        "judged": True,
        "dimension_scores_100": {"request_response": 60.0,
                                "driving_rationality": 90.0, "environmental_impact": 75.0},
    }])
    assert summary["dimension_scores_100"] == {}
    assert summary["overall_score_100"] is None
    assert summary["legacy_unscored_count"] == 1


def test_personal_prompt_excludes_destination_requests_not_road_context():
    runtime = PersonalAgentRuntime(
        "ego", SimpleNamespace(model="fake"))
    assert "never request a new destination" in runtime.system_prompt
    assert "not limited to cabin requests" in runtime.system_prompt


def test_structured_pa_offers_distinct_request_tools_and_compiles_one_source_of_truth():
    client = _SequencedClient("pa-model", [
        _tool_response("send_triggered_request", {
            "message": (
                "Please set the cabin to 23 degrees now, then lower the music "
                "to 20 percent after two seconds."),
            "immediate_request": {
                "core": ["Set the cabin temperature to 23 degrees"],
                "secondary": [],
            },
            "triggered_request": {
                "core": ["Lower the music volume to 20 percent"],
                "secondary": [],
            },
            "judge_trigger": {
                "trigger_type": "time", "condition": "after_delay",
                "after_s": 2, "timeout_s": 4,
            },
            "request_kind": "one_shot", "expected_response_s": 2,
            "valid_for_s": 5,
        }),
    ], [], "personal")
    runtime = PersonalAgentRuntime("ego", client)
    request = runtime.generate({
        "sim_time_s": 0, "wake_id": "wake-000000",
        "temporal_task_context": {
            "remaining_episode_s": 20, "max_after_delay_s": 10},
        "request_design": {
            "min_explicit_outcomes": 2, "min_core_outcomes": 2,
            "judge_trigger": "required",
            "phase_structure": "immediate_and_triggered",
            "allowed_trigger_conditions": ["after_delay"],
        },
    })
    tool_names = {tool["function"]["name"] for tool in client.requests[0]["tools"]}
    assert {"send_immediate_request", "send_triggered_request"} <= tool_names
    assert "send_passenger_request" not in tool_names
    assert request is not None
    assert request.acceptance_criteria["core"] == [
        "Set the cabin temperature to 23 degrees",
        "Lower the music volume to 20 percent",
    ]
    assert request.message == (
        "Please set the cabin to 23 degrees now, then lower the music "
        "to 20 percent after two seconds.")
    assert runtime.state["errors"] == []


def test_structured_pa_retries_with_precise_field_feedback():
    invalid = {
        "message": "Please set the temperature.",
        "immediate_request": {"core": ["A" * 120], "secondary": []},
        "request_kind": "one_shot", "expected_response_s": 2,
        "valid_for_s": 5,
    }
    valid = {
        **invalid,
        "message": "Could you set the temperature to 23 degrees?",
        "immediate_request": {"core": ["Set the temperature to 23"],
                              "secondary": []},
    }
    client = _SequencedClient("pa-model", [
        _tool_response("send_immediate_request", invalid),
        _tool_response("send_immediate_request", valid),
    ], [], "personal")
    runtime = PersonalAgentRuntime("ego", client)
    request = runtime.generate({
        "sim_time_s": 0, "wake_id": "wake-000000",
        "request_design": {"min_explicit_outcomes": 1,
                           "judge_trigger": "forbidden"},
    })
    assert request is not None
    assert request.message == "Could you set the temperature to 23 degrees?"
    receipt = json.loads(client.requests[1]["messages"][-1]["content"])
    assert receipt["error"] == "invalid_passenger_request"
    assert "immediate_request.core[0]" in receipt["detail"]
    assert receipt["retry_allowed"] is True


def test_personal_observation_caps_delay_by_remaining_route_eta():
    observation = build_personal_observation(
        VehicleWorld(), vehicle_id="ego", sim_time_s=10.0, tick_index=100,
        vehicle_state={
            "current_speed_kmh": 0.0, "target_speed_kmh": 0.0,
            "desired_speed_kmh": 36.0,
        }, wake_events=[], recent_motion=[], episode_total_time_s=100.0,
        navigation_status={"status": "active", "remaining_distance_m": 50.0},
    )
    context = observation["temporal_task_context"]
    assert context["estimated_free_flow_remaining_s"] == 5.0
    assert context["max_after_delay_s"] == 4.0


def test_personal_agent_retains_matching_tool_receipt_across_wakes():
    order = []
    client = _SequencedClient("pa-model", [
        _tool_response("send_passenger_request", {"message": "开稳一点"}),
        _tool_response("finish", {}),
    ], order, "personal")
    runtime = PersonalAgentRuntime("ego", client)
    observation = {
        "sim_time_s": 0.0, "wake_id": "wake-000000",
        "current_events": [],
    }
    runtime.generate(observation)
    tool_call_id = runtime.previous_turn[1]["tool_calls"][0]["id"]
    previous_payload = json.loads(runtime.previous_turn[0]["content"])
    assert previous_payload["type"] == "previous_personal_wake"
    assert "observation" not in previous_payload
    assert runtime.previous_turn[2]["role"] == "tool"
    assert runtime.previous_turn[2]["tool_call_id"] == tool_call_id
    observation["sim_time_s"] = 1.0
    runtime.generate(observation)
    assert runtime.state["errors"] == []


def test_incomplete_judge_tool_call_is_not_scored_as_zero():
    order = []
    client = _SequencedClient("judge-model", [
        _tool_response("submit_passenger_judgement", {}),
        _tool_response("submit_passenger_judgement", {}),
    ], order, "judge")
    runtime = PassengerJudgeRuntime("ego", client)
    result = runtime.judge({
        "request": {
            "request_id": "req-1", "created_at_s": 0.0,
            "message": "请慢一点",
        },
        "judged_at_s": 1.0,
    })
    assert result["judged"] is False
    assert result["error"] == "invalid request satisfaction grade or evidence"
    assert result["grade"] is None
    assert result["overall_score_100"] is None
    assert result["judge_retry_count"] == 1
    assert result["judge_retry_exhausted"] is True
    assert len(result["judge_attempt_ids"]) == 2


def test_judge_retries_invalid_submission_once_and_recovers():
    client = _SequencedClient("judge-model", [
        _tool_response("submit_passenger_judgement", {}),
        _tool_response("submit_passenger_judgement", {
            "grade": "A", "reason": "Request fulfilled.",
        }),
    ], [], "judge")
    runtime = PassengerJudgeRuntime("ego", client)

    result = runtime.judge({
        "request": {
            "request_id": "req-retry", "created_at_s": 0.0,
            "message": "Keep driving smoothly.",
        },
        "judged_at_s": 1.0,
    })

    assert result["judged"] is True
    assert result["overall_score_100"] == 100
    assert result["judge_status"] == "succeeded_after_retry"
    assert result["judge_retry_count"] == 1
    assert len(result["judge_attempt_ids"]) == 2
    assert len(runtime.state["input_log"]) == 2
    assert len(runtime.state["evidence_log"]) == 1
    assert {item["evidence_log_index"] for item in runtime.state[
        "input_log"]} == {0}
    assert runtime.state["input_log"][1]["retry_of_attempt_id"] == \
        runtime.state["input_log"][0]["attempt_id"]
    repair = json.loads(runtime.state["input_log"][1][
        "request_kwargs"]["messages"][-1]["content"])
    assert repair["type"] == "judge_submission_repair"
    assert repair["previous_error"] == (
        "invalid request satisfaction grade or evidence")
    assert runtime.state["errors"][0]["recovered"] is True
    assert runtime.state["errors"][0]["recovered_by_attempt_id"] == result[
        "judge_attempt_id"]
    assert runtime.state["judgements"] == [result]


def test_judge_persists_exact_replayable_input():
    client = _SequencedClient("judge-model", [
        _tool_response("submit_passenger_judgement", {
            "grade": "A", "reason": "Request fulfilled.",
        }),
    ], [], "judge")
    runtime = PassengerJudgeRuntime("ego", client)
    evidence = {
        "request": {
            "request_id": "req-replay", "created_at_s": 0.0,
            "message": "Keep driving smoothly.",
        },
        "judged_at_s": 1.0,
        "observation_when_requested": {
            "cabin": {"shared_settings": {"temperature": 22}}},
        "physical_execution": {
            "ego_motion_trace": [
                {"time_s": 0.0, "speed_kmh": 20.0},
                {"time_s": 1.0, "speed_kmh": 18.0},
            ],
        },
    }

    result = runtime.judge(evidence)

    assert result["judged"] is True
    assert len(runtime.state["input_log"]) == 1
    saved = runtime.state["input_log"][0]
    sent = client.requests[0]
    assert saved["schema"] == "vehiclearena-passenger-judge-input-v1"
    assert saved["attempt_id"] == "ego:req-replay:check-legacy:call-1"
    assert saved["request_kwargs"]["messages"] == sent["messages"]
    assert saved["request_kwargs"]["tools"] == sent["tools"]
    canonical = json.dumps(
        saved["request_kwargs"], ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False)
    assert saved["request_sha256"] == hashlib.sha256(
        canonical.encode("utf-8")).hexdigest()
    actual_user_input = json.loads(
        saved["request_kwargs"]["messages"][1]["content"])
    assert actual_user_input["observation_when_requested"]["cabin"][
        "current_state"] == {"temperature": 22}
    assert actual_user_input["physical_execution"][
        "numeric_trace_summary"]["speed_kmh"]["max"] == 20.0
    assert runtime.state["response_log"][0]["judge_attempt_id"] == saved[
        "attempt_id"]
    assert runtime.state["model_call_log"][0]["request_sha256"] == saved[
        "request_sha256"]


def test_judge_persists_replayable_input_before_provider_failure():
    class FailingJudgeClient:
        model = "judge-model"
        api_base = "https://judge.invalid/v1"
        last_call_metadata = {"ok": False, "attempts": 3}

        def build_tool_chat_request(self, messages, tools):
            return {
                "model": self.model,
                "messages": messages,
                "tools": tools,
                "tool_choice": "auto",
                "temperature": 0.0,
            }

        def chat_with_tools(self, messages, tools):
            raise RuntimeError("provider unavailable")

    runtime = PassengerJudgeRuntime("ego", FailingJudgeClient())
    result = runtime.judge({
        "request": {
            "request_id": "req-failed", "created_at_s": 0.0,
            "message": "Lower the temperature.",
        },
        "judged_at_s": 1.0,
    })

    assert result["judged"] is False
    assert result["grade"] is None
    assert result["overall_score_100"] is None
    assert result["judge_retry_count"] == 1
    assert len(runtime.state["input_log"]) == 2
    first, saved = runtime.state["input_log"]
    assert first["request_sha256"] == saved["request_sha256"]
    assert saved["request_id"] == "req-failed"
    assert saved["api_base"] == "https://judge.invalid/v1"
    assert saved["request_kwargs"]["temperature"] == 0.0
    assert saved["call_metadata"] == {"ok": False, "attempts": 3}
    assert runtime.state["errors"][-1]["judge_attempt_id"] == saved[
        "attempt_id"]


def test_personal_observation_excludes_media_catalogs_and_static_tables():
    observation = build_personal_observation(
        VehicleWorld(), vehicle_id="ego", sim_time_s=0.0, tick_index=0,
        vehicle_state={"current_speed_kmh": 20.0},
        wake_events=[], recent_motion=[], episode_total_time_s=12.5)
    encoded = json.dumps(observation, ensure_ascii=False)
    assert len(encoded) < 5000
    assert "current_playlist" not in encoded
    assert "downloaded_videos" not in encoded
    assert "unit_ranges" not in encoded
    assert "contacts" not in encoded
    assert "preset_angles" not in encoded
    assert observation["cabin"]["current_module_state"]["music"][
        "current_track"]["title"]
    assert "airConditioner" in observation["cabin"]["available_modules"]
    assert observation["temporal_task_context"]["remaining_episode_s"] == 12.5
    assert observation["temporal_task_context"]["max_after_delay_s"] == 9.0


def test_personal_observation_lists_only_installed_verifiable_devices():
    vw = VehicleWorld(equipment_profile="economy")
    vw.wiper.carcontrol_wiperBlade_switch(True, "front")
    observation = build_personal_observation(
        vw, vehicle_id="ego", sim_time_s=0.0, tick_index=0,
        vehicle_state={}, wake_events=[], recent_motion=[])
    cabin = observation["cabin"]
    assert "wiper" in cabin["available_modules"]
    assert cabin["current_module_state"]["wiper"]["front"]["is_on"] is True
    assert "sunroof" not in cabin["available_modules"]
    assert "sunroof" not in cabin["current_module_state"]


def test_driver_evidence_never_promotes_ambiguous_or_error_results():
    assert not _tool_succeeded({"result": {"error": "failed"}})
    assert not _tool_succeeded({"result": {"status": "error"}})
    assert _tool_succeeded({"result": {"status": "success"}})
    assert _tool_succeeded({"result": {"success": True}})


def test_judge_preserves_request_score_without_visible_agent_response():
    order = []
    client = _SequencedClient("judge-model", [
        _tool_response("submit_passenger_judgement", {
            "grade": "A",
            "reason": "车辆减速了。",
        }),
    ], order, "judge")
    runtime = PassengerJudgeRuntime("ego", client)
    evidence = {
        "request": {
            "request_id": "req-1", "created_at_s": 0.0,
            "message": "请减速",
        },
        "judged_at_s": 1.0,
        "vehicle_agent_response": {
            "has_passenger_visible_response": False,
        },
    }
    result = runtime.judge(evidence)
    assert result["dimension_scores_100"]["request_response"] == 100.0
    assert result["overall_score_100"] == 100.0
    assert result["score_constraints"] == []
    assert result["reason"] == result["judge_reason_raw"] == "车辆减速了。"
    assert runtime.state["evidence_log"] == [evidence]


def test_driver_response_preserves_assistant_response_timestamps():
    response = _compact_driver_response(
        actions=[], tool_calls=[], protocol_events=[], wakes=[{
            "time_s": 2.5,
            "messages": [{
                "role": "assistant",
                "content": "This vehicle cannot perform that operation.",
            }],
        }])

    assert response["assistant_texts"] == [
        "This vehicle cannot perform that operation."]
    assert response["assistant_responses"] == [{
        "time_s": 2.5,
        "text": "This vehicle cannot perform that operation.",
    }]
    assert response["has_passenger_visible_response"] is True


@pytest.mark.parametrize("status", ["completed", "pending", "uncertain"])
@pytest.mark.parametrize("kind", ["one_shot", "ongoing"])
def test_bounded_judge_verdict_is_not_overridden_by_missing_reply(status, kind):
    client = _SequencedClient("judge-model", [
        _tool_response("submit_passenger_judgement", {
            "grade": "A" if status == "completed" else "D" if status == "pending" else "NA",
            "na_reason": "insufficient_evidence",
            "reason": "依据所提供的真实状态和轨迹验收。",
            "completion_status": status,
            "request_kind": kind,
            "completion_reason": "不要求重复操作或口头确认。",
        }),
    ], [], "judge")
    runtime = PassengerJudgeRuntime("ego", client)
    result = runtime.judge({
        "schema_version": "passenger-judge-evidence-v3",
        "request": {
            "request_id": "req-1", "created_at_s": 0.0,
            "message": "请打开雾灯" if kind == "one_shot" else "请保持平稳驾驶",
        },
        "judged_at_s": 3.0,
        "final_window": True,
        "vehicle_agent_response": {
            "has_passenger_visible_response": False,
            "assistant_texts": [], "tool_calls": [],
        },
    })
    assert result["judged"] is True
    assert result["completion_status"] == status
    assert result["request_kind"] == kind
    assert result["overall_score_100"] == {"completed": 100, "pending": 40, "uncertain": None}[status]
    assert result["score_constraints"] == []
    assert "No spoken reply is needed" in runtime.system_prompt
    assert "Already satisfied state counts" in runtime.system_prompt


@pytest.mark.parametrize("bounded", [False, True])
@pytest.mark.parametrize("status", ["completed", "pending", "uncertain"])
def test_judge_prompt_and_evidence_accept_persistent_control_without_new_action(bounded, status):
    response = _compact_driver_response(actions=[], tool_calls=[], wakes=[], protocol_events=[])
    client = _SequencedClient("judge-model", [
        _tool_response("submit_passenger_judgement", {
            "grade": "A" if status == "completed" else "D" if status == "pending" else "NA",
            "na_reason": "insufficient_evidence",
            "reason": "依据窗口内的运动轨迹验收，不要求重复设置速度。",
            "completion_status": status,
            "request_kind": "ongoing",
            "completion_reason": "当前唤醒没有新动作不代表旧控制失效。",
        }),
    ], [], "judge")
    runtime = PassengerJudgeRuntime("ego", client)
    evidence = {
        "request": {"request_id": "persistent-1", "created_at_s": 0.0,
                    "message": "请保持平稳驾驶"},
        "judged_at_s": 3.0,
        "final_window": True,
        "vehicle_agent_response": response,
        "physical_execution": {"ego_motion_trace": [
            {"time_s": 0.0, "speed_kmh": 20.0, "acceleration_mps2": 0.0},
            {"time_s": 3.0, "speed_kmh": 20.0, "acceleration_mps2": 0.0},
        ]},
    }
    if bounded:
        evidence["schema_version"] = "passenger-judge-evidence-v3"
    result = runtime.judge(evidence)
    messages = client.requests[-1]["messages"]
    contract = json.loads(messages[1]["content"])["vehicle_agent_response"]["attribution_contract"]
    assert contract == _DRIVING_OUTCOME_CONTRACT
    assert contract in messages[0]["content"]
    assert "previously issued controls can remain active across wakes" in contract
    assert "not a prerequisite for completion or scoring" in contract
    assert "must not be credited" not in contract
    assert "No spoken reply is needed" in messages[0]["content"]
    assert not response["has_passenger_visible_response"]
    assert result["score_constraints"] == []
    assert result["overall_score_100"] == {"completed": 100, "pending": 40, "uncertain": None}[status]
    if bounded:
        assert result["completion_status"] == status  # No programmatic verdict override.


def test_simulation_end_judges_pending_request_without_pa_or_driver_call():
    order = []
    pa_client = _SequencedClient("pa-model", [
        _tool_response("send_passenger_request", {"message": "请慢一点"}),
    ], order, "personal")
    judge_client = _SequencedClient("judge-model", [
        _tool_response("submit_passenger_judgement", {
            "grade": "A",
            "reason": "已处理。",
            "completion_status": "completed",
            "request_kind": "one_shot",
            "completion_reason": "Observed request fulfillment.",
        }),
    ], order, "judge")

    def driver(_vw, _t, _messages, _memory, _tick, **_kwargs):
        order.append("vehicle")
        driver._state["all_messages"].append({
            "messages": [{"role": "assistant", "content": "好的。"}]})
        return []

    driver._state = {
        "all_messages": [], "tool_call_log": [], "protocol_events": []}
    callback = PersonalAgentVehicleCallback(
        "ego", driver, PersonalAgentRuntime("ego", pa_client),
        PassengerJudgeRuntime("ego", judge_client))
    common = {
        "_vehicle_state": {"current_speed_kmh": 20.0},
        "_navigation_status": {"status": "active"},
        "_recent_motion": [],
        "_passenger_judge_world": {},
    }
    callback(
        VehicleWorld(), 0.0, [], None, 0,
        _wake_events=[{"event_type": "simulation_start"}], **common)
    callback(
        VehicleWorld(), 1.0, [], None, 1,
        _wake_events=[{"event_type": "simulation_ended"}], **common)
    assert order == ["personal", "vehicle", "judge"]
    assert callback.personal_agent.state["turn_log"][-1]["skipped"] is True
    assert len(callback.judge.state["judgements"]) == 1


def test_engine_runs_judge_timer_without_waking_driver():
    order = []
    personal_client = _SequencedClient("pa-model", [
        _tool_response("send_passenger_request", {
            "message": "请保持平稳驾驶"}),
    ], order, "personal")
    judge_client = _SequencedClient("judge-model", [
        _tool_response("submit_passenger_judgement", {
            "grade": "A",
            "reason": "观察窗内驾驶平稳。",
            "completion_status": "completed",
            "request_kind": "one_shot",
            "completion_reason": "Observed smooth driving.",
        }),
    ], order, "judge")
    driver_wakes = []

    def driver(_vw, time_s, _messages, _memory, _tick, **_kwargs):
        driver_wakes.append(round(time_s, 6))
        order.append("vehicle")
        return []

    driver._state = {
        "all_messages": [], "tool_call_log": [], "protocol_events": []}
    callback = PersonalAgentVehicleCallback(
        "ego", driver, PersonalAgentRuntime("ego", personal_client),
        PassengerJudgeRuntime("ego", judge_client), judge_window_s=1.0)
    callback._agent_type = "llm"
    scenario = MultiScenario.from_dict({
        "scenario_id": "passenger_judge_timer",
        "road_network_id": "beijing_guomao",
        "total_time_s": 1.2,
        "vehicles": [{
            "vehicle_id": "ego",
            "initial_node": "n317995521",
            "destination_node": "n1634509692",
            "agent_config": {
                "type": "llm", "heartbeat_interval_s": 10.0},
            "initial_physical_state": {
                "lane_id": "n1634509692_n317995521::lane_3",
                "progress": 0.54,
            },
        }],
    })
    engine = MultiSimEngine(scenario)
    engine.run({"ego": callback})
    assert driver_wakes == [0.0]
    assert order == ["personal", "vehicle", "judge"]
    assert callback._state["passenger_evaluations"][0][
        "judged_at_s"] == 1.0
    judge_event_ids = {
        event.event_id for event in engine.wake_broker.log
        if event.event_type == "passenger_judge_due"}
    judge_deliveries = [
        item for item in engine.wake_broker.delivery_log
        if judge_event_ids.intersection(item["event_ids"])
    ]
    assert len(judge_deliveries) == 1
