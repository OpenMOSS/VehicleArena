"""Regression tests for the unified vehicle/pedestrian LLM harness."""

from __future__ import annotations

import copy
import hashlib
import json
from types import SimpleNamespace

import pytest

from evaluation.agent_client import AgentClient, AgentClientError
from evaluation.multi_agent_runner import (
    make_llm_agent_callback, resolve_agent_specs, run_multi_scenario,
)
from evaluation import multi_agent_runner
from evaluation.context_runtime import (
    RollingWakeContext, TodoStore, build_vehicle_self_now,
    normalize_new_events,
)
from evaluation.driving_eval import (
    _assistant_message_has_payload,
    _message_to_dict,
    generate_driving_instruction,
)
from evaluation.pedestrian_agent import make_llm_pedestrian_callback
from tool_utils import dispatch
from tool_utils import generate_tools_schema
from vehiclearena import VehicleWorld
from simulation.multi_sim_engine import (
    MultiScenario, MultiSimEngine, MultiSimResult,
)


def test_runtime_llm_override_is_committed_before_engine_construction():
    captured = {}

    class CaptureEngine:
        def __init__(self, scenario):
            captured["agent_type"] = scenario.vehicles[0].agent_type
            self.scenario = scenario

        def run(self, _callbacks):
            return MultiSimResult(
                scenario_id=self.scenario.scenario_id,
                vehicle_results={},
            )

    run_multi_scenario({
        "scenario_id": "authority_sync",
        "road_network_id": "beijing_guomao",
        "total_time_s": 0.0,
        "vehicles": [{
            "vehicle_id": "ego",
            "initial_node": "n33399858",
            "agent_config": {"type": "sumo"},
        }],
    }, agent_overrides={"ego": {
           "type": "llm", "api_key": "test-key",
           "api_base": "https://example.invalid/v1",
       }},
       engine_class=CaptureEngine)

    assert captured["agent_type"] == "llm"


@pytest.mark.parametrize("entity_key, legacy_field", [
    ("vehicles", "driver_plugin"),
    ("pedestrians", "pedestrian_plugin"),
])
def test_removed_npc_personality_fields_are_rejected(
    entity_key, legacy_field,
):
    entity = (
        {"vehicle_id": "ego", "initial_node": "n33399858"}
        if entity_key == "vehicles"
        else {"ped_id": "walker", "initial_node": "n33399858"}
    )
    entity[legacy_field] = "normal"
    raw = {
        "scenario_id": "no_npc_personality",
        "road_network_id": "beijing_guomao",
        "vehicles": ([entity] if entity_key == "vehicles" else [{
            "vehicle_id": "ego", "initial_node": "n33399858",
        }]),
        "pedestrians": ([entity] if entity_key == "pedestrians" else []),
    }
    with pytest.raises(ValueError, match="Unknown .*Config fields"):
        MultiScenario.from_dict(raw)


@pytest.mark.parametrize("mutation, message", [
    (lambda raw: raw.update({"episode_seed": 7}), "Unknown scenario fields"),
    (lambda raw: raw.update({"sumo_config": {"seed": 7}}),
     "unknown SUMO physics settings"),
    (lambda raw: raw["vehicles"][0]["agent_config"].update({"seed": 7}),
     "Unknown vehicle agent_config fields"),
])
def test_runtime_random_seed_configuration_is_rejected(mutation, message):
    raw = {
        "scenario_id": "no_runtime_seed",
        "road_network_id": "beijing_guomao",
        "vehicles": [{
            "vehicle_id": "ego",
            "initial_node": "n33399858",
            "agent_config": {"type": "sumo"},
        }],
    }
    mutation(raw)
    with pytest.raises(ValueError, match=message):
        MultiScenario.from_dict(raw)


class _FailingCompletions:
    def __init__(self):
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        raise OSError("offline")


def test_agent_client_failure_is_not_silent():
    client = object.__new__(AgentClient)
    client.api_base = "http://invalid"
    client.model = "mock-model"
    client.temperature = 0.7
    client.max_tokens = 32
    client.last_call_metadata = {}
    completions = _FailingCompletions()
    client.client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions))
    with pytest.raises(AgentClientError):
        client.chat_with_tools([], [])
    assert completions.calls == 3
    assert client.last_call_metadata["ok"] is False
    assert client.last_call_metadata["attempts"] == 3


def test_query_classifier_does_not_treat_suffix_get_as_action():
    assert multi_agent_runner._classify_vehicle_tool(
        "speedLimit__speed_limit_get") == "query"
    assert multi_agent_runner._classify_vehicle_tool(
        "navigation__navigation_set_speed") == "action"
    assert multi_agent_runner._classify_vehicle_tool(
        "music__music_currentDetail_view") == "query"
    assert multi_agent_runner._classify_vehicle_tool(
        "radio__radio_history_view") == "query"


def test_device_state_query_exposes_compact_current_state():
    vw = VehicleWorld()
    vw.wiper.carcontrol_wiperBlade_switch(True, "front")
    schema = multi_agent_runner._make_device_state_tool(vw)
    assert schema["function"]["name"] == "get_device_state"
    result = multi_agent_runner._dispatch_device_state(
        vw, {"modules": ["wiper", "music"]})
    assert result["success"] is True
    assert result["device_state"]["wiper"]["front"]["is_on"] is True
    assert result["device_state"]["music"]["is_playing"] is True


def test_passenger_response_authorization_survives_delayed_wakes():
    assert multi_agent_runner._passenger_response_authorized(
        ["请稍后播报路况"], ())
    assert multi_agent_runner._passenger_response_authorized(
        [], ("pa-ego-0001",))
    assert not multi_agent_runner._passenger_response_authorized([], ())


class _QuotaError(RuntimeError):
    status_code = 403


class _QuotaCompletions:
    def __init__(self):
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        raise _QuotaError("insufficient_quota")


def test_agent_client_does_not_retry_terminal_quota_error():
    client = object.__new__(AgentClient)
    client.api_base = "https://provider.invalid/v1"
    client.model = "mock-model"
    client.temperature = 0.7
    client.max_tokens = 32
    client.last_call_metadata = {}
    completions = _QuotaCompletions()
    client.client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions))
    with pytest.raises(AgentClientError):
        client.chat_with_tools([], [])
    assert completions.calls == 1
    assert client.last_call_metadata["attempts"] == 1


class _SuccessfulCompletions:
    def __init__(self):
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        message = SimpleNamespace(content="", tool_calls=[])
        choice = SimpleNamespace(message=message, finish_reason="stop")
        return SimpleNamespace(
            choices=[choice], usage=None, id="test-response",
            system_fingerprint=None)


def test_agent_client_forwards_thinking_mode_and_reasoning_effort():
    client = object.__new__(AgentClient)
    client.api_base = "https://provider.invalid/v1"
    client.model = "visual-model"
    client.temperature = 0.7
    client.max_tokens = 32
    client.thinking_mode = "enabled"
    client.reasoning_effort = "low"
    client.chat_template_enable_thinking = False
    client.last_call_metadata = {}
    completions = _SuccessfulCompletions()
    client.client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions))

    messages = [{"role": "user", "content": "judge this"}]
    tools = [{"type": "function", "function": {
        "name": "submit", "parameters": {"type": "object"}}}]
    expected_request = client.build_tool_chat_request(messages, tools)

    client.chat_with_tools(messages, tools)

    assert completions.kwargs == expected_request
    assert completions.kwargs["messages"] == messages
    assert completions.kwargs["tools"] == tools
    assert completions.kwargs["tool_choice"] == "auto"
    assert completions.kwargs["extra_body"] == {
        "thinking": {"type": "enabled"},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    assert completions.kwargs["reasoning_effort"] == "low"
    assert client.last_call_metadata["thinking_mode"] == "enabled"
    assert client.last_call_metadata["reasoning_effort"] == "low"
    assert client.last_call_metadata[
        "chat_template_enable_thinking"] is False


def test_qwen3_explicit_thinking_never_sends_conflicting_false_flag():
    client = object.__new__(AgentClient)
    client.api_base = "https://provider.invalid/v1"
    client.model = "Qwen3-VL-8B-Instruct"
    client.temperature = 0.7
    client.max_tokens = 32768
    client.thinking_mode = "enabled"
    client.reasoning_effort = "xhigh"
    client.chat_template_enable_thinking = True
    client.last_call_metadata = {}
    completions = _SuccessfulCompletions()
    client.client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions))

    client.chat([{"role": "user", "content": "drive"}])

    assert completions.kwargs["extra_body"] == {
        "enable_thinking": True,
        "thinking": {"type": "enabled"},
        "chat_template_kwargs": {"enable_thinking": True},
    }
    assert completions.kwargs["reasoning_effort"] == "xhigh"

    tool_request = client.build_tool_chat_request([], [])
    assert tool_request["extra_body"] == completions.kwargs["extra_body"]


def test_simulation_stops_after_first_fatal_agent_callback():
    scenario = MultiScenario.from_dict({
        "scenario_id": "fatal_callback_stops_run",
        "road_network_id": "beijing_guomao",
        "total_time_s": 10.0,
        "tick_interval_s": 0.5,
        "vehicles": [{
            "vehicle_id": "ego", "initial_node": "n33399858",
            "destination_node": "n35722739",
            "agent_config": {"type": "llm"},
        }],
    })
    calls = []

    def callback(*args, **kwargs):
        calls.append(args[1])
        raise AgentClientError("provider unavailable")

    engine = MultiSimEngine(scenario)
    engine.run({"ego": callback})
    assert calls == [0.0]
    assert len(engine.agent_callback_errors) == 1


def test_provider_parameters_wrapper_is_normalized_at_dispatch():
    vw = VehicleWorld()
    look = dispatch(
        vw, "navigation__navigation_minimap",
        {"parameters": '{"scope":"route"}'})
    assert look["success"] is True
    speed = dispatch(
        vw, "navigation__navigation_set_speed",
        {"parameters": '{"speed_kmh": 12}'})
    # Parsing succeeds, but standalone VW has no physical driving backend.
    assert speed == {"success": False, "reason": "driving_backend_unavailable"}


def test_speed_tool_schema_declares_numeric_control_arguments():
    schema = next(
        item["function"] for item in generate_tools_schema(
            modules=["navigation"])
        if item["function"]["name"] == (
            "navigation__navigation_set_speed"))
    properties = schema["parameters"]["properties"]
    assert properties["speed_kmh"]["type"] == "number"
    assert properties["acceleration_mps2"]["type"] == "number"
    assert properties["deceleration_mps2"]["type"] == "number"


def test_specs_include_llm_pedestrians_and_effective_sampling():
    scenario = MultiScenario.from_dict({
        "scenario_id": "agent_specs",
        "road_network_id": "beijing_guomao",
        "vehicles": [{
            "vehicle_id": "ego", "initial_node": "n33399858",
            "agent_config": {"type": "sumo"},
        }],
        "pedestrians": [{
            "ped_id": "walker", "initial_node": "n33399858",
            "agent_config": {
                "type": "llm",
                "model": "ped-model", "max_turns": 7,
                "max_tokens": 333, "temperature": 0.2,
                "context_window_tokens": 8192,
            },
        }],
    })
    specs = resolve_agent_specs(scenario)
    assert specs["ego"].entity_type == "vehicle"
    assert specs["walker"].entity_type == "pedestrian"
    assert specs["walker"].model == "ped-model"
    assert specs["walker"].max_turns == 7
    assert specs["walker"].max_tokens == 333
    assert specs["walker"].temperature == 0.2
    assert specs["walker"].context_window_tokens == 8192


def test_run_level_llm_context_config_overrides_scenario_defaults():
    scenario = MultiScenario.from_dict({
        "scenario_id": "runtime_llm_config",
        "road_network_id": "beijing_guomao",
        "total_time_s": 3.0,
        "vehicles": [{
            "vehicle_id": "ego", "initial_node": "n33399858",
            "agent_config": {
                "type": "llm", "model": "scenario-model",
                "api_base": "https://scenario.invalid/v1",
                "max_tokens": 100, "context_window_tokens": 1000,
            },
        }],
    })
    specs = resolve_agent_specs(scenario, llm_runtime_config={
        "model": "runtime-model",
        "api_base": "https://runtime.invalid/v1",
        "max_tokens": 256,
        "context_window_tokens": 4096,
        "todo_max_ttl_s": 600,
        "thinking_mode": "disabled",
        "reasoning_effort": "low",
        "chat_template_enable_thinking": False,
    })
    spec = specs["ego"]
    assert spec.model == "runtime-model"
    assert spec.api_base == "https://runtime.invalid/v1"
    assert spec.max_tokens == 256
    assert spec.context_window_tokens == 4096
    # Todo state is episode-scoped already; a short diagnostic horizon must
    # not force the model to renew its persistent goal every wake.
    assert spec.todo_max_ttl_s == 600
    assert spec.thinking_mode == "disabled"
    assert spec.reasoning_effort == "low"
    assert spec.chat_template_enable_thinking is False


def test_driver_prompt_override_reaches_prompt_and_audit(monkeypatch):
    scenario = MultiScenario.from_dict({
        "scenario_id": "driver_roles",
        "road_network_id": "beijing_guomao",
        "total_time_s": 0.0,
        "vehicles": [{
            "vehicle_id": "ego", "initial_node": "n33399858",
            "destination_node": "n35722739",
            "agent_config": {"type": "llm"},
        }],
    })
    role = "Prefer smooth acceleration and ample following distance."
    specs = resolve_agent_specs(scenario, overrides={
        "ego": {"driver_prompt": role},
    }, llm_runtime_config={"driver_prompt": "Default role"})
    client = _PerceptionClient()
    monkeypatch.setattr(multi_agent_runner, "AgentClient", lambda **kw: client)
    callbacks, metadata = multi_agent_runner.build_callbacks(specs)
    MultiSimEngine(scenario).run(callbacks)
    state = callbacks["ego"]._state
    assert role in state["instruction"]
    assert "Default role" not in state["instruction"]
    assert state["effective_config"]["driver_prompt"] == role
    assert metadata["ego"]["driver_prompt"] == role
    assert any(role in str(call["messages"]) for call in client.calls)
    with pytest.raises(ValueError, match="driver_prompt must be a string"):
        resolve_agent_specs(scenario, overrides={"ego": {"driver_prompt": {}}})


class _PerceptionClient:
    model = "mock-perception"
    api_base = "local"
    temperature = 0.0
    max_tokens = 64

    def __init__(self):
        self.calls = []
        self.last_call_metadata = {}

    def chat_with_tools(self, messages, tools):
        self.calls.append({
            "messages": copy.deepcopy(messages),
            "tools": copy.deepcopy(tools),
        })
        self.last_call_metadata = {
            "ok": True, "model": self.model,
            "finish_reason": "tool_calls",
        }
        if len(self.calls) == 1:
            tool_call = SimpleNamespace(
                id="route-1", type="function",
                function=SimpleNamespace(
                    name="navigation__navigation_route_plan",
                    arguments='{"address":"n35722739"}'))
        elif len(self.calls) == 2:
            tool_call = SimpleNamespace(
                id="minimap-1", type="function",
                function=SimpleNamespace(
                    name="navigation__navigation_minimap",
                    arguments='{"scope":"route"}'))
        else:
            tool_call = SimpleNamespace(
                id="finish-1", type="function",
                function=SimpleNamespace(
                    name="finish", arguments='{"reason":"done"}'))
        message = SimpleNamespace(
            role="assistant", content="", tool_calls=[tool_call])
        return message, 2, 1, 1


class _RollingWakeClient(_PerceptionClient):
    def chat(self, messages):
        self.last_call_metadata = {
            "ok": True, "model": self.model,
            "finish_reason": "stop", "prompt_tokens": 2,
            "completion_tokens": 1, "total_tokens": 3,
        }
        return "已保持当前持续控制和工具状态。", 3, 2, 1


def test_failed_lane_plan_clears_minimap_highlight_after_cached_success(monkeypatch):
    from simulation.traffic_manager import TrafficCoordinator
    from visualization.agent_visual_renderer import AgentVisualRenderer
    import visualization.web3d_camera

    class NoCamera:
        def render(self, *args, **kwargs):
            return None

        def close(self):
            pass

    monkeypatch.setattr(visualization.web3d_camera, "Web3DCameraRenderer", NoCamera)
    client = _PerceptionClient()
    original = TrafficCoordinator.navigation_route_preview
    cached = []

    def preview(manager, *args, **kwargs):
        if client.calls:
            return None  # model-requested replan now has no lane-level path
        plan = original(manager, *args, **kwargs)
        cached.append(plan)
        return plan

    monkeypatch.setattr(TrafficCoordinator, "navigation_route_preview", preview)
    images = []
    original_render = AgentVisualRenderer.render_minimap

    def render(renderer, *args, **kwargs):
        image = original_render(renderer, *args, **kwargs)
        images.append(image)
        return image

    monkeypatch.setattr(AgentVisualRenderer, "render_minimap", render)
    scenario = MultiScenario.from_dict({
        "scenario_id": "failed_plan_minimap", "road_network_id": "beijing_guomao",
        "total_time_s": 0, "vehicles": [{
            "vehicle_id": "ego", "initial_node": "n33399858",
            "destination_node": "n35722739", "agent_config": {"type": "llm"},
        }],
    })
    callback = make_llm_agent_callback("ego", client, max_turns=5)
    engine = MultiSimEngine(scenario)
    engine.run({"ego": callback})
    assert cached and all(cached)
    logs = callback._state["tool_call_log"]
    assert logs[0]["result"]["reason"] == "no_lane_level_route_from_current_pose"
    assert logs[1]["result"]["success"] is True
    assert logs[1]["result"]["image_attached"] is True
    assert logs[1]["result"]["route_available"] is False
    assert len(images) == 1 and images[0].route_available is False
    assert engine._vw["ego"].navigation._lane_route_preview is None


def test_minimap_preserves_saved_route_until_explicit_replan(monkeypatch):
    from simulation.traffic_manager import TrafficCoordinator
    from visualization.agent_visual_renderer import AgentVisualRenderer
    import visualization.web3d_camera

    class NoCamera:
        def render(self, *args, **kwargs):
            return None

        def close(self):
            pass

    # Repeated views, successful replacement, failure, repeated empty views,
    # and explicit recovery. A view must never invoke the planning provider.
    actions = [
        ("navigation_minimap", {"scope": "route"}),
        ("navigation_minimap", {"scope": "local"}),
        ("navigation_route_plan", {"address": "n35722739"}),
        ("navigation_minimap", {"scope": "route"}),
        ("navigation_route_plan", {"address": "n35722739"}),
        ("navigation_minimap", {"scope": "route"}),
        ("navigation_minimap", {"scope": "local"}),
        ("navigation_route_plan", {"address": "n35722739"}),
        ("navigation_minimap", {"scope": "local"}),
    ]

    class Client(_PerceptionClient):
        def chat_with_tools(self, messages, tools):
            message, total, prompt, completion = super().chat_with_tools(
                messages, tools)
            index = len(self.calls) - 1
            if index < len(actions):
                name, args = actions[index]
                message.tool_calls = [SimpleNamespace(
                    id=f"navigation-{index}", type="function",
                    function=SimpleNamespace(
                        name=f"navigation__{name}",
                        arguments=json.dumps(args)))]
            return message, total, prompt, completion

    monkeypatch.setattr(visualization.web3d_camera, "Web3DCameraRenderer", NoCamera)
    planner_calls = []
    original_preview = TrafficCoordinator.navigation_route_preview

    def preview(manager, *args, **kwargs):
        planner_calls.append((args, kwargs))
        if len(planner_calls) == 3:
            return None
        plan = copy.deepcopy(original_preview(manager, *args, **kwargs))
        assert plan
        plan["_test_generation"] = len(planner_calls)
        return plan

    monkeypatch.setattr(TrafficCoordinator, "navigation_route_preview", preview)
    snapshots = []
    images = []
    original_render = AgentVisualRenderer.render_minimap

    def render(renderer, *args, **kwargs):
        snapshots.append(copy.deepcopy(kwargs["route_preview"]))
        image = original_render(renderer, *args, **kwargs)
        images.append(image)
        return image

    monkeypatch.setattr(AgentVisualRenderer, "render_minimap", render)
    scenario = MultiScenario.from_dict({
        "scenario_id": "saved_route_minimap", "road_network_id": "beijing_guomao",
        "total_time_s": 0, "vehicles": [{
            "vehicle_id": "ego", "initial_node": "n33399858",
            "destination_node": "n35722739", "agent_config": {"type": "llm"},
        }],
    })
    callback = make_llm_agent_callback("ego", Client(), max_turns=12)
    engine = MultiSimEngine(scenario)
    engine.run({"ego": callback})
    assert len(planner_calls) == 4  # initialization plus three explicit plans
    assert [s["_test_generation"] if s else None for s in snapshots] == [
        1, 1, 2, None, None, 4]
    assert snapshots[0] == snapshots[1]
    assert [image.route_available for image in images] == [
        True, True, True, False, False, True]
    logs = callback._state["tool_call_log"]
    assert logs[4]["result"]["reason"] == "no_lane_level_route_from_current_pose"
    assert engine._vw["ego"].navigation._lane_route_preview == snapshots[-1]


class _TruncatedThenFinishClient:
    model = "mock-truncation"
    api_base = "local"
    temperature = 0.0
    max_tokens = 32
    thinking_mode = "disabled"

    def __init__(self):
        self.calls = 0
        self.requests = []
        self.last_call_metadata = {}

    def chat_with_tools(self, messages, tools):
        self.calls += 1
        self.requests.append(copy.deepcopy(messages))
        if self.calls == 1:
            self.last_call_metadata = {
                "ok": True, "model": self.model,
                "finish_reason": "length",
            }
            message = SimpleNamespace(
                role="assistant", content=None, tool_calls=[])
        else:
            self.last_call_metadata = {
                "ok": True, "model": self.model,
                "finish_reason": "tool_calls",
            }
            call = SimpleNamespace(
                id="finish-after-truncation", type="function",
                function=SimpleNamespace(
                    name="finish", arguments='{"reason":"done"}'))
            message = SimpleNamespace(
                role="assistant", content="", tool_calls=[call])
        return message, 1, 1, 0


class _SetSpeedThenFinishClient(_PerceptionClient):
    """Issue one persistent actuator command, then finish every wake."""

    def chat_with_tools(self, messages, tools):
        self.calls.append({
            "messages": copy.deepcopy(messages),
            "tools": copy.deepcopy(tools),
        })
        self.last_call_metadata = {
            "ok": True, "model": self.model,
            "finish_reason": "tool_calls",
        }
        if len(self.calls) == 1:
            name = "navigation__navigation_set_speed"
            arguments = '{"speed_kmh":18,"reason":"clear road"}'
        else:
            name = "finish"
            arguments = '{"reason":"command remains active"}'
        call = SimpleNamespace(
            id=f"call-{len(self.calls)}", type="function",
            function=SimpleNamespace(name=name, arguments=arguments))
        return SimpleNamespace(
            role="assistant", content="", tool_calls=[call]), 2, 1, 1


class _AlwaysTruncatedClient(_TruncatedThenFinishClient):
    def chat_with_tools(self, messages, tools):
        self.calls += 1
        self.requests.append(copy.deepcopy(messages))
        self.last_call_metadata = {
            "ok": True, "model": self.model, "finish_reason": "length",
        }
        return SimpleNamespace(
            role="assistant", content=None, tool_calls=[]), 1, 1, 0


class _SetHeartbeatThenFinishClient(_PerceptionClient):
    """Choose a slower cadence on the first wake, then only finish."""

    def chat_with_tools(self, messages, tools):
        self.calls.append({
            "messages": copy.deepcopy(messages),
            "tools": copy.deepcopy(tools),
        })
        self.last_call_metadata = {
            "ok": True, "model": self.model,
            "finish_reason": "tool_calls",
        }
        if len(self.calls) == 1:
            name = "set_heartbeat_interval"
            arguments = '{"interval_s":3,"reason":"steady observation"}'
        else:
            name = "finish"
            arguments = '{"reason":"no new action"}'
        call = SimpleNamespace(
            id=f"heartbeat-{len(self.calls)}", type="function",
            function=SimpleNamespace(name=name, arguments=arguments))
        return SimpleNamespace(
            role="assistant", content="", tool_calls=[call]), 2, 1, 1


def test_truncated_response_continues_same_wake_instead_of_finishing():
    scenario = MultiScenario.from_dict({
        "scenario_id": "truncated_response",
        "road_network_id": "beijing_guomao",
        "total_time_s": 0.0,
        "vehicles": [{
            "vehicle_id": "ego", "initial_node": "n33399858",
            "destination_node": "n35722739",
            "agent_config": {"type": "llm"},
        }],
    })
    client = _TruncatedThenFinishClient()
    callback = make_llm_agent_callback("ego", client, max_turns=3)

    engine = MultiSimEngine(scenario)
    engine.run({"ego": callback})

    assert client.calls == 2
    event_types = [
        event["type"] for event in callback._state["protocol_events"]]
    assert "response_truncated" in event_types
    assert "explicit_finish" in event_types
    assert "implicit_finish" not in event_types
    replayed_assistant = [
        message for message in client.requests[1]
        if message.get("role") == "assistant"]
    assert all(_assistant_message_has_payload(message)
               for message in replayed_assistant)
    messages = callback._state["all_messages"][0]["messages"]
    assert any(
        message.get("role") == "user"
        and "reached the output limit" in str(message.get("content", ""))
        for message in messages)


def test_reasoning_only_assistant_payload_is_preserved_for_replay():
    message = SimpleNamespace(
        role="assistant", content=None, tool_calls=[],
        reasoning_content="partial reasoning")
    serialized = _message_to_dict(message)
    assert serialized == {
        "role": "assistant", "reasoning_content": "partial reasoning"}
    assert _assistant_message_has_payload(serialized)


def test_consecutive_empty_truncations_never_replay_empty_assistant():
    scenario = MultiScenario.from_dict({
        "scenario_id": "consecutive_truncations",
        "road_network_id": "beijing_guomao",
        "total_time_s": 0.0,
        "vehicles": [{
            "vehicle_id": "ego", "initial_node": "n33399858",
            "destination_node": "n35722739",
            "agent_config": {"type": "llm"},
        }],
    })
    client = _AlwaysTruncatedClient()
    callback = make_llm_agent_callback("ego", client, max_turns=3)
    MultiSimEngine(scenario).run({"ego": callback})

    assert client.calls == 3
    assert all(
        _assistant_message_has_payload(message)
        for request in client.requests
        for message in request
        if message.get("role") == "assistant"
    )
    assert callback._state["protocol_events"][-1]["type"] == (
        "turn_limit_reached")


def test_committed_control_is_injected_on_the_next_wake():
    scenario = MultiScenario.from_dict({
        "scenario_id": "active_control_feedback",
        "road_network_id": "beijing_guomao",
        "total_time_s": 1.0,
        "vehicles": [{
            "vehicle_id": "ego", "initial_node": "n33399858",
            "destination_node": "n35722739",
            "agent_config": {
                "type": "llm", "heartbeat_interval_s": 1.0},
        }],
    })
    client = _SetSpeedThenFinishClient()
    callback = make_llm_agent_callback("ego", client, max_turns=3)

    MultiSimEngine(scenario).run({"ego": callback})

    assert len(callback._state["all_messages"]) == 2
    first_wake = json.loads(
        callback._state["all_messages"][0]["messages"][0]["content"])
    second_wake = json.loads(
        callback._state["all_messages"][1]["messages"][0]["content"])
    assert first_wake["active_control"] == {}
    longitudinal = second_wake["active_control"]["longitudinal"]
    assert longitudinal["command"] == "navigation_set_speed"
    assert longitudinal["target_speed_kmh"] == 18.0
    assert longitudinal["emergency_brake"] is False
    assert longitudinal["command_id"].startswith("command-")
    speed_result = callback._state["tool_call_log"][0]["result"]
    assert speed_result["accepted_target_speed_kmh"] == 18.0
    assert speed_result["current_speed_kmh"] == 0.0
    assert "actual_speed" not in speed_result


def test_llm_heartbeat_request_controls_periodic_wake_cadence():
    scenario = MultiScenario.from_dict({
        "scenario_id": "model_owned_heartbeat",
        "road_network_id": "beijing_guomao",
        "total_time_s": 3.0,
        "vehicles": [{
            "vehicle_id": "ego", "initial_node": "n33399858",
            "destination_node": "n35722739",
            "agent_config": {"type": "llm"},
        }],
    })
    client = _SetHeartbeatThenFinishClient()
    callback = make_llm_agent_callback("ego", client, max_turns=3)

    MultiSimEngine(scenario).run({"ego": callback})

    wakes = callback._state["all_messages"]
    assert [wake["time_s"] for wake in wakes] == [0.0, 3.0]
    first = json.loads(wakes[0]["messages"][0]["content"])
    second = json.loads(wakes[1]["messages"][0]["content"])
    assert first["wake_policy"] == {
        "heartbeat_interval_s": 3.0,
        "event_wake_can_preempt": True,
    }
    assert second["wake_policy"] == {
        "heartbeat_interval_s": 3.0,
        "event_wake_can_preempt": True,
    }
    assert callback._state["heartbeat_interval_s"] == 3.0
    audit = callback._state["heartbeat_interval_audit_log"]
    assert len(audit) == 1
    assert audit[0]["result"]["success"] is True
    schemas = {
        tool["function"]["name"]: tool["function"]
        for tool in callback._state["initial_tools"]
    }
    interval = schemas["set_heartbeat_interval"]["parameters"][
        "properties"]["interval_s"]
    assert interval == {
        "type": "number",
        "minimum": 0.1,
        "maximum": 30.0,
        "default": 3.0,
        "description": (
            "Seconds until the next periodic wake if no event wakes the "
            "vehicle first."),
    }


def test_pushed_event_preempts_model_selected_heartbeat():
    scenario = MultiScenario.from_dict({
        "scenario_id": "event_preempts_model_heartbeat",
        "road_network_id": "beijing_guomao",
        "total_time_s": 0.6,
        "weather_keyframes": [
            {"t": 0.0, "condition": "sunny"},
            {"t": 0.01, "condition": "rainy"},
        ],
        "vehicles": [{
            "vehicle_id": "ego", "initial_node": "n33399858",
            "destination_node": "n35722739",
            "agent_config": {"type": "llm"},
        }],
    })
    callback = make_llm_agent_callback(
        "ego", _SetHeartbeatThenFinishClient(), max_turns=3)

    engine = MultiSimEngine(scenario)
    engine.run({"ego": callback})

    assert [wake["time_s"] for wake in callback._state[
        "all_messages"]] == [0.0, 0.6]
    assert any(
        event.event_type == "weather_changed"
        and event.entity_id == "ego"
        for event in engine.wake_broker.log)


def test_formal_vehicle_callback_has_visual_and_structured_wake(tmp_path):
    scenario = MultiScenario.from_dict({
        "scenario_id": "formal_perception",
        "road_network_id": "beijing_guomao",
        "total_time_s": 0.0,
        "tick_interval_s": 100.0,
        "vehicles": [{
            "vehicle_id": "ego",
            "initial_node": "n33399858",
            "destination_node": "n35722739",
            "destination_name": "Guomao test destination",
            "agent_config": {"type": "llm"},
        }],
    })
    client = _PerceptionClient()
    callback = make_llm_agent_callback("ego", client, max_turns=5)
    callback._state["visual_observation_output_dir"] = str(tmp_path)
    MultiSimEngine(scenario).run({"ego": callback})
    initial_names = {
        tool["function"]["name"] for tool in client.calls[0]["tools"]}
    assert "road_perception__look_ahead" not in initial_names
    assert "navigation__navigation_look" not in initial_names
    assert "navigation__navigation_drive" not in initial_names
    assert {
        "get_module_api", "load_tools", "todo_manage", "memory_search",
        "finish",
    } <= initial_names
    assert {
        "navigation__navigation_route_plan",
        "navigation__navigation_minimap",
        "navigation__navigation_set_speed",
        "navigation__navigation_emergency_stop",
        "navigation__navigation_select_maneuver",
        "frontRadar__scan", "rearRadar__scan",
    } <= initial_names
    assert "memory_search" not in callback._state["core_tool_names"]
    assert "navigation__navigation_minimap" in initial_names
    user_messages = [msg for msg in client.calls[0]["messages"]
                     if msg["role"] == "user"]
    assert len(user_messages) == 3
    wake = json.loads(user_messages[0]["content"])
    assert wake["decision_budget"] == {
        "current_turn": 1,
        "max_turns": 5,
        "turns_remaining_including_current": 5,
        "final_turn_rule": (
            "Prioritize a necessary action or finish; a query cannot be "
            "followed by another decision."),
    }
    wake = json.loads(user_messages[0]["content"])
    assert wake["type"] == "wake"
    assert wake["entity"] == {
        "agent_id": "ego", "entity_type": "vehicle"}
    assert "speed_kmh" in wake["self_now"]
    assert wake["self_now"]["own_signals"] == {
        "left_indicator": False,
        "right_indicator": False,
        "hazard": False,
        "brake_light": False,
        "low_beam": False,
        "high_beam": False,
        "front_fog_light": False,
        "rear_fog_light": False,
        "position_light": False,
        "tail_light": False,
    }
    assert "target_speed_kmh" not in wake["self_now"]
    assert "llm_driver_intent" not in wake["self_now"]
    assert wake["active_control"] == {}
    assert "navigation" not in wake
    assert wake["todo"] == {
        "revision": 0, "long_term_goal": None, "subgoals": []}
    start_event = next(
        event for event in wake["new_events"]
        if event.get("event_type") == "simulation_start")
    assert "destination_node" not in start_event["details"]
    task_event = next(
        event for event in wake["new_events"]
        if event.get("event_type") == "driving_task_assigned")
    assert task_event["details"] == {
        "task_type": "drive_to_destination",
        "destination_name": "Guomao test destination",
        "destination_node": "n35722739",
        "instruction": (
            "Drive safely to the assigned destination and complete the trip."),
    }
    assert user_messages[1]["content"][0]["text"].startswith(
        "[CameraVisual]")
    assert user_messages[1]["content"][1]["image_url"]["url"].startswith(
        "data:image/png;base64,")
    loaded = json.loads(user_messages[2]["content"])
    assert loaded["type"] == "loaded_capabilities"
    assert "frontRadar__scan" in loaded["tools"]
    assert [item["kind"] for item in callback._state["tool_call_log"]] == [
        "query", "query", "harness"]
    assert callback._state["tool_call_log"][1]["result"][
        "image_attached"] is True
    route_receipt = callback._state["tool_call_log"][0]["result"]
    assert route_receipt["success"] is True
    assert route_receipt["minimap_available"] is True
    assert "route_status" not in route_receipt
    assert "destination" not in route_receipt
    assert callback._state["tool_call_log"][1]["result"][
        "route_available"] is True
    assert "data:image/png;base64," not in json.dumps(
        callback._state["all_messages"])
    assert callback._state["protocol_events"][-1]["type"] == \
        "explicit_finish"
    assert callback._state["model_call_log"][0]["finish_reason"] == \
        "tool_calls"
    observations = callback._state["visual_observation_log"]
    assert [item["label"] for item in observations] == [
        "CameraVisual", "NavigationMinimap"]
    for observation in observations:
        persisted = tmp_path / observation["path"].split("/")[-1]
        assert persisted.is_file()
        assert hashlib.sha256(persisted.read_bytes()).hexdigest() == \
            observation["sha256"]


def test_previous_wake_rolls_once_and_capabilities_persist():
    scenario = MultiScenario.from_dict({
        "scenario_id": "rolling_wake_harness_state",
        "road_network_id": "beijing_guomao",
        "total_time_s": 3.0,
        "tick_interval_s": 100.0,
        "vehicles": [{
            "vehicle_id": "ego", "initial_node": "n33399858",
            "destination_node": "n35722739",
            "agent_config": {"type": "llm", "heartbeat_interval_s": 1.0},
        }],
    })
    client = _RollingWakeClient()
    callback = make_llm_agent_callback(
        "ego", client, max_turns=5, heartbeat_interval_s=1.0)
    MultiSimEngine(scenario).run({"ego": callback})
    state = callback._state
    assert len(state["all_messages"]) == 4
    second_wake_request = client.calls[-1]
    second_user_messages = [
        message for message in second_wake_request["messages"]
        if message.get("role") == "user"]
    assert len(second_user_messages) == 3
    assert json.loads(second_user_messages[0]["content"])[
        "sim_time_s"] == 2.0
    assert json.loads(second_user_messages[1]["content"])[
        "sim_time_s"] == 3.0
    image_messages = [
        message for message in second_wake_request["messages"]
        if isinstance(message.get("content"), list)]
    assert len(image_messages) == 1
    capability_messages = []
    for message in second_wake_request["messages"]:
        if message.get("role") != "user" or not isinstance(
                message.get("content"), str):
            continue
        try:
            payload = json.loads(message["content"])
        except json.JSONDecodeError:
            continue
        if payload.get("type") == "loaded_capabilities":
            capability_messages.append(payload)
    assert capability_messages == []
    assert all("Session Summary" not in str(message.get("content", ""))
               for message in second_wake_request["messages"])
    assert "navigation__navigation_minimap" in {
        tool["function"]["name"]
        for tool in second_wake_request["tools"]}
    # After the second wake finishes, only that wake remains online.
    assert json.loads(state["previous_wake_messages"][0]["content"])[
        "sim_time_s"] == 3.0


def test_todo_store_is_atomic_bounded_and_extendable():
    todo = TodoStore(max_ttl_s=100.0)
    result = todo.apply([
        {"op": "set_long_term", "text": "前往目的地", "ttl_s": 80},
        {"op": "add_subgoal", "text": "观察路口", "ttl_s": 10},
    ], now_s=5.0, expected_revision=0)
    assert result["success"] is True
    visible = todo.to_context(5.0)
    assert visible["revision"] == 1
    assert visible["long_term_goal"]["remaining_s"] == 80.0
    subgoal_id = visible["subgoals"][0]["id"]
    extended = todo.apply([
        {"op": "extend", "id": subgoal_id, "extra_s": 20},
    ], now_s=16.0, expected_revision=1)
    assert extended["success"] is True
    assert todo.to_context(16.0)["subgoals"][0]["remaining_s"] == 20.0
    before = todo.audit_dict()
    rejected = todo.apply([
        {"op": "add_subgoal", "text": "x" * 100, "ttl_s": 5},
    ], now_s=16.0, expected_revision=2)
    assert rejected["success"] is False
    assert rejected["open_todos"] == todo.to_context(16.0)
    assert todo.audit_dict() == before


def test_todo_close_is_idempotent_and_receipts_restore_open_state():
    todo = TodoStore(max_ttl_s=100.0)
    created = todo.apply([
        {"op": "add_subgoal", "text": "close window", "ttl_s": 20},
    ], now_s=0.0)
    item_id = created["modified_ids"][0]
    closed = todo.apply([
        {"op": "complete", "id": item_id},
    ], now_s=1.0, expected_revision=1)
    assert closed["success"] is True
    repeated = todo.apply([
        {"op": "complete", "id": item_id},
    ], now_s=2.0, expected_revision=2)
    assert repeated["success"] is True
    assert repeated["revision"] == 2
    assert repeated["already_closed_ids"] == [item_id]
    assert repeated["open_todos"] == todo.to_context(2.0)


def test_todo_schema_exposes_runtime_bounds():
    todo = TodoStore(
        max_long_text_chars=80, max_subgoal_text_chars=48,
        max_ttl_s=100.0)
    operation = todo.tool_schema()["function"]["parameters"][
        "properties"]["operations"]["items"]
    properties = operation["properties"]
    assert properties["text"]["minLength"] == 1
    assert properties["text"]["maxLength"] == 80
    assert "48" in properties["text"]["description"]
    assert properties["ttl_s"]["maximum"] == 100.0
    assert properties["extra_s"]["maximum"] == 100.0
    description = todo.tool_schema()["function"]["description"]
    assert "authoritative success" in description
    assert "same wake" in description


def test_driver_instruction_requires_todo_completion_after_tool_success():
    instruction = generate_driving_instruction()
    assert "action tools for a Todo item return authoritative success" in (
        instruction)
    assert "same wake" in instruction


def test_driver_instruction_does_not_promise_navigation_decision_wakes():
    instruction = generate_driving_instruction()
    assert "navigation_decision_required" not in instruction
    assert "no special decision reminder or automatic braking" in instruction
    assert "normal wakes" in instruction


def test_vehicle_self_now_exposes_persistent_own_signal_state():
    current = build_vehicle_self_now({
        "current_speed_kmh": 20.0,
        "signal_state": {
            "left_indicator": True,
            "low_beam": True,
            "tail_light": True,
        },
    })
    assert current["speed_kmh"] == 20.0
    assert current["own_signals"]["left_indicator"] is True
    assert current["own_signals"]["low_beam"] is True
    assert current["own_signals"]["tail_light"] is True
    assert current["own_signals"]["high_beam"] is False


def test_vehicle_self_now_distinguishes_endpoint_removal_from_stopping():
    current = build_vehicle_self_now({
        "current_speed_kmh": 0.0,
        "arrived": True,
        "present_in_physics_world": False,
        "terminal_crossing_speed_kmh": 25.0,
    })
    assert current["has_arrived"] is True
    assert current["present_in_physics_world"] is False
    assert current["terminal_crossing_speed_kmh"] == 25.0
    assert current["is_disabled"] is True


def test_context_limit_unloads_all_dynamic_capabilities_together():
    context = RollingWakeContext(
        "short system", context_window_tokens=1000,
        max_output_tokens=100)
    wake_messages = [{"role": "user", "content": "current wake"}]
    tools = [
        {"type": "function", "function": {
            "name": "finish", "description": "done",
            "parameters": {"type": "object", "properties": {}}}},
        {"type": "function", "function": {
            "name": "large_dynamic_tool", "description": "x" * 3000,
            "parameters": {"type": "object", "properties": {}}}},
    ]
    skills = {"large_skill": "y" * 1200}
    messages, record = context.prepare_request(
        wake_messages=wake_messages,
        tools=tools,
        core_tool_names={"finish"},
        loaded_skills=skills,
        entity_id="ego", sim_time_s=1.0, tick_index=1, turn_index=1)
    assert record["unloaded_tools"] == ["large_dynamic_tool"]
    assert record["unloaded_skills"] == ["large_skill"]
    assert [item["function"]["name"] for item in tools] == ["finish"]
    assert skills == {}
    assert context.capability_epoch == 1
    assert record["estimated_component_tokens_after"][
        "tool_schemas"] > 0
    assert record["previous_wake_ids"] == []
    assert any("capabilities_unloaded" in item.get("content", "")
               for item in messages)


def test_context_limit_drops_previous_wake_before_failing_current_wake():
    context = RollingWakeContext(
        "short system", context_window_tokens=1000,
        max_output_tokens=100)
    context.previous_wake_messages = [{
        "role": "user",
        "content": json.dumps({
            "type": "wake", "wake_id": "ego:old", "detail": "x" * 2400,
        }),
    }]
    wake_messages = [{"role": "user", "content": "current wake"}]
    tools = [{"type": "function", "function": {
        "name": "finish", "description": "done",
        "parameters": {"type": "object", "properties": {}}}}]

    messages, record = context.prepare_request(
        wake_messages=wake_messages,
        tools=tools,
        core_tool_names={"finish"},
        loaded_skills={},
        entity_id="ego", sim_time_s=2.0, tick_index=2, turn_index=1)

    assert context.previous_wake_messages == []
    assert record["dropped_previous_wake_ids"] == ["ego:old"]
    assert record["dropped_previous_wake_message_count"] == 1
    assert record["estimated_input_tokens_after"] <= record[
        "input_budget_tokens"]
    assert any("previous_wake_context_dropped" in item.get("content", "")
               for item in messages)


def test_mid_wake_context_limit_ends_wake_without_invalidating_episode(
    monkeypatch,
):
    original = RollingWakeContext.prepare_request
    calls = 0

    def fail_second_prepare(self, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            # multi_agent_runner imports the compatibility top-level module;
            # raise the identical class object that its callback catches.
            raise multi_agent_runner.ContextWindowExceeded(
                "synthetic current-wake overflow")
        return original(self, **kwargs)

    monkeypatch.setattr(
        multi_agent_runner.RollingWakeContext,
        "prepare_request", fail_second_prepare)
    scenario = MultiScenario.from_dict({
        "scenario_id": "graceful_context_turn_boundary",
        "road_network_id": "beijing_guomao",
        "total_time_s": 0.0,
        "vehicles": [{
            "vehicle_id": "ego", "initial_node": "n33399858",
            "agent_config": {"type": "llm"},
        }],
    })
    client = _NeverFinishesClient()
    callback = make_llm_agent_callback("ego", client, max_turns=3)

    MultiSimEngine(scenario).run({"ego": callback})

    assert len(client.calls) == 1
    assert callback._state["infrastructure_errors"] == []
    assert callback._state["protocol_events"][-1]["type"] == \
        "wake_context_limit_reached"


def test_world_wake_events_and_live_pa_requests_stay_separate():
    event = {
        "event_id": "event-1",
        "event_type": "heartbeat",
        "details": {},
    }
    normalized = normalize_new_events(["去科技园"], [event])
    assert normalized == [event, {
        "event_type": "passenger_request",
        "message": "去科技园",
    }]


class _NeverFinishesClient(_PerceptionClient):
    def chat_with_tools(self, messages, tools):
        self.calls.append({
            "messages": copy.deepcopy(messages),
            "tools": copy.deepcopy(tools),
        })
        self.last_call_metadata = {
            "ok": True, "model": self.model,
            "finish_reason": "tool_calls",
        }
        tool_call = SimpleNamespace(
            id=f"loop-{len(self.calls)}", type="function",
            function=SimpleNamespace(
                name="get_module_api",
                arguments='{"module":"road_perception"}'))
        return SimpleNamespace(
            role="assistant", content="", tool_calls=[tool_call]), 2, 1, 1


def test_vehicle_callback_enforces_hard_turn_limit():
    scenario = MultiScenario.from_dict({
        "scenario_id": "hard_turn_limit",
        "road_network_id": "beijing_guomao",
        "total_time_s": 0.0,
        "vehicles": [{
            "vehicle_id": "ego", "initial_node": "n33399858",
            "agent_config": {"type": "llm"},
        }],
    })
    client = _NeverFinishesClient()
    callback = make_llm_agent_callback("ego", client, max_turns=100)
    MultiSimEngine(scenario).run({"ego": callback})
    assert len(client.calls) == 16
    assert callback._state["effective_config"]["max_turns"] == 16
    assert callback._state["protocol_events"][-1]["type"] == \
        "turn_limit_reached"


class _PedestrianTodoClient(_PerceptionClient):
    def chat_with_tools(self, messages, tools):
        self.calls.append({
            "messages": copy.deepcopy(messages),
            "tools": copy.deepcopy(tools),
        })
        self.last_call_metadata = {
            "ok": True, "model": self.model, "finish_reason": "tool_calls",
        }
        if len(self.calls) == 1:
            tool_call = SimpleNamespace(
                id="todo-1", type="function",
                function=SimpleNamespace(
                    name="todo_manage",
                    arguments=json.dumps({
                        "expected_revision": 0,
                        "operations": [{
                            "op": "set_long_term",
                            "text": "步行前往目的地",
                            "ttl_s": 60,
                        }],
                    }, ensure_ascii=False)))
            message = SimpleNamespace(
                role="assistant", content="", tool_calls=[tool_call])
        elif len(self.calls) == 2:
            tool_call = SimpleNamespace(
                id="finish-ped", type="function",
                function=SimpleNamespace(
                    name="finish", arguments='{"reason":"planned"}'))
            message = SimpleNamespace(
                role="assistant", content="", tool_calls=[tool_call])
        else:
            message = SimpleNamespace(
                role="assistant", content="继续观察", tool_calls=[])
        return message, 2, 1, 1


def test_pedestrian_uses_same_todo_and_previous_wake_contract():
    client = _PedestrianTodoClient()
    callback = make_llm_pedestrian_callback(
        "ped-1", client, max_turns=4, context_window_tokens=8192)
    state = SimpleNamespace(
        has_arrived=False,
        as_dict=lambda: {
            "speed": 1.2, "is_waiting": True, "is_walking": False,
            "is_on_crosswalk": False, "crossing_progress": 0.0,
            "walking_progress": 0.0, "is_crashed": False,
            "has_arrived": False, "destination": "hidden-destination",
        })
    callback(
        state, 0.0,
        [{"event_id": "start", "event_type": "simulation_start",
          "details": {"destination_node": "n2"}}],
        None, 0, SimpleNamespace())
    callback(state, 1.0, [], None, 1, SimpleNamespace())
    second_request = client.calls[2]
    user_messages = [
        item for item in second_request["messages"]
        if item.get("role") == "user"]
    assert len(user_messages) == 2
    current = json.loads(user_messages[-1]["content"])
    assert current["todo"]["long_term_goal"]["text"] == \
        "步行前往目的地"
    assert "destination" not in current["self_now"]
    assert callback._state["todo_state"]["revision"] == 1
    assert callback._state["next_todo_deadline_s"] == 60.0
    assert "pedestrian_walk" not in callback._state["core_tool_names"]
    assert set(callback._state["core_tool_names"]) == {
        "finish", "get_module_api", "load_tools", "todo_manage"}


def test_loaded_skills_share_first_system_message_without_mutating_history():
    context = RollingWakeContext(
        "driver contract", context_window_tokens=32768, max_output_tokens=4096)
    previous = [
        {"role": "user", "content": "previous wake"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "old"}]},
        {"role": "tool", "tool_call_id": "old", "content": "receipt"},
    ]
    context.previous_wake_messages = copy.deepcopy(previous)
    wake = [{"role": "user", "content": "current wake"}]
    skills = {"driving_control": "driving body", "navigation": "navigation body"}
    messages = context.compose(wake, skills)
    assert messages[0] == {
        "role": "system",
        "content": "driver contract\n\n[LoadedSkill:driving_control]\ndriving body"
                   "\n\n[LoadedSkill:navigation]\nnavigation body",
    }
    assert [item["role"] for item in messages].count("system") == 1
    assert messages[1:] == previous + wake
    messages[1]["content"] = "changed"
    messages[-1]["content"] = "changed"
    assert context.previous_wake_messages == previous
    assert wake == [{"role": "user", "content": "current wake"}]
    assert context.compose(wake, {})[0]["content"] == "driver contract"


class _SkillClient(_PerceptionClient):
    def chat_with_tools(self, messages, tools):
        self.calls.append({
            "messages": copy.deepcopy(messages),
            "tools": copy.deepcopy(tools),
        })
        self.last_call_metadata = {
            "ok": True, "model": self.model, "finish_reason": "tool_calls",
        }
        if len(self.calls) == 1:
            tool_call = SimpleNamespace(
                id="skill-1", type="function",
                function=SimpleNamespace(
                    name="load_skill",
                    arguments='{"skill_name":"driving_control"}'))
            message = SimpleNamespace(
                role="assistant", content="", tool_calls=[tool_call])
        elif len(self.calls) == 2:
            tool_call = SimpleNamespace(
                id="finish-skill", type="function",
                function=SimpleNamespace(
                    name="finish", arguments='{"reason":"loaded"}'))
            message = SimpleNamespace(
                role="assistant", content="", tool_calls=[tool_call])
        else:
            message = SimpleNamespace(
                role="assistant", content="继续", tool_calls=[])
        return message, 2, 1, 1


def test_loaded_skill_body_persists_but_load_receipt_stays_compact():
    scenario = MultiScenario.from_dict({
        "scenario_id": "skill_persistence",
        "road_network_id": "beijing_guomao",
        "total_time_s": 1.0,
        "tick_interval_s": 100.0,
        "vehicles": [{
            "vehicle_id": "ego", "initial_node": "n33399858",
            "destination_node": "n35722739",
            "agent_config": {"type": "llm"},
        }],
    })
    client = _SkillClient()
    callback = make_llm_agent_callback(
        "ego", client, max_turns=4, context_window_tokens=32768)
    MultiSimEngine(scenario).run({"ego": callback})
    second_call = client.calls[1]["messages"]
    skill_contexts = [
        item["content"] for item in second_call
        if item.get("role") == "system"
        and "[LoadedSkill:driving_control]" in item.get("content", "")]
    assert len(skill_contexts) == 1
    assert [item["role"] for item in second_call].count("system") == 1
    assert second_call[0]["role"] == "system"
    tool_receipt = next(
        item["content"] for item in second_call
        if item.get("role") == "tool"
        and "loaded_skill" in item.get("content", ""))
    assert "content_sha256" in tool_receipt
    assert "# Driving Control" not in tool_receipt
    assert any(
        item.get("role") == "system"
        and "[LoadedSkill:driving_control]" in item.get("content", "")
        for item in client.calls[2]["messages"])
    assert callback._state["loaded_skills"] == ["driving_control"]
