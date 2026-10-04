"""Regression tests for per-vehicle equipment and chassis capabilities."""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)

from capabilities import resolve_vehicle_capabilities  # noqa: E402
from evaluation.cabin_evaluator import CabinYamlEvaluator  # noqa: E402
from evaluation.driving_eval import generate_driving_instruction  # noqa: E402
from evaluation.driving_eval import _make_memory_search_tool  # noqa: E402
from rules.rule_loader import Rule  # noqa: E402
from skills.skill_loader import get_skill_loader  # noqa: E402
from simulation.multi_sim_engine import MultiScenario, MultiSimEngine  # noqa: E402
from simulation.scenario_generator import ScenarioGenerator  # noqa: E402
from tool_utils import (  # noqa: E402
    dispatch,
    dispatch_lazy_discovery,
    generate_lazy_discovery_tools,
)
from vehiclearena import VehicleWorld  # noqa: E402


def test_profile_instantiates_only_installed_modules():
    economy = VehicleWorld(
        equipment_profile="economy", chassis_profile="compact")
    executive = VehicleWorld(equipment_profile="executive")

    assert not economy.has_module("HUD")
    assert not hasattr(economy, "HUD")
    assert "HUD" not in economy.get_modules()
    assert economy.has_module("navigation")
    assert economy.has_module("weather")
    assert executive.has_module("HUD")
    assert len(executive.available_module_names()) > len(
        economy.available_module_names())
    assert economy.capabilities.chassis.length_m == 4.25


def test_unavailable_module_is_hidden_and_cannot_be_called():
    economy = VehicleWorld(equipment_profile="economy")
    tools = generate_lazy_discovery_tools()

    prompt = generate_driving_instruction(economy)
    assert "HUD" not in prompt
    assert "`navigation`" in prompt

    api_result = dispatch_lazy_discovery(
        economy, "get_module_api", {"module": "HUD"}, tools)
    assert api_result["error"] == "capability_not_available"

    load_result = dispatch_lazy_discovery(
        economy,
        "load_tools",
        {"tools": ["HUD__carcontrol_HUD_switch"]},
        tools,
    )
    assert load_result["error"] == "capability_not_available"
    assert all(
        item["function"]["name"] != "HUD__carcontrol_HUD_switch"
        for item in tools)

    direct_result = dispatch(
        economy,
        "HUD__carcontrol_HUD_switch",
        {"switch": True},
    )
    assert direct_result["error"] == "capability_not_available"

    skill_text = dispatch_lazy_discovery(
        economy,
        "load_skill",
        {"skill_name": "weather_transition"},
        tools,
        skill_loader=get_skill_loader(),
    )
    assert "HUD" not in skill_text
    assert "sunroof" not in skill_text


def test_extra_lazy_tool_can_be_restored_after_context_unload():
    vehicle = VehicleWorld(equipment_profile="economy")
    tools = generate_lazy_discovery_tools()
    schema = _make_memory_search_tool()
    catalog = {"session_history": "memory_search: query session history"}
    brief = dispatch_lazy_discovery(
        vehicle, "get_module_api", {"module": "session_history"}, tools,
        extra_schemas=[schema], virtual_module_catalogs=catalog)
    assert "memory_search" in brief
    loaded = dispatch_lazy_discovery(
        vehicle, "load_tools", {"tools": ["memory_search"]}, tools,
        extra_schemas=[schema], virtual_module_catalogs=catalog)
    assert loaded["success"] is True
    assert loaded["loaded"] == ["memory_search"]
    assert "schema_sha256" in loaded
    assert any(
        item["function"]["name"] == "memory_search" for item in tools)


def test_required_driver_contract_cannot_be_disabled():
    try:
        resolve_vehicle_capabilities(
            equipment_profile="economy",
            disable_modules=["navigation"],
        )
    except ValueError as exc:
        assert "Required modules cannot be disabled" in str(exc)
    else:
        raise AssertionError("navigation disable must fail")


def test_yaml_rule_requirements_are_explicit_or_inferred():
    inferred = Rule.from_dict({
        "id": "hud_test",
        "domain": "user_intent",
        "trigger": {"user_intent": {"messages": ["HUD"]}},
        "expect": {
            "actions": ["vw.HUD.carcontrol_HUD_switch(True)"],
        },
    })
    explicit = Rule.from_dict({
        "id": "custom_test",
        "domain": "user_intent",
        "trigger": {"user_intent": {"messages": ["custom"]}},
        "requires_modules": ["video", "overheadScreen"],
        "expect": {"actions": []},
    })
    assert inferred.required_modules == ["HUD"]
    assert explicit.required_modules == ["video", "overheadScreen"]


def test_generator_supports_default_and_per_vehicle_overrides():
    vehicles = [
        {"vehicle_id": "ego"},
        {"vehicle_id": "freight_1"},
    ]
    ScenarioGenerator._apply_vehicle_capabilities(vehicles, {
        "*": {
            "equipment_profile": "economy",
            "chassis_profile": "compact",
        },
        "freight_1": {
            "equipment_profile": "freight",
            "chassis_profile": "freight",
        },
    })
    assert vehicles[0]["equipment_profile"] == "economy"
    assert vehicles[0]["chassis_profile"] == "compact"
    assert vehicles[1]["equipment_profile"] == "freight"
    assert vehicles[1]["chassis_profile"] == "freight"


def test_missing_equipment_ground_truth_is_not_scored():
    economy = VehicleWorld(equipment_profile="economy")
    before = VehicleWorld(capability_set=economy.capabilities)
    checkpoint = CabinYamlEvaluator().evaluate(
        vehicle_id="ego",
        time_s=0.0,
        pre_agent_vw=before,
        post_agent_vw=economy,
        ground_truth_lines=[
            "vw.HUD.carcontrol_HUD_switch(True)",
        ],
    )
    assert checkpoint is None


def test_multisim_records_equipment_chassis_and_filters_request():
    scenario = MultiScenario.from_dict({
        "scenario_id": "capability_integration",
        "name": "capability integration",
        "road_network_id": "beijing_guomao",
        "total_time_s": 0.1,
        "tick_interval_s": 100.0,
        "inits_code": "vw.HUD.carcontrol_HUD_switch(True)",
        "vehicles": [
            {
                "vehicle_id": "ego",
                "initial_node": "n33399858",
                "destination_node": "n35722739",
                "equipment_profile": "economy",
                "chassis_profile": "compact",
                "agent_config": {"type": "llm"},
            },
            {
                "vehicle_id": "freight_1",
                "initial_node": "n35553582",
                "destination_node": "n35722739",
                "equipment_profile": "freight",
                "chassis_profile": "freight",
                "agent_config": {"type": "sumo"},
                "is_evaluated": False,
            },
        ],
    })
    delivered_messages = []

    def callback(vw, time_s, messages, memory, tick_index, **kwargs):
        delivered_messages.extend(messages)
        return []

    engine = MultiSimEngine(scenario)
    result = engine.run({"ego": callback})
    reports = result.to_dict()["vehicles"]
    report = reports["ego"]
    physical = engine.traffic_mgr.get_state("ego")
    freight_report = reports["freight_1"]
    freight_physical = engine.traffic_mgr.get_state("freight_1")

    assert report["capabilities"]["equipment_profile"] == "economy"
    assert report["capabilities"]["chassis_profile"] == "compact"
    assert "HUD" not in report["capabilities"]["modules"]
    assert physical.length_m == 4.25
    assert physical.width_m == 1.78
    assert physical.max_acceleration_mps2 == 3.2
    assert freight_report["capabilities"]["equipment_profile"] == "freight"
    assert freight_report["capabilities"]["chassis_profile"] == "freight"
    assert freight_physical.length_m == 7.5
    assert freight_physical.width_m == 2.45
    assert freight_physical.max_acceleration_mps2 == 1.8
    assert delivered_messages == []
    assert any(
        event["type"] == "initial_condition_skipped"
        and event["module"] == "HUD"
        for event in report["capability_events"])
