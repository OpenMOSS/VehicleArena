"""Offline checks that driver instructions match actual tools and weather scoring."""

import ast
import itertools
import re

import pytest
from jsonschema import Draft202012Validator

from capabilities import action_module_name
from evaluation.cabin_evaluator import execute_yaml_action
from evaluation.driving_process_score import _equipment_snapshot
from simulation.ground_truth_rules import WorldSnapshot, _apply_weather_rules
from skills.skill_loader import SkillLoader
from tool_utils import dispatch, dispatch_lazy_discovery, generate_tools_schema
from vehiclearena import VehicleWorld
from weather_safety import (
    SUPPORTED_WEATHER_CONDITIONS, WEATHER_SAFETY_MODULES,
    apply_weather_safety_transition, weather_equipment_action_lines,
    weather_score_requirements,
)


def _guide_calls(text):
    return re.findall(r"`(\w+\.\w+\([^`\n]*\))`", text)


def test_driving_guide_describes_persistent_control_and_no_automatic_rescue():
    body = SkillLoader().load_skill('driving_control')
    assert 'actuator gate stops' not in body
    assert 'driver must call' not in body
    assert 'unique legal straight connector' in body
    assert 'does not brake or choose a turn for you' in body
    assert 'selection is consumed' in body
    assert 'straight green arrow does not permit' in body


def _weather_guide_actions(text, previous, condition):
    """Read the actual loaded prose, including both entering and cleanup rules."""
    actions = []
    sections = re.split(r"^### (.+)$", text, flags=re.MULTILINE)
    for title, body in zip(sections[1::2], sections[2::2]):
        conditions = set(re.findall(r"`(\w+)`", title))
        applies = (
            title.startswith("Entering ") and condition in conditions
        ) or (
            title.startswith("Leaving ")
            and previous in conditions and condition not in conditions
        )
        if applies:
            actions.extend("vw." + call for call in _guide_calls(body))
    return actions


@pytest.mark.parametrize("skill", ["daynight_transition", "weather_transition"])
def test_operational_guide_examples_have_loadable_names_and_valid_arguments(skill):
    vw = VehicleWorld()
    schemas = {t["function"]["name"]: t["function"]["parameters"]
               for t in generate_tools_schema()}
    body = SkillLoader().load_skill(skill)
    assert "{{WEATHER_SAFETY_RULES}}" not in body
    calls = _guide_calls(body)
    assert calls
    for example in calls:
        call = ast.parse(example, mode="eval").body
        name = f"{call.func.value.id}__{call.func.attr}"
        assert name in schemas, example
        schema = schemas[name]
        positional_names = list(schema["properties"])
        assert len(call.args) <= len(positional_names)
        arguments = dict(zip(positional_names, map(ast.literal_eval, call.args)))
        arguments.update({kw.arg: ast.literal_eval(kw.value) for kw in call.keywords})
        Draft202012Validator(schema).validate(arguments)
        result = dispatch_lazy_discovery(vw, "load_tools", {"tools": [name]}, [])
        assert result["success"], (example, result)


def test_daynight_display_example_really_decreases_brightness():
    vw = VehicleWorld()
    body = SkillLoader().load_skill("daynight_transition")
    assert "centerInformationDisplay.brightness_decrease(degree='large')" in body
    assert "carcontrol_centerInformationDisplay_brightness_decrease" not in body
    before = vw.centerInformationDisplay.brightness_settings.brightness_level
    result = dispatch(vw, "centerInformationDisplay__brightness_decrease", {"degree": "large"})
    assert result["success"]
    assert vw.centerInformationDisplay.brightness_settings.brightness_level < before


def _equipment(vw):
    return {**_equipment_snapshot(vw),
            "high_beam_on": vw.highBeamHeadlight.high_beam_on}


def _execute(vw, actions):
    for action in actions:
        result = execute_yaml_action(vw, action)
        if isinstance(result, dict):
            assert result.get("success", True), (action, result)


@pytest.mark.parametrize("previous,condition", list(itertools.product(
    sorted(SUPPORTED_WEATHER_CONDITIONS), repeat=2)))
def test_weather_guide_cabin_checkpoint_and_scorer_agree(previous, condition):
    guided, reference, checkpoint = (VehicleWorld() for _ in range(3))
    _, _, state, initial = apply_weather_safety_transition("sunny", previous)
    for vw in (guided, reference, checkpoint):
        _execute(vw, ["vw.window.carcontrol_window_switch(['all'], True)",
                      "vw.sunroof.carcontrol_sunroof_switch('open')"])
        _execute(vw, weather_equipment_action_lines(initial))
    _, _, final_state, updates = apply_weather_safety_transition(previous, condition, state)
    _execute(reference, weather_equipment_action_lines(updates))
    body = SkillLoader().load_skill("weather_transition")
    _execute(guided, _weather_guide_actions(body, previous, condition))
    lines = []
    _apply_weather_rules(WorldSnapshot(weather_condition=previous),
                         WorldSnapshot(weather_condition=condition), lines)
    _execute(checkpoint, [line for line in lines
                         if action_module_name(line) in WEATHER_SAFETY_MODULES])
    actual = _equipment(guided)
    assert actual == _equipment(reference) == _equipment(checkpoint)
    for field, value in final_state.to_dict().items():
        if value is not None:
            assert actual[field] == value
    requirements = weather_score_requirements(previous, condition)
    if requirements["front_wiper_required"]:
        assert actual["front_wiper_on"] is True
    if requirements["front_wiper_must_be_off"]:
        assert actual["front_wiper_on"] is False
    if requirements["openings_must_be_closed"]:
        assert actual["all_windows_closed"] and actual["sunroof_closed"]
    if requirements["fog_visibility_lights_required"]:
        assert all(actual[key] for key in (
            "low_beam_on", "front_fog_on", "rear_fog_on", "position_light_on"))
    if requirements["fog_and_position_lights_must_be_off"]:
        assert not any(actual[key] for key in (
            "front_fog_on", "rear_fog_on", "position_light_on"))
    if requirements["high_beam_must_be_off"]:
        assert actual["high_beam_on"] is False
    if requirements["low_beam_required"]:
        assert actual["low_beam_on"] is True


def test_rain_to_hail_guidance_turns_wipers_off_and_keeps_openings_closed():
    actions = _weather_guide_actions(SkillLoader().load_skill("weather_transition"),
                                     "rainy", "hail")
    assert "vw.wiper.carcontrol_wiperBlade_switch(False, 'front')" in actions
    assert "vw.wiper.carcontrol_wiperBlade_switch(True, 'front')" not in actions
    assert "vw.window.carcontrol_window_switch(['all'], False)" in actions
    assert "vw.sunroof.carcontrol_sunroof_switch('close')" in actions


def test_weather_guidance_filters_each_uninstalled_equipment_independently():
    vw = VehicleWorld(equipment_profile="economy")
    body = SkillLoader().load_skill("weather_transition", vw.available_module_names())
    assert "HUD" not in body and "sunroof" not in body
    assert "wiper.carcontrol_wiperBlade_switch" in body
    assert "window.carcontrol_window_switch" in body
    assert "keep positionLight ON" in body


@pytest.mark.parametrize("period", ["dusk", "night", "dawn"])
def test_weather_cleanup_retains_night_position_lights(period):
    vw = VehicleWorld()
    _, _, _, updates = apply_weather_safety_transition("sunny", "foggy")
    _execute(vw, weather_equipment_action_lines(updates))
    lines = []
    _apply_weather_rules(WorldSnapshot(weather_condition="foggy", daynight_period=period),
                         WorldSnapshot(weather_condition="hail", daynight_period=period), lines)
    _execute(vw, [line for line in lines
                  if action_module_name(line) in WEATHER_SAFETY_MODULES])
    assert vw.positionLight.is_on
    assert not vw.fogLight.front_light.is_on
    assert not vw.fogLight.rear_light.is_on


def test_shared_weather_rules_preserve_yaml_cabin_heating_actions():
    lines = []
    _apply_weather_rules(WorldSnapshot(weather_condition="rainy"),
                         WorldSnapshot(weather_condition="hail"), lines)
    assert "vw.steeringWheel.carcontrol_steeringWheel_heater_switch(True)" in lines
    assert "vw.rearviewMirror.mode_heating(True)" in lines
    assert "vw.seat.carcontrol_carSeat_heater_switch(True, ['all'])" in lines
