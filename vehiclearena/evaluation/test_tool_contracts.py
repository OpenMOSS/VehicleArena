"""Schema/runtime regressions for malformed calls seen in paused trajectories."""
import inspect
from typing import Optional, List, Union, Literal

import pytest
from jsonschema import Draft202012Validator

from vehiclearena import VehicleWorld
from tool_utils import (
    _annotation_schema, _build_param_schema, _extract_enum_values,
    _parse_docstring_params, dispatch, generate_tools_schema,
)
from evaluation.context_runtime import TodoStore
from tool_utils import dispatch_lazy_discovery


def test_unknown_air_conditioner_names_return_loadable_candidates_transactionally():
    vw = VehicleWorld()
    live = []
    unknown = "airConditioner__carcontrol_airConditioner_switch"
    result = dispatch_lazy_discovery(vw, "load_tools", {
        "tools": ["airConditioner__temperature_set", unknown]}, live)
    assert not result["success"]
    assert live == []
    assert "airConditioner__switch" in result["suggested_tools"][unknown]
    for candidate in result["suggested_tools"][unknown]:
        assert dispatch_lazy_discovery(vw, "load_tools", {"tools": [candidate]}, [])["success"]
    recovered = dispatch_lazy_discovery(vw, "load_tools", {"tools": ["airConditioner__switch"]}, live)
    assert recovered["success"]


def test_compact_cabin_alias_loads_and_legacy_name_remains_compatible():
    vw = VehicleWorld()
    compact = []
    loaded = dispatch_lazy_discovery(
        vw, "load_tools", {"tools": ["fogLight__switch"]}, compact)
    assert loaded["success"] is True
    assert compact[0]["function"]["name"] == "fogLight__switch"
    assert dispatch(vw, "fogLight__switch", {
        "switch": True, "position": "front"})["success"] is True
    legacy = []
    assert dispatch_lazy_discovery(vw, "load_tools", {"tools": [
        "fogLight__carcontrol_fogLight_switch"]}, legacy)["success"] is True
    assert dispatch(vw, "fogLight__carcontrol_fogLight_switch", {
        "switch": False, "position": "front"})["success"] is True


def test_discovery_suggestions_never_include_denied_tools():
    vw = VehicleWorld()
    denied = {"airConditioner__switch"}
    unknown = "airConditioner__carcontrol_airConditioner_switch"
    result = dispatch_lazy_discovery(vw, "load_tools", {"tools": [unknown]}, [],
                                     denied_tool_names=denied)
    assert not denied.intersection(result["suggested_tools"][unknown])
    assert dispatch_lazy_discovery(vw, "load_tools", {"tools": list(denied)}, [],
                                  denied_tool_names=denied)["error"] == "visual_information_is_image_only"


@pytest.mark.parametrize("names", ["airConditioner__switch", [{}], [None]])
def test_discovery_rejects_malformed_names_without_crashing(names):
    live = []
    assert dispatch_lazy_discovery(VehicleWorld(), "load_tools", {"tools": names}, live)["error"] == "invalid_tool_arguments"
    assert not live


def test_optional_and_nested_annotations_preserve_numeric_types():
    assert _annotation_schema(Optional[float]) == {"type": "number"}
    assert _annotation_schema(float | None) == {"type": "number"}
    assert _annotation_schema(List[float]) == {"type": "array", "items": {"type": "number"}}
    assert _annotation_schema(Optional[List[float]])["items"]["type"] == "number"
    assert _annotation_schema(Union[str, float]) == {
        "anyOf": [{"type": "string"}, {"type": "number"}]}
    assert _annotation_schema(Literal["a", "b"])["enum"] == ["a", "b"]
    param = inspect.Parameter("value", inspect.Parameter.KEYWORD_ONLY, annotation=List[float])
    assert _build_param_schema("value", param, None)["items"]["type"] == "number"
    union_param = param.replace(annotation=Union[str, float])
    assert "type" not in _build_param_schema("value", union_param, {"type_str": "str"})
    assert _annotation_schema("SomeInterestingType") == {}
    assert _extract_enum_values("A continuous range [0, 100].") is None


def test_untyped_google_args_preserve_multiline_multiword_enum():
    params = _parse_docstring_params('''Set a shade.
    Args:
        position: Shade position.
            Must be one of ['front row', 'rear row', 'all'].
        value (float): Opening value.
    Raises:
        ValueError: Incorrect value.
    ''')
    assert set(params) == {"position", "value"}
    assert params["position"]["enum"] == ["front row", "rear row", "all"]
    assert params["value"]["type_str"] == "float"


def test_all_registered_tool_schemas_are_valid():
    VehicleWorld()
    schemas = generate_tools_schema()
    assert len(schemas) > 200
    for tool in schemas:
        Draft202012Validator.check_schema(tool["function"]["parameters"])


@pytest.mark.parametrize("bullet", ["", "- "])
def test_parameters_docstrings_preserve_types_and_multiline_enums(bullet):
    params = _parse_docstring_params(f'''Set heating.
    Parameters:
        {bullet}switch (bool): Enable heating.
        {bullet}value (float, optional): Requested value.
        {bullet}unit (str, optional): Unit of value.
            Enum values: ['celsius', 'level', 'percentage']
        {bullet}count: int Number of steps.
    Returns:
        dict: Result.
    ''')
    assert set(params) == {"switch", "value", "unit", "count"}
    assert params["switch"]["type_str"] == "bool"
    assert params["value"]["type_str"] == "float, optional"
    assert params["unit"]["enum"] == ["celsius", "level", "percentage"]
    assert params["count"]["type_str"] == "int"


def test_steering_heater_schema_and_dispatch_agree():
    vw = VehicleWorld()
    schemas = {tool["function"]["name"]: tool["function"]["parameters"]
               for tool in generate_tools_schema(modules=["steeringWheel"])}
    switch = "steeringWheel__carcontrol_steeringWheel_heater_switch"
    assert schemas[switch]["properties"]["switch"]["type"] == "boolean"
    assert schemas[switch]["required"] == ["switch"]
    assert schemas["steeringWheel__carcontrol_steeringWheel_view_switch"]["properties"]["switch"]["type"] == "boolean"
    for invalid in ["true", "false", "on", 1, None]:
        result = dispatch(vw, switch, {"switch": invalid})
        assert result["error"] == "invalid_tool_arguments"
        assert not vw.steeringWheel.is_heater_on
    assert dispatch(vw, switch, {"switch": True})["success"]
    assert vw.steeringWheel.is_heater_on
    heater_set = "steeringWheel__carcontrol_steeringWheel_heater_set"
    assert schemas[heater_set]["properties"]["value"]["type"] == "number"
    assert schemas[heater_set]["properties"]["unit"]["enum"] == ["celsius", "level", "percentage"]
    assert dispatch(vw, heater_set, {"value": 5, "unit": "level"})["success"]
    assert vw.steeringWheel.heater_level == 5
    assert dispatch(vw, switch, {"switch": False})["success"]
    assert not vw.steeringWheel.is_heater_on


def test_sunshade_schema_and_dispatch_agree_without_partial_mutation():
    vw = VehicleWorld()
    schemas = {tool["function"]["name"]: tool["function"]["parameters"]
               for tool in generate_tools_schema(modules=["sunshade"])}
    schema = schemas["sunshade__carcontrol_sunshade_switch"]
    assert schema["properties"]["position"]["enum"] == ["front row", "rear row", "all"]
    assert schema["required"] == ["action"]
    assert schemas["sunshade__carcontrol_sunshade_openDegree_increase"]["properties"]["value"]["type"] == "number"
    result = dispatch(vw, "sunshade__carcontrol_sunshade_switch", {"action": "open", "position": "front"})
    assert result["error"] == "invalid_tool_arguments"
    assert result["expected"]["enum"] == ["front row", "rear row", "all"]
    assert not vw.sunshade.front_row_status.is_open
    bad_value = dispatch(vw, "sunshade__carcontrol_sunshade_openDegree_increase",
                         {"position": "front row", "value": "30", "unit": "percentage"})
    assert bad_value["parameter"] == "value"
    assert not vw.sunshade.front_row_status.is_open
    assert dispatch(vw, "sunshade__carcontrol_sunshade_switch",
                    {"action": "open", "position": "front row"})["success"]


def test_minimap_schema_requires_explicit_replanning():
    schemas = {tool["function"]["name"]: tool["function"]
               for tool in generate_tools_schema(modules=["navigation"])}
    description = schemas["navigation__navigation_minimap"]["description"]
    assert "without replanning" in description
    assert "explicitly call navigation_route_plan again" in description


def test_engine_adapter_keeps_public_contract(monkeypatch):
    vw = VehicleWorld()
    calls = []

    def adapter(**kwargs):
        calls.append(kwargs)
        return {"success": True}

    monkeypatch.setattr(vw.navigation, "navigation_set_speed", adapter)
    assert dispatch(vw, "navigation__navigation_set_speed", {"speed_kmh": 20})["success"]
    assert dispatch(vw, "navigation__navigation_set_speed", {"speed_kmh": "20"})["error"] == "invalid_tool_arguments"
    assert calls == [{"speed_kmh": 20}]


def test_dispatch_normalizes_legacy_errors_and_radio_status_results():
    vw = VehicleWorld()
    missing = dispatch(vw, "window__carcontrol_window_switch", {
        "position": ["driver's seat"],
    })
    assert missing["success"] is False
    assert "error" in missing

    original_channel = vw.settings.sound_channel
    invalid_radio = dispatch(vw, "radio__radio_soundVolume_set", {})
    assert invalid_radio["success"] is False
    assert invalid_radio["status"] == "error"
    assert vw.settings.sound_channel == original_channel

    played = dispatch(vw, "radio__radio_play", {
        "radioName": "Test FM", "radioValue": "99.0 MHz",
    })
    assert played["success"] is True
    assert played["status"] == "success"


def test_media_playback_has_one_active_source_and_video_stop_is_truthful():
    vw = VehicleWorld()
    assert vw.music.get_is_playing() is True
    assert dispatch(vw, "radio__radio_play", {
        "radioName": "Test FM", "radioValue": "99.0 MHz",
    })["success"]
    assert vw.radio.get_is_playing() is True
    assert vw.music.get_is_playing() is False
    assert vw.settings.sound_channel == "radio"

    assert dispatch(vw, "music__music_local_play", {})["success"]
    assert vw.music.get_is_playing() is True
    assert vw.radio.get_is_playing() is False
    assert vw.settings.sound_channel == "music"

    already_stopped = dispatch(vw, "video__video_play_stop", {})
    assert already_stopped["success"] is True
    assert already_stopped["status"] == "Already stopped"


def test_multi_door_open_is_atomic_when_one_target_is_locked():
    vw = VehicleWorld()
    assert dispatch(vw, "door__carcontrol_carDoor_lock_switch", {
        "switch": True, "position": ["passenger seat"],
    })["success"]
    result = dispatch(vw, "door__carcontrol_carDoor_switch", {
        "action": "open", "position": ["all"],
    })
    assert result["success"] is False
    assert result["updated_doors"] == {}
    assert all(door.status == "closed" for door in vw.door._doors.values())


@pytest.mark.parametrize("operation,valid", [
    ({"op": "add_subgoal", "text": "test"}, False),
    ({"op": "set_long_term", "text": "test"}, False),
    ({"op": "add_subgoal", "text": "test", "ttl_s": 1}, True),
    ({"op": "add_subgoal", "text": "x" * 49, "ttl_s": 1}, True),
    ({"op": "add_subgoal", "text": "x" * 97, "ttl_s": 1}, False),
    ({"op": "set_long_term", "text": "x" * 49, "ttl_s": 1}, True),
    ({"op": "update", "id": "todo-0001"}, False),
    ({"op": "update", "id": "todo-0001", "ttl_s": 1}, True),
    ({"op": "extend", "id": "todo-0001"}, False),
    ({"op": "extend", "id": "todo-0001", "extra_s": 1}, True),
    ({"op": "complete"}, False),
    ({"op": "complete", "id": "todo-0001"}, True),
    ({"op": "cancel", "id": "todo-0001"}, True),
])
def test_todo_schema_requires_operation_specific_fields(operation, valid):
    schema = TodoStore().tool_schema()["function"]["parameters"]
    Draft202012Validator.check_schema(schema)
    assert Draft202012Validator(schema).is_valid({"operations": [operation]}) == valid
