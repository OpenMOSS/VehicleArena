"""Window height/opening and weather warning regressions from Basic195."""
from types import SimpleNamespace

import pytest

from vehiclearena import VehicleWorld
from module.weather import Weather
from tool_utils import dispatch, generate_tools_schema
from constraints import ConstraintEngine
from event_bus import EventBus
from coupling_rules.climate import _register_weather_window_rules


@pytest.mark.parametrize("height,opening", [(0,100), (10,90), (20,80), (80,20), (90,10), (100,0)])
def test_numeric_height_receipt_matches_physical_opening(height, opening):
    vw = VehicleWorld()
    result = dispatch(vw, "window__carcontrol_window_height_set", {
        "position": ["all"], "value": height, "unit": "percentage"})
    assert result["success"]
    assert len(result["updated_states"]) == 6
    for position, state in result["updated_states"].items():
        assert state == {"height_percent": height, "open_percent": opening}
        assert vw.window.get_window_state(position).open_degree == opening


@pytest.mark.parametrize("degree,height", [("max",100), ("high",75), ("medium",50), ("low",25), ("min",0)])
def test_presets_agree_with_equivalent_numeric_height(degree, height):
    window = VehicleWorld().window
    preset = window.carcontrol_window_height_set(["all"], degree=degree)
    numeric = window.carcontrol_window_height_set(["all"], value=height, unit="percentage")
    assert preset["updated_states"] == numeric["updated_states"]


def test_raise_lower_and_switch_report_actual_opening():
    window = VehicleWorld().window
    position = ["driver's seat"]
    window.carcontrol_window_height_set(position, value=80, unit="percentage")
    raised = window.carcontrol_window_height_increase(position, value=10, unit="percentage")
    assert raised["updated_states"][position[0]]["open_percent"] == 10
    lowered = window.carcontrol_window_height_decrease(position, value=20, unit="percentage")
    assert lowered["updated_states"][position[0]]["open_percent"] == 30
    assert window.carcontrol_window_switch(position, True)["updated_states"][position[0]]["open_percent"] == 30
    assert window.carcontrol_window_switch(position, False)["updated_states"][position[0]]["open_percent"] == 0
    assert window.carcontrol_window_switch(position, True)["updated_states"][position[0]]["open_percent"] == 10


def test_llm_schema_explains_height_not_opening():
    VehicleWorld()
    tool = next(t["function"] for t in generate_tools_schema(modules=["window"])
                if t["function"]["name"] == "window__carcontrol_window_height_set")
    assert "not opening" in tool["description"]
    assert "90=10% open" in tool["parameters"]["properties"]["value"]["description"]
    assert tool["parameters"]["properties"]["degree"]["enum"] == ["max", "high", "medium", "low", "min"]


def test_rainy_window_close_through_real_dispatch_has_no_open_warning():
    vw = VehicleWorld()
    vw.externalWorld.weather.condition = Weather.Condition.RAINY
    opened = dispatch(vw, "window__carcontrol_window_switch", {
        "position": ["all"], "switch": True})
    assert opened["success"]
    assert any('weather_window_rain_check' in w for w in opened.get('_coupling_warnings', []))
    closed = dispatch(vw, "window__carcontrol_window_switch", {
        "position": ["all"], "switch": False})
    assert closed["success"]
    assert not any('weather_window_rain_check' in w for w in closed.get('_coupling_warnings', []))
    assert all(s['open_percent'] == 0 for s in closed['updated_states'].values())


@pytest.mark.parametrize("weather", ["rainy", "heavy_rain", "snowy", "heavy_snow", "hail", "sunny"])
@pytest.mark.parametrize("positional", [False, True])
def test_weather_warning_is_only_for_opening(weather, positional):
    engine = ConstraintEngine()
    _register_weather_window_rules(EventBus(), engine)
    world = SimpleNamespace(externalWorld=SimpleNamespace(
        weather=SimpleNamespace(condition=SimpleNamespace(value=weather))))
    checks = {c.name: c.check_fn for c in engine._pre_constraints}
    window_check = checks["weather_window_rain_check"]
    roof_check = checks["weather_sunroof_rain_check"]
    for switch in [True, False]:
        args, kwargs = ((["all"], switch), {}) if positional else ((), {"position": ["all"], "switch": switch})
        result = window_check(None, "carcontrol_window_switch", args, kwargs, world)
        assert bool(result) == (weather != "sunny" and switch)
    for action in ["open", "close", "pause", "Tilt"]:
        args, kwargs = ((action,), {}) if positional else ((), {"action": action})
        result = roof_check(None, "carcontrol_sunroof_switch", args, kwargs, world)
        assert bool(result) == (weather != "sunny" and action in ("open", "Tilt"))
