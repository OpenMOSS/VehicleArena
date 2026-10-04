"""Shared weather equipment rules for scoring, NPCs, and driver guidance."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Dict, Mapping, Optional


SUPPORTED_WEATHER_CONDITIONS = frozenset({
    "sunny", "cloudy", "rainy", "heavy_rain",
    "foggy", "snowy", "heavy_snow", "hail",
})
CLEAR_WEATHER_CONDITIONS = frozenset({"sunny", "cloudy"})
RAIN_CONDITIONS = frozenset({"rainy", "heavy_rain"})
SNOW_CONDITIONS = frozenset({"snowy", "heavy_snow"})
FOG_VISIBILITY_CONDITIONS = frozenset({"foggy", "heavy_snow"})
WIPER_CONDITIONS = RAIN_CONDITIONS | SNOW_CONDITIONS
OPENING_CLOSE_CONDITIONS = WIPER_CONDITIONS | frozenset({"hail"})
WEATHER_SAFETY_MODULES = frozenset({
    "wiper", "fogLight", "lowBeamHeadlight", "highBeamHeadlight",
    "positionLight", "window", "sunroof",
})

_WEATHER_CONDITION_ALIASES = {
    "clear": "sunny",
    "fog": "foggy",
}


def normalize_weather_condition(condition: str) -> str:
    """Return one of the eight weather conditions accepted by tasks."""
    normalized = str(condition).strip().lower()
    normalized = _WEATHER_CONDITION_ALIASES.get(normalized, normalized)
    if normalized not in SUPPORTED_WEATHER_CONDITIONS:
        raise ValueError(f"unsupported weather condition: {condition!r}")
    return normalized


@dataclass(frozen=True)
class WeatherSafetyEquipmentState:
    """Known weather-controlled state for one evaluated reference NPC.

    None means weather rules have not taken ownership of that setting.
    This is important for sunny weather: it performs no blanket reset.
    """

    front_wiper_on: Optional[bool] = None
    front_fog_on: Optional[bool] = None
    rear_fog_on: Optional[bool] = None
    low_beam_on: Optional[bool] = None
    high_beam_on: Optional[bool] = None
    position_light_on: Optional[bool] = None
    all_windows_closed: Optional[bool] = None
    sunroof_closed: Optional[bool] = None

    def to_dict(self) -> Dict[str, Optional[bool]]:
        return asdict(self)


def apply_weather_safety_transition(
    previous_condition: str,
    condition: str,
    current_state: Optional[
        WeatherSafetyEquipmentState | Mapping[str, Any]
    ] = None,
) -> tuple[
    str,
    str,
    WeatherSafetyEquipmentState,
    Dict[str, bool],
]:
    """Apply only the settings owned by one weather transition.

    Cleanup is deliberately selective:

    * leaving fog/heavy snow turns off fog and position lights when the next
      weather does not need them, but retains low beam;
    * leaving rain/snow turns off the front wiper when the next weather does
      not need it;
    * windows and sunroof are never reopened automatically.
    """
    previous = normalize_weather_condition(previous_condition)
    normalized = normalize_weather_condition(condition)
    if isinstance(current_state, WeatherSafetyEquipmentState):
        state = current_state
    elif current_state:
        allowed = {
            key: value for key, value in current_state.items()
            if key in WeatherSafetyEquipmentState.__dataclass_fields__
        }
        state = WeatherSafetyEquipmentState(**allowed)
    else:
        state = WeatherSafetyEquipmentState()

    updates: Dict[str, bool] = {}
    if (previous in FOG_VISIBILITY_CONDITIONS
            and normalized not in FOG_VISIBILITY_CONDITIONS):
        updates.update({
            "front_fog_on": False,
            "rear_fog_on": False,
            "position_light_on": False,
        })
    if previous in WIPER_CONDITIONS and normalized not in WIPER_CONDITIONS:
        updates["front_wiper_on"] = False

    if normalized == "foggy":
        updates.update({
            "low_beam_on": True,
            "high_beam_on": False,
            "front_fog_on": True,
            "rear_fog_on": True,
            "position_light_on": True,
        })
    elif normalized in RAIN_CONDITIONS:
        updates.update({
            "front_wiper_on": True,
            "all_windows_closed": True,
            "sunroof_closed": True,
        })
    elif normalized == "snowy":
        updates.update({
            "front_wiper_on": True,
            "all_windows_closed": True,
            "sunroof_closed": True,
            "low_beam_on": True,
        })
    elif normalized == "heavy_snow":
        updates.update({
            "front_wiper_on": True,
            "all_windows_closed": True,
            "sunroof_closed": True,
            "low_beam_on": True,
            "high_beam_on": False,
            "front_fog_on": True,
            "rear_fog_on": True,
            "position_light_on": True,
        })
    elif normalized == "hail":
        updates.update({
            "all_windows_closed": True,
            "sunroof_closed": True,
        })

    return previous, normalized, replace(state, **updates), updates


def weather_score_requirements(
    previous_condition: str,
    condition: str,
) -> Dict[str, Any]:
    """Expose verifiable requirements for one weather transition."""
    previous = normalize_weather_condition(previous_condition)
    normalized = normalize_weather_condition(condition)
    return {
        "previous_condition": previous,
        "condition": normalized,
        "fog_visibility_lights_required": (
            normalized in FOG_VISIBILITY_CONDITIONS),
        "low_beam_required": normalized == "snowy",
        "high_beam_must_be_off": (
            normalized in FOG_VISIBILITY_CONDITIONS),
        "front_wiper_required": normalized in WIPER_CONDITIONS,
        "openings_must_be_closed": normalized in OPENING_CLOSE_CONDITIONS,
        "fog_and_position_lights_must_be_off": (
            previous in FOG_VISIBILITY_CONDITIONS
            and normalized not in FOG_VISIBILITY_CONDITIONS),
        "front_wiper_must_be_off": (
            previous in WIPER_CONDITIONS
            and normalized not in WIPER_CONDITIONS),
    }


def weather_equipment_action_lines(
    updates: Mapping[str, bool], *, dark_period: bool = False,
) -> list[str]:
    """Translate shared state updates into executable public module calls.

    The same calls feed driver guidance and the legacy cabin checkpoint;
    neither may introduce a second weather policy. Night lighting takes
    precedence over weather cleanup, as in the driving process scorer.
    """
    templates = {
        "front_wiper_on": "wiper.carcontrol_wiperBlade_switch({value}, 'front')",
        "front_fog_on": "fogLight.carcontrol_fogLight_switch({value}, 'front')",
        "rear_fog_on": "fogLight.carcontrol_fogLight_switch({value}, 'rear')",
        "low_beam_on": "lowBeamHeadlight.switch('{mode}')",
        "high_beam_on": "highBeamHeadlight.switch({value})",
        "position_light_on": "positionLight.carcontrol_positionLight_switch({value})",
        "all_windows_closed": "window.carcontrol_window_switch(['all'], {opened})",
        "sunroof_closed": "sunroof.carcontrol_sunroof_switch('{opening_mode}')",
    }
    actions = []
    for field, value in updates.items():
        if field == "position_light_on" and not value and dark_period:
            continue
        actions.append("vw." + templates[field].format(
            value=value, mode="on" if value else "off", opened=not value,
            opening_mode="close" if value else "open"))
    return actions


def render_weather_transition_rules() -> str:
    """Render installed-equipment instructions from the scoring/NPC policy."""
    lines = [
        "## Required external equipment",
        "",
        "Apply the entering-weather settings and any applicable cleanup below. "
        "Only operate installed equipment; an already satisfied state needs no "
        "repeat action. These instructions use the same rules as scoring and NPCs.",
    ]

    def section(title, updates):
        lines.extend(["", title, ""])
        for action in weather_equipment_action_lines(updates):
            lines.append(f"- `{action.removeprefix('vw.')}`")

    for condition in sorted(SUPPORTED_WEATHER_CONDITIONS):
        _, _, _, updates = apply_weather_safety_transition("sunny", condition)
        if updates:
            section(f"### Entering `{condition}`", updates)
    for conditions, example in (
        (WIPER_CONDITIONS, "rainy"),
        (FOG_VISIBILITY_CONDITIONS, "foggy"),
    ):
        names = ", ".join(f"`{item}`" for item in sorted(conditions))
        _, _, _, updates = apply_weather_safety_transition(example, "sunny")
        section(f"### Leaving {names} for a condition outside that set", updates)
    lines.extend([
        "",
        "During dusk/night/dawn, keep positionLight ON despite weather cleanup; "
        "the day/night lighting requirement takes precedence.",
        "Weather cleanup does not turn lowBeamHeadlight off.",
        "Weather cleanup does not reopen window or sunroof.",
        "Sunny/cloudy adds no entering-weather action; apply only relevant cleanup.",
    ])
    return "\n".join(lines)
