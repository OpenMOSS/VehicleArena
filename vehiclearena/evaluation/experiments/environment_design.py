"""Deterministic climate-aware weather and forward-only day/night design."""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


CONFIG_PATH = Path(__file__).with_name("location_weather_profiles.yaml")
WEATHER_CONDITIONS = (
    "sunny", "cloudy", "rainy", "heavy_rain",
    "foggy", "snowy", "heavy_snow", "hail",
)
WEATHER_TARGET_CONDITIONS = tuple(
    condition for condition in WEATHER_CONDITIONS if condition != "sunny")
WEATHER_PATTERNS = ("onset", "temporary", "evolve")
WEATHER_FAMILY = {
    "sunny": "fair",
    "cloudy": "fair",
    "rainy": "rain",
    "heavy_rain": "rain",
    "foggy": "fog",
    "snowy": "snow",
    "heavy_snow": "snow",
    "hail": "hail",
}
WEATHER_TRANSITION_EDGES = frozenset({
    ("sunny", "cloudy"),
    ("cloudy", "sunny"),
    ("cloudy", "foggy"),
    ("foggy", "cloudy"),
    ("cloudy", "rainy"),
    ("rainy", "cloudy"),
    ("rainy", "heavy_rain"),
    ("heavy_rain", "rainy"),
    ("cloudy", "snowy"),
    ("snowy", "cloudy"),
    ("snowy", "heavy_snow"),
    ("heavy_snow", "snowy"),
    ("rainy", "hail"),
    ("hail", "rainy"),
    ("heavy_rain", "hail"),
    ("hail", "heavy_rain"),
})

# A 20-slot weighted cycle keeps ordinary rain dominant while giving the new
# severe states explicit, reproducible coverage. Slots 10--19 contain every
# target at least once, which also covers the ten MultiLLM weather treatments.
WEATHER_PRIMARY_CYCLE = (
    "rainy", "heavy_rain", "cloudy", "foggy", "snowy",
    "heavy_snow", "rainy", "heavy_rain", "cloudy", "foggy",
    "hail", "snowy", "rainy", "heavy_rain", "cloudy",
    "foggy", "heavy_snow", "rainy", "heavy_rain", "snowy",
)

DAYNIGHT_PERIODS = (
    "dawn", "morning", "noon", "afternoon", "dusk", "night")
DAYNIGHT_PATTERNS = ("single_step", "forward_two_step", "light_boundary")
DAYNIGHT_EDGES = frozenset(
    (period, DAYNIGHT_PERIODS[(index + 1) % len(DAYNIGHT_PERIODS)])
    for index, period in enumerate(DAYNIGHT_PERIODS))

WEATHER_DESCRIPTION = {
    "sunny": "Clouds clear and dry sunny conditions return.",
    "cloudy": "Cloud cover increases without precipitation.",
    "rainy": "Rain begins and the road becomes wet.",
    "heavy_rain": "Rain intensifies into a heavy downpour.",
    "foggy": "Dense fog forms and visibility drops sharply.",
    "snowy": "Snow begins and the road may become icy.",
    "heavy_snow": "Snow intensifies and visibility drops sharply.",
    "hail": "A short hailstorm begins.",
}
WEATHER_HUMIDITY = {
    "sunny": 45, "cloudy": 65, "rainy": 84, "heavy_rain": 94,
    "foggy": 95, "snowy": 83, "heavy_snow": 91, "hail": 88,
}
WEATHER_WIND_KMH = {
    "sunny": 5, "cloudy": 7, "rainy": 12, "heavy_rain": 22,
    "foggy": 3, "snowy": 11, "heavy_snow": 24, "hail": 25,
}
WEATHER_INTENSITY = {
    "sunny": (0.0, 0.0, 0.0),
    "cloudy": (0.0, 0.0, 0.05),
    "rainy": (0.5, 0.0, 0.1),
    "heavy_rain": (0.9, 0.0, 0.2),
    "foggy": (0.0, 0.0, 0.9),
    "snowy": (0.0, 0.4, 0.1),
    "heavy_snow": (0.0, 0.9, 0.25),
    "hail": (0.8, 0.3, 0.1),
}
TEMPERATURE_OFFSETS = {
    "sunny": 2, "cloudy": 0, "rainy": -1, "heavy_rain": -2,
    "foggy": -1, "snowy": -1, "heavy_snow": -3, "hail": -2,
}
DAYNIGHT_DESCRIPTION = {
    "dawn": "Dawn begins and ambient light gradually increases.",
    "morning": "Morning daylight becomes established.",
    "noon": "The local time advances to bright noon daylight.",
    "afternoon": "The local time advances into afternoon daylight.",
    "dusk": "Dusk begins and ambient light falls.",
    "night": "Night falls and the road becomes dark.",
}


def _stable_number(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:8], 16)


@lru_cache(maxsize=1)
def load_location_weather_profiles() -> dict[str, Any]:
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if payload.get("schema") != "vehiclearena-location-weather-v1":
        raise ValueError("unsupported location weather profile schema")
    profiles = payload.get("profiles") or {}
    locations = payload.get("locations") or {}
    for location, profile_id in locations.items():
        if profile_id not in profiles:
            raise ValueError(
                f"{location} references unknown climate profile {profile_id}")
    for profile_id, profile in profiles.items():
        eligibility = profile.get("eligibility") or {}
        if set(eligibility) != set(WEATHER_CONDITIONS):
            raise ValueError(
                f"{profile_id} does not classify all weather conditions")
        if any(value not in {"common", "plausible", "forbidden"}
               for value in eligibility.values()):
            raise ValueError(f"{profile_id} has invalid eligibility")
    return payload


def location_weather_profile(network: str) -> tuple[str, Mapping[str, Any]]:
    payload = load_location_weather_profiles()
    profile_id = (payload.get("locations") or {}).get(network)
    if profile_id is None:
        raise ValueError(f"missing climate profile for {network}")
    return str(profile_id), payload["profiles"][profile_id]


def weather_condition_allowed(network: str, condition: str) -> bool:
    _, profile = location_weather_profile(network)
    return profile["eligibility"].get(condition) != "forbidden"


def weather_condition_month_allowed(
    network: str, condition: str, month: int,
) -> bool:
    _, profile = location_weather_profile(network)
    return (
        weather_condition_allowed(network, condition)
        and int(month) in profile["months"][WEATHER_FAMILY[condition]]
    )


def select_weather_primary(network: str, selection_index: int,
                           scene_id: str) -> str:
    """Choose a weighted target, falling back only within the local climate."""
    preferred = WEATHER_PRIMARY_CYCLE[selection_index % len(
        WEATHER_PRIMARY_CYCLE)]
    if weather_condition_allowed(network, preferred):
        return preferred

    _, profile = location_weather_profile(network)
    eligibility = profile["eligibility"]
    common = [condition for condition in WEATHER_TARGET_CONDITIONS
              if eligibility[condition] == "common"]
    candidates = common or [
        condition for condition in WEATHER_TARGET_CONDITIONS
        if eligibility[condition] == "plausible"]
    if not candidates:
        raise ValueError(f"{network} has no active weather condition")
    return candidates[
        _stable_number(f"{scene_id}:local-weather-fallback")
        % len(candidates)]


def weather_sequence(network: str, primary: str,
                     pattern: str) -> tuple[str, ...]:
    if pattern not in WEATHER_PATTERNS:
        raise ValueError(f"unsupported weather pattern {pattern!r}")
    sequences = {
        "cloudy": {
            "onset": ("sunny", "cloudy"),
            "temporary": ("sunny", "cloudy", "sunny"),
            "evolve": ("sunny", "cloudy", "rainy"),
        },
        "rainy": {
            "onset": ("cloudy", "rainy"),
            "temporary": ("cloudy", "rainy", "cloudy"),
            "evolve": ("cloudy", "rainy", "heavy_rain"),
        },
        "heavy_rain": {
            "onset": ("rainy", "heavy_rain"),
            "temporary": ("rainy", "heavy_rain", "rainy"),
            "evolve": ("cloudy", "rainy", "heavy_rain"),
        },
        "foggy": {
            "onset": ("cloudy", "foggy"),
            "temporary": ("cloudy", "foggy", "cloudy"),
            "evolve": ("sunny", "cloudy", "foggy"),
        },
        "snowy": {
            "onset": ("cloudy", "snowy"),
            "temporary": ("cloudy", "snowy", "cloudy"),
            "evolve": ("sunny", "cloudy", "snowy"),
        },
        "heavy_snow": {
            "onset": ("snowy", "heavy_snow"),
            "temporary": ("snowy", "heavy_snow", "snowy"),
            "evolve": ("cloudy", "snowy", "heavy_snow"),
        },
        "hail": {
            "onset": ("rainy", "hail"),
            "temporary": ("rainy", "hail", "rainy"),
            "evolve": ("cloudy", "rainy", "hail"),
        },
    }
    sequence = sequences[primary][pattern]
    if any(not weather_condition_allowed(network, condition)
           for condition in sequence):
        raise ValueError(
            f"weather sequence {sequence} is incompatible with {network}")
    if any(edge not in WEATHER_TRANSITION_EDGES
           for edge in zip(sequence, sequence[1:])):
        raise ValueError(f"weather sequence contains a non-adjacent edge: {sequence}")
    return sequence


def climate_month(network: str, conditions: Sequence[str],
                  scene_id: str) -> int:
    """Return a reproducible month supported by every state in a sequence."""
    _, profile = location_weather_profile(network)
    months = set(range(1, 13))
    for condition in conditions:
        family = WEATHER_FAMILY[condition]
        months &= set(profile["months"][family])
    if not months:
        raise ValueError(
            f"no common climate month for {network}: {tuple(conditions)}")
    ordered = sorted(months)
    return ordered[_stable_number(f"{scene_id}:climate-month") % len(ordered)]


def weather_parameters(network: str, condition: str,
                       sequence: Sequence[str], scene_id: str) -> dict[str, Any]:
    """Build locally plausible, internally continuous keyframe parameters."""
    _, profile = location_weather_profile(network)
    if any(item in {"snowy", "heavy_snow"} for item in sequence):
        system_family = "snow"
    elif "hail" in sequence:
        system_family = "hail"
    elif any(item in {"rainy", "heavy_rain"} for item in sequence):
        system_family = "rain"
    elif "foggy" in sequence:
        system_family = "fog"
    else:
        system_family = "fair"
    lower, upper = profile["temperature_c"][system_family]
    base = int(lower) + _stable_number(
        f"{scene_id}:{system_family}:temperature") % (
            int(upper) - int(lower) + 1)
    temperature = base + TEMPERATURE_OFFSETS[condition]
    if condition in {"snowy", "heavy_snow"}:
        temperature = min(0, temperature)
    rain, snow, fog = WEATHER_INTENSITY[condition]
    return {
        "temperature": temperature,
        "humidity": WEATHER_HUMIDITY[condition],
        "wind_speed": WEATHER_WIND_KMH[condition],
        "description": WEATHER_DESCRIPTION[condition],
        "rain_intensity": rain,
        "snow_intensity": snow,
        "fog_density": fog,
    }


def daynight_sequence(selection_index: int, pattern: str) -> tuple[str, ...]:
    if pattern not in DAYNIGHT_PATTERNS:
        raise ValueError(f"unsupported day/night pattern {pattern!r}")
    if pattern == "single_step":
        start = selection_index % len(DAYNIGHT_PERIODS)
        return (
            DAYNIGHT_PERIODS[start],
            DAYNIGHT_PERIODS[(start + 1) % len(DAYNIGHT_PERIODS)],
        )
    if pattern == "forward_two_step":
        start = selection_index % len(DAYNIGHT_PERIODS)
    else:
        # Each sequence crosses a safety-relevant light boundary, including
        # the valid night -> dawn -> morning progression across midnight.
        start = (3 + selection_index % 3) % len(DAYNIGHT_PERIODS)
    return tuple(
        DAYNIGHT_PERIODS[(start + offset) % len(DAYNIGHT_PERIODS)]
        for offset in range(3))


def daynight_sequence_is_forward(periods: Sequence[str]) -> bool:
    return bool(periods) and all(
        edge in DAYNIGHT_EDGES for edge in zip(periods, periods[1:]))
