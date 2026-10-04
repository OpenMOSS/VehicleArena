"""Typed time-varying inputs shared by VehicleArena scenarios."""

from dataclasses import dataclass


@dataclass
class WeatherKeyframe:
    """Weather state authored at a simulation minute."""

    t: int
    condition: str
    temperature: int = 25
    humidity: int = 50
    wind_speed: int = 5
    description: str = ""
    rain_intensity: float = 0.0
    snow_intensity: float = 0.0
    fog_density: float = 0.0
    transition_minutes: int = 0


@dataclass
class DayNightKeyframe:
    """Day/night state authored at a simulation minute."""

    t: int
    period: str
    description: str = ""
    transition_minutes: int = 0
