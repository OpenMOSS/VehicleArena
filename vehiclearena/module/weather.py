from enum import Enum
from typing import Dict, Any, Optional
from utils import api
from module.base_module import BaseModule
from registry import register_module


@register_module("weather", description="Weather condition and environment related", category="environment", is_external=True)
class Weather(BaseModule):
    """
    Weather module for simulating external weather conditions.
    Weather changes can trigger vehicle module reactions such as wipers, fog lights, and AC.
    """

    class Condition(Enum):
        """Weather condition enumeration."""
        SUNNY = "sunny"
        CLOUDY = "cloudy"
        RAINY = "rainy"
        HEAVY_RAIN = "heavy_rain"
        FOGGY = "foggy"
        SNOWY = "snowy"
        HEAVY_SNOW = "heavy_snow"
        HAIL = "hail"

    # Visibility mapping for weather conditions (meters)
    _VISIBILITY_MAP = {
        Condition.SUNNY: 10000,
        Condition.CLOUDY: 8000,
        Condition.RAINY: 3000,
        Condition.HEAVY_RAIN: 1000,
        Condition.FOGGY: 50,
        Condition.SNOWY: 1500,
        Condition.HEAVY_SNOW: 300,
        Condition.HAIL: 800,
    }

    # Road surface mapping for weather conditions
    _ROAD_SURFACE_MAP = {
        Condition.SUNNY: "dry",
        Condition.CLOUDY: "dry",
        Condition.RAINY: "wet",
        Condition.HEAVY_RAIN: "flooded",
        Condition.FOGGY: "wet",
        Condition.SNOWY: "icy",
        Condition.HEAVY_SNOW: "icy",
        Condition.HAIL: "wet",
    }

    # Intensity defaults keyed by condition
    _RAIN_INTENSITY_MAP = {
        Condition.SUNNY: 0.0,
        Condition.CLOUDY: 0.0,
        Condition.RAINY: 0.5,
        Condition.HEAVY_RAIN: 0.9,
        Condition.FOGGY: 0.0,
        Condition.SNOWY: 0.0,
        Condition.HEAVY_SNOW: 0.0,
        Condition.HAIL: 0.8,
    }
    _SNOW_INTENSITY_MAP = {
        Condition.SUNNY: 0.0,
        Condition.CLOUDY: 0.0,
        Condition.RAINY: 0.0,
        Condition.HEAVY_RAIN: 0.0,
        Condition.FOGGY: 0.0,
        Condition.SNOWY: 0.4,
        Condition.HEAVY_SNOW: 0.9,
        Condition.HAIL: 0.3,
    }
    _FOG_DENSITY_MAP = {
        Condition.SUNNY: 0.0,
        Condition.CLOUDY: 0.05,
        Condition.RAINY: 0.1,
        Condition.HEAVY_RAIN: 0.2,
        Condition.FOGGY: 0.9,
        Condition.SNOWY: 0.10,
        Condition.HEAVY_SNOW: 0.25,
        Condition.HAIL: 0.1,
    }

    def __init__(self):
        self._condition = Weather.Condition.SUNNY
        self._temperature = 25  # Celsius
        self._humidity = 50     # Percentage (0-100)
        self._visibility = 10000  # Meters
        self._wind_speed = 5    # km/h
        # Continuous intensity values (Level 2)
        self._rain_intensity = 0.0   # 0.0-1.0
        self._snow_intensity = 0.0   # 0.0-1.0
        self._fog_density = 0.0      # 0.0-1.0

    # --- Properties ---
    @property
    def condition(self) -> 'Weather.Condition':
        return self._condition

    @condition.setter
    def condition(self, value: 'Weather.Condition'):
        if not isinstance(value, Weather.Condition):
            raise ValueError("Condition must be a Weather.Condition enum value")
        self._condition = value

    @property
    def temperature(self) -> int:
        return self._temperature

    @temperature.setter
    def temperature(self, value: int):
        if not isinstance(value, (int, float)):
            raise TypeError("Temperature must be a number")
        self._temperature = int(value)

    @property
    def humidity(self) -> int:
        return self._humidity

    @humidity.setter
    def humidity(self, value: int):
        if not isinstance(value, (int, float)):
            raise TypeError("Humidity must be a number")
        self._humidity = max(0, min(100, int(value)))

    @property
    def visibility(self) -> int:
        return self._visibility

    @visibility.setter
    def visibility(self, value: int):
        if not isinstance(value, (int, float)):
            raise TypeError("Visibility must be a number")
        self._visibility = max(0, min(10000, int(value)))

    @property
    def wind_speed(self) -> int:
        return self._wind_speed

    @wind_speed.setter
    def wind_speed(self, value: int):
        if not isinstance(value, (int, float)):
            raise TypeError("Wind speed must be a number")
        self._wind_speed = max(0, int(value))

    @property
    def rain_intensity(self) -> float:
        return self._rain_intensity

    @rain_intensity.setter
    def rain_intensity(self, value: float):
        self._rain_intensity = max(0.0, min(1.0, float(value)))

    @property
    def snow_intensity(self) -> float:
        return self._snow_intensity

    @snow_intensity.setter
    def snow_intensity(self, value: float):
        self._snow_intensity = max(0.0, min(1.0, float(value)))

    @property
    def fog_density(self) -> float:
        return self._fog_density

    @fog_density.setter
    def fog_density(self, value: float):
        self._fog_density = max(0.0, min(1.0, float(value)))

    # --- API Methods ---
    @api("weather")
    def weather_set_condition(self, condition: str) -> Dict[str, Any]:
        """
        Set the current weather condition. This affects visibility, road surface, and may
        trigger vehicle module advisories (e.g., wipers for rain, fog lights for fog).

        Parameters:
        - condition (string): Weather condition, enum values: "sunny", "cloudy", "rainy",
          "heavy_rain", "foggy", "snowy", "heavy_snow", "hail"

        Returns:
        - dict: Contains operation result and updated weather state
        """
        # Validate condition
        condition_enum = None
        for c in Weather.Condition:
            if c.value == condition:
                condition_enum = c
                break

        if condition_enum is None:
            valid = [c.value for c in Weather.Condition]
            raise ValueError(f"Invalid condition: {condition}. Must be one of: {valid}")

        old_condition = self._condition.value
        self.condition = condition_enum

        # Auto-adjust visibility, intensities, and sync to Environment
        self._visibility = self._VISIBILITY_MAP.get(condition_enum, 10000)
        self._rain_intensity = self._RAIN_INTENSITY_MAP.get(condition_enum, 0.0)
        self._snow_intensity = self._SNOW_INTENSITY_MAP.get(condition_enum, 0.0)
        self._fog_density = self._FOG_DENSITY_MAP.get(condition_enum, 0.0)
        road_surface = self._ROAD_SURFACE_MAP.get(condition_enum, "dry")

        # Emit weather.changed event
        try:
            from event_bus import Event
            bus = getattr(self, '_event_bus', None)
            if bus:
                bus.publish(Event("weather.changed", "weather", {
                    "old_condition": old_condition,
                    "new_condition": condition,
                    "visibility": self._visibility,
                    "road_surface": road_surface
                }))
        except (ImportError, AttributeError):
            pass

        return {
            "success": True,
            "operation": "set_condition",
            "old_condition": old_condition,
            "new_condition": condition,
            "current_state": {
                "condition": self._condition.value,
                "visibility": self._visibility,
                "road_surface": road_surface,
                "temperature": self._temperature
            }
        }

    @api("weather")
    def weather_get_condition(self) -> Dict[str, Any]:
        """
        Get the current weather condition and related information.

        Returns:
        - dict: Current weather state including condition, temperature, visibility, humidity, wind speed
        """
        return {
            "success": True,
            "condition": self._condition.value,
            "temperature": self._temperature,
            "humidity": self._humidity,
            "visibility": self._visibility,
            "wind_speed": self._wind_speed,
            "rain_intensity": self._rain_intensity,
            "snow_intensity": self._snow_intensity,
            "fog_density": self._fog_density,
        }

    @api("weather")
    def weather_set_temperature(self, value: int) -> Dict[str, Any]:
        """
        Set the outdoor temperature. This syncs to the Environment temperature and may
        trigger AC advisories (e.g., cooling when too hot, heating when too cold).

        Parameters:
        - value (int): Temperature in Celsius

        Returns:
        - dict: Contains operation result and updated temperature
        """
        old_temp = self._temperature
        self.temperature = value

        # Emit temperature event
        try:
            from event_bus import Event
            bus = getattr(self, '_event_bus', None)
            if bus:
                bus.publish(Event("weather.temperature_changed", "weather", {
                    "old_temperature": old_temp,
                    "new_temperature": value
                }))
        except (ImportError, AttributeError):
            pass

        return {
            "success": True,
            "operation": "set_temperature",
            "old_temperature": old_temp,
            "new_temperature": value
        }

    @classmethod
    def init1(cls) -> 'Weather':
        """Sunny day, 25C, clear visibility."""
        instance = cls()
        instance._condition = Weather.Condition.SUNNY
        instance._temperature = 25
        instance._humidity = 40
        instance._visibility = 10000
        instance._wind_speed = 5
        return instance

    @classmethod
    def init2(cls) -> 'Weather':
        """Rainy day, 18C, reduced visibility."""
        instance = cls()
        instance._condition = Weather.Condition.RAINY
        instance._temperature = 18
        instance._humidity = 85
        instance._visibility = 3000
        instance._wind_speed = 15
        return instance

    @classmethod
    def init3(cls) -> 'Weather':
        """Dense fog, 10C, very low visibility."""
        instance = cls()
        instance._condition = Weather.Condition.FOGGY
        instance._temperature = 10
        instance._humidity = 95
        instance._visibility = 50
        instance._wind_speed = 2
        return instance

    @classmethod
    def init4(cls) -> 'Weather':
        """Light snow, -5C, reduced visibility."""
        instance = cls()
        instance._condition = Weather.Condition.SNOWY
        instance._temperature = -5
        instance._humidity = 70
        instance._visibility = 1500
        instance._wind_speed = 20
        instance._snow_intensity = 0.4
        instance._fog_density = 0.1
        return instance
