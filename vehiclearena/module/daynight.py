from enum import Enum
from typing import Dict, Any, Optional
from utils import api
from module.base_module import BaseModule
from registry import register_module


@register_module("dayNight", description="Day/night cycle and lighting related", category="environment", is_external=True)
class DayNight(BaseModule):
    """
    Day/Night cycle module for simulating ambient lighting conditions.
    Daylight changes can trigger automatic headlight and position light advisories.
    """

    class TimePeriod(Enum):
        """Time of day period enumeration."""
        DAWN = "dawn"
        MORNING = "morning"
        NOON = "noon"
        AFTERNOON = "afternoon"
        DUSK = "dusk"
        NIGHT = "night"

    # Daylight level mapping for each time period (0-100)
    _DAYLIGHT_MAP = {
        TimePeriod.DAWN: 25,
        TimePeriod.MORNING: 70,
        TimePeriod.NOON: 100,
        TimePeriod.AFTERNOON: 85,
        TimePeriod.DUSK: 30,
        TimePeriod.NIGHT: 5,
    }

    # Time period progression order
    _PERIOD_ORDER = [
        TimePeriod.DAWN,
        TimePeriod.MORNING,
        TimePeriod.NOON,
        TimePeriod.AFTERNOON,
        TimePeriod.DUSK,
        TimePeriod.NIGHT,
    ]

    def __init__(self):
        self._time_of_day = DayNight.TimePeriod.NOON
        self._daylight_level = 100  # 0-100
        self._is_dark = False

    # --- Properties ---
    @property
    def time_of_day(self) -> 'DayNight.TimePeriod':
        return self._time_of_day

    @time_of_day.setter
    def time_of_day(self, value: 'DayNight.TimePeriod'):
        if not isinstance(value, DayNight.TimePeriod):
            raise ValueError("time_of_day must be a DayNight.TimePeriod enum value")
        self._time_of_day = value

    @property
    def daylight_level(self) -> int:
        return self._daylight_level

    @daylight_level.setter
    def daylight_level(self, value: int):
        if not isinstance(value, (int, float)):
            raise TypeError("Daylight level must be a number")
        self._daylight_level = max(0, min(100, int(value)))

    @property
    def is_dark(self) -> bool:
        return self._is_dark

    @is_dark.setter
    def is_dark(self, value: bool):
        if not isinstance(value, bool):
            raise TypeError("is_dark must be a boolean")
        self._is_dark = value

    # --- API Methods ---
    @api("dayNight")
    def daynight_set_time(self, time_of_day: str) -> Dict[str, Any]:
        """
        Set the current time period. This automatically adjusts daylight level and
        may trigger headlight advisories (e.g., turn on lights at dusk/night).

        Parameters:
        - time_of_day (string): Time period, enum values: "dawn", "morning", "noon",
          "afternoon", "dusk", "night"

        Returns:
        - dict: Contains operation result and updated day/night state
        """
        period_enum = None
        for p in DayNight.TimePeriod:
            if p.value == time_of_day:
                period_enum = p
                break

        if period_enum is None:
            valid = [p.value for p in DayNight.TimePeriod]
            raise ValueError(f"Invalid time_of_day: {time_of_day}. Must be one of: {valid}")

        old_period = self._time_of_day.value
        old_daylight = self._daylight_level
        self.time_of_day = period_enum
        self._daylight_level = self._DAYLIGHT_MAP.get(period_enum, 50)
        self._is_dark = self._daylight_level < 30

        # Emit event
        try:
            from event_bus import Event
            bus = getattr(self, '_event_bus', None)
            if bus:
                bus.publish(Event("daynight.changed", "dayNight", {
                    "old_period": old_period,
                    "new_period": time_of_day,
                    "old_daylight": old_daylight,
                    "new_daylight": self._daylight_level,
                    "is_dark": self._is_dark
                }))
        except (ImportError, AttributeError):
            pass

        return {
            "success": True,
            "operation": "set_time",
            "old_period": old_period,
            "new_period": time_of_day,
            "daylight_level": self._daylight_level,
            "is_dark": self._is_dark
        }

    @api("dayNight")
    def daynight_get_status(self) -> Dict[str, Any]:
        """
        Get the current day/night status including time period, daylight level, and darkness flag.

        Returns:
        - dict: Current day/night state
        """
        return {
            "success": True,
            "time_of_day": self._time_of_day.value,
            "daylight_level": self._daylight_level,
            "is_dark": self._is_dark
        }

    @api("dayNight")
    def daynight_advance_time(self) -> Dict[str, Any]:
        """
        Advance to the next time period in the day/night cycle
        (dawn → morning → noon → afternoon → dusk → night → dawn).

        Returns:
        - dict: Contains operation result and updated day/night state
        """
        current_idx = self._PERIOD_ORDER.index(self._time_of_day)
        next_idx = (current_idx + 1) % len(self._PERIOD_ORDER)
        next_period = self._PERIOD_ORDER[next_idx]

        # Use set_time to handle all syncing and event emission
        return self.daynight_set_time(next_period.value)

    @classmethod
    def init1(cls) -> 'DayNight':
        """Noon, bright daylight (100)."""
        instance = cls()
        instance._time_of_day = DayNight.TimePeriod.NOON
        instance._daylight_level = 100
        instance._is_dark = False
        return instance

    @classmethod
    def init2(cls) -> 'DayNight':
        """Night, very low light (5)."""
        instance = cls()
        instance._time_of_day = DayNight.TimePeriod.NIGHT
        instance._daylight_level = 5
        instance._is_dark = True
        return instance

    @classmethod
    def init3(cls) -> 'DayNight':
        """Dusk, transitional light (30)."""
        instance = cls()
        instance._time_of_day = DayNight.TimePeriod.DUSK
        instance._daylight_level = 30
        instance._is_dark = True
        return instance
