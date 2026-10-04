from enum import Enum
from typing import Dict, Any, Optional
from utils import api


class TrafficLight:
    """
    Traffic light module for simulating traffic signals at intersections.
    Signal changes can trigger navigation/HUD advisories.
    """

    class Signal(Enum):
        """Traffic light signal enumeration."""
        RED = "red"
        YELLOW = "yellow"
        GREEN = "green"

    def __init__(self):
        self._current_signal = TrafficLight.Signal.GREEN
        self._remaining_seconds = 30
        self._is_flashing = False
        self._intersection_name = ""

    # --- Properties ---
    @property
    def current_signal(self) -> 'TrafficLight.Signal':
        return self._current_signal

    @current_signal.setter
    def current_signal(self, value: 'TrafficLight.Signal'):
        if not isinstance(value, TrafficLight.Signal):
            raise ValueError("Signal must be a TrafficLight.Signal enum value")
        self._current_signal = value

    @property
    def remaining_seconds(self) -> int:
        return self._remaining_seconds

    @remaining_seconds.setter
    def remaining_seconds(self, value: int):
        if not isinstance(value, (int, float)):
            raise TypeError("Remaining seconds must be a number")
        self._remaining_seconds = max(0, int(value))

    @property
    def is_flashing(self) -> bool:
        return self._is_flashing

    @is_flashing.setter
    def is_flashing(self, value: bool):
        if not isinstance(value, bool):
            raise TypeError("is_flashing must be a boolean")
        self._is_flashing = value

    @property
    def intersection_name(self) -> str:
        return self._intersection_name

    @intersection_name.setter
    def intersection_name(self, value: str):
        self._intersection_name = str(value)

    # --- API Methods ---
    @api("trafficLight")
    def traffic_light_change(self, signal: str) -> Dict[str, Any]:
        """
        Change the traffic light signal. This may trigger navigation/HUD advisories
        (e.g., red light → "prepare to stop", green → "safe to proceed").

        Parameters:
        - signal (string): Traffic signal, enum values: "red", "yellow", "green"

        Returns:
        - dict: Contains operation result and updated signal state
        """
        signal_enum = None
        for s in TrafficLight.Signal:
            if s.value == signal:
                signal_enum = s
                break

        if signal_enum is None:
            valid = [s.value for s in TrafficLight.Signal]
            raise ValueError(f"Invalid signal: {signal}. Must be one of: {valid}")

        old_signal = self._current_signal.value
        self.current_signal = signal_enum

        # Default countdown per signal
        default_countdown = {"red": 45, "yellow": 5, "green": 30}
        self._remaining_seconds = default_countdown.get(signal, 30)

        # Emit event
        try:
            from event_bus import Event
            bus = getattr(self, '_event_bus', None)
            if bus:
                bus.publish(Event("trafficlight.changed", "trafficLight", {
                    "old_signal": old_signal,
                    "new_signal": signal,
                    "remaining_seconds": self._remaining_seconds,
                    "intersection_name": self._intersection_name
                }))
        except (ImportError, AttributeError):
            pass

        return {
            "success": True,
            "operation": "change_signal",
            "old_signal": old_signal,
            "new_signal": signal,
            "remaining_seconds": self._remaining_seconds
        }

    @api("trafficLight")
    def traffic_light_get_status(self) -> Dict[str, Any]:
        """
        Get the current traffic light status including signal, countdown, and intersection info.

        Returns:
        - dict: Current traffic light state
        """
        return {
            "success": True,
            "current_signal": self._current_signal.value,
            "remaining_seconds": self._remaining_seconds,
            "is_flashing": self._is_flashing,
            "intersection_name": self._intersection_name
        }

    @api("trafficLight")
    def traffic_light_set_countdown(self, seconds: int) -> Dict[str, Any]:
        """
        Set the remaining countdown time for the current signal.

        Parameters:
        - seconds (int): Remaining seconds for the current signal

        Returns:
        - dict: Contains operation result and updated countdown
        """
        if not isinstance(seconds, (int, float)):
            raise TypeError("Seconds must be a number")

        old_remaining = self._remaining_seconds
        self.remaining_seconds = seconds

        return {
            "success": True,
            "operation": "set_countdown",
            "old_remaining": old_remaining,
            "new_remaining": self._remaining_seconds,
            "current_signal": self._current_signal.value
        }

    @classmethod
    def init1(cls) -> 'TrafficLight':
        """Green light with 30 seconds remaining."""
        instance = cls()
        instance._current_signal = TrafficLight.Signal.GREEN
        instance._remaining_seconds = 30
        instance._intersection_name = "Main St & 1st Ave"
        return instance

    @classmethod
    def init2(cls) -> 'TrafficLight':
        """Red light with 45 seconds remaining."""
        instance = cls()
        instance._current_signal = TrafficLight.Signal.RED
        instance._remaining_seconds = 45
        instance._intersection_name = "Highway Exit 12"
        return instance
