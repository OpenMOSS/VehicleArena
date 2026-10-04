from enum import Enum
from typing import Dict, Any, Optional
from utils import api
from module.base_module import BaseModule
from registry import register_module


@register_module("speedLimit", description="Speed limit zone related", category="environment", is_external=True)
class SpeedLimit(BaseModule):
    """
    Speed limit module for simulating speed limit zones.
    Speed limit changes can trigger navigation/HUD display updates.
    """

    class ZoneType(Enum):
        """Speed limit zone type enumeration."""
        SCHOOL = "school"
        RESIDENTIAL = "residential"
        HIGHWAY = "highway"
        CONSTRUCTION = "construction"
        URBAN = "urban"

    def __init__(self):
        self._current_limit = 120  # km/h
        self._zone_type = SpeedLimit.ZoneType.HIGHWAY
        self._is_active = True

    # --- Properties ---
    @property
    def current_limit(self) -> int:
        return self._current_limit

    @current_limit.setter
    def current_limit(self, value: int):
        if not isinstance(value, (int, float)):
            raise TypeError("Speed limit must be a number")
        self._current_limit = max(0, int(value))

    @property
    def zone_type(self) -> 'SpeedLimit.ZoneType':
        return self._zone_type

    @zone_type.setter
    def zone_type(self, value: 'SpeedLimit.ZoneType'):
        if not isinstance(value, SpeedLimit.ZoneType):
            raise ValueError("Zone type must be a SpeedLimit.ZoneType enum value")
        self._zone_type = value

    @property
    def is_active(self) -> bool:
        return self._is_active

    @is_active.setter
    def is_active(self, value: bool):
        if not isinstance(value, bool):
            raise TypeError("is_active must be a boolean")
        self._is_active = value

    # --- API Methods ---
    @api("speedLimit")
    def speed_limit_set(self, limit: int, zone_type: str = "highway") -> Dict[str, Any]:
        """
        Set the current speed limit and zone type. This triggers navigation/HUD updates
        to display the new speed limit.

        Parameters:
        - limit (int): Speed limit in km/h
        - zone_type (string): Zone type, enum values: "school", "residential", "highway",
          "construction", "urban", default is "highway"

        Returns:
        - dict: Contains operation result and updated speed limit state
        """
        zone_enum = None
        for z in SpeedLimit.ZoneType:
            if z.value == zone_type:
                zone_enum = z
                break

        if zone_enum is None:
            valid = [z.value for z in SpeedLimit.ZoneType]
            raise ValueError(f"Invalid zone_type: {zone_type}. Must be one of: {valid}")

        old_limit = self._current_limit
        old_zone = self._zone_type.value
        self.current_limit = limit
        self.zone_type = zone_enum
        self.is_active = True

        # Emit event
        try:
            from event_bus import Event
            bus = getattr(self, '_event_bus', None)
            if bus:
                bus.publish(Event("speedlimit.changed", "speedLimit", {
                    "old_limit": old_limit,
                    "new_limit": limit,
                    "zone_type": zone_type
                }))
        except (ImportError, AttributeError):
            pass

        return {
            "success": True,
            "operation": "set_speed_limit",
            "old_limit": old_limit,
            "new_limit": limit,
            "zone_type": zone_type,
            "is_active": self._is_active
        }

    @api("speedLimit")
    def speed_limit_get(self) -> Dict[str, Any]:
        """
        Get the current speed limit information including limit value, zone type, and active state.

        Returns:
        - dict: Current speed limit state
        """
        return {
            "success": True,
            "current_limit": self._current_limit,
            "zone_type": self._zone_type.value,
            "is_active": self._is_active
        }

    @api("speedLimit")
    def speed_limit_clear(self) -> Dict[str, Any]:
        """
        Exit the current speed limit zone, deactivating the speed limit.

        Returns:
        - dict: Contains operation result confirming speed limit cleared
        """
        old_limit = self._current_limit
        self.is_active = False

        # Emit event
        try:
            from event_bus import Event
            bus = getattr(self, '_event_bus', None)
            if bus:
                bus.publish(Event("speedlimit.cleared", "speedLimit", {
                    "cleared_limit": old_limit,
                    "zone_type": self._zone_type.value
                }))
        except (ImportError, AttributeError):
            pass

        return {
            "success": True,
            "operation": "clear_speed_limit",
            "cleared_limit": old_limit,
            "is_active": False
        }

    @classmethod
    def init1(cls) -> 'SpeedLimit':
        """Highway zone, 120 km/h limit."""
        instance = cls()
        instance._current_limit = 120
        instance._zone_type = SpeedLimit.ZoneType.HIGHWAY
        instance._is_active = True
        return instance

    @classmethod
    def init2(cls) -> 'SpeedLimit':
        """School zone, 30 km/h limit."""
        instance = cls()
        instance._current_limit = 30
        instance._zone_type = SpeedLimit.ZoneType.SCHOOL
        instance._is_active = True
        return instance
