from enum import Enum
from typing import Dict, Any, Optional
from utils import api


class Road:
    """
    Road condition module for simulating road surface and traffic conditions.
    Road changes can trigger navigation rerouting, hazard light advisories, and speed warnings.
    """

    class Surface(Enum):
        """Road surface condition enumeration."""
        DRY = "dry"
        WET = "wet"
        ICY = "icy"
        MUDDY = "muddy"
        FLOODED = "flooded"

    class Condition(Enum):
        """Road traffic condition enumeration."""
        CLEAR = "clear"
        CONGESTED = "congested"
        CONSTRUCTION = "construction"
        ACCIDENT = "accident"

    class RoadType(Enum):
        """Road type enumeration."""
        HIGHWAY = "highway"
        URBAN = "urban"
        RURAL = "rural"

    def __init__(self):
        self._surface = Road.Surface.DRY
        self._condition = Road.Condition.CLEAR
        self._lanes = 3
        self._road_type = Road.RoadType.HIGHWAY

    # --- Properties ---
    @property
    def surface(self) -> 'Road.Surface':
        return self._surface

    @surface.setter
    def surface(self, value: 'Road.Surface'):
        if not isinstance(value, Road.Surface):
            raise ValueError("Surface must be a Road.Surface enum value")
        self._surface = value

    @property
    def condition(self) -> 'Road.Condition':
        return self._condition

    @condition.setter
    def condition(self, value: 'Road.Condition'):
        if not isinstance(value, Road.Condition):
            raise ValueError("Condition must be a Road.Condition enum value")
        self._condition = value

    @property
    def lanes(self) -> int:
        return self._lanes

    @lanes.setter
    def lanes(self, value: int):
        if not isinstance(value, int):
            raise TypeError("Lanes must be an integer")
        self._lanes = max(1, min(8, value))

    @property
    def road_type(self) -> 'Road.RoadType':
        return self._road_type

    @road_type.setter
    def road_type(self, value: 'Road.RoadType'):
        if not isinstance(value, Road.RoadType):
            raise ValueError("Road type must be a Road.RoadType enum value")
        self._road_type = value

    # --- API Methods ---
    @api("road")
    def road_set_condition(self, condition: str) -> Dict[str, Any]:
        """
        Set the current road traffic condition. Conditions like accident or construction
        may trigger navigation rerouting and hazard light advisories.

        Parameters:
        - condition (string): Road condition, enum values: "clear", "congested", "construction", "accident"

        Returns:
        - dict: Contains operation result and updated road state
        """
        condition_enum = None
        for c in Road.Condition:
            if c.value == condition:
                condition_enum = c
                break

        if condition_enum is None:
            valid = [c.value for c in Road.Condition]
            raise ValueError(f"Invalid condition: {condition}. Must be one of: {valid}")

        old_condition = self._condition.value
        self.condition = condition_enum

        # Emit event
        try:
            from event_bus import Event
            bus = getattr(self, '_event_bus', None)
            if bus:
                bus.publish(Event("road.condition_changed", "road", {
                    "old_condition": old_condition,
                    "new_condition": condition,
                    "road_type": self._road_type.value
                }))
        except (ImportError, AttributeError):
            pass

        return {
            "success": True,
            "operation": "set_condition",
            "old_condition": old_condition,
            "new_condition": condition,
            "current_state": {
                "surface": self._surface.value,
                "condition": self._condition.value,
                "road_type": self._road_type.value,
                "lanes": self._lanes
            }
        }

    @api("road")
    def road_set_surface(self, surface: str) -> Dict[str, Any]:
        """
        Set the current road surface condition. Icy or flooded surfaces may trigger
        speed reduction and traction control advisories.

        Parameters:
        - surface (string): Road surface, enum values: "dry", "wet", "icy", "muddy", "flooded"

        Returns:
        - dict: Contains operation result and updated road state
        """
        surface_enum = None
        for s in Road.Surface:
            if s.value == surface:
                surface_enum = s
                break

        if surface_enum is None:
            valid = [s.value for s in Road.Surface]
            raise ValueError(f"Invalid surface: {surface}. Must be one of: {valid}")

        old_surface = self._surface.value
        self.surface = surface_enum

        # Emit event
        try:
            from event_bus import Event
            bus = getattr(self, '_event_bus', None)
            if bus:
                bus.publish(Event("road.surface_changed", "road", {
                    "old_surface": old_surface,
                    "new_surface": surface,
                    "road_type": self._road_type.value
                }))
        except (ImportError, AttributeError):
            pass

        return {
            "success": True,
            "operation": "set_surface",
            "old_surface": old_surface,
            "new_surface": surface,
            "current_state": {
                "surface": self._surface.value,
                "condition": self._condition.value,
                "road_type": self._road_type.value,
                "lanes": self._lanes
            }
        }

    @api("road")
    def road_get_status(self) -> Dict[str, Any]:
        """
        Get the current road status including surface condition, traffic condition,
        road type, and number of lanes.

        Returns:
        - dict: Current road state
        """
        return {
            "success": True,
            "surface": self._surface.value,
            "condition": self._condition.value,
            "road_type": self._road_type.value,
            "lanes": self._lanes
        }

    @classmethod
    def init1(cls) -> 'Road':
        """Dry highway, clear conditions, 3 lanes."""
        instance = cls()
        instance._surface = Road.Surface.DRY
        instance._condition = Road.Condition.CLEAR
        instance._lanes = 3
        instance._road_type = Road.RoadType.HIGHWAY
        return instance

    @classmethod
    def init2(cls) -> 'Road':
        """Wet urban road, congested, 2 lanes."""
        instance = cls()
        instance._surface = Road.Surface.WET
        instance._condition = Road.Condition.CONGESTED
        instance._lanes = 2
        instance._road_type = Road.RoadType.URBAN
        return instance
