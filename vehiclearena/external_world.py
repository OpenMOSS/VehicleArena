"""
ExternalWorld - Vehicle External World Simulation Engine

Peer class to VehicleWorld that models the vehicle's external environment:
weather, map (including traffic lights, road conditions, congestion, construction),
speed limits, and day/night cycles.

External world changes trigger vehicle module reactions via the EventBus/ConstraintEngine
coupling system (e.g., rain → wiper advisory, fog → fog lights, night → headlights).
"""

from typing import Mapping, Optional
from module.weather import Weather
from module.speedlimit import SpeedLimit
from module.daynight import DayNight
from module.map_module import MapModule


class ExternalWorld:
    """
    Container for all external world simulation modules.
    Access sub-modules via ew.weather, ew.map, ew.speedLimit, ew.dayNight.
    """

    def __init__(self, module_descriptors: Optional[Mapping] = None):
        """Instantiate the selected registry-owned external modules.

        ``module_descriptors`` is supplied by :class:`VehicleWorld`, so a
        third-party external module follows the same equipment selection and
        wiring path as the built-ins.
        """
        if module_descriptors is None:
            from registry import ModuleRegistry
            module_descriptors = {
                name: descriptor
                for name, descriptor in
                ModuleRegistry.instance().all_modules().items()
                if descriptor.is_external
            }
        for name, descriptor in module_descriptors.items():
            if not descriptor.is_external:
                raise ValueError(
                    f"ExternalWorld received internal module {name!r}")
            setattr(self, name, descriptor.cls())

    @classmethod
    def init1(cls) -> 'ExternalWorld':
        """
        Sunny daytime — city normal driving.
        Good weather, green light ahead, light congestion.
        """
        instance = cls()
        instance.weather = Weather.init1()       # sunny, 25C
        instance.speedLimit = SpeedLimit.init1()  # 120 km/h highway
        instance.dayNight = DayNight.init1()     # noon, daylight=100
        instance.map = MapModule.init1()         # city, green light, light congestion
        return instance

    @classmethod
    def init2(cls) -> 'ExternalWorld':
        """
        Rainy urban congested — evening rush hour.
        Rain, heavy congestion, red lights, construction.
        """
        instance = cls()
        instance.weather = Weather.init2()       # rainy, 18C
        instance.speedLimit = SpeedLimit.init2()  # 30 km/h school zone
        instance.dayNight = DayNight.init1()     # noon
        instance.map = MapModule.init3()         # heavy congestion, red lights, construction
        return instance

    @classmethod
    def init3(cls) -> 'ExternalWorld':
        """
        Foggy night highway — expressway with construction.
        Dense fog, very low visibility, dark, construction ahead.
        """
        instance = cls()
        instance.weather = Weather.init3()       # foggy, 10C, visibility=50m
        instance.speedLimit = SpeedLimit.init1()  # 120 km/h
        instance.dayNight = DayNight.init2()     # night, daylight=5
        instance.map = MapModule.init2()         # highway, construction ahead
        return instance

    @classmethod
    def init4(cls) -> 'ExternalWorld':
        """
        Snowy dusk — accident ahead.
        Snow, icy roads, dusk transition, multi-vehicle accident blocking lanes.
        """
        instance = cls()
        instance.weather = Weather.init4()       # snowy, -5C
        instance.speedLimit = SpeedLimit()
        instance.speedLimit._current_limit = 40
        instance.speedLimit._zone_type = SpeedLimit.ZoneType.URBAN
        instance.dayNight = DayNight.init3()     # dusk, daylight=30
        instance.map = MapModule.init4()         # accident, flashing yellow, wet road
        return instance
