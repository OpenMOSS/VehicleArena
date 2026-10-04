"""
Map Module — Core external-world spatial module.

Models the vehicle's complete driving context on the road network:
  - Current location & road info (type, lanes, speed limit, surface, congestion)
  - Upcoming intersections with traffic lights
  - Road events ahead (construction, accidents, congestion zones, flooding)
  - Speed cameras
  - POI search

Traffic lights belong to intersections and per-segment road properties belong
directly to the map context.
"""

import random
from enum import Enum
from typing import Dict, Any, List, Optional
from utils import api
from module.base_module import BaseModule
from registry import register_module


@register_module("map", description="Map, navigation, traffic, POI search", category="environment", is_external=True)
class MapModule(BaseModule):
    """
    Core spatial simulation module.  Holds the vehicle's current road context
    and everything the agent can discover about the road ahead.
    """

    # ── Enumerations ────────────────────────────────────────────

    class RoadType(Enum):
        HIGHWAY = "highway"
        URBAN = "urban"
        RURAL = "rural"
        RESIDENTIAL = "residential"

    class CongestionLevel(Enum):
        FREE = "free"               # 畅通
        LIGHT = "light"             # 轻微拥堵
        MODERATE = "moderate"       # 中度拥堵
        HEAVY = "heavy"             # 严重拥堵
        GRIDLOCK = "gridlock"       # 堵死

    class TrafficSignal(Enum):
        RED = "red"
        YELLOW = "yellow"
        GREEN = "green"
        FLASHING_YELLOW = "flashing_yellow"

    class RoadEventType(Enum):
        CONSTRUCTION = "construction"     # 施工
        ACCIDENT = "accident"             # 事故
        CONGESTION = "congestion"         # 拥堵路段
        FLOODING = "flooding"             # 积水
        ROAD_CLOSURE = "road_closure"     # 道路封闭

    class Severity(Enum):
        MINOR = "minor"
        MODERATE = "moderate"
        SEVERE = "severe"

    class POICategory(Enum):
        GAS_STATION = "gas_station"
        RESTAURANT = "restaurant"
        PARKING = "parking"
        HOSPITAL = "hospital"
        HOTEL = "hotel"
        CHARGING_STATION = "charging_station"
        SHOPPING = "shopping"
        SCENIC = "scenic"
        ALL = "all"

    # ── Constructor ─────────────────────────────────────────────

    def __init__(self):
        # Current vehicle position
        self._current_location = {
            "lat": 42.8150, "lng": 87.6320, "name": "Lumina City Center"
        }

        # Current road segment
        self._current_road = {
            "name": "Maple Boulevard",
            "type": "urban",
            "lanes": 4,
            "speed_limit": 60,
            "surface": "dry",
            "congestion": "free",
            "estimated_speed": 55,        # actual flow speed in km/h
        }

        # Upcoming intersections (ordered by distance) — each carries a traffic light
        self._intersections_ahead = []

        # Road events ahead (construction / accident / congestion zone / flooding)
        self._road_events_ahead = []

        # Speed cameras ahead
        self._speed_cameras_ahead = []

        # Navigation state (set by sim engine when vehicle is navigating)
        self._is_navigating = False

    # ── Properties ──────────────────────────────────────────────

    @property
    def current_location(self) -> Dict[str, Any]:
        return self._current_location

    @current_location.setter
    def current_location(self, value: Dict[str, Any]):
        if not isinstance(value, dict):
            raise TypeError("Location must be a dict with lat, lng, name")
        self._current_location = value

    @property
    def current_road(self) -> Dict[str, Any]:
        return self._current_road

    @current_road.setter
    def current_road(self, value: Dict[str, Any]):
        if not isinstance(value, dict):
            raise TypeError("current_road must be a dict")
        self._current_road = value

    @property
    def current_road_name(self) -> str:
        return self._current_road.get("name", "")

    @current_road_name.setter
    def current_road_name(self, value: str):
        self._current_road["name"] = str(value)

    @property
    def intersections_ahead(self) -> List[Dict]:
        return self._intersections_ahead

    @property
    def road_events_ahead(self) -> List[Dict]:
        return self._road_events_ahead

    @property
    def speed_cameras_ahead(self) -> List[Dict]:
        return self._speed_cameras_ahead

    # ── Query APIs ──────────────────────────────────────────────

    @api("map")
    def map_get_current_location(self) -> Dict[str, Any]:
        """
        Get the vehicle's current GPS location.

        Returns:
        - dict: Current location with latitude, longitude, and place name
        """
        return {
            "success": True,
            "location": self._current_location
        }

    @api("map")
    def map_query_road(self, road_from: str, road_to: str) -> Dict[str, Any]:
        """
        Query detailed information about a road segment between two adjacent
        intersections, including current events (congestion, weather, road
        events) and traffic light state.

        Can query any road segment on the map, not just the current route.
        Returns real-time data for the current tick (future events not visible).

        Parameters:
        - road_from (string): Starting intersection ID (e.g. "C2")
        - road_to (string): Ending intersection ID (e.g. "C3")

        Returns:
        - dict: Road segment details with current events and traffic light state.
                Populated by sim_engine at runtime from RoadNetwork data.
        """
        # Placeholder — sim_engine intercepts this call and injects real data
        return {
            "success": True,
            "road": {
                "from": road_from,
                "to": road_to,
                "road_name": "",
                "road_type": "",
                "distance_meters": 0,
                "speed_limit": 0,
            },
            "current_events": [],
            "traffic_light": None,
        }

    @api("map")
    def map_search_poi(self, query: str, category: str = "all") -> Dict[str, Any]:
        """
        Search for Points of Interest (POI) by keyword and category.

        Parameters:
        - query (string): Search text (e.g., "gas station", "cafe", "hospital")
        - category (string): POI category filter, enum values: "gas_station",
          "restaurant", "parking", "hospital", "hotel", "charging_station",
          "shopping", "scenic", "all". Default is "all"

        Returns:
        - dict: Search results with matching POIs
        """
        self._validate_poi_category(category)
        all_pois = self._get_simulated_pois()
        results = [
            p for p in all_pois
            if (category == "all" or p.get("category") == category)
            and (query.lower() in p.get("name", "").lower()
                 or query.lower() in p.get("category", "").lower())
        ]
        return {
            "success": True,
            "query": query,
            "category": category,
            "result_count": len(results),
            "results": results[:10]
        }

    @api("map")
    def map_get_nearby(self, category: str = "all", radius: int = 5000) -> Dict[str, Any]:
        """
        Get nearby POIs within a specified radius from the current location.

        Parameters:
        - category (string): POI category filter, enum values: "gas_station",
          "restaurant", "parking", "hospital", "hotel", "charging_station",
          "shopping", "scenic", "all". Default is "all"
        - radius (int): Search radius in meters, default is 5000

        Returns:
        - dict: Nearby POIs within radius, sorted by distance
        """
        self._validate_poi_category(category)
        all_pois = self._get_simulated_pois()
        results = [
            p for p in all_pois
            if (category == "all" or p.get("category") == category)
            and p.get("distance_meters", 99999) <= radius
        ]
        results.sort(key=lambda x: x.get("distance_meters", 99999))
        return {
            "success": True,
            "category": category,
            "radius": radius,
            "current_location": self._current_location,
            "result_count": len(results),
            "results": results[:10]
        }

    # ── Action APIs ─────────────────────────────────────────────

    @api("map")
    def map_update_location(self, lat: float, lng: float, name: str = "") -> Dict[str, Any]:
        """
        Update the vehicle's current location. This may trigger navigation
        events such as approaching-intersection or speed-camera warnings.

        Parameters:
        - lat (float): Latitude of the new location
        - lng (float): Longitude of the new location
        - name (string): Optional place name or description

        Returns:
        - dict: Operation result with old and new location
        """
        old_location = self._current_location.copy()
        self._current_location = {"lat": lat, "lng": lng, "name": name}

        self._emit_event("map.location_updated", {
            "old_location": old_location,
            "new_location": self._current_location
        })

        # Check proximity-based events
        for intersection in self._intersections_ahead:
            if intersection.get("distance_meters", 9999) <= 200:
                self._emit_event("map.approaching_intersection", {
                    "intersection": intersection
                })
                break

        for camera in self._speed_cameras_ahead:
            if camera.get("distance_meters", 9999) <= 500:
                self._emit_event("map.speed_camera_ahead", {"camera": camera})
                break

        for event in self._road_events_ahead:
            if event.get("distance_meters", 9999) <= 300:
                self._emit_event("map.approaching_road_event", {"road_event": event})
                break

        return {
            "success": True,
            "operation": "update_location",
            "old_location": old_location,
            "new_location": self._current_location
        }


    # ── Private helpers ─────────────────────────────────────────

    def _validate_poi_category(self, category: str):
        valid = [c.value for c in MapModule.POICategory]
        if category not in valid:
            raise ValueError(f"Invalid category: {category}. Must be one of: {valid}")

    def _emit_event(self, event_name: str, data: dict):
        try:
            from event_bus import Event
            bus = getattr(self, '_event_bus', None)
            if bus:
                bus.publish(Event(event_name, "map", data))
        except (ImportError, AttributeError):
            pass

    def _get_simulated_pois(self) -> List[Dict]:
        """Simulated POI database (all fictional names)."""
        return [
            {"name": "Starway Gas Station", "category": "gas_station",
             "distance_meters": 500, "address": "100 Amber Lane"},
            {"name": "Northpeak Fuel Stop", "category": "gas_station",
             "distance_meters": 1200, "address": "250 Crestview Ave"},
            {"name": "Golden Spoon Diner", "category": "restaurant",
             "distance_meters": 300, "address": "50 Coral Plaza"},
            {"name": "Ember Grill", "category": "restaurant",
             "distance_meters": 800, "address": "80 Thornfield Blvd"},
            {"name": "Driftwood Cafe", "category": "restaurant",
             "distance_meters": 150, "address": "10 Willow Bend"},
            {"name": "Clearview General Hospital", "category": "hospital",
             "distance_meters": 2000, "address": "1 Redstone Way"},
            {"name": "Stonebridge Underground Parking", "category": "parking",
             "distance_meters": 200, "address": "5 Quarry Road"},
            {"name": "Silverpine Hotel", "category": "hotel",
             "distance_meters": 3000, "address": "88 Foxglove Blvd"},
            {"name": "Voltline Supercharger", "category": "charging_station",
             "distance_meters": 1500, "address": "200 Circuit Drive"},
            {"name": "Crystalgate Mall", "category": "shopping",
             "distance_meters": 1000, "address": "168 Opal Street"},
            {"name": "Moonridge Scenic Park", "category": "scenic",
             "distance_meters": 5000, "address": "1 Horizon Trail"},
        ]

    # ── Serialization ───────────────────────────────────────────

    # ── Init Presets ────────────────────────────────────────────

    @classmethod
    def init1(cls) -> 'MapModule':
        """
        Urban normal driving — Maple Boulevard, Lumina City
        Green light intersection ahead, light congestion, one speed camera.
        """
        inst = cls()
        inst._current_location = {
            "lat": 42.8150, "lng": 87.6320, "name": "Maple Boulevard, Thornfield District"
        }
        inst._current_road = {
            "name": "Maple Boulevard",
            "type": "urban",
            "lanes": 4,
            "speed_limit": 60,
            "surface": "dry",
            "congestion": "light",
            "estimated_speed": 45,
        }
        inst._intersections_ahead = [
            {
                "name": "Maple Blvd & Alder Street",
                "distance_meters": 150,
                "traffic_light": {
                    "signal": "green",
                    "remaining_seconds": 25,
                    "is_flashing": False,
                },
            },
            {
                "name": "Maple Blvd & Cobalt Lane",
                "distance_meters": 600,
                "traffic_light": {
                    "signal": "red",
                    "remaining_seconds": 32,
                    "is_flashing": False,
                },
            },
        ]
        inst._road_events_ahead = []
        inst._speed_cameras_ahead = [
            {"distance_meters": 200, "speed_limit": 60, "type": "fixed"},
        ]
        return inst

    @classmethod
    def init2(cls) -> 'MapModule':
        """
        Highway construction zone — H7 Northridge Expressway
        Construction 1.5km ahead, speed reduced to 60, speed cameras present.
        """
        inst = cls()
        inst._current_location = {
            "lat": 43.2810, "lng": 88.1450, "name": "H7 Northridge Expressway KM85"
        }
        inst._current_road = {
            "name": "H7 Northridge Expressway",
            "type": "highway",
            "lanes": 3,
            "speed_limit": 120,
            "surface": "dry",
            "congestion": "free",
            "estimated_speed": 110,
        }
        inst._intersections_ahead = []
        inst._road_events_ahead = [
            {
                "type": "construction",
                "distance_meters": 1500,
                "length_meters": 800,
                "description": "Bridge deck repair, right lane closed",
                "severity": "moderate",
                "lanes_affected": 1,
                "detour_available": False,
                "speed_limit_in_zone": 60,
                "estimated_delay_minutes": 8,
            },
        ]
        inst._speed_cameras_ahead = [
            {"distance_meters": 1400, "speed_limit": 60, "type": "temporary"},
            {"distance_meters": 3000, "speed_limit": 120, "type": "fixed"},
        ]
        return inst

    @classmethod
    def init3(cls) -> 'MapModule':
        """
        Urban heavy congestion — Ironwood Elevated Road, evening rush hour
        Two consecutive red lights, heavy congestion, construction detour ahead.
        """
        inst = cls()
        inst._current_location = {
            "lat": 42.7980, "lng": 87.5870, "name": "Ironwood Elevated, Greystone Quarter"
        }
        inst._current_road = {
            "name": "Ironwood Elevated Road",
            "type": "urban",
            "lanes": 3,
            "speed_limit": 50,
            "surface": "dry",
            "congestion": "heavy",
            "estimated_speed": 8,
        }
        inst._intersections_ahead = [
            {
                "name": "Ironwood Rd & Flint Avenue",
                "distance_meters": 80,
                "traffic_light": {
                    "signal": "red",
                    "remaining_seconds": 55,
                    "is_flashing": False,
                },
            },
            {
                "name": "Ironwood Rd & Garnet Street",
                "distance_meters": 350,
                "traffic_light": {
                    "signal": "red",
                    "remaining_seconds": 40,
                    "is_flashing": False,
                },
            },
            {
                "name": "Ironwood Rd & Jasper Road",
                "distance_meters": 700,
                "traffic_light": {
                    "signal": "green",
                    "remaining_seconds": 15,
                    "is_flashing": False,
                },
            },
        ]
        inst._road_events_ahead = [
            {
                "type": "construction",
                "distance_meters": 500,
                "length_meters": 300,
                "description": "Water pipe replacement, middle lane closed",
                "severity": "severe",
                "lanes_affected": 1,
                "detour_available": True,
                "speed_limit_in_zone": 30,
                "estimated_delay_minutes": 20,
            },
            {
                "type": "congestion",
                "distance_meters": 0,
                "length_meters": 2000,
                "description": "Evening rush hour congestion",
                "severity": "severe",
                "lanes_affected": 3,
                "detour_available": True,
                "speed_limit_in_zone": None,
                "estimated_delay_minutes": 25,
            },
        ]
        inst._speed_cameras_ahead = [
            {"distance_meters": 400, "speed_limit": 50, "type": "fixed"},
        ]
        return inst

    @classmethod
    def init4(cls) -> 'MapModule':
        """
        Accident section — Orion Ring Road, rear-end collision ahead
        Accident 300m ahead blocking 2 lanes, flashing yellow light, detour needed.
        """
        inst = cls()
        inst._current_location = {
            "lat": 42.7600, "lng": 87.5100, "name": "Orion Ring Road, Westbrook"
        }
        inst._current_road = {
            "name": "Orion Ring Road",
            "type": "urban",
            "lanes": 4,
            "speed_limit": 80,
            "surface": "wet",
            "congestion": "heavy",
            "estimated_speed": 12,
        }
        inst._intersections_ahead = [
            {
                "name": "Orion Ring & Sycamore Road",
                "distance_meters": 250,
                "traffic_light": {
                    "signal": "flashing_yellow",
                    "remaining_seconds": 0,
                    "is_flashing": True,
                },
            },
        ]
        inst._road_events_ahead = [
            {
                "type": "accident",
                "distance_meters": 300,
                "length_meters": 100,
                "description": "Multi-vehicle rear-end collision, "
                               "2 right lanes blocked, police on scene",
                "severity": "severe",
                "lanes_affected": 2,
                "detour_available": True,
                "speed_limit_in_zone": 20,
                "estimated_delay_minutes": 35,
            },
        ]
        inst._speed_cameras_ahead = []
        return inst

    @classmethod
    def init5(cls) -> 'MapModule':
        """
        School zone — low speed limit, heavy pedestrian traffic, red light
        """
        inst = cls()
        inst._current_location = {
            "lat": 42.9200, "lng": 87.7500, "name": "Near Crestwood Academy, Hazel Park"
        }
        inst._current_road = {
            "name": "Birchwood Lane",
            "type": "residential",
            "lanes": 2,
            "speed_limit": 30,
            "surface": "dry",
            "congestion": "moderate",
            "estimated_speed": 20,
        }
        inst._intersections_ahead = [
            {
                "name": "Birchwood Ln & Fern Crescent",
                "distance_meters": 100,
                "traffic_light": {
                    "signal": "red",
                    "remaining_seconds": 30,
                    "is_flashing": False,
                },
            },
            {
                "name": "Birchwood Ln & Hawthorn Way",
                "distance_meters": 450,
                "traffic_light": {
                    "signal": "green",
                    "remaining_seconds": 18,
                    "is_flashing": False,
                },
            },
        ]
        inst._road_events_ahead = [
            {
                "type": "congestion",
                "distance_meters": 50,
                "length_meters": 500,
                "description": "School dismissal time, heavy pedestrian "
                               "and bicycle traffic",
                "severity": "moderate",
                "lanes_affected": 2,
                "detour_available": False,
                "speed_limit_in_zone": 30,
                "estimated_delay_minutes": 10,
            },
        ]
        inst._speed_cameras_ahead = [
            {"distance_meters": 300, "speed_limit": 30, "type": "fixed"},
        ]
        return inst

    @classmethod
    def init6(cls) -> 'MapModule':
        """
        Flooded road — heavy rain flooding in urban area
        Flooded road section + flashing yellow light + congestion.
        """
        inst = cls()
        inst._current_location = {
            "lat": 42.8300, "lng": 87.6900, "name": "Driftstone Avenue, Eastport"
        }
        inst._current_road = {
            "name": "Driftstone Avenue",
            "type": "urban",
            "lanes": 3,
            "speed_limit": 50,
            "surface": "flooded",
            "congestion": "heavy",
            "estimated_speed": 10,
        }
        inst._intersections_ahead = [
            {
                "name": "Driftstone Ave & Pebblebrook Road",
                "distance_meters": 200,
                "traffic_light": {
                    "signal": "flashing_yellow",
                    "remaining_seconds": 0,
                    "is_flashing": True,
                },
            },
        ]
        inst._road_events_ahead = [
            {
                "type": "flooding",
                "distance_meters": 100,
                "length_meters": 400,
                "description": "Road flooded with 15-20cm standing water "
                               "after heavy rain, vehicles wading slowly",
                "severity": "severe",
                "lanes_affected": 3,
                "detour_available": True,
                "speed_limit_in_zone": 15,
                "estimated_delay_minutes": 30,
            },
        ]
        inst._speed_cameras_ahead = []
        return inst

    # ── Dynamic Generation ─────────────────────────────────────

    @classmethod
    def generate(
        cls,
        scale: str = "medium",
        road_type: str = None,
        congestion: str = None,
        surface: str = None,
    ) -> 'MapModule':
        """
        Dynamically generate a MapModule with controllable scale and optional overrides.

        Parameters:
            scale: Map complexity level — "small", "medium", "large", or "xlarge".
                   Controls the number of intersections, road events, speed cameras, and POIs.
            road_type: Override road type — "highway", "urban", "rural", "residential".
                       If None, randomly selected.
            congestion: Override congestion level — "free", "light", "moderate", "heavy", "gridlock".
                        If None, derived from road_type and events.
            surface: Override road surface — "dry", "wet", "icy", "flooded".
                     If None, randomly selected with weighted probabilities.
        Returns:
            A fully populated MapModule instance.
        """
        return MapGenerator.generate(
            cls, scale=scale, road_type=road_type,
            congestion=congestion, surface=surface
        )


class MapGenerator:
    """
    Dynamic map scenario generator with data pools and scale-aware element counts.

    Scale presets control how many map elements are generated:
        small  : 0-1 intersections, 0-1 events, 0-1 cameras
        medium : 1-3 intersections, 0-2 events, 0-2 cameras
        large  : 2-5 intersections, 1-3 events, 1-3 cameras
        xlarge : 4-8 intersections, 2-5 events, 2-4 cameras
    """

    # ── Scale definitions: (min, max) for each element type ──

    SCALE_PARAMS = {
        "small":  {"intersections": (0, 1), "events": (0, 1), "cameras": (0, 1)},
        "medium": {"intersections": (1, 3), "events": (0, 2), "cameras": (0, 2)},
        "large":  {"intersections": (2, 5), "events": (1, 3), "cameras": (1, 3)},
        "xlarge": {"intersections": (4, 8), "events": (2, 5), "cameras": (2, 4)},
    }

    # ── Road data pools (all fictional) ────────────────────────
    #
    # Fictional city: Lumina City and surrounding areas.
    # Coordinates are synthetic and do not correspond to real locations.
    # Format: (road_name, lat, lng, location_name, lanes, speed_limit)

    ROAD_POOL = {
        "highway": [
            ("H7 Northridge Expressway", 43.2810, 88.1450, "H7 Northridge Expressway KM85, Cloverdale County", 3, 120),
            ("H3 Silvercoast Expressway", 42.5200, 87.3100, "H3 Silvercoast Expressway KM220, Briarwood Pass", 4, 120),
            ("H12 Stormvale Expressway", 43.0500, 88.4600, "H12 Stormvale Expressway KM45, Redthorn Valley", 3, 120),
            ("R20 Lumina Outer Ring", 42.6900, 87.8200, "R20 Outer Ring KM30, Foxglove Junction", 3, 100),
            ("H5 Ironpeak Expressway", 42.4100, 88.0800, "H5 Ironpeak Expressway KM110, Ashford Flats", 4, 120),
            ("R8 Dawnfield Link", 42.1800, 87.4500, "R8 Dawnfield Link KM20, Havenbrook", 2, 100),
        ],
        "urban": [
            ("Maple Boulevard", 42.8150, 87.6320, "Maple Boulevard, Thornfield District", 4, 60),
            ("Ironwood Elevated Road", 42.7980, 87.5870, "Ironwood Elevated, Greystone Quarter", 3, 50),
            ("Coral Crescent", 42.8095, 87.6154, "Coral Crescent, Ambervale Center", 4, 50),
            ("Granite Parkway", 42.8200, 87.5500, "Granite Parkway, Stonecrest Park", 4, 60),
            ("Prism Avenue", 42.8350, 87.7250, "Prism Avenue, Glasswater Financial District", 6, 60),
            ("Driftstone Avenue", 42.8300, 87.6900, "Driftstone Avenue, Eastport", 3, 50),
            ("Orion Ring Road", 42.7600, 87.5100, "Orion Ring Road, Westbrook", 4, 80),
            ("Skyline Elevated Road", 42.8100, 87.6600, "Skyline Elevated, Midtown", 3, 60),
            ("Quartzhill Road", 42.7900, 87.6400, "Quartzhill Road, Coppergate", 3, 50),
            ("Willowmere Road", 42.8500, 87.7100, "Willowmere Road, Northvale", 3, 50),
        ],
        "rural": [
            ("Thistledown County Road", 43.3200, 88.2000, "Thistledown, Fernhollow Island", 2, 60),
            ("Bramblewood Village Road", 42.1300, 87.0400, "Bramblewood Countryside", 2, 40),
            ("Clearwater Lake Road", 42.6500, 87.1200, "Clearwater Lake Area", 2, 50),
            ("Sandpiper Coastal Road", 42.3600, 87.4700, "Sandpiper Beach, Windmere", 2, 60),
            ("Pinecrest Mountain Road", 43.0300, 87.2100, "Pinecrest Ridge Area", 2, 40),
        ],
        "residential": [
            ("Birchwood Lane", 42.9200, 87.7500, "Near Crestwood Academy, Hazel Park", 2, 30),
            ("Rosemary Drive", 42.9100, 87.7600, "Rosemary Drive, Larkspur Heights", 2, 30),
            ("Tanglewood Street", 42.8700, 87.6800, "Tanglewood, Briarcliff Residential", 2, 30),
            ("Heatherfield Road", 42.8200, 87.5200, "Heatherfield, Ivydale Residential", 2, 40),
            ("Cloverdale Circle", 42.7100, 87.4800, "Cloverdale, Millbrook District", 2, 30),
            ("Aldercroft Way", 43.0000, 87.8500, "Aldercroft, Stonebriar Residential", 2, 30),
        ],
    }

    # Cross-road name pools for generating intersection names (all fictional)
    CROSS_ROADS = {
        "highway": [
            "Exit 15 (Cloverdale)", "Exit 22 (Ashford North)", "Foxglove Service Area",
            "Exit 8 (Windmere)", "Stonebriar Toll Station", "Exit 31 (Thornvale)",
            "Pinecrest Rest Area", "Exit 12 (Millbrook)", "Exit 18 (Havenbrook South)",
        ],
        "urban": [
            "Alder Street", "Cobalt Lane", "Flint Avenue", "Garnet Street",
            "Jasper Road", "Onyx Boulevard", "Quartz Drive", "Topaz Way",
            "Basalt Road", "Feldspar Lane", "Mica Street", "Beryl Avenue",
            "Cinnabar Road", "Dolomite Drive", "Galena Lane", "Hematite Street",
            "Lazurite Road", "Pyrite Avenue", "Rutile Boulevard", "Spinel Way",
            "Obsidian Crescent", "Pumice Road", "Calcite Lane", "Zircon Drive",
        ],
        "rural": [
            "Meadow Trail", "Windmill Road", "Creek Bridge", "Chapel Lane",
            "Orchard Path", "Millpond Road", "Stone Arch Bridge", "Ridgeview Road",
        ],
        "residential": [
            "Fern Crescent", "Hawthorn Way", "Ivy Close", "Juniper Court",
            "Linden Terrace", "Magnolia Path", "Nettle Row", "Primrose Circle",
            "Sage Walk", "Thyme Gardens", "Violet Mews", "Wisteria Place",
        ],
    }

    # ── Road event data pools ────────────────────────────────

    CONSTRUCTION_DESCRIPTIONS = [
        "Bridge deck repair, {lanes} lane(s) closed",
        "Water pipe replacement, {lanes} lane(s) blocked",
        "Road resurfacing work in progress",
        "Power cable installation, partial road closure",
        "Sidewalk reconstruction, traffic narrowed",
        "Gas pipeline repair, {lanes} lane(s) closed",
        "Overpass structural maintenance, reduced speed",
        "Drainage system upgrade, temporary barriers",
        "Metro station construction, detour signs posted",
        "Utility tunnel excavation, single-lane traffic",
    ]

    ACCIDENT_DESCRIPTIONS = [
        "Multi-vehicle rear-end collision, {lanes} lane(s) blocked, police on scene",
        "Two-car side collision, {lanes} lane(s) blocked, tow truck en route",
        "Truck rollover, {lanes} lane(s) blocked, emergency crews clearing",
        "Minor fender bender, shoulder lane occupied",
        "Vehicle breakdown on {lanes} lane(s), hazard lights on",
        "Motorcycle accident, {lanes} lane(s) blocked, ambulance on scene",
        "Bus and sedan collision, {lanes} lane(s) blocked",
    ]

    CONGESTION_DESCRIPTIONS = [
        "Evening rush hour congestion",
        "Morning commute traffic backup",
        "Congestion due to nearby event dispersal",
        "Heavy traffic near shopping district",
        "School zone congestion, slow-moving vehicles",
        "Holiday travel congestion",
        "Congestion caused by lane merging ahead",
        "Traffic backup from upstream accident",
    ]

    FLOODING_DESCRIPTIONS = [
        "Road flooded with {depth}cm standing water after heavy rain, vehicles wading slowly",
        "Underpass flooded with {depth}cm water, caution advised",
        "Low-lying section waterlogged, {depth}cm depth, slow passage only",
        "Drainage overflow causing {depth}cm ponding on road surface",
    ]

    CLOSURE_DESCRIPTIONS = [
        "Road fully closed for emergency repair",
        "Road closed due to sinkhole, detour required",
        "Road closed for scheduled maintenance until further notice",
        "Section closed due to structural safety inspection",
    ]

    # ── Speed camera data pools ──────────────────────────────

    CAMERA_TYPES = ["fixed", "temporary", "mobile"]

    # ── POI data pools ───────────────────────────────────────

    POI_POOL = {
        "gas_station": [
            "Starway Gas Station", "Northpeak Fuel Stop", "Ironclad Energy Station",
            "Brightwell Petroleum", "Quartzfield Gas Station", "Ridgeline Fuel Depot",
        ],
        "restaurant": [
            "Golden Spoon Diner", "Ember Grill", "Driftwood Cafe", "Cobblestone Bistro",
            "Hearthfire Kitchen", "Jade Bowl Noodle House", "Thornberry Pizza", "Mapleleaf Sushi",
        ],
        "parking": [
            "Stonebridge Underground Parking", "Foxglove Parking Lot", "Skyview Multi-story Car Park",
            "Crystalgate Mall Parking", "Coppergate Public Garage",
        ],
        "hospital": [
            "Clearview General Hospital", "Thornfield Community Hospital", "Briarwood Children's Hospital",
            "Eastport Emergency Center", "Hazel Park Health Center",
        ],
        "hotel": [
            "Silverpine Hotel", "Willowmere Inn", "Coppergate Lodge", "Stormvale Grand Hotel",
            "Moonridge Suites", "Cloverdale Express Hotel",
        ],
        "charging_station": [
            "Voltline Supercharger", "Sparkpoint Charging Hub", "Gridwell Power Station",
            "Amberflow Charge Point", "Lumina Public Charging",
        ],
        "shopping": [
            "Crystalgate Mall", "Thornfield Market", "Prism Plaza", "Stonebriar Superstore",
            "Coppergate Shopping Center", "Northvale Outlet",
        ],
        "scenic": [
            "Moonridge Scenic Park", "Clearwater Botanical Garden", "Willowmere Riverside Walk",
            "Pinecrest Arboretum", "Lumina Heritage Museum",
        ],
    }

    # ── Congestion-speed mapping ─────────────────────────────

    CONGESTION_SPEED_RATIO = {
        "free": (0.85, 0.95),
        "light": (0.60, 0.80),
        "moderate": (0.35, 0.55),
        "heavy": (0.10, 0.25),
        "gridlock": (0.0, 0.08),
    }

    # ── Surface probability weights by scenario ──────────────

    SURFACE_WEIGHTS = {
        "default": {"dry": 60, "wet": 25, "icy": 5, "flooded": 10},
        "highway": {"dry": 70, "wet": 25, "icy": 5, "flooded": 0},
        "residential": {"dry": 65, "wet": 25, "icy": 5, "flooded": 5},
    }

    # ── Main generation logic ────────────────────────────────

    @classmethod
    def generate(
        cls,
        map_cls,
        scale: str = "medium",
        road_type: str = None,
        congestion: str = None,
        surface: str = None,
    ) -> 'MapModule':
        """Core generation entry point."""

        rng = random.Random()

        if scale not in cls.SCALE_PARAMS:
            raise ValueError(f"Invalid scale '{scale}'. Choose from: {list(cls.SCALE_PARAMS.keys())}")

        params = cls.SCALE_PARAMS[scale]

        # ── 1. Choose road type ──
        if road_type is None:
            road_type = rng.choice(["highway", "urban", "rural", "residential"])
        if road_type not in cls.ROAD_POOL:
            raise ValueError(f"Invalid road_type '{road_type}'. Choose from: {list(cls.ROAD_POOL.keys())}")

        is_highway = (road_type == "highway")

        # ── 2. Pick road from pool ──
        road_info = rng.choice(cls.ROAD_POOL[road_type])
        road_name, lat, lng, loc_name, lanes, speed_limit = road_info

        # ── 3. Choose surface ──
        if surface is None:
            weight_key = road_type if road_type in cls.SURFACE_WEIGHTS else "default"
            surfaces, weights = zip(*cls.SURFACE_WEIGHTS[weight_key].items())
            surface = rng.choices(surfaces, weights=weights, k=1)[0]

        # ── 4. Decide congestion ──
        if congestion is None:
            congestion = cls._pick_congestion(rng, road_type, surface)

        # ── 5. Compute estimated speed ──
        lo, hi = cls.CONGESTION_SPEED_RATIO.get(congestion, (0.5, 0.7))
        ratio = rng.uniform(lo, hi)
        estimated_speed = max(0, int(speed_limit * ratio))

        # ── 6. Build the MapModule instance ──
        inst = map_cls()
        inst._current_location = {"lat": lat, "lng": lng, "name": loc_name}
        inst._current_road = {
            "name": road_name,
            "type": road_type,
            "lanes": lanes,
            "speed_limit": speed_limit,
            "surface": surface,
            "congestion": congestion,
            "estimated_speed": estimated_speed,
        }

        # ── 7. Generate intersections (highways have none) ──
        if is_highway:
            inst._intersections_ahead = []
        else:
            n_inter = rng.randint(*params["intersections"])
            inst._intersections_ahead = cls._generate_intersections(
                rng, road_name, road_type, n_inter
            )

        # ── 8. Generate road events ──
        n_events = rng.randint(*params["events"])
        inst._road_events_ahead = cls._generate_road_events(
            rng, road_type, lanes, n_events, surface
        )

        # ── 9. Generate speed cameras ──
        n_cameras = rng.randint(*params["cameras"])
        inst._speed_cameras_ahead = cls._generate_cameras(
            rng, speed_limit, n_cameras, is_highway
        )

        return inst

    # ── Element generators ───────────────────────────────────

    @classmethod
    def _pick_congestion(cls, rng, road_type, surface):
        """Pick congestion level with road-type-aware probabilities."""
        if road_type == "highway":
            choices = ["free", "free", "free", "light", "light", "moderate", "heavy"]
        elif road_type == "urban":
            choices = ["free", "light", "light", "moderate", "moderate", "heavy", "heavy", "gridlock"]
        elif road_type == "residential":
            choices = ["free", "free", "light", "moderate", "moderate"]
        else:  # rural
            choices = ["free", "free", "free", "free", "light"]

        # Flooding/icy always escalates congestion
        if surface in ("flooded", "icy"):
            choices = ["moderate", "heavy", "heavy", "gridlock"]

        return rng.choice(choices)

    @classmethod
    def _generate_intersections(cls, rng, main_road, road_type, count):
        """Generate a list of intersections with traffic lights."""
        if count == 0:
            return []

        cross_pool = list(cls.CROSS_ROADS.get(road_type, cls.CROSS_ROADS["urban"]))
        rng.shuffle(cross_pool)

        intersections = []
        base_dist = rng.randint(60, 200)

        for i in range(count):
            cross = cross_pool[i % len(cross_pool)]
            dist = base_dist + i * rng.randint(150, 400)

            # Traffic light generation
            signal = rng.choices(
                ["red", "green", "yellow", "flashing_yellow"],
                weights=[35, 40, 10, 15],
                k=1
            )[0]

            if signal == "flashing_yellow":
                remaining = 0
                is_flashing = True
            elif signal == "yellow":
                remaining = rng.randint(1, 5)
                is_flashing = False
            elif signal == "red":
                remaining = rng.randint(10, 90)
                is_flashing = False
            else:  # green
                remaining = rng.randint(5, 45)
                is_flashing = False

            intersections.append({
                "name": f"{main_road.split(' (')[0]} & {cross}",
                "distance_meters": dist,
                "traffic_light": {
                    "signal": signal,
                    "remaining_seconds": remaining,
                    "is_flashing": is_flashing,
                },
            })

        return intersections

    @classmethod
    def _generate_road_events(cls, rng, road_type, total_lanes, count, surface):
        """Generate road events appropriate for the road type and conditions."""
        if count == 0:
            return []

        # Determine available event types based on context
        if road_type == "highway":
            type_weights = {
                "construction": 35, "accident": 25, "congestion": 30,
                "road_closure": 10, "flooding": 0,
            }
        elif road_type == "urban":
            type_weights = {
                "construction": 25, "accident": 20, "congestion": 35,
                "flooding": 10, "road_closure": 10,
            }
        elif road_type == "residential":
            type_weights = {
                "construction": 20, "accident": 10, "congestion": 50,
                "flooding": 10, "road_closure": 10,
            }
        else:  # rural
            type_weights = {
                "construction": 30, "accident": 15, "congestion": 10,
                "flooding": 20, "road_closure": 25,
            }

        # Boost flooding if surface is already wet/flooded
        if surface in ("wet", "flooded"):
            type_weights["flooding"] = type_weights.get("flooding", 0) + 30

        event_types = list(type_weights.keys())
        weights = [type_weights[t] for t in event_types]

        events = []
        base_dist = rng.randint(50, 500)

        for i in range(count):
            etype = rng.choices(event_types, weights=weights, k=1)[0]
            dist = base_dist + i * rng.randint(300, 1000)
            lanes_affected = min(rng.randint(1, 2), total_lanes - 1) if total_lanes > 1 else 1

            event = cls._build_event(rng, etype, dist, lanes_affected, total_lanes)
            events.append(event)

        return events

    @classmethod
    def _build_event(cls, rng, etype, distance, lanes_affected, total_lanes):
        """Build a single road event dict."""
        severity = rng.choices(
            ["minor", "moderate", "severe"],
            weights=[25, 45, 30],
            k=1
        )[0]

        if etype == "construction":
            desc = rng.choice(cls.CONSTRUCTION_DESCRIPTIONS).format(lanes=lanes_affected)
            length = rng.randint(200, 1200)
            speed_in_zone = rng.choice([20, 30, 40, 60])
            delay = rng.randint(5, 30)
            detour = rng.choice([True, False])
        elif etype == "accident":
            desc = rng.choice(cls.ACCIDENT_DESCRIPTIONS).format(lanes=lanes_affected)
            length = rng.randint(50, 300)
            speed_in_zone = rng.choice([10, 15, 20, 30])
            delay = rng.randint(10, 45)
            detour = True if severity == "severe" else rng.choice([True, False])
        elif etype == "congestion":
            desc = rng.choice(cls.CONGESTION_DESCRIPTIONS)
            length = rng.randint(500, 3000)
            speed_in_zone = None
            delay = rng.randint(5, 35)
            lanes_affected = total_lanes
            detour = rng.choice([True, True, False])
        elif etype == "flooding":
            depth = rng.choice([10, 15, 20, 25, 30, 40])
            desc = rng.choice(cls.FLOODING_DESCRIPTIONS).format(depth=depth)
            length = rng.randint(100, 600)
            speed_in_zone = rng.choice([10, 15, 20])
            delay = rng.randint(10, 40)
            detour = True if depth >= 25 else rng.choice([True, False])
        elif etype == "road_closure":
            desc = rng.choice(cls.CLOSURE_DESCRIPTIONS)
            length = rng.randint(200, 1000)
            speed_in_zone = 0
            delay = rng.randint(15, 60)
            lanes_affected = total_lanes
            detour = True
            severity = "severe"
        else:
            desc = "Unknown road event"
            length = 100
            speed_in_zone = 30
            delay = 10
            detour = False

        return {
            "type": etype,
            "distance_meters": distance,
            "length_meters": length,
            "description": desc,
            "severity": severity,
            "lanes_affected": lanes_affected,
            "detour_available": detour,
            "speed_limit_in_zone": speed_in_zone,
            "estimated_delay_minutes": delay,
        }

    @classmethod
    def _generate_cameras(cls, rng, road_speed_limit, count, is_highway):
        """Generate speed cameras at reasonable distances."""
        if count == 0:
            return []

        cameras = []
        base_dist = rng.randint(100, 800) if not is_highway else rng.randint(500, 2000)

        for i in range(count):
            dist = base_dist + i * rng.randint(400, 1500)
            cam_type = rng.choices(
                ["fixed", "temporary", "mobile"],
                weights=[50, 30, 20] if is_highway else [60, 25, 15],
                k=1
            )[0]

            # Camera limit matches road limit or nearby construction zone limit
            cam_limit = rng.choice([road_speed_limit, road_speed_limit,
                                    road_speed_limit - 20]) if road_speed_limit > 40 else road_speed_limit

            cameras.append({
                "distance_meters": dist,
                "speed_limit": max(20, cam_limit),
                "type": cam_type,
            })

        return cameras
