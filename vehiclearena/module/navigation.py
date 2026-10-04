"""Navigation route planning and mini-map display surface.

Navigation exposes route state and a highlighted mini-map.  It deliberately
has no voice guidance, road-condition broadcast, speed-camera alert, traffic
overlay, or audio-volume controls. Live road facts belong to CameraVisual and
installed physical sensors.
"""

from __future__ import annotations

from utils import api
from module.base_module import BaseModule
from registry import register_module


@register_module(
    "navigation",
    description="Route planning and route mini-map",
    category="navigation",
    needs_settings=True,
)
class Navigation(BaseModule):
    class RouteInfo:
        def __init__(
            self,
            departure: str = "Current location",
            destination: str = "",
        ):
            self._departure = departure
            self._destination = destination

        @property
        def departure(self):
            return self._departure

        @departure.setter
        def departure(self, value):
            self._departure = value

        @property
        def destination(self):
            return self._destination

        @destination.setter
        def destination(self, value):
            self._destination = value

    def __init__(self):
        self._settings = None
        self._is_active = False
        self._current_route = self.RouteInfo()
        self._waypoints = []

        # These are command state, not an autonomous driving policy.  The
        # simulation engine intercepts the public methods and commits them to
        # SUMO at the next fixed 0.1-second boundary.
        self._desired_speed = -1.0
        self._is_stopped = False
        self._current_target_node = ""
        self._waypoints_driving = []

    @property
    def is_active(self):
        return self._is_active

    @is_active.setter
    def is_active(self, value):
        self._is_active = bool(value)

    @property
    def current_route(self):
        return self._current_route

    @current_route.setter
    def current_route(self, value):
        self._current_route = value

    @property
    def waypoints(self):
        return self._waypoints

    @waypoints.setter
    def waypoints(self, value):
        self._waypoints = list(value)

    @api("navigation")
    def navigation_get_destination(self):
        """Return only the configured destination label, not road facts."""
        if not self.is_active or self.current_route is None:
            return {"success": False, "error": "No active navigation route"}
        return {
            "success": True,
            "destinationInfo": self.current_route.destination,
        }

    @api("navigation")
    def navigation_exit(self):
        """Clear the displayed route."""
        if not self.is_active:
            return {"success": True, "message": "Navigation is already inactive"}
        self.is_active = False
        self.current_route = None
        self.waypoints = []
        return {"success": True, "message": "Navigation exited successfully"}

    @api("navigation")
    def navigation_route_plan(
        self, address, placeOfDeparture="Current location",
    ):
        """Explicitly create or replace the saved route shown by navigation_minimap; this does not steer the vehicle."""
        if not address:
            return {"success": False, "error": "Destination address is required"}
        self.current_route = self.RouteInfo(
            destination=address, departure=placeOfDeparture)
        self.is_active = True
        self.waypoints = []
        return {"success": True, "minimap_available": True}

    @api("navigation")
    def navigation_minimap(self, scope="route"):
        """Refresh the saved route at the latest vehicle pose, removing already-traveled highlights without replanning; to change or recalculate navigation, explicitly call navigation_route_plan again.

        ``route`` shows a wider forward driving window; ``local`` shows a
        closer junction/lane window. Neither scope compresses the whole trip
        into one frame or exposes live traffic and signal answers. Changing
        scope only changes the viewing window. Driving off the saved route
        does not automatically replace it or prove it is still reachable.
        Each call redraws the remaining route from current progress; future
        route sections that turn back behind the vehicle remain visible.
        If an explicit plan fails to find a path, no route is highlighted
        until a subsequent explicit plan succeeds.
        """
        if scope not in ("route", "local"):
            return {
                "success": False,
                "error": "scope must be 'route' or 'local'",
            }
        return {"success": True, "scope": scope}

    @api("navigation")
    def navigation_select_maneuver(self, direction):
        """Choose the next junction movement from the current physical lane.

        ``direction`` must be ``left``, ``straight``, ``right`` or
        ``u_turn``.  The simulation engine validates the corresponding
        lane-to-lane connector and commits only that one movement to SUMO.
        Unique straight continuation is automatic. Before junction entry this
        command overrides default straight; after a turn, default following
        resumes on the new lane. It does not repeat the turn at later junctions.
        """
        normalized = str(direction or "").strip().lower().replace("-", "_")
        if normalized not in {"left", "straight", "right", "u_turn", "uturn"}:
            return {
                "success": False,
                "reason": "direction_must_be_left_straight_right_or_u_turn",
            }
        return {"success": False, "reason": "driving_backend_unavailable"}

    @api("navigation")
    def navigation_reroute(self):
        """Replan displayed guidance; this does not steer the vehicle."""
        if not self.is_active:
            return {"success": False, "error": "Navigation is not active"}
        return {"success": True, "minimap_available": True}

    @api("navigation")
    def navigation_destination_change(self, address):
        """Change the route destination without producing voice guidance."""
        if not self.is_active or self.current_route is None:
            return {"success": False, "error": "Navigation is not active"}
        if not address:
            return {"success": False, "error": "New destination is required"}
        old_destination = self.current_route.destination
        self.current_route.destination = address
        self.waypoints = []
        return {
            "success": True,
            "old_destination": old_destination,
            "new_destination": address,
            "minimap_available": True,
        }

    @api("navigation")
    def navigation_midWay_add(self, midway):
        """Add route waypoints without producing a textual route broadcast."""
        if not self.is_active or self.current_route is None:
            return {"success": False, "error": "No active navigation route"}
        if not isinstance(midway, list) or not midway:
            return {"success": False, "error": "Waypoints must be a non-empty list"}
        self.waypoints.extend(midway)
        return {
            "success": True,
            "added_waypoints": list(midway),
            "minimap_available": True,
        }

    @api("navigation")
    def navigation_midWay_delete(self, address=None, number=None):
        """Remove one route waypoint."""
        if not self.is_active or self.current_route is None:
            return {"success": False, "error": "No active navigation route"}
        if address is not None and number is not None:
            return {"success": False, "error": "Choose address or number"}
        if address is not None:
            if address not in self.waypoints:
                return {"success": False, "error": "Waypoint not found"}
            self.waypoints.remove(address)
        elif number is not None:
            if not isinstance(number, int) or not 0 <= number < len(self.waypoints):
                return {"success": False, "error": "Waypoint index out of range"}
            self.waypoints.pop(number)
        else:
            return {"success": False, "error": "Address or number is required"}
        return {"success": True, "minimap_available": True}

    # The following methods are command requests.  Their physical semantics
    # are installed by MultiSimEngine; these local bodies make the plugin
    # independently discoverable and testable.

    @api("navigation")
    def navigation_set_speed(
        self, speed_kmh: float, reason: str = "",
        acceleration_mps2: float | None = None,
        deceleration_mps2: float | None = None,
    ):
        """Submit a persistent longitudinal speed target.

        Args:
            speed_kmh (float): Target speed in km/h; values below zero stop.
            reason (str, optional): Short reason for the driving decision.
            acceleration_mps2 (float, optional): Positive acceleration limit.
            deceleration_mps2 (float, optional): Positive braking limit.
        """
        return {"success": False, "reason": "driving_backend_unavailable"}

    @api("navigation")
    def navigation_emergency_stop(self, reason=""):
        """Request emergency braking, not an instantaneous physical stop."""
        return {"success": False, "reason": "driving_backend_unavailable"}

    @api("navigation")
    def navigation_change_lane(self, direction):
        """Request an adjacent lane change; observe the road to verify it."""
        return {"success": False, "reason": "driving_backend_unavailable"}

    @api("navigation")
    def navigation_u_turn(self):
        """Request a map-defined U-turn; acceptance does not mean completion."""
        return {"success": False, "reason": "driving_backend_unavailable"}

    @classmethod
    def init1(cls):
        instance = cls()
        instance.is_active = True
        instance.current_route = cls.RouteInfo(destination="Shanghai")
        return instance

    @classmethod
    def init2(cls):
        return cls()
