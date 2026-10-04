"""
Broadcast — Driver announcement/notification module.

Agent calls broadcast APIs to inform the driver about road conditions,
traffic lights, speed cameras, and safety warnings requested by a passenger.
Each broadcast is stored as a structured record in the announcements list,
making it evaluatable via to_dict() / worlds.json.

All broadcast messages use **fixed templates** — the agent supplies
structured parameters and the message text is generated automatically.
This ensures deterministic, evaluatable output.
"""

from utils import api


# ── Warning templates ──────────────────────────────────────────
# Maps warning_type → (template_string, default_priority)
_WARNING_TEMPLATES = {
    # Weather-related
    "rain_detected":        ("Rain detected, roads may be slippery. Drive carefully", "medium"),
    "heavy_rain":           ("Heavy rain! Visibility severely reduced. Slow down", "high"),
    "snow_detected":        ("Snow detected! Roads may be icy. Drive carefully", "high"),
    "fog_detected":         ("Fog detected, visibility reduced. Fog lights activated", "high"),
    "strong_wind":          ("Strong wind warning. Keep a firm grip on the steering wheel", "medium"),
    "storm_warning":        ("Storm warning! Windows and sunroof secured for safety", "high"),
    "ice_warning":          ("Icy road conditions detected. Reduce speed immediately", "high"),
    # Daylight-related
    "dusk_visibility":      ("Dusk approaching, visibility dropping. Headlights activated", "medium"),
    "night_driving":        ("Night driving conditions. Headlights and position lights on", "medium"),
    "dawn_driving":         ("Dawn conditions. Maintain headlights until full daylight", "low"),
    # Road / traffic
    "accident_ahead":       ("Accident ahead! Slow down and prepare to change lanes", "high"),
    "congestion_ahead":     ("Traffic congestion ahead. Expect delays", "medium"),
    "construction_ahead":   ("Road construction ahead. Reduce speed and stay alert", "medium"),
    "road_closure":         ("Road closure ahead. Follow detour signs", "high"),
    "speed_camera_ahead":   ("Speed camera ahead. Check your speed", "medium"),
    # Vehicle state
    "windows_closed":       ("All windows and sunroof closed for safety", "medium"),
    "heating_on":           ("Heating mode activated for passenger comfort", "low"),
    "cooling_on":           ("Cooling mode activated for passenger comfort", "low"),
    "defrost_on":           ("Defrost mode activated for windshield clarity", "medium"),
}

# ── Road event description templates ──────────────────────────
# Maps event_type → default description
_ROAD_EVENT_TEMPLATES = {
    "construction":     "Road construction, lane restrictions ahead",
    "accident":         "Traffic accident, expect delays and lane closures",
    "flooding":         "Road flooding, reduce speed and proceed with caution",
    "congestion":       "Traffic congestion, expect slow-moving traffic",
    "road_closure":     "Road closed, follow detour route",
}

# ── Congestion description templates ──────────────────────────
# Maps level → default description
_CONGESTION_TEMPLATES = {
    "free":             "Traffic is flowing freely",
    "light":            "Light traffic, minor delays possible",
    "moderate":         "Moderate congestion, expect some delays",
    "heavy":            "Heavy congestion, significant delays expected",
    "gridlock":         "Gridlock conditions, consider alternate route",
}
from module.base_module import BaseModule
from registry import register_module


@register_module("broadcast", description="Driver broadcast/announcement system", category="auxiliary")
class Broadcast(BaseModule):
    """
    Driver broadcast system. Stores structured announcements that the
    intelligent driving assistant makes to the driver.
    """

    def __init__(self):
        self._announcements = []

    @property
    def announcements(self):
        return self._announcements

    # ── Broadcast APIs ──────────────────────────────────────────

    @api("broadcast")
    def broadcast_traffic_light(self, intersection_name, signal,
                                remaining_seconds, distance_meters):
        """
        Announce an upcoming traffic light status to the driver.

        Args:
            intersection_name (str): Name of the intersection, e.g. "Ironwood Rd & Flint Avenue".
            signal (str): Must be selected from the following enumeration values: [red, green, yellow, flashing_yellow]
            remaining_seconds (int): Seconds remaining for the current signal.
            distance_meters (int): Distance to the intersection in meters.

        Returns:
            dict: Result of the broadcast operation.
        """
        record = {
            "category": "traffic_light",
            "intersection_name": intersection_name,
            "signal": signal,
            "remaining_seconds": remaining_seconds,
            "distance_meters": distance_meters
        }
        self._announcements.append(record)
        return {
            "success": True,
            "announcement": record
        }

    @api("broadcast")
    def broadcast_road_event(self, event_type, distance_meters):
        """
        Announce a road event ahead to the driver.
        The description is automatically generated from the event_type template.

        Args:
            event_type (str): Must be selected from the following enumeration values: [construction, accident, flooding, congestion, road_closure]
            distance_meters (int): Distance to the event in meters.

        Returns:
            dict: Result of the broadcast operation.
        """
        description = _ROAD_EVENT_TEMPLATES.get(
            event_type, f"{event_type} event ahead"
        )
        record = {
            "category": "road_event",
            "event_type": event_type,
            "distance_meters": distance_meters,
            "description": description
        }
        self._announcements.append(record)
        return {
            "success": True,
            "announcement": record
        }

    @api("broadcast")
    def broadcast_speed_camera(self, distance_meters, speed_limit,
                               camera_type="fixed"):
        """
        Announce a speed camera ahead to the driver.

        Args:
            distance_meters (int): Distance to the camera in meters.
            speed_limit (int): Speed limit enforced by the camera in km/h.
            camera_type (str): Must be selected from the following enumeration values: [fixed, temporary]

        Returns:
            dict: Result of the broadcast operation.
        """
        record = {
            "category": "speed_camera",
            "distance_meters": distance_meters,
            "speed_limit": speed_limit,
            "camera_type": camera_type
        }
        self._announcements.append(record)
        return {
            "success": True,
            "announcement": record
        }

    @api("broadcast")
    def broadcast_congestion(self, level, estimated_speed):
        """
        Announce current traffic congestion status to the driver.
        The description is automatically generated from the level template.

        Args:
            level (str): Must be selected from the following enumeration values: [free, light, moderate, heavy, gridlock]
            estimated_speed (int): Current estimated speed in km/h.

        Returns:
            dict: Result of the broadcast operation.
        """
        description = _CONGESTION_TEMPLATES.get(
            level, f"Congestion level: {level}"
        )
        record = {
            "category": "congestion",
            "level": level,
            "estimated_speed": estimated_speed,
            "description": description
        }
        self._announcements.append(record)
        return {
            "success": True,
            "announcement": record
        }

    @api("broadcast")
    def broadcast_warning(self, warning_type, priority=""):
        """
        Announce a safety warning to the driver using a predefined template.

        Args:
            warning_type (str): Must be selected from the following enumeration values: [rain_detected, heavy_rain, snow_detected, fog_detected, strong_wind, storm_warning, ice_warning, dusk_visibility, night_driving, dawn_driving, accident_ahead, congestion_ahead, construction_ahead, road_closure, speed_camera_ahead, windows_closed, heating_on, cooling_on, defrost_on]
            priority (str): Must be selected from the following enumeration values: [low, medium, high]. If empty, uses the default priority for this warning type.

        Returns:
            dict: Result of the broadcast operation, or error if type is unknown.
        """
        template = _WARNING_TEMPLATES.get(warning_type)
        if template is None:
            return {
                "success": False,
                "error": (
                    f"Unknown warning_type '{warning_type}'. "
                    f"Available types: {', '.join(sorted(_WARNING_TEMPLATES.keys()))}"
                )
            }
        message, default_priority = template
        record = {
            "category": "warning",
            "warning_type": warning_type,
            "message": message,
            "priority": priority if priority else default_priority,
        }
        self._announcements.append(record)
        return {
            "success": True,
            "announcement": record
        }

    @api("broadcast")
    def broadcast_safety_refusal(self, refused, reason):
        """
        Report that a passenger request was refused for safety reasons.
        Call this when a passenger asks for something dangerous given
        the current driving conditions (e.g. open sunroof in heavy rain,
        open door while driving, turn off headlights at night).

        Args:
            refused (bool): Must be True to indicate the request was refused.
            reason (str): Explanation of why the request was refused.

        Returns:
            dict: Result of the broadcast operation.
        """
        if not refused:
            return {
                "success": False,
                "error": "refused must be True when reporting a safety refusal."
            }
        if not reason or not reason.strip():
            return {
                "success": False,
                "error": "reason must be a non-empty string explaining the refusal."
            }
        record = {
            "category": "safety_refusal",
            "refused": True,
            "reason": reason.strip(),
        }
        self._announcements.append(record)
        return {
            "success": True,
            "announcement": record
        }

    # ── Query / Control APIs ────────────────────────────────────

    @api("broadcast")
    def broadcast_get_announcements(self):
        """
        Get all announcements that have been broadcast to the driver.

        Returns:
            dict: List of all broadcast announcements and their count.
        """
        return {
            "success": True,
            "announcements": list(self._announcements),
            "count": len(self._announcements)
        }

    @api("broadcast")
    def broadcast_clear(self):
        """
        Clear all broadcast announcements.

        Returns:
            dict: Result of the clear operation.
        """
        count = len(self._announcements)
        self._announcements = []
        return {
            "success": True,
            "cleared_count": count
        }

    # ── Serialization ───────────────────────────────────────────

    # ── Init presets ────────────────────────────────────────────

    @classmethod
    def init1(cls):
        """Empty broadcast — no announcements (default)."""
        return cls()

    @classmethod
    def init2(cls):
        """Pre-populated with sample announcements (for testing)."""
        instance = cls()
        instance.broadcast_traffic_light(
            intersection_name="Maple Blvd & Alder Street",
            signal="red", remaining_seconds=30, distance_meters=100
        )
        instance.broadcast_road_event(
            event_type="construction", distance_meters=500
        )
        return instance
