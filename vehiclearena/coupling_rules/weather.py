"""
Coupling Rules — Weather Reactions
"""

import logging
from event_bus import EventBus, Event, EventPriority
from constraints import (
    ConstraintEngine, Constraint, ConstraintLevel, ConstraintResult,
)

logger = logging.getLogger(__name__)


def _get_coupling_state(constraint_engine):
    """Get per-instance coupling state dict (thread-safe)."""
    if not hasattr(constraint_engine, '_coupling_state'):
        constraint_engine._coupling_state = {}
    return constraint_engine._coupling_state




# ──────────────────────────────────────────────
# Rule 7: Weather → Wiper (rain advisory)
# ──────────────────────────────────────────────
def _register_weather_wiper_rules(event_bus: EventBus, constraint_engine: ConstraintEngine):
    """When weather changes to rainy/heavy_rain, advise turning on wipers."""

    def on_weather_changed_wiper(event: Event):
        vw = constraint_engine._vw
        if vw is None:
            return

        new_condition = event.data.get('new_condition', '')
        if new_condition in ('rainy', 'heavy_rain'):
            try:
                wiper_dict = vw.wiper.to_dict()
                # Check if front wiper is off
                front_state = wiper_dict.get("front_wiper", {}).get("value", {})
                is_on = front_state.get("is_on", {}).get("value", False) if isinstance(front_state, dict) else False
                if not is_on:
                    logger.info(
                        f"Weather-Wiper advisory: It's {new_condition}, consider turning on wipers."
                    )
                    return {
                        "advisory": "weather_wiper",
                        "message": f"It's {new_condition}. Consider turning on the wipers for better visibility."
                    }
            except Exception as e:
                logger.debug(f"Weather-Wiper check error: {e}")

    event_bus.subscribe("weather.changed", on_weather_changed_wiper, EventPriority.NORMAL)




# ──────────────────────────────────────────────
# Rule 8: Weather → Fog Light (fog advisory)
# ──────────────────────────────────────────────
def _register_weather_foglight_rules(event_bus: EventBus, constraint_engine: ConstraintEngine):
    """When weather changes to foggy, advise turning on fog lights."""

    def on_weather_changed_foglight(event: Event):
        vw = constraint_engine._vw
        if vw is None:
            return

        new_condition = event.data.get('new_condition', '')
        if new_condition == 'foggy':
            try:
                fog_dict = vw.fogLight.to_dict()
                front_on = fog_dict.get("front_light", {}).get("value", {}).get("is_on", False)
                if not front_on:
                    logger.info(
                        "Weather-FogLight advisory: Foggy conditions detected, "
                        "consider turning on front fog lights for visibility."
                    )
                    return {
                        "advisory": "weather_foglight",
                        "message": "Foggy conditions detected. Consider turning on front fog lights "
                                   "to improve visibility."
                    }
            except Exception as e:
                logger.debug(f"Weather-FogLight check error: {e}")

    event_bus.subscribe("weather.changed", on_weather_changed_foglight, EventPriority.NORMAL)



def register_weather_rules(event_bus: EventBus, constraint_engine: ConstraintEngine):
    """Register all weather reactions rules."""
    _register_weather_wiper_rules(event_bus, constraint_engine)
    _register_weather_foglight_rules(event_bus, constraint_engine)
