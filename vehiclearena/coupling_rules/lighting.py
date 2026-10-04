"""
Coupling Rules — Lighting
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
# Rule 4: Lighting Mutex
# ──────────────────────────────────────────────
def _register_lighting_rules(event_bus: EventBus, constraint_engine: ConstraintEngine):
    """
    High beam and front fog light should not be on simultaneously (in normal conditions).
    This is a SOFT constraint - warns but doesn't block.
    """

    def check_high_beam_fog_conflict(module_instance, method_name, args, kwargs, vw):
        """Check if turning on high beam conflicts with front fog light."""
        if vw is None:
            return None

        # Only check when turning ON high beam
        if len(args) > 1:
            switch = args[1]
        else:
            switch = kwargs.get('switch', None)

        if switch is not True:
            return None

        try:
            fog_dict = vw.fogLight.to_dict()
            front_fog_on = fog_dict.get("front_light", {}).get("value", {}).get("is_on", {}).get("value", False)
            if front_fog_on:
                return ConstraintResult(
                    passed=False,
                    level=ConstraintLevel.SOFT,
                    message="Front fog lights are ON. Using high beam and fog lights simultaneously "
                            "may cause excessive glare. Consider turning off fog lights."
                )
        except Exception as e:
            logger.debug(f"Lighting constraint check error: {e}")

        return None

    def check_fog_high_beam_conflict(module_instance, method_name, args, kwargs, vw):
        """Check if turning on front fog light conflicts with high beam."""
        if vw is None:
            return None

        # Only check when turning ON fog light
        if len(args) > 1:
            switch = args[1]
        else:
            switch = kwargs.get('switch', None)

        if switch is not True:
            return None

        # Check position - only front fog light conflicts
        if len(args) > 2:
            position = args[2]
        else:
            position = kwargs.get('position', None)

        if position == 'rear':
            return None

        try:
            hb_dict = vw.highBeamHeadlight.to_dict()
            high_beam_on = hb_dict.get("high_beam_on", {}).get("value", False)
            if high_beam_on:
                return ConstraintResult(
                    passed=False,
                    level=ConstraintLevel.SOFT,
                    message="High beam headlights are ON. Using fog lights and high beam simultaneously "
                            "may cause excessive glare."
                )
        except Exception as e:
            logger.debug(f"Lighting constraint check error: {e}")

        return None

    # Register as pre-execution constraints
    constraint_engine.register(Constraint(
        name="high_beam_fog_conflict",
        level=ConstraintLevel.SOFT,
        check_fn=check_high_beam_fog_conflict,
        phase="pre",
        target_modules=["highBeamHeadlight"],
        target_methods=["carcontrol_highBeamHeadlight_switch"]
    ))

    constraint_engine.register(Constraint(
        name="fog_high_beam_conflict",
        level=ConstraintLevel.SOFT,
        check_fn=check_fog_high_beam_conflict,
        phase="pre",
        target_modules=["fogLight"],
        target_methods=["carcontrol_fogLight_switch"]
    ))




# ──────────────────────────────────────────────
# Rule 12 (was 14): DayNight → Lights (auto headlights)
# ──────────────────────────────────────────────
def _register_daynight_rules(event_bus: EventBus, constraint_engine: ConstraintEngine):
    """When daylight drops (dusk/night), advise turning on headlights. When bright, advise off."""

    def on_daynight_changed(event: Event):
        vw = constraint_engine._vw
        if vw is None:
            return

        new_period = event.data.get('new_period', '')
        new_daylight = event.data.get('new_daylight', 50)
        is_dark = event.data.get('is_dark', False)

        if is_dark:  # daylight_level < 30 (dusk/night)
            logger.info(
                f"DayNight-Lights advisory: It's {new_period} (daylight={new_daylight}). "
                f"Consider turning on low beam headlights."
            )
            return {
                "advisory": "daynight_lights_on",
                "message": f"It's getting dark ({new_period}). Consider turning on low beam headlights "
                           f"and position lights for safety."
            }
        elif new_daylight > 70:  # bright enough (morning/noon/afternoon)
            logger.info(
                f"DayNight-Lights advisory: It's {new_period} (daylight={new_daylight}). "
                f"Headlights may not be needed."
            )
            return {
                "advisory": "daynight_lights_off",
                "message": f"Good daylight conditions ({new_period}). "
                           f"You may turn off headlights if they are on."
            }

    event_bus.subscribe("daynight.changed", on_daynight_changed, EventPriority.NORMAL)




# ──────────────────────────────────────────────
# Rule 16: DayNight → HUD Brightness Auto-Adjust
# ──────────────────────────────────────────────
def _register_daynight_hud_rules(event_bus: EventBus, constraint_engine: ConstraintEngine):
    """
    When transitioning to night → advise dimming HUD to reduce glare.
    When transitioning to day → advise brightening HUD for readability.
    """

    def on_daynight_changed_hud(event: Event):
        vw = constraint_engine._vw
        if vw is None:
            return

        is_dark = event.data.get('is_dark', False)
        new_period = event.data.get('new_period', '')

        try:
            hud = vw.HUD
            if not hasattr(hud, '_is_on') or not hud._is_on:
                return  # HUD is off, no need to advise

            if is_dark:
                logger.info(
                    f"DayNight-HUD advisory: It's {new_period}. "
                    f"Consider dimming HUD to reduce glare."
                )
                return {
                    "advisory": "daynight_hud_dim",
                    "message": f"It's getting dark ({new_period}). Consider reducing HUD brightness "
                               f"to avoid windshield glare."
                }
            else:
                logger.info(
                    f"DayNight-HUD advisory: It's {new_period}. "
                    f"Consider increasing HUD brightness for visibility."
                )
                return {
                    "advisory": "daynight_hud_brighten",
                    "message": f"Daylight conditions ({new_period}). Consider increasing HUD brightness "
                               f"for better readability."
                }
        except Exception as e:
            logger.debug(f"DayNight-HUD check error: {e}")

    event_bus.subscribe("daynight.changed", on_daynight_changed_hud, EventPriority.LOW)



def register_lighting_rules(event_bus: EventBus, constraint_engine: ConstraintEngine):
    """Register all lighting rules."""
    _register_lighting_rules(event_bus, constraint_engine)
    _register_daynight_rules(event_bus, constraint_engine)
    _register_daynight_hud_rules(event_bus, constraint_engine)

