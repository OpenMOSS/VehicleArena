"""
Coupling Rules — Safety Constraints (HARD)
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




# ══════════════════════════════════════════════
#  HARD Safety Constraints (Rules 22-27)
# ══════════════════════════════════════════════


# ──────────────────────────────────────────────
# Rule 22: [HARD] Door Open while Driving
# ──────────────────────────────────────────────
def _register_door_driving_hard_rules(event_bus: EventBus, constraint_engine: ConstraintEngine):
    """
    Block opening any car door while vehicle is in motion (navigation active).
    This is a critical safety constraint — opening doors at speed can be fatal.
    """

    def check_door_open_while_driving(module_instance, method_name, args, kwargs, vw):
        if vw is None:
            return None

        # carcontrol_carDoor_switch(self, action, position=...)
        # Only block "open" action
        action = args[1] if len(args) > 1 else kwargs.get('action', None)
        if action != 'open':
            return None

        try:
            nav = vw.navigation
            if hasattr(nav, '_is_active') and nav._is_active:
                return ConstraintResult(
                    passed=False,
                    level=ConstraintLevel.HARD,
                    message="Vehicle is in motion (navigation active). "
                            "Opening the door is BLOCKED for safety. "
                            "Please stop the vehicle first."
                )
        except Exception as e:
            logger.debug(f"Door-driving hard check error: {e}")
        return None

    def check_door_unlock_while_driving(module_instance, method_name, args, kwargs, vw):
        """Also block unlocking doors while driving."""
        if vw is None:
            return None

        # carcontrol_carDoor_lock_switch(self, switch, position=...)
        # switch=False means unlock
        switch = args[1] if len(args) > 1 else kwargs.get('switch', None)
        if switch is not False:  # Only block unlock, not lock
            return None

        try:
            nav = vw.navigation
            if hasattr(nav, '_is_active') and nav._is_active:
                return ConstraintResult(
                    passed=False,
                    level=ConstraintLevel.HARD,
                    message="Vehicle is in motion (navigation active). "
                            "Unlocking doors is BLOCKED for safety. "
                            "Please stop the vehicle first."
                )
        except Exception as e:
            logger.debug(f"Door-unlock-driving hard check error: {e}")
        return None

    constraint_engine.register(Constraint(
        name="door_open_while_driving",
        level=ConstraintLevel.HARD,
        check_fn=check_door_open_while_driving,
        phase="pre",
        target_modules=["door"],
        target_methods=["carcontrol_carDoor_switch"]
    ))

    constraint_engine.register(Constraint(
        name="door_unlock_while_driving",
        level=ConstraintLevel.HARD,
        check_fn=check_door_unlock_while_driving,
        phase="pre",
        target_modules=["door"],
        target_methods=["carcontrol_carDoor_lock_switch"]
    ))




# ──────────────────────────────────────────────
# Rule 23: [HARD] Fuel Port Open while Driving
# ──────────────────────────────────────────────
def _register_fuelport_driving_hard_rules(event_bus: EventBus, constraint_engine: ConstraintEngine):
    """
    Block opening fuel port while vehicle is in motion.
    Opening fuel port at speed is a fire/spill hazard.
    """

    def check_fuelport_open_while_driving(module_instance, method_name, args, kwargs, vw):
        if vw is None:
            return None

        # FuelPort.switch(self, switch) — True = open
        switch = args[1] if len(args) > 1 else kwargs.get('switch', None)
        if switch is not True:
            return None

        try:
            nav = vw.navigation
            if hasattr(nav, '_is_active') and nav._is_active:
                return ConstraintResult(
                    passed=False,
                    level=ConstraintLevel.HARD,
                    message="Vehicle is in motion (navigation active). "
                            "Opening the fuel port is BLOCKED for safety. "
                            "Please stop and park the vehicle first."
                )
        except Exception as e:
            logger.debug(f"FuelPort-driving hard check error: {e}")
        return None

    constraint_engine.register(Constraint(
        name="fuelport_open_while_driving",
        level=ConstraintLevel.HARD,
        check_fn=check_fuelport_open_while_driving,
        phase="pre",
        target_modules=["fuelPort"],
        target_methods=["switch"]
    ))




# ──────────────────────────────────────────────
# Rule 24: [HARD] Turn Off Headlights at Night
# ──────────────────────────────────────────────
def _register_headlight_night_hard_rules(event_bus: EventBus, constraint_engine: ConstraintEngine):
    """
    Block turning off low-beam headlights when it is dark (night/dusk).
    Driving without headlights in darkness is extremely dangerous.
    """

    def check_lowbeam_off_at_night(module_instance, method_name, args, kwargs, vw):
        if vw is None:
            return None

        # LowBeamHeadlight.switch(self, action) — action="off" to turn off
        action = args[1] if len(args) > 1 else kwargs.get('action', None)
        if action != 'off':
            return None

        try:
            daynight = vw.externalWorld.dayNight
            if daynight.is_dark:
                return ConstraintResult(
                    passed=False,
                    level=ConstraintLevel.HARD,
                    message=f"It is currently dark (daylight level: {daynight.daylight_level}%). "
                            f"Turning off low-beam headlights is BLOCKED for safety. "
                            f"Driving without headlights in darkness is illegal and dangerous."
                )
        except Exception as e:
            logger.debug(f"Headlight-night hard check error: {e}")
        return None

    def check_positionlight_off_at_night(module_instance, method_name, args, kwargs, vw):
        """Also block turning off position lights at night."""
        if vw is None:
            return None

        switch = args[1] if len(args) > 1 else kwargs.get('switch', None)
        if switch is not False:
            return None

        try:
            daynight = vw.externalWorld.dayNight
            if daynight.is_dark:
                return ConstraintResult(
                    passed=False,
                    level=ConstraintLevel.HARD,
                    message=f"It is currently dark (daylight level: {daynight.daylight_level}%). "
                            f"Turning off position lights is BLOCKED for safety. "
                            f"Vehicle must remain visible to other drivers."
                )
        except Exception as e:
            logger.debug(f"PositionLight-night hard check error: {e}")
        return None

    constraint_engine.register(Constraint(
        name="lowbeam_off_at_night",
        level=ConstraintLevel.HARD,
        check_fn=check_lowbeam_off_at_night,
        phase="pre",
        target_modules=["lowBeamHeadlight"],
        target_methods=["switch"]
    ))

    constraint_engine.register(Constraint(
        name="positionlight_off_at_night",
        level=ConstraintLevel.HARD,
        check_fn=check_positionlight_off_at_night,
        phase="pre",
        target_modules=["positionLight"],
        target_methods=["carcontrol_positionLight_switch"]
    ))




# ──────────────────────────────────────────────
# Rule 25: [HARD] Turn Off Wiper in Heavy Rain
# ──────────────────────────────────────────────
def _register_wiper_rain_hard_rules(event_bus: EventBus, constraint_engine: ConstraintEngine):
    """
    Block turning off windshield wiper during heavy rain.
    Driving in heavy rain without wipers means near-zero visibility.
    """

    def check_wiper_off_in_heavy_rain(module_instance, method_name, args, kwargs, vw):
        if vw is None:
            return None

        # carcontrol_wiperBlade_switch(self, switch, position="all")
        switch = args[1] if len(args) > 1 else kwargs.get('switch', None)
        if switch is not False:  # Only block turning OFF
            return None

        try:
            weather = vw.externalWorld.weather
            if weather.condition.value == 'heavy_rain':
                return ConstraintResult(
                    passed=False,
                    level=ConstraintLevel.HARD,
                    message="Heavy rain detected. Turning off the windshield wiper is BLOCKED. "
                            "Visibility without wipers in heavy rain is near zero. "
                            "Keep wipers running for safety."
                )
        except Exception as e:
            logger.debug(f"Wiper-rain hard check error: {e}")
        return None

    constraint_engine.register(Constraint(
        name="wiper_off_in_heavy_rain",
        level=ConstraintLevel.HARD,
        check_fn=check_wiper_off_in_heavy_rain,
        phase="pre",
        target_modules=["wiper"],
        target_methods=["carcontrol_wiperBlade_switch"]
    ))




# ──────────────────────────────────────────────
# Rule 26: [HARD] Open Sunroof during Hail
# ──────────────────────────────────────────────
def _register_sunroof_hail_hard_rules(event_bus: EventBus, constraint_engine: ConstraintEngine):
    """
    Block opening sunroof during hail.
    Hail can cause serious injury to occupants and damage to vehicle interior.
    """

    def check_sunroof_open_in_hail(module_instance, method_name, args, kwargs, vw):
        if vw is None:
            return None

        # carcontrol_sunroof_switch(self, action) — "open" / "close"
        action = args[1] if len(args) > 1 else kwargs.get('action', None)
        if action != 'open':
            return None

        try:
            weather = vw.externalWorld.weather
            if weather.condition.value == 'hail':
                return ConstraintResult(
                    passed=False,
                    level=ConstraintLevel.HARD,
                    message="Hail is occurring. Opening the sunroof is BLOCKED. "
                            "Hailstones can cause serious injury to occupants "
                            "and damage the vehicle interior."
                )
        except Exception as e:
            logger.debug(f"Sunroof-hail hard check error: {e}")
        return None

    constraint_engine.register(Constraint(
        name="sunroof_open_in_hail",
        level=ConstraintLevel.HARD,
        check_fn=check_sunroof_open_in_hail,
        phase="pre",
        target_modules=["sunroof"],
        target_methods=["carcontrol_sunroof_switch"]
    ))




# ──────────────────────────────────────────────
# Rule 27: [HARD] Disable Child Safety Lock while Driving
# ──────────────────────────────────────────────
def _register_child_lock_driving_hard_rules(event_bus: EventBus, constraint_engine: ConstraintEngine):
    """
    Block disabling child safety lock on doors/windows while vehicle is in motion.
    Disabling child lock while driving could allow children to open doors/windows.
    """

    def check_door_child_lock_off_driving(module_instance, method_name, args, kwargs, vw):
        if vw is None:
            return None

        # carcontrol_carDoor_mode_childSafetyLock(self, switch, position=...)
        switch = args[1] if len(args) > 1 else kwargs.get('switch', None)
        if switch is not False:  # Only block disabling (switch=False)
            return None

        try:
            nav = vw.navigation
            if hasattr(nav, '_is_active') and nav._is_active:
                return ConstraintResult(
                    passed=False,
                    level=ConstraintLevel.HARD,
                    message="Vehicle is in motion (navigation active). "
                            "Disabling child safety lock on doors is BLOCKED. "
                            "Children could accidentally open doors while driving."
                )
        except Exception as e:
            logger.debug(f"ChildLock-door-driving hard check error: {e}")
        return None

    def check_window_child_lock_off_driving(module_instance, method_name, args, kwargs, vw):
        if vw is None:
            return None

        switch = args[1] if len(args) > 1 else kwargs.get('switch', None)
        if switch is not False:
            return None

        try:
            nav = vw.navigation
            if hasattr(nav, '_is_active') and nav._is_active:
                return ConstraintResult(
                    passed=False,
                    level=ConstraintLevel.HARD,
                    message="Vehicle is in motion (navigation active). "
                            "Disabling child safety lock on windows is BLOCKED. "
                            "Children could accidentally open windows while driving."
                )
        except Exception as e:
            logger.debug(f"ChildLock-window-driving hard check error: {e}")
        return None

    constraint_engine.register(Constraint(
        name="door_child_lock_off_while_driving",
        level=ConstraintLevel.HARD,
        check_fn=check_door_child_lock_off_driving,
        phase="pre",
        target_modules=["door"],
        target_methods=["carcontrol_carDoor_mode_childSafetyLock"]
    ))

    constraint_engine.register(Constraint(
        name="window_child_lock_off_while_driving",
        level=ConstraintLevel.HARD,
        check_fn=check_window_child_lock_off_driving,
        phase="pre",
        target_modules=["window"],
        target_methods=["carcontrol_window_mode_childSafetyLock"]
    ))



def register_safety_rules(event_bus: EventBus, constraint_engine: ConstraintEngine):
    """Register all safety constraints (hard) rules."""
    _register_door_driving_hard_rules(event_bus, constraint_engine)
    _register_fuelport_driving_hard_rules(event_bus, constraint_engine)
    _register_headlight_night_hard_rules(event_bus, constraint_engine)
    _register_wiper_rain_hard_rules(event_bus, constraint_engine)
    _register_sunroof_hail_hard_rules(event_bus, constraint_engine)
    _register_child_lock_driving_hard_rules(event_bus, constraint_engine)
