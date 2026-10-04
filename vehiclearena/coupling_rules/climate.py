"""
Coupling Rules — Climate & Comfort
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
# Rule 3: Window-AC Conflict
# ──────────────────────────────────────────────
def _register_window_ac_rules(event_bus: EventBus, constraint_engine: ConstraintEngine):
    """
    When a window is opened, if AC is running, emit an advisory warning.
    This doesn't block the operation but informs the user.
    """

    def on_window_opened(event: Event):
        """Check AC status when a window is opened."""
        vw = constraint_engine._vw
        if vw is None:
            return

        # Check if any AC zone is active
        try:
            ac_dict = vw.airConditioner.to_dict()
            ac_mode = ac_dict.get("ac_mode", {}).get("value", "off")
            if ac_mode != "off":
                position = event.data.get('position', 'unknown')
                logger.info(
                    f"Window-AC conflict: Window at {position} opened while AC is in '{ac_mode}' mode. "
                    f"AC efficiency may be reduced."
                )
                return {
                    "advisory": "window_ac_conflict",
                    "message": f"Air conditioning is running in '{ac_mode}' mode. "
                               f"Opening the window may reduce AC efficiency."
                }
        except Exception as e:
            logger.debug(f"Window-AC check error: {e}")

    def on_sunroof_opened(event: Event):
        """Check AC status when sunroof is opened."""
        vw = constraint_engine._vw
        if vw is None:
            return

        try:
            ac_dict = vw.airConditioner.to_dict()
            ac_mode = ac_dict.get("ac_mode", {}).get("value", "off")
            if ac_mode != "off":
                logger.info(
                    f"Sunroof-AC conflict: Sunroof opened while AC is in '{ac_mode}' mode."
                )
                return {
                    "advisory": "sunroof_ac_conflict",
                    "message": f"Air conditioning is running. Opening the sunroof may reduce AC efficiency."
                }
        except Exception as e:
            logger.debug(f"Sunroof-AC check error: {e}")

    event_bus.subscribe("window.opened", on_window_opened, EventPriority.NORMAL)
    event_bus.subscribe("sunroof.opened", on_sunroof_opened, EventPriority.NORMAL)




# ──────────────────────────────────────────────
# Rule 5: Door/Sunroof Safety
# ──────────────────────────────────────────────
def _register_door_sunroof_rules(event_bus: EventBus, constraint_engine: ConstraintEngine):
    """
    Opening a door at high speed should be blocked (HARD constraint).
    Door opened -> auto turn on reading light (if auto mode is on).
    """

    def on_door_opened(event: Event):
        """When a door is opened, auto-enable the corresponding reading light if auto mode is on."""
        vw = constraint_engine._vw
        if vw is None:
            return

        try:
            if vw.readingLight.auto_mode:
                position = event.data.get('position', '')
                # Map door positions to reading light positions
                from module.readinglight import ReadingLight
                for pos_enum in ReadingLight.Position:
                    if pos_enum.value == position:
                        light = vw.readingLight._lights.get(pos_enum)
                        if light and not light.is_on:
                            light.is_on = True
                            logger.info(
                                f"Auto reading light: Turned on {position} reading light "
                                f"because door was opened (auto mode)"
                            )
                        break
        except Exception as e:
            logger.debug(f"Door-readinglight coupling error: {e}")

    def on_door_closed(event: Event):
        """When a door is closed, auto-disable the corresponding reading light if auto mode is on."""
        vw = constraint_engine._vw
        if vw is None:
            return

        try:
            if vw.readingLight.auto_mode:
                position = event.data.get('position', '')
                from module.readinglight import ReadingLight
                for pos_enum in ReadingLight.Position:
                    if pos_enum.value == position:
                        light = vw.readingLight._lights.get(pos_enum)
                        if light and light.is_on:
                            light.is_on = False
                            logger.info(
                                f"Auto reading light: Turned off {position} reading light "
                                f"because door was closed (auto mode)"
                            )
                        break
        except Exception as e:
            logger.debug(f"Door-readinglight coupling error: {e}")

    event_bus.subscribe("door.opened", on_door_opened, EventPriority.NORMAL)
    event_bus.subscribe("door.closed", on_door_closed, EventPriority.NORMAL)




# ──────────────────────────────────────────────
# Rule 9: Weather → AC (extreme temperature advisory)
# ──────────────────────────────────────────────
def _register_weather_ac_rules(event_bus: EventBus, constraint_engine: ConstraintEngine):
    """When temperature is extreme, advise AC mode."""

    def on_temperature_changed(event: Event):
        vw = constraint_engine._vw
        if vw is None:
            return

        new_temp = event.data.get('new_temperature', None)
        if new_temp is None:
            return

        if new_temp > 35:
            logger.info(
                f"Weather-AC advisory: Outdoor temperature is {new_temp}°C. "
                f"Consider enabling AC cooling mode."
            )
            return {
                "advisory": "weather_ac_hot",
                "message": f"Outdoor temperature is {new_temp}°C. Consider enabling AC cooling mode."
            }
        elif new_temp < 0:
            logger.info(
                f"Weather-AC advisory: Outdoor temperature is {new_temp}°C. "
                f"Consider enabling AC heating mode."
            )
            return {
                "advisory": "weather_ac_cold",
                "message": f"Outdoor temperature is {new_temp}°C. Consider enabling AC heating mode."
            }

    event_bus.subscribe("weather.temperature_changed", on_temperature_changed, EventPriority.NORMAL)




# ──────────────────────────────────────────────
# Rule 10: Weather → Sunroof/Window (rain/snow close advisory)
# ──────────────────────────────────────────────
def _register_weather_window_rules(event_bus: EventBus, constraint_engine: ConstraintEngine):
    """When it's raining/snowing, warn about opening windows/sunroof."""

    def check_weather_before_window(module_instance, method_name, args, kwargs, vw):
        """Pre-check: warn if opening window during rain/snow."""
        switch = kwargs.get("switch", args[1] if len(args) > 1 else None)
        if vw is None or switch is not True:
            return None

        try:
            weather = vw.externalWorld.weather.condition.value
            if weather in (
                    'rainy', 'heavy_rain', 'snowy', 'heavy_snow', 'hail'):
                return ConstraintResult(
                    passed=False,
                    level=ConstraintLevel.SOFT,
                    message=f"Current weather is {weather}. Opening the window is not recommended "
                            f"as it may let rain/snow into the vehicle."
                )
        except Exception as e:
            logger.debug(f"Weather-Window constraint error: {e}")
        return None

    def check_weather_before_sunroof(module_instance, method_name, args, kwargs, vw):
        """Pre-check: warn if opening sunroof during rain/snow."""
        action = kwargs.get("action", args[0] if args else None)
        action = getattr(action, "value", action)
        if vw is None or action not in ("open", "Tilt"):
            return None

        try:
            weather = vw.externalWorld.weather.condition.value
            if weather in (
                    'rainy', 'heavy_rain', 'snowy', 'heavy_snow', 'hail'):
                return ConstraintResult(
                    passed=False,
                    level=ConstraintLevel.SOFT,
                    message=f"Current weather is {weather}. Opening the sunroof is not recommended."
                )
        except Exception as e:
            logger.debug(f"Weather-Sunroof constraint error: {e}")
        return None

    constraint_engine.register(Constraint(
        name="weather_window_rain_check",
        level=ConstraintLevel.SOFT,
        check_fn=check_weather_before_window,
        phase="pre",
        target_modules=["window"],
        target_methods=["carcontrol_window_switch"]
    ))

    constraint_engine.register(Constraint(
        name="weather_sunroof_rain_check",
        level=ConstraintLevel.SOFT,
        check_fn=check_weather_before_sunroof,
        phase="pre",
        target_modules=["sunroof"],
        target_methods=["carcontrol_sunroof_switch"]
    ))




# ──────────────────────────────────────────────
# Rule 14: Seat Heater + Ventilation Mutex
# ──────────────────────────────────────────────
def _register_seat_heater_ventilation_rules(event_bus: EventBus, constraint_engine: ConstraintEngine):
    """
    Seat heater and seat ventilation on the same seat are mutually exclusive.
    Turning on heater when ventilation is active → SOFT warning.
    Turning on ventilation when heater is active → SOFT warning.
    """

    def check_heater_when_ventilation_on(module_instance, method_name, args, kwargs, vw):
        """Warn if turning on seat heater while ventilation is active on same seat."""
        if vw is None:
            return None

        # Extract switch param
        switch = args[1] if len(args) > 1 else kwargs.get('switch', None)
        if switch is not True:
            return None

        # Extract position
        position = args[2] if len(args) > 2 else kwargs.get('position', None)
        positions = [position] if position else ['driver', 'passenger']

        try:
            for pos in positions:
                seat_obj = vw.seat._seats.get(pos)
                if seat_obj and seat_obj.ventilation.is_on:
                    return ConstraintResult(
                        passed=False,
                        level=ConstraintLevel.SOFT,
                        message=f"Seat ventilation is active on {pos} seat. "
                                f"Running heater and ventilation simultaneously is inefficient. "
                                f"Consider turning off ventilation first."
                    )
        except Exception as e:
            logger.debug(f"Seat heater-ventilation check error: {e}")
        return None

    def check_ventilation_when_heater_on(module_instance, method_name, args, kwargs, vw):
        """Warn if turning on seat ventilation while heater is active on same seat."""
        if vw is None:
            return None

        switch = args[1] if len(args) > 1 else kwargs.get('switch', None)
        if switch is not True:
            return None

        position = args[2] if len(args) > 2 else kwargs.get('position', None)
        positions = [position] if position else ['driver', 'passenger']

        try:
            for pos in positions:
                seat_obj = vw.seat._seats.get(pos)
                if seat_obj and seat_obj.heater.is_on:
                    return ConstraintResult(
                        passed=False,
                        level=ConstraintLevel.SOFT,
                        message=f"Seat heater is active on {pos} seat. "
                                f"Running ventilation and heater simultaneously is inefficient. "
                                f"Consider turning off heater first."
                    )
        except Exception as e:
            logger.debug(f"Seat ventilation-heater check error: {e}")
        return None

    constraint_engine.register(Constraint(
        name="seat_heater_ventilation_conflict",
        level=ConstraintLevel.SOFT,
        check_fn=check_heater_when_ventilation_on,
        phase="pre",
        target_modules=["seat"],
        target_methods=["carcontrol_carSeat_heater_switch"]
    ))

    constraint_engine.register(Constraint(
        name="seat_ventilation_heater_conflict",
        level=ConstraintLevel.SOFT,
        check_fn=check_ventilation_when_heater_on,
        phase="pre",
        target_modules=["seat"],
        target_methods=["carcontrol_carSeat_ventilation_switch"]
    ))




# ──────────────────────────────────────────────
# Rule 15: Seat Heater + AC Cooling Conflict
# ──────────────────────────────────────────────
def _register_seat_heater_ac_rules(event_bus: EventBus, constraint_engine: ConstraintEngine):
    """
    Turning on seat heater while AC is in cooling mode is contradictory.
    """

    def check_seat_heater_vs_ac_cooling(module_instance, method_name, args, kwargs, vw):
        if vw is None:
            return None

        switch = args[1] if len(args) > 1 else kwargs.get('switch', None)
        if switch is not True:
            return None

        try:
            ac = vw.airConditioner
            if hasattr(ac, '_ac_mode') and ac._ac_mode == 'cooling':
                return ConstraintResult(
                    passed=False,
                    level=ConstraintLevel.SOFT,
                    message="Air conditioning is in cooling mode. Turning on seat heater "
                            "while cooling is active is contradictory and wastes energy."
                )
        except Exception as e:
            logger.debug(f"Seat heater-AC check error: {e}")
        return None

    constraint_engine.register(Constraint(
        name="seat_heater_ac_cooling_conflict",
        level=ConstraintLevel.SOFT,
        check_fn=check_seat_heater_vs_ac_cooling,
        phase="pre",
        target_modules=["seat"],
        target_methods=["carcontrol_carSeat_heater_switch"]
    ))




# ──────────────────────────────────────────────
# Rule 19: Defrost + Recirculation Conflict
# ──────────────────────────────────────────────
def _register_defrost_recirculation_rules(event_bus: EventBus, constraint_engine: ConstraintEngine):
    """
    Defrost mode works best with fresh outside air.
    If user turns on defrost while recirculation is on → SOFT warning.
    If user turns on recirculation while defrost is on → SOFT warning.
    """

    def check_defrost_vs_recirculation(module_instance, method_name, args, kwargs, vw):
        """Warn if enabling defrost while in recirculation mode."""
        if vw is None:
            return None

        switch = args[1] if len(args) > 1 else kwargs.get('switch', None)
        if switch is not True:
            return None

        try:
            ac = vw.airConditioner
            if hasattr(ac, '_recycle_mode') and ac._recycle_mode == 'recirculation':
                return ConstraintResult(
                    passed=False,
                    level=ConstraintLevel.SOFT,
                    message="Air recirculation mode is active. Defrost works best with fresh outside air. "
                            "Consider switching to fresh air mode for effective defrosting."
                )
        except Exception as e:
            logger.debug(f"Defrost-recirculation check error: {e}")
        return None

    def check_recirculation_vs_defrost(module_instance, method_name, args, kwargs, vw):
        """Warn if switching to recirculation while defrost is active."""
        if vw is None:
            return None

        mode = args[1] if len(args) > 1 else kwargs.get('mode', None)
        if mode != 'recirculation':
            return None

        try:
            ac = vw.airConditioner
            if hasattr(ac, '_defrost_on') and ac._defrost_on:
                return ConstraintResult(
                    passed=False,
                    level=ConstraintLevel.SOFT,
                    message="Defrost mode is active. Switching to recirculation will reduce "
                            "defrost effectiveness. Fresh air mode is recommended during defrost."
                )
        except Exception as e:
            logger.debug(f"Recirculation-defrost check error: {e}")
        return None

    constraint_engine.register(Constraint(
        name="defrost_recirculation_conflict",
        level=ConstraintLevel.SOFT,
        check_fn=check_defrost_vs_recirculation,
        phase="pre",
        target_modules=["airConditioner"],
        target_methods=["defrost_mode_switch"]
    ))

    constraint_engine.register(Constraint(
        name="recirculation_defrost_conflict",
        level=ConstraintLevel.SOFT,
        check_fn=check_recirculation_vs_defrost,
        phase="pre",
        target_modules=["airConditioner"],
        target_methods=["recycle_mode_switch"]
    ))




# ──────────────────────────────────────────────
# Rule 20: Weather → Seat Heater Advisory
# ──────────────────────────────────────────────
def _register_weather_seat_heater_rules(event_bus: EventBus, constraint_engine: ConstraintEngine):
    """
    When outdoor temperature drops below 5°C, advise turning on seat heater.
    When outdoor temperature rises above 30°C, advise seat ventilation.
    """

    def on_temperature_changed_seat(event: Event):
        vw = constraint_engine._vw
        if vw is None:
            return

        new_temp = event.data.get('new_temperature', None)
        if new_temp is None:
            return

        try:
            if new_temp < 5:
                # Check if any seat heater is already on
                any_heater_on = False
                for pos, seat_obj in vw.seat._seats.items():
                    if seat_obj.heater.is_on:
                        any_heater_on = True
                        break
                if not any_heater_on:
                    logger.info(
                        f"Weather-Seat advisory: Temperature is {new_temp}°C. "
                        f"Consider turning on seat heater for comfort."
                    )
                    return {
                        "advisory": "weather_seat_heater",
                        "message": f"Outdoor temperature is {new_temp}°C. "
                                   f"Consider turning on seat heater for comfort."
                    }
            elif new_temp > 30:
                any_vent_on = False
                for pos, seat_obj in vw.seat._seats.items():
                    if seat_obj.ventilation.is_on:
                        any_vent_on = True
                        break
                if not any_vent_on:
                    logger.info(
                        f"Weather-Seat advisory: Temperature is {new_temp}°C. "
                        f"Consider turning on seat ventilation for comfort."
                    )
                    return {
                        "advisory": "weather_seat_ventilation",
                        "message": f"Outdoor temperature is {new_temp}°C. "
                                   f"Consider turning on seat ventilation to stay cool."
                    }
        except Exception as e:
            logger.debug(f"Weather-Seat check error: {e}")

    event_bus.subscribe("weather.temperature_changed", on_temperature_changed_seat, EventPriority.LOW)



def register_climate_rules(event_bus: EventBus, constraint_engine: ConstraintEngine):
    """Register all climate & comfort rules."""
    _register_window_ac_rules(event_bus, constraint_engine)
    _register_door_sunroof_rules(event_bus, constraint_engine)
    _register_weather_ac_rules(event_bus, constraint_engine)
    _register_weather_window_rules(event_bus, constraint_engine)
    _register_seat_heater_ventilation_rules(event_bus, constraint_engine)
    _register_seat_heater_ac_rules(event_bus, constraint_engine)
    _register_defrost_recirculation_rules(event_bus, constraint_engine)
    _register_weather_seat_heater_rules(event_bus, constraint_engine)
