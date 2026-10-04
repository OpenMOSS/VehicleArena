from utils import api
import sys
from module.base_module import BaseModule
from registry import register_module


@register_module("seat", description="Seat related", category="body", needs_settings=True)
class Seat(BaseModule):
    """
    A class representing a car seat with all its controllable functionalities.
    """

    class HeaterSystem:
        """
        Inner class representing the heating system of a car seat.
        """

        def __init__(self):
            self._is_on = False
            self._temperature_level = 1  # 1-5 (min-max)
            # Temperature can be specified in different units
            self._temperature_value = 20.0
            self._temperature_unit = "celsius"  # celsius, gear, percentage

        @property
        def is_on(self):
            return self._is_on

        @is_on.setter
        def is_on(self, value):
            self._is_on = value

        @property
        def temperature_level(self):
            return self._temperature_level

        @temperature_level.setter
        def temperature_level(self, value):
            if 1 <= value <= 5:
                self._temperature_level = value

        @property
        def temperature_value(self):
            return self._temperature_value

        @temperature_value.setter
        def temperature_value(self, value):
            self._temperature_value = value

        @property
        def temperature_unit(self):
            return self._temperature_unit

        @temperature_unit.setter
        def temperature_unit(self, value):
            self._temperature_unit = value


    class MassageSystem:
        """
        Inner class representing the massage system of a car seat.
        """

        def __init__(self):
            self._is_on = False
            self._intensity_level = 1  # 1-5 (min-max)
            self._intensity_value = 0.0
            self._intensity_unit = "gear"  # gear, percentage
            self._active_mode = None  # Current active massage mode

        @property
        def is_on(self):
            return self._is_on

        @is_on.setter
        def is_on(self, value):
            self._is_on = value

        @property
        def intensity_level(self):
            return self._intensity_level

        @intensity_level.setter
        def intensity_level(self, value):
            if 1 <= value <= 5:
                self._intensity_level = value

        @property
        def intensity_value(self):
            return self._intensity_value

        @intensity_value.setter
        def intensity_value(self, value):
            self._intensity_value = value

        @property
        def intensity_unit(self):
            return self._intensity_unit

        @intensity_unit.setter
        def intensity_unit(self, value):
            self._intensity_unit = value

        @property
        def active_mode(self):
            return self._active_mode

        @active_mode.setter
        def active_mode(self, value):
            valid_modes = [
                "wave",
                "cat step",
                "stretch",
                "snake",
                "butterfly",
                "shoulder",
                "upper back",
                "waist",
                "full back",
                "random",
            ]
            if value in valid_modes or value is None:
                self._active_mode = value


    class VentilationSystem:
        """
        Inner class representing the ventilation system of a car seat.
        """

        def __init__(self):
            self._is_on = False
            self._airflow_level = 1  # 1-5 (min-max)
            self._airflow_value = 0.0
            self._airflow_unit = "gear"  # gear, percentage

        @property
        def is_on(self):
            return self._is_on

        @is_on.setter
        def is_on(self, value):
            self._is_on = value

        @property
        def airflow_level(self):
            return self._airflow_level

        @airflow_level.setter
        def airflow_level(self, value):
            if 1 <= value <= 5:
                self._airflow_level = value

        @property
        def airflow_value(self):
            return self._airflow_value

        @airflow_value.setter
        def airflow_value(self, value):
            self._airflow_value = value

        @property
        def airflow_unit(self):
            return self._airflow_unit

        @airflow_unit.setter
        def airflow_unit(self, value):
            self._airflow_unit = value


    class PositionSystem:
        """
        Inner class representing the positioning system of a car seat.
        """

        def __init__(self):
            # Basic positioning
            self._horizontal_position = 50  # 0-100%
            self._vertical_position = 50  # 0-100%
            self._is_folded = False
            self._cushion_length = 50  # 0-100%

            # New positioning attributes from additional APIs
            self._cushion_angle = 50  # 0-100%
            self._backrest_angle = 50  # 0-100%
            self._leg_rest_height = 0  # 0-100%
            self._feet_rest_height = 0  # 0-100%
            self._headrest_height = 50  # 0-100%
            self._guest_welcome_mode = False

        @property
        def horizontal_position(self):
            return self._horizontal_position

        @horizontal_position.setter
        def horizontal_position(self, value):
            if 0 <= value <= 100:
                self._horizontal_position = value

        @property
        def vertical_position(self):
            return self._vertical_position

        @vertical_position.setter
        def vertical_position(self, value):
            if 0 <= value <= 100:
                self._vertical_position = value

        @property
        def is_folded(self):
            return self._is_folded

        @is_folded.setter
        def is_folded(self, value):
            self._is_folded = value

        @property
        def cushion_length(self):
            return self._cushion_length

        @cushion_length.setter
        def cushion_length(self, value):
            if 0 <= value <= 100:
                self._cushion_length = value

        @property
        def cushion_angle(self):
            return self._cushion_angle

        @cushion_angle.setter
        def cushion_angle(self, value):
            if 0 <= value <= 100:
                self._cushion_angle = value

        @property
        def backrest_angle(self):
            return self._backrest_angle

        @backrest_angle.setter
        def backrest_angle(self, value):
            if 0 <= value <= 100:
                self._backrest_angle = value

        @property
        def leg_rest_height(self):
            return self._leg_rest_height

        @leg_rest_height.setter
        def leg_rest_height(self, value):
            if 0 <= value <= 100:
                self._leg_rest_height = value

        @property
        def feet_rest_height(self):
            return self._feet_rest_height

        @feet_rest_height.setter
        def feet_rest_height(self, value):
            if 0 <= value <= 100:
                self._feet_rest_height = value

        @property
        def headrest_height(self):
            return self._headrest_height

        @headrest_height.setter
        def headrest_height(self, value):
            if 0 <= value <= 100:
                self._headrest_height = value

        @property
        def guest_welcome_mode(self):
            return self._guest_welcome_mode

        @guest_welcome_mode.setter
        def guest_welcome_mode(self, value):
            self._guest_welcome_mode = value


    def __init__(self):
        self._settings = None
        # Initialize each seat position with its own systems
        self._seats = {
            "driver's seat": {
                "heater": Seat.HeaterSystem(),
                "massager": Seat.MassageSystem(),
                "position": Seat.PositionSystem(),
                "ventilation": Seat.VentilationSystem(),
            },
            "passenger seat": {
                "heater": Seat.HeaterSystem(),
                "massager": Seat.MassageSystem(),
                "position": Seat.PositionSystem(),
                "ventilation": Seat.VentilationSystem(),
            },
            "second row left": {
                "heater": Seat.HeaterSystem(),
                "massager": Seat.MassageSystem(),
                "position": Seat.PositionSystem(),
                "ventilation": Seat.VentilationSystem(),
            },
            "second row right": {
                "heater": Seat.HeaterSystem(),
                "massager": Seat.MassageSystem(),
                "position": Seat.PositionSystem(),
                "ventilation": Seat.VentilationSystem(),
            },
            "third row left": {
                "heater": Seat.HeaterSystem(),
                "massager": Seat.MassageSystem(),
                "position": Seat.PositionSystem(),
                "ventilation": Seat.VentilationSystem(),
            },
            "third row right": {
                "heater": Seat.HeaterSystem(),
                "massager": Seat.MassageSystem(),
                "position": Seat.PositionSystem(),
                "ventilation": Seat.VentilationSystem(),
            },
        }

        # Control view page state
        self._view_page_open = False


    def set_settings(self, settings):
        """Inject shared VehicleSettings instance."""
        self._settings = settings

    @classmethod
    def init1(cls):
        """
        Initialize the Seat system with a comfort-oriented preset.
        All seats are configured with heated, ventilated, and massage functions
        optimized for maximum comfort.
        
        Returns:
            Seat: A new Seat instance with comfort-oriented settings
        """
        instance = cls()
        
        # Configure all seats with comfort settings
        for seat_pos in instance._seats:
            # Heater configuration - warm settings
            instance._seats[seat_pos]["heater"].is_on = False
            instance._seats[seat_pos]["heater"].temperature_level = 3
            instance._seats[seat_pos]["heater"].temperature_value = 25.0
            instance._seats[seat_pos]["heater"].temperature_unit = "celsius"
            
            # Massage configuration - gentle wave massage
            instance._seats[seat_pos]["massager"].is_on = False
            instance._seats[seat_pos]["massager"].intensity_level = 2
            instance._seats[seat_pos]["massager"].intensity_value = 40.0
            instance._seats[seat_pos]["massager"].intensity_unit = "percentage"
            instance._seats[seat_pos]["massager"].active_mode = "wave"
            
            # Ventilation configuration - light cooling
            instance._seats[seat_pos]["ventilation"].is_on = False
            instance._seats[seat_pos]["ventilation"].airflow_level = 2
            instance._seats[seat_pos]["ventilation"].airflow_value = 40.0
            instance._seats[seat_pos]["ventilation"].airflow_unit = "percentage"
            
            # Position configuration - relaxed position
            position = instance._seats[seat_pos]["position"]
            position.horizontal_position = 65  # More reclined
            position.vertical_position = 40    # Slightly lower
            position.is_folded = False
            position.cushion_length = 70      # Extended
            position.cushion_angle = 30       # Slightly angled
            position.backrest_angle = 70      # More reclined
            position.leg_rest_height = 40     # Partially raised
            position.feet_rest_height = 30    # Partially raised
            position.headrest_height = 60     # Properly positioned
            position.guest_welcome_mode = False
        
        # Set view page to closed
        instance._view_page_open = False
        
        return instance

    @classmethod
    def init2(cls):
        """
        Initialize the Seat system with a driving-focused preset.
        Front seats are configured for optimal driving position with moderate
        heating, while rear seats are configured for passenger comfort.
        
        Returns:
            Seat: A new Seat instance with driving-focused settings
        """
        instance = cls()
        
        # Configure front seats (driver and passenger) for driving
        for seat_pos in ["driver's seat", "passenger seat"]:
            # Heater configuration - mild warmth
            instance._seats[seat_pos]["heater"].is_on = True
            instance._seats[seat_pos]["heater"].temperature_level = 2
            instance._seats[seat_pos]["heater"].temperature_value = 22.0
            instance._seats[seat_pos]["heater"].temperature_unit = "celsius"
            
            # Massage configuration - off for driver, light for passenger
            instance._seats[seat_pos]["massager"].is_on = False if seat_pos == "driver's seat" else True
            instance._seats[seat_pos]["massager"].intensity_level = 1
            instance._seats[seat_pos]["massager"].intensity_value = 20.0
            instance._seats[seat_pos]["massager"].intensity_unit = "percentage"
            instance._seats[seat_pos]["massager"].active_mode = None if seat_pos == "driver's seat" else "shoulder"
            
            # Ventilation configuration - off
            instance._seats[seat_pos]["ventilation"].is_on = False
            instance._seats[seat_pos]["ventilation"].airflow_level = 1
            instance._seats[seat_pos]["ventilation"].airflow_value = 0.0
            instance._seats[seat_pos]["ventilation"].airflow_unit = "gear"
            
            # Position configuration - upright driving position
            position = instance._seats[seat_pos]["position"]
            position.horizontal_position = 40    # Forward for control
            position.vertical_position = 60      # Higher for visibility
            position.is_folded = False
            position.cushion_length = 50         # Standard
            position.cushion_angle = 20          # Slight angle
            position.backrest_angle = 30         # Upright
            position.leg_rest_height = 0         # Not raised
            position.feet_rest_height = 0        # Not raised
            position.headrest_height = 70        # Higher for safety
            position.guest_welcome_mode = False
        
        # Configure rear seats for passenger comfort
        for seat_pos in ["second row left", "second row right", "third row left", "third row right"]:
            # Heater configuration - cozy
            instance._seats[seat_pos]["heater"].is_on = True
            instance._seats[seat_pos]["heater"].temperature_level = 3
            instance._seats[seat_pos]["heater"].temperature_value = 24.0
            instance._seats[seat_pos]["heater"].temperature_unit = "celsius"
            
            # Massage configuration - comfort massage
            instance._seats[seat_pos]["massager"].is_on = True
            instance._seats[seat_pos]["massager"].intensity_level = 3
            instance._seats[seat_pos]["massager"].intensity_value = 60.0
            instance._seats[seat_pos]["massager"].intensity_unit = "percentage"
            instance._seats[seat_pos]["massager"].active_mode = "full back"
            
            # Ventilation configuration - gentle
            instance._seats[seat_pos]["ventilation"].is_on = True
            instance._seats[seat_pos]["ventilation"].airflow_level = 2
            instance._seats[seat_pos]["ventilation"].airflow_value = 40.0
            instance._seats[seat_pos]["ventilation"].airflow_unit = "percentage"
            
            # Position configuration - comfortable passenger position
            position = instance._seats[seat_pos]["position"]
            position.horizontal_position = 60    # More reclined
            position.vertical_position = 45      # Lower
            position.is_folded = False
            position.cushion_length = 65         # Extended
            position.cushion_angle = 25          # Comfortable angle
            position.backrest_angle = 60         # Reclined for comfort
            position.leg_rest_height = 30        # Partially raised
            position.feet_rest_height = 20       # Slightly raised
            position.headrest_height = 55        # Comfortable position
            position.guest_welcome_mode = True   # Welcome mode for passengers
        
        # Set view page to open
        instance._view_page_open = True
        
        return instance
    def _get_target_positions(self, position):
        """
        Determine which seat positions to target based on the position parameter.

        Args:
            position (list): List of seat positions to target, or ["all"] for all seats

        Returns:
            list: List of seat positions to adjust
        """
        if position is None:
            # Default to the current speaker's position
            speaker = self._settings.speaker
            return [speaker] if speaker in self._seats else ["driver's seat"]

        if "all" in position:
            return list(self._seats.keys())

        return [pos for pos in position if pos in self._seats]

    def _convert_degree_to_level(self, degree):
        """
        Convert a textual degree to a numeric level.

        Args:
            degree (str): The degree string ("min", "low", "medium", "high", "max",
                         or "tiny", "little", "large")

        Returns:
            int: The corresponding level (1-5) or adjustment value
        """
        if degree in ["min", "low", "medium", "high", "max"]:
            degree_map = {"min": 1, "low": 2, "medium": 3, "high": 4, "max": 5}
            return degree_map.get(degree, 3)  # Default to medium if not recognized
        elif degree in ["tiny", "little", "large"]:
            adjustment_map = {"tiny": 1, "little": 2, "large": 3}
            return adjustment_map.get(degree, 2)  # Default to little if not recognized
        return None

    def _adjust_value_by_degree(self, current_value, degree, min_val=0, max_val=100):
        """
        Adjust a current value by a degree of change.

        Args:
            current_value (float/int): The current value
            degree (str): The degree of adjustment ("tiny", "little", "large")
            min_val (float/int): Minimum allowable value
            max_val (float/int): Maximum allowable value

        Returns:
            float/int: The adjusted value
        """
        adjustment = self._convert_degree_to_level(degree)
        if adjustment:
            step = (max_val - min_val) / 10  # Divide range into 10 steps
            change = step * adjustment
            return max(min_val, min(max_val, current_value + change))
        return current_value

    def _adjust_value_by_inverse_degree(
        self, current_value, degree, min_val=0, max_val=100
    ):
        """
        Adjust a current value by a degree of change in the negative direction.

        Args:
            current_value (float/int): The current value
            degree (str): The degree of adjustment ("tiny", "little", "large")
            min_val (float/int): Minimum allowable value
            max_val (float/int): Maximum allowable value

        Returns:
            float/int: The adjusted value
        """
        adjustment = self._convert_degree_to_level(degree)
        if adjustment:
            step = (max_val - min_val) / 10  # Divide range into 10 steps
            change = step * adjustment
            return max(min_val, min(max_val, current_value - change))
        return current_value


    # ── Position property configs: (attr_name, cm_scale) ──
    _POSITION_PROPS = {
        "horizontal_position":  2.0,     # 100% = 50cm
        "vertical_position":    3.33,    # 100% = 30cm
        "cushion_length":       4.0,     # 100% = 25cm
        "cushion_angle":        1.0,     # degrees (no cm conversion)
        "backrest_angle":       1.0,     # degrees
        "leg_rest_height":      3.33,    # 100% = 30cm
        "feet_rest_height":     5.0,     # 100% = 20cm
        "headrest_height":      6.67,    # 100% = 15cm
    }

    def _adjust_position_prop(self, prop_name, direction, position=None,
                              value=None, unit=None, degree=None, action="adjusted"):
        """Generic position property adjustment.

        Args:
            prop_name: Attribute name on the PositionSystem (e.g. "horizontal_position").
            direction: +1 for increase, -1 for decrease, 0 for absolute set.
            position: Seat position filter.
            value/unit/degree: Standard adjustment parameters.
            action: Action label for the result dict.

        Returns:
            dict: Standard operation result.
        """
        targets = self._get_target_positions(position)
        results = {}
        cm_scale = self._POSITION_PROPS.get(prop_name, 1.0)

        for pos in targets:
            pos_sys = self._seats[pos]["position"]
            current = getattr(pos_sys, prop_name)

            if direction == 0:
                # ── Absolute set ──
                if value is not None and unit is not None:
                    if unit == "gear":
                        gear = max(1, min(5, int(value)))
                        new_val = (gear - 1) * 25
                    elif unit == "percentage":
                        new_val = max(0, min(100, value))
                    elif unit == "centimeter":
                        new_val = max(0, min(100, value * cm_scale))
                    else:
                        new_val = current
                elif degree is not None:
                    position_map = {"min": 0, "low": 25, "medium": 50, "high": 75, "max": 100}
                    new_val = position_map.get(degree, 50)
                else:
                    new_val = current
            else:
                # ── Relative adjust ──
                if value is not None and unit is not None:
                    if unit == "gear":
                        delta = int(value) * 20
                    elif unit == "percentage":
                        delta = value
                    elif unit == "centimeter":
                        delta = value * cm_scale
                    else:
                        delta = 0
                elif degree is not None:
                    if direction > 0:
                        new_val = self._adjust_value_by_degree(current, degree, 0, 100)
                        setattr(pos_sys, prop_name, new_val)
                        results[pos] = {prop_name: new_val}
                        continue
                    else:
                        new_val = self._adjust_value_by_inverse_degree(current, degree, 0, 100)
                        setattr(pos_sys, prop_name, new_val)
                        results[pos] = {prop_name: new_val}
                        continue
                else:
                    delta = 10  # default step
                new_val = max(0, min(100, current + direction * delta))

            setattr(pos_sys, prop_name, new_val)
            results[pos] = {prop_name: getattr(pos_sys, prop_name)}

        return {
            "success": True,
            "action": action,
            "affected_positions": targets,
            "states": results,
        }

    # API Methods Implementation

    @api("seat")
    def carcontrol_carSeat_get_info(self):
        """
        Get current seat information including speaker position and per-seat states.

        Returns:
            dict: Speaker position and per-seat heater/massager/ventilation status
        """
        info = {"speaker": self._settings.speaker, "seats": {}}
        for pos, seat_data in self._seats.items():
            info["seats"][pos] = {
                "heater": {"is_on": seat_data["heater"].is_on},
                "massager": {"is_on": seat_data["massager"].is_on},
                "ventilation": {"is_on": seat_data["ventilation"].is_on},
            }
        return info

    @api("seat")
    def carcontrol_carSeat_heater_switch(self, switch, position=None):
        """
        Turn on or off the car seat heating function.

        Args:
            switch (bool): True to turn heating on, False to turn it off
            position (list): List of seat positions to adjust, defaults to current speaker position
                            Enum values: ["driver's seat", "passenger seat", "second row left",
                                          "second row right", "third row left", "third row right", "all"]

        Returns:
            dict: Operation result and updated states
        """
        targets = self._get_target_positions(position)
        results = {}

        for pos in targets:
            self._seats[pos]["heater"].is_on = switch
            results[pos] = {"heater_state": self._seats[pos]["heater"].is_on}

        return {
            "success": True,
            "action": "heater_switch_" + ("on" if switch else "off"),
            "affected_positions": targets,
            "states": results,
        }

    @api("seat")
    def carcontrol_carSeat_heater_increase(
        self, position=None, value=None, unit=None, degree=None
    ):
        """
        Increase car seat heating temperature.

        Args:
            position (list): List of seat positions to adjust, defaults to current speaker position
                            Enum values: ["driver's seat", "passenger seat", "second row left",
                                          "second row right", "third row left", "third row right", "all"]
            value (float, optional): Specific numerical value for adjustment
            unit (str, optional): Unit for the value
                                 Enum values: ["gear", "percentage", "celsius"]
            degree (str, optional): Degree of adjustment if not using specific value
                                   Enum values: ["large", "little", "tiny"]

        Returns:
            dict: Operation result and updated states
        """
        targets = self._get_target_positions(position)
        results = {}

        for pos in targets:
            heater = self._seats[pos]["heater"]

            # Turn on heater if it's off
            if not heater.is_on:
                heater.is_on = True

            if value is not None and unit is not None:
                # Handle specific value adjustments
                if unit == "gear":
                    # Assuming gears are 1-5
                    heater.temperature_level = min(
                        5, heater.temperature_level + int(value)
                    )
                elif unit == "percentage":
                    # Adjust percentage and convert to level
                    current_pct = (
                        heater.temperature_level - 1
                    ) * 25  # Convert level to percentage (0-100)
                    new_pct = min(100, current_pct + value)
                    heater.temperature_level = min(5, max(1, int(new_pct / 25) + 1))
                elif unit == "celsius":
                    heater.temperature_value = heater.temperature_value + value
                    heater.temperature_unit = "celsius"
            elif degree is not None:
                # Handle degree-based adjustments
                adjustment = self._convert_degree_to_level(degree)
                if adjustment:
                    heater.temperature_level = min(
                        5, heater.temperature_level + adjustment
                    )
            else:
                # Default increase by 1 level if no specifics provided
                heater.temperature_level = min(5, heater.temperature_level + 1)

            results[pos] = {
                "heater_state": heater.is_on,
                "temperature_level": heater.temperature_level,
                "temperature_value": heater.temperature_value,
                "temperature_unit": heater.temperature_unit,
            }

        return {
            "success": True,
            "action": "heater_temperature_increased",
            "affected_positions": targets,
            "states": results,
        }

    @api("seat")
    def carcontrol_carSeat_heater_decrease(
        self, position=None, value=None, unit=None, degree=None
    ):
        """
        Decrease car seat heating temperature.

        Args:
            position (list): List of seat positions to adjust, defaults to current speaker position
                            Enum values: ["driver's seat", "passenger seat", "second row left",
                                          "second row right", "third row left", "third row right", "all"]
            value (float, optional): Specific numerical value for adjustment
            unit (str, optional): Unit for the value
                                 Enum values: ["gear", "percentage", "celsius"]
            degree (str, optional): Degree of adjustment if not using specific value
                                   Enum values: ["large", "little", "tiny"]

        Returns:
            dict: Operation result and updated states
        """
        targets = self._get_target_positions(position)
        results = {}

        for pos in targets:
            heater = self._seats[pos]["heater"]

            # Skip if heater is already off
            if not heater.is_on:
                results[pos] = {"status": "heater already off"}
                continue

            if value is not None and unit is not None:
                # Handle specific value adjustments
                if unit == "gear":
                    # Assuming gears are 1-5
                    heater.temperature_level = max(
                        1, heater.temperature_level - int(value)
                    )
                    # Turn off if reduced to minimum
                    if heater.temperature_level == 1 and value >= 1:
                        heater.is_on = False
                elif unit == "percentage":
                    # Adjust percentage and convert to level
                    current_pct = (
                        heater.temperature_level - 1
                    ) * 25  # Convert level to percentage (0-100)
                    new_pct = max(0, current_pct - value)
                    heater.temperature_level = max(1, int(new_pct / 25) + 1)
                    # Turn off if reduced to 0%
                    if new_pct == 0:
                        heater.is_on = False
                elif unit == "celsius":
                    heater.temperature_value = max(0, heater.temperature_value - value)
                    heater.temperature_unit = "celsius"
            elif degree is not None:
                # Handle degree-based adjustments
                adjustment = self._convert_degree_to_level(degree)
                if adjustment:
                    heater.temperature_level = max(
                        1, heater.temperature_level - adjustment
                    )
                    # Turn off if reduced to minimum with large adjustment
                    if heater.temperature_level == 1 and degree == "large":
                        heater.is_on = False
            else:
                # Default decrease by 1 level if no specifics provided
                heater.temperature_level = max(1, heater.temperature_level - 1)
                # Turn off if reduced to minimum
                if heater.temperature_level == 1:
                    heater.is_on = False

            results[pos] = {
                "heater_state": heater.is_on,
                "temperature_level": heater.temperature_level,
                "temperature_value": heater.temperature_value,
                "temperature_unit": heater.temperature_unit,
            }

        return {
            "success": True,
            "action": "heater_temperature_decreased",
            "affected_positions": targets,
            "states": results,
        }

    @api("seat")
    def carcontrol_carSeat_heater_set(
        self, position=None, value=None, unit=None, degree=None
    ):
        """
        Set car seat heating temperature to a specified value.

        Args:
            position (list): List of seat positions to adjust, defaults to current speaker position
                            Enum values: ["driver's seat", "passenger seat", "second row left",
                                          "second row right", "third row left", "third row right", "all"]
            value (float, optional): Specific numerical value to set
            unit (str, optional): Unit for the value
                                 Enum values: ["gear", "percentage", "celsius"]
            degree (str, optional): Predefined level if not using specific value
                                   Enum values: ["max", "high", "medium", "low", "min"]

        Returns:
            dict: Operation result and updated states
        """
        targets = self._get_target_positions(position)
        results = {}

        for pos in targets:
            heater = self._seats[pos]["heater"]

            # Turn on heater
            heater.is_on = True

            if value is not None and unit is not None:
                # Handle specific value settings
                if unit == "gear":
                    # Gears are typically 1-5
                    heater.temperature_level = max(1, min(5, int(value)))
                elif unit == "percentage":
                    # Convert percentage to level (1-5)
                    level = max(1, min(5, int(value / 25) + 1))
                    heater.temperature_level = level
                elif unit == "celsius":
                    heater.temperature_value = value
                    heater.temperature_unit = "celsius"
            elif degree is not None:
                # Handle predefined level settings
                level_map = {"min": 1, "low": 2, "medium": 3, "high": 4, "max": 5}
                heater.temperature_level = level_map.get(
                    degree, 3
                )  # Default to medium if not recognized

            results[pos] = {
                "heater_state": heater.is_on,
                "temperature_level": heater.temperature_level,
                "temperature_value": heater.temperature_value,
                "temperature_unit": heater.temperature_unit,
            }

        return {
            "success": True,
            "action": "heater_temperature_set",
            "affected_positions": targets,
            "states": results,
        }

    @api("seat")
    def carcontrol_carSeat_massager_switch(self, switch, position=None):
        """
        Turn on or off the car seat massage function.

        Args:
            switch (bool): True to turn massage on, False to turn it off
            position (list): List of seat positions to adjust, defaults to current speaker position
                            Enum values: ["driver's seat", "passenger seat", "second row left",
                                          "second row right", "third row left", "third row right", "all"]

        Returns:
            dict: Operation result and updated states
        """
        targets = self._get_target_positions(position)
        results = {}

        for pos in targets:
            self._seats[pos]["massager"].is_on = switch
            results[pos] = {"massager_state": self._seats[pos]["massager"].is_on}

        return {
            "success": True,
            "action": "massager_switch_" + ("on" if switch else "off"),
            "affected_positions": targets,
            "states": results,
        }

    @api("seat")
    def carcontrol_carSeat_massager_increase(
        self, position=None, value=None, unit=None, degree=None
    ):
        """
        Increase car seat massage intensity.

        Args:
            position (list): List of seat positions to adjust, defaults to current speaker position
                            Enum values: ["driver's seat", "passenger seat", "second row left",
                                          "second row right", "third row left", "third row right", "all"]
            value (float, optional): Specific numerical value for adjustment
            unit (str, optional): Unit for the value
                                 Enum values: ["gear", "percentage"]
            degree (str, optional): Degree of adjustment if not using specific value
                                   Enum values: ["large", "little", "tiny"]

        Returns:
            dict: Operation result and updated states
        """
        targets = self._get_target_positions(position)
        results = {}

        for pos in targets:
            massager = self._seats[pos]["massager"]

            # Turn on massager if it's off
            if not massager.is_on:
                massager.is_on = True

            if value is not None and unit is not None:
                # Handle specific value adjustments
                if unit == "gear":
                    # Assuming gears are 1-5
                    massager.intensity_level = min(
                        5, massager.intensity_level + int(value)
                    )
                elif unit == "percentage":
                    # Adjust percentage and convert to level
                    current_pct = (
                        massager.intensity_level - 1
                    ) * 25  # Convert level to percentage (0-100)
                    new_pct = min(100, current_pct + value)
                    massager.intensity_level = min(5, max(1, int(new_pct / 25) + 1))
            elif degree is not None:
                # Handle degree-based adjustments
                adjustment = self._convert_degree_to_level(degree)
                if adjustment:
                    massager.intensity_level = min(
                        5, massager.intensity_level + adjustment
                    )
            else:
                # Default increase by 1 level if no specifics provided
                massager.intensity_level = min(5, massager.intensity_level + 1)

            results[pos] = {
                "massager_state": massager.is_on,
                "intensity_level": massager.intensity_level,
            }

        return {
            "success": True,
            "action": "massager_intensity_increased",
            "affected_positions": targets,
            "states": results,
        }

    @api("seat")
    def carcontrol_carSeat_massager_decrease(
        self, position=None, value=None, unit=None, degree=None
    ):
        """
        Decrease car seat massage intensity.

        Args:
            position (list): List of seat positions to adjust, defaults to current speaker position
                            Enum values: ["driver's seat", "passenger seat", "second row left",
                                          "second row right", "third row left", "third row right", "all"]
            value (float, optional): Specific numerical value for adjustment
            unit (str, optional): Unit for the value
                                 Enum values: ["gear", "percentage"]
            degree (str, optional): Degree of adjustment if not using specific value
                                   Enum values: ["large", "little", "tiny"]

        Returns:
            dict: Operation result and updated states
        """
        targets = self._get_target_positions(position)
        results = {}

        for pos in targets:
            massager = self._seats[pos]["massager"]

            # Skip if massager is already off
            if not massager.is_on:
                results[pos] = {"status": "massager already off"}
                continue

            if value is not None and unit is not None:
                # Handle specific value adjustments
                if unit == "gear":
                    # Assuming gears are 1-5
                    massager.intensity_level = max(
                        1, massager.intensity_level - int(value)
                    )
                    # Turn off if reduced to minimum
                    if massager.intensity_level == 1 and value >= 1:
                        massager.is_on = False
                elif unit == "percentage":
                    # Adjust percentage and convert to level
                    current_pct = (
                        massager.intensity_level - 1
                    ) * 25  # Convert level to percentage (0-100)
                    new_pct = max(0, current_pct - value)
                    massager.intensity_level = max(1, int(new_pct / 25) + 1)
                    # Turn off if reduced to 0%
                    if new_pct == 0:
                        massager.is_on = False
            elif degree is not None:
                # Handle degree-based adjustments
                adjustment = self._convert_degree_to_level(degree)
                if adjustment:
                    massager.intensity_level = max(
                        1, massager.intensity_level - adjustment
                    )
                    # Turn off if reduced to minimum with large adjustment
                    if massager.intensity_level == 1 and degree == "large":
                        massager.is_on = False
            else:
                # Default decrease by 1 level if no specifics provided
                massager.intensity_level = max(1, massager.intensity_level - 1)
                # Turn off if reduced to minimum
                if massager.intensity_level == 1:
                    massager.is_on = False

            results[pos] = {
                "massager_state": massager.is_on,
                "intensity_level": massager.intensity_level,
            }

        return {
            "success": True,
            "action": "massager_intensity_decreased",
            "affected_positions": targets,
            "states": results,
        }

    @api("seat")
    def carcontrol_carSeat_massager_set(
        self, position=None, value=None, unit=None, degree=None
    ):
        """
        Set car seat massage intensity to a specified value.

        Args:
            position (list): List of seat positions to adjust, defaults to current speaker position
                            Enum values: ["driver's seat", "passenger seat", "second row left",
                                          "second row right", "third row left", "third row right", "all"]
            value (float, optional): Specific numerical value to set
            unit (str, optional): Unit for the value
                                 Enum values: ["gear", "percentage"]
            degree (str, optional): Predefined level if not using specific value
                                   Enum values: ["max", "high", "medium", "low", "min"]

        Returns:
            dict: Operation result and updated states
        """
        targets = self._get_target_positions(position)
        results = {}

        for pos in targets:
            massager = self._seats[pos]["massager"]

            # Turn on massager
            massager.is_on = True

            if value is not None and unit is not None:
                # Handle specific value settings
                if unit == "gear":
                    # Gears are typically 1-5
                    massager.intensity_level = max(1, min(5, int(value)))
                    massager.intensity_value = value
                    massager.intensity_unit = "gear"
                elif unit == "percentage":
                    # Convert percentage to level (1-5)
                    level = max(1, min(5, int(value / 25) + 1))
                    massager.intensity_level = level
                    massager.intensity_value = value
                    massager.intensity_unit = "percentage"
            elif degree is not None:
                # Handle predefined level settings
                level_map = {"min": 1, "low": 2, "medium": 3, "high": 4, "max": 5}
                massager.intensity_level = level_map.get(
                    degree, 3
                )  # Default to medium if not recognized

            results[pos] = {
                "massager_state": massager.is_on,
                "intensity_level": massager.intensity_level,
                "intensity_value": massager.intensity_value,
                "intensity_unit": massager.intensity_unit,
            }

        return {
            "success": True,
            "action": "massager_intensity_set",
            "affected_positions": targets,
            "states": results,
        }

    @api("seat")
    def carcontrol_carSeat_massager_mode(self, switch, mode, position=None):
        """
        Turn on or off a specific massage mode for the car seat massage function.

        Args:
            switch (bool): True to turn the mode on, False to turn it off
            mode (str): The massage mode to activate
                       Enum values: ["wave", "cat step", "stretch", "snake", "butterfly",
                                    "shoulder", "upper back", "waist", "full back", "random"]
            position (list): List of seat positions to adjust, defaults to current speaker position
                            Enum values: ["driver's seat", "passenger seat", "second row left",
                                          "second row right", "third row left", "third row right", "all"]

        Returns:
            dict: Operation result and updated states
        """
        targets = self._get_target_positions(position)
        results = {}

        valid_modes = [
            "wave",
            "cat step",
            "stretch",
            "snake",
            "butterfly",
            "shoulder",
            "upper back",
            "waist",
            "full back",
            "random",
        ]

        if mode not in valid_modes:
            return {
                "success": False,
                "error": f"Invalid massage mode: {mode}. Valid modes are: {', '.join(valid_modes)}",
            }

        for pos in targets:
            massager = self._seats[pos]["massager"]

            if switch:
                # Turn on the massager and set the mode
                massager.is_on = True
                massager.active_mode = mode
            else:
                # If turning off the specified mode
                if massager.active_mode == mode:
                    massager.active_mode = None
                    # Turn off massage system if no mode is active
                    if massager.active_mode is None:
                        massager.is_on = False

            results[pos] = {
                "massager_state": massager.is_on,
                "active_mode": massager.active_mode,
            }

        return {
            "success": True,
            "action": f"massager_mode_{mode}_"
            + ("activated" if switch else "deactivated"),
            "affected_positions": targets,
            "states": results,
        }

    @api("seat")
    def carcontrol_carSeat_switch(self, action, position=None):
        """
        Open (unfold) or close (fold) the car seat.

        Args:
            action (str): Action to perform on the seat
                         Enum values: ["open", "close", "pause"]
            position (list): List of seat positions to adjust, defaults to current speaker position
                            Enum values: ["driver's seat", "passenger seat", "second row left",
                                          "second row right", "third row left", "third row right", "all"]

        Returns:
            dict: Operation result and updated states
        """
        targets = self._get_target_positions(position)
        results = {}

        valid_actions = ["open", "close", "pause"]

        if action not in valid_actions:
            return {
                "success": False,
                "error": f"Invalid seat action: {action}. Valid actions are: {', '.join(valid_actions)}",
            }

        for pos in targets:
            position_system = self._seats[pos]["position"]

            if action == "open":
                position_system.is_folded = False
            elif action == "close":
                position_system.is_folded = True
            # For "pause", we don't change the folding state

            results[pos] = {"is_folded": position_system.is_folded}

        return {
            "success": True,
            "action": f"seat_{action}",
            "affected_positions": targets,
            "states": results,
        }

    @api("seat")
    def carcontrol_carSeat_horizontal_forward(
        self, position=None, value=None, unit=None, degree=None
    ):
        """
        Move the car seat horizontal position forward.

        Args:
            position (list): List of seat positions to adjust, defaults to current speaker position
                            Enum values: ["driver's seat", "passenger seat", "second row left",
                                          "second row right", "third row left", "third row right", "all"]
            value (float, optional): Specific numerical value for adjustment
            unit (str, optional): Unit for the value
                                 Enum values: ["gear", "percentage", "centimeter"]
            degree (str, optional): Degree of adjustment if not using specific value
                                   Enum values: ["large", "little", "tiny"]

        Returns:
            dict: Operation result and updated states
        """
        return self._adjust_position_prop(
            "horizontal_position", -1, position, value, unit, degree,
            action="seat_moved_forward"
        )

    @api("seat")
    def carcontrol_carSeat_horizontal_backward(
        self, position=None, value=None, unit=None, degree=None
    ):
        """
        Move the car seat horizontal position backward.

        Args:
            position (list): List of seat positions to adjust, defaults to current speaker position
                            Enum values: ["driver's seat", "passenger seat", "second row left",
                                          "second row right", "third row left", "third row right", "all"]
            value (float, optional): Specific numerical value for adjustment
            unit (str, optional): Unit for the value
                                 Enum values: ["gear", "percentage", "centimeter"]
            degree (str, optional): Degree of adjustment if not using specific value
                                   Enum values: ["large", "little", "tiny"]

        Returns:
            dict: Operation result and updated states
        """
        return self._adjust_position_prop(
            "horizontal_position", 1, position, value, unit, degree,
            action="seat_moved_backward"
        )

    @api("seat")
    def carcontrol_carSeat_horizontal_set(
        self, position=None, value=None, unit=None, degree=None
    ):
        """
        Set the car seat horizontal position to a specified value.

        Args:
            position (list): List of seat positions to adjust, defaults to current speaker position
                            Enum values: ["driver's seat", "passenger seat", "second row left",
                                          "second row right", "third row left", "third row right", "all"]
            value (float, optional): Specific numerical value for adjustment
            unit (str, optional): Unit for the value
                                 Enum values: ["gear", "percentage", "centimeter"]
            degree (str, optional): Predefined position level
                                   Enum values: ["max", "high", "medium", "low", "min"]

        Returns:
            dict: Operation result and updated states
        """
        return self._adjust_position_prop(
            "horizontal_position", 0, position, value, unit, degree,
            action="seat_horizontal_position_set"
        )

    @api("seat")
    def carcontrol_carSeat_height_increase(
        self, position=None, value=None, unit=None, degree=None
    ):
        """
        Increase the car seat height.

        Args:
            position (list): List of seat positions to adjust, defaults to current speaker position
                            Enum values: ["driver's seat", "passenger seat", "second row left",
                                          "second row right", "third row left", "third row right", "all"]
            value (float, optional): Specific numerical value for adjustment
            unit (str, optional): Unit for the value
                                 Enum values: ["gear", "percentage", "centimeter"]
            degree (str, optional): Degree of adjustment if not using specific value
                                   Enum values: ["large", "little", "tiny"]

        Returns:
            dict: Operation result and updated states
        """
        return self._adjust_position_prop(
            "vertical_position", 1, position, value, unit, degree,
            action="seat_height_increased"
        )

    @api("seat")
    def carcontrol_carSeat_height_decrease(
        self, position=None, value=None, unit=None, degree=None
    ):
        """
        Decrease the car seat height.

        Args:
            position (list): List of seat positions to adjust, defaults to current speaker position
                            Enum values: ["driver's seat", "passenger seat", "second row left",
                                          "second row right", "third row left", "third row right", "all"]
            value (float, optional): Specific numerical value for adjustment
            unit (str, optional): Unit for the value
                                 Enum values: ["gear", "percentage", "centimeter"]
            degree (str, optional): Degree of adjustment if not using specific value
                                   Enum values: ["large", "little", "tiny"]

        Returns:
            dict: Operation result and updated states
        """
        return self._adjust_position_prop(
            "vertical_position", -1, position, value, unit, degree,
            action="seat_height_decreased"
        )

    @api("seat")
    def carcontrol_carSeat_height_set(
        self, position=None, value=None, unit=None, degree=None
    ):
        """
        Set the car seat height to a specified value.

        Args:
            position (list): List of seat positions to adjust, defaults to current speaker position
                            Enum values: ["driver's seat", "passenger seat", "second row left",
                                          "second row right", "third row left", "third row right", "all"]
            value (float, optional): Specific numerical value for adjustment
            unit (str, optional): Unit for the value
                                 Enum values: ["gear", "percentage", "centimeter"]
            degree (str, optional): Predefined position level
                                   Enum values: ["max", "high", "medium", "low", "min"]

        Returns:
            dict: Operation result and updated states
        """
        return self._adjust_position_prop(
            "vertical_position", 0, position, value, unit, degree,
            action="seat_height_set"
        )

    @api("seat")
    def carcontrol_carSeatCushion_length_increase(
        self, position=None, value=None, unit=None, degree=None
    ):
        """
        Increase the car seat cushion length.

        Args:
            position (list): List of seat positions to adjust, defaults to current speaker position
                            Enum values: ["driver's seat", "passenger seat", "second row left",
                                          "second row right", "third row left", "third row right", "all"]
            value (float, optional): Specific numerical value for adjustment
            unit (str, optional): Unit for the value
                                 Enum values: ["gear", "percentage", "centimeter"]
            degree (str, optional): Degree of adjustment if not using specific value
                                   Enum values: ["large", "little", "tiny"]

        Returns:
            dict: Operation result and updated states
        """
        return self._adjust_position_prop(
            "cushion_length", 1, position, value, unit, degree,
            action="seat_cushion_length_increased"
        )

    @api("seat")
    def carcontrol_carSeatCushion_length_decrease(
        self, position=None, value=None, unit=None, degree=None
    ):
        """
        Decrease the car seat cushion length.

        Args:
            position (list): List of seat positions to adjust, defaults to current speaker position
                            Enum values: ["driver's seat", "passenger seat", "second row left",
                                          "second row right", "third row left", "third row right", "all"]
            value (float, optional): Specific numerical value for adjustment
            unit (str, optional): Unit for the value
                                 Enum values: ["gear", "percentage", "centimeter"]
            degree (str, optional): Degree of adjustment if not using specific value
                                   Enum values: ["large", "little", "tiny"]

        Returns:
            dict: Operation result and updated states
        """
        return self._adjust_position_prop(
            "cushion_length", -1, position, value, unit, degree,
            action="seat_cushion_length_decreased"
        )

    @api("seat")
    def carcontrol_carSeatCushion_length_set(
        self, position=None, value=None, unit=None, degree=None
    ):
        """
        Set the car seat cushion length to a specified value.

        Args:
            position (list): List of seat positions to adjust, defaults to current speaker position
                            Enum values: ["driver's seat", "passenger seat", "second row left",
                                          "second row right", "third row left", "third row right", "all"]
            value (float, optional): Specific numerical value for adjustment
            unit (str, optional): Unit for the value
                                 Enum values: ["gear", "percentage", "centimeter"]
            degree (str, optional): Predefined position level
                                   Enum values: ["max", "high", "medium", "low", "min"]

        Returns:
            dict: Operation result and updated states
        """
        return self._adjust_position_prop(
            "cushion_length", 0, position, value, unit, degree,
            action="seat_cushion_length_set"
        )

    @api("seat")
    def carcontrol_carSeatCushion_angle_increase(
        self, position=None, value=None, unit=None, degree=None
    ):
        """
        Increase the car seat cushion angle.

        Args:
            position (list): List of seat positions to adjust, defaults to current speaker position
                            Enum values: ["driver's seat", "passenger seat", "second row left",
                                          "second row right", "third row left", "third row right", "all"]
            value (float, optional): Specific numerical value for adjustment
            unit (str, optional): Unit for the value
                                 Enum values: ["gear", "percentage", "centimeter"]
            degree (str, optional): Degree of adjustment if not using specific value
                                   Enum values: ["large", "little", "tiny"]

        Returns:
            dict: Operation result and updated states
        """
        return self._adjust_position_prop(
            "cushion_angle", 1, position, value, unit, degree,
            action="seat_cushion_angle_increased"
        )

    @api("seat")
    def carcontrol_carSeatCushion_angle_decrease(
        self, position=None, value=None, unit=None, degree=None
    ):
        """
        Decrease the car seat cushion angle.

        Args:
            position (list): List of seat positions to adjust, defaults to current speaker position
                            Enum values: ["driver's seat", "passenger seat", "second row left",
                                          "second row right", "third row left", "third row right", "all"]
            value (float, optional): Specific numerical value for adjustment
            unit (str, optional): Unit for the value
                                 Enum values: ["gear", "percentage", "centimeter"]
            degree (str, optional): Degree of adjustment if not using specific value
                                   Enum values: ["large", "little", "tiny"]

        Returns:
            dict: Operation result and updated states
        """
        return self._adjust_position_prop(
            "cushion_angle", -1, position, value, unit, degree,
            action="seat_cushion_angle_decreased"
        )

    @api("seat")
    def carcontrol_carSeatCushion_angle_set(
        self, position=None, value=None, unit=None, degree=None
    ):
        """
        Set the car seat cushion angle to a specified value.

        Args:
            position (list): List of seat positions to adjust, defaults to current speaker position
                            Enum values: ["driver's seat", "passenger seat", "second row left",
                                          "second row right", "third row left", "third row right", "all"]
            value (float, optional): Specific numerical value for adjustment
            unit (str, optional): Unit for the value
                                 Enum values: ["gear", "percentage", "centimeter"]
            degree (str, optional): Predefined position level
                                   Enum values: ["max", "high", "medium", "low", "min"]

        Returns:
            dict: Operation result and updated states
        """
        return self._adjust_position_prop(
            "cushion_angle", 0, position, value, unit, degree,
            action="seat_cushion_angle_set"
        )

    @api("seat")
    def carcontrol_carSeatBackrest_angle_increase(
        self, position=None, value=None, unit=None, degree=None
    ):
        """
        Increase the car seat backrest angle.

        Args:
            position (list): List of seat positions to adjust, defaults to current speaker position
                            Enum values: ["driver's seat", "passenger seat", "second row left",
                                          "second row right", "third row left", "third row right", "all"]
            value (float, optional): Specific numerical value for adjustment
            unit (str, optional): Unit for the value
                                 Enum values: ["gear", "percentage", "centimeter"]
            degree (str, optional): Degree of adjustment if not using specific value
                                   Enum values: ["large", "little", "tiny"]

        Returns:
            dict: Operation result and updated states
        """
        return self._adjust_position_prop(
            "backrest_angle", 1, position, value, unit, degree,
            action="seat_backrest_angle_increased"
        )

    @api("seat")
    def carcontrol_carSeatBackrest_angle_decrease(
        self, position=None, value=None, unit=None, degree=None
    ):
        """
        Decrease the car seat backrest angle.

        Args:
            position (list): List of seat positions to adjust, defaults to current speaker position
                            Enum values: ["driver's seat", "passenger seat", "second row left",
                                          "second row right", "third row left", "third row right", "all"]
            value (float, optional): Specific numerical value for adjustment
            unit (str, optional): Unit for the value
                                 Enum values: ["gear", "percentage", "centimeter"]
            degree (str, optional): Degree of adjustment if not using specific value
                                   Enum values: ["large", "little", "tiny"]

        Returns:
            dict: Operation result and updated states
        """
        return self._adjust_position_prop(
            "backrest_angle", -1, position, value, unit, degree,
            action="seat_backrest_angle_decreased"
        )

    @api("seat")
    def carcontrol_carSeatBackrest_angle_set(
        self, position=None, value=None, unit=None, degree=None
    ):
        """
        Set the car seat backrest angle to a specified value.

        Args:
            position (list): List of seat positions to adjust, defaults to current speaker position
                            Enum values: ["driver's seat", "passenger seat", "second row left",
                                          "second row right", "third row left", "third row right", "all"]
            value (float, optional): Specific numerical value for adjustment
            unit (str, optional): Unit for the value
                                 Enum values: ["gear", "percentage", "centimeter"]
            degree (str, optional): Predefined position level
                                   Enum values: ["max", "high", "medium", "low", "min"]

        Returns:
            dict: Operation result and updated states
        """
        return self._adjust_position_prop(
            "backrest_angle", 0, position, value, unit, degree,
            action="seat_backrest_angle_set"
        )

    @api("seat")
    def carcontrol_carSeatLegRest_height_increase(
        self, position=None, value=None, unit=None, degree=None
    ):
        """
        Increase the car seat leg rest height.

        Args:
            position (list): List of seat positions to adjust, defaults to current speaker position
                            Enum values: ["driver's seat", "passenger seat", "second row left",
                                          "second row right", "third row left", "third row right", "all"]
            value (float, optional): Specific numerical value for adjustment
            unit (str, optional): Unit for the value
                                 Enum values: ["gear", "percentage", "centimeter"]
            degree (str, optional): Degree of adjustment if not using specific value
                                   Enum values: ["large", "little", "tiny"]

        Returns:
            dict: Operation result and updated states
        """
        return self._adjust_position_prop(
            "leg_rest_height", 1, position, value, unit, degree,
            action="seat_leg_rest_height_increased"
        )

    @api("seat")
    def carcontrol_carSeatLegRest_height_decrease(
        self, position=None, value=None, unit=None, degree=None
    ):
        """
        Decrease the car seat leg rest height.

        Args:
            position (list): List of seat positions to adjust, defaults to current speaker position
                            Enum values: ["driver's seat", "passenger seat", "second row left",
                                          "second row right", "third row left", "third row right", "all"]
            value (float, optional): Specific numerical value for adjustment
            unit (str, optional): Unit for the value
                                 Enum values: ["gear", "percentage", "centimeter"]
            degree (str, optional): Degree of adjustment if not using specific value
                                   Enum values: ["large", "little", "tiny"]

        Returns:
            dict: Operation result and updated states
        """
        return self._adjust_position_prop(
            "leg_rest_height", -1, position, value, unit, degree,
            action="seat_leg_rest_height_decreased"
        )

    @api("seat")
    def carcontrol_carSeatLegRest_height_set(
        self, position=None, value=None, unit=None, degree=None
    ):
        """
        Set the car seat leg rest height to a specified value.

        Args:
            position (list): List of seat positions to adjust, defaults to current speaker position
                            Enum values: ["driver's seat", "passenger seat", "second row left",
                                          "second row right", "third row left", "third row right", "all"]
            value (float, optional): Specific numerical value for adjustment
            unit (str, optional): Unit for the value
                                 Enum values: ["gear", "percentage", "centimeter"]
            degree (str, optional): Predefined position level
                                   Enum values: ["max", "high", "medium", "low", "min"]

        Returns:
            dict: Operation result and updated states
        """
        return self._adjust_position_prop(
            "leg_rest_height", 0, position, value, unit, degree,
            action="seat_leg_rest_height_set"
        )

    @api("seat")
    def carcontrol_carSeatFeetRest_height_increase(
        self, position=None, value=None, unit=None, degree=None
    ):
        """
        Increase the car seat feet rest height.

        Args:
            position (list): List of seat positions to adjust, defaults to current speaker position
                            Enum values: ["driver's seat", "passenger seat", "second row left",
                                          "second row right", "third row left", "third row right", "all"]
            value (float, optional): Specific numerical value for adjustment
            unit (str, optional): Unit for the value
                                 Enum values: ["gear", "percentage", "centimeter"]
            degree (str, optional): Degree of adjustment if not using specific value
                                   Enum values: ["large", "little", "tiny"]

        Returns:
            dict: Operation result and updated states
        """
        return self._adjust_position_prop(
            "feet_rest_height", 1, position, value, unit, degree,
            action="seat_feet_rest_height_increased"
        )

    @api("seat")
    def carcontrol_carSeatFeetRest_height_decrease(
        self, position=None, value=None, unit=None, degree=None
    ):
        """
        Decrease the car seat feet rest height.

        Args:
            position (list): List of seat positions to adjust, defaults to current speaker position
                            Enum values: ["driver's seat", "passenger seat", "second row left",
                                          "second row right", "third row left", "third row right", "all"]
            value (float, optional): Specific numerical value for adjustment
            unit (str, optional): Unit for the value
                                 Enum values: ["gear", "percentage", "centimeter"]
            degree (str, optional): Degree of adjustment if not using specific value
                                   Enum values: ["large", "little", "tiny"]

        Returns:
            dict: Operation result and updated states
        """
        return self._adjust_position_prop(
            "feet_rest_height", -1, position, value, unit, degree,
            action="seat_feet_rest_height_decreased"
        )

    @api("seat")
    def carcontrol_carSeatFeetRest_height_set(
        self, position=None, value=None, unit=None, degree=None
    ):
        """
        Set the car seat feet rest height to a specified value.

        Args:
            position (list): List of seat positions to adjust, defaults to current speaker position
                            Enum values: ["driver's seat", "passenger seat", "second row left",
                                          "second row right", "third row left", "third row right", "all"]
            value (float, optional): Specific numerical value for adjustment
            unit (str, optional): Unit for the value
                                 Enum values: ["gear", "percentage", "centimeter"]
            degree (str, optional): Predefined position level
                                   Enum values: ["max", "high", "medium", "low", "min"]

        Returns:
            dict: Operation result and updated states
        """
        return self._adjust_position_prop(
            "feet_rest_height", 0, position, value, unit, degree,
            action="seat_feet_rest_height_set"
        )

    @api("seat")
    def carcontrol_carSeatHeadRest_height_increase(
        self, position=None, value=None, unit=None, degree=None
    ):
        """
        Increase the car seat headrest height.

        Args:
            position (list): List of seat positions to adjust, defaults to current speaker position
                            Enum values: ["driver's seat", "passenger seat", "second row left",
                                          "second row right", "third row left", "third row right", "all"]
            value (float, optional): Specific numerical value for adjustment
            unit (str, optional): Unit for the value
                                 Enum values: ["gear", "percentage", "centimeter"]
            degree (str, optional): Degree of adjustment if not using specific value
                                   Enum values: ["large", "little", "tiny"]

        Returns:
            dict: Operation result and updated states
        """
        return self._adjust_position_prop(
            "headrest_height", 1, position, value, unit, degree,
            action="seat_headrest_height_increased"
        )

    @api("seat")
    def carcontrol_carSeatHeadRest_height_decrease(
        self, position=None, value=None, unit=None, degree=None
    ):
        """
        Decrease the car seat headrest height.

        Args:
            position (list): List of seat positions to adjust, defaults to current speaker position
                            Enum values: ["driver's seat", "passenger seat", "second row left",
                                          "second row right", "third row left", "third row right", "all"]
            value (float, optional): Specific numerical value for adjustment
            unit (str, optional): Unit for the value
                                 Enum values: ["gear", "percentage", "centimeter"]
            degree (str, optional): Degree of adjustment if not using specific value
                                   Enum values: ["large", "little", "tiny"]

        Returns:
            dict: Operation result and updated states
        """
        return self._adjust_position_prop(
            "headrest_height", -1, position, value, unit, degree,
            action="seat_headrest_height_decreased"
        )

    @api("seat")
    def carcontrol_carSeatHeadRest_height_set(
        self, position=None, value=None, unit=None, degree=None
    ):
        """
        Set the car seat headrest height to a specified value.

        Args:
            position (list): List of seat positions to adjust, defaults to current speaker position
                            Enum values: ["driver's seat", "passenger seat", "second row left",
                                          "second row right", "third row left", "third row right", "all"]
            value (float, optional): Specific numerical value for adjustment
            unit (str, optional): Unit for the value
                                 Enum values: ["gear", "percentage", "centimeter"]
            degree (str, optional): Predefined position level
                                   Enum values: ["max", "high", "medium", "low", "min"]

        Returns:
            dict: Operation result and updated states
        """
        return self._adjust_position_prop(
            "headrest_height", 0, position, value, unit, degree,
            action="seat_headrest_height_set"
        )

    @api("seat")
    def carcontrol_carSeat_greetGuestMode(self, switch, position=None):
        """
        Turn on or off the car seat guest welcome mode.

        Args:
            switch (bool): True to turn welcome mode on, False to turn it off
            position (list, optional): List of seat positions to adjust, defaults to all positions
                                      Enum values: ["driver's seat", "passenger seat", "second row left",
                                                    "second row right", "third row left", "third row right", "all"]

        Returns:
            dict: Operation result and updated states
        """
        # For guest welcome mode, default to all positions if none specified
        if position is None:
            position = ["all"]

        targets = self._get_target_positions(position)
        results = {}

        for pos in targets:
            position_system = self._seats[pos]["position"]
            position_system.guest_welcome_mode = switch

            results[pos] = {"guest_welcome_mode": position_system.guest_welcome_mode}

        return {
            "success": True,
            "action": "seat_guest_welcome_mode_"
            + ("activated" if switch else "deactivated"),
            "affected_positions": targets,
            "states": results,
        }

    @api("seat")
    def carcontrol_carSeat_ventilation_switch(self, switch, position=None):
        """
        Turn on or off the car seat ventilation function.

        Args:
            switch (bool): True to turn ventilation on, False to turn it off
            position (list): List of seat positions to adjust, defaults to current speaker position
                            Enum values: ["driver's seat", "passenger seat", "second row left",
                                          "second row right", "third row left", "third row right", "all"]

        Returns:
            dict: Operation result and updated states
        """
        targets = self._get_target_positions(position)
        results = {}

        for pos in targets:
            self._seats[pos]["ventilation"].is_on = switch
            results[pos] = {"ventilation_state": self._seats[pos]["ventilation"].is_on}

        return {
            "success": True,
            "action": "ventilation_switch_" + ("on" if switch else "off"),
            "affected_positions": targets,
            "states": results,
        }

    @api("seat")
    def carcontrol_carSeat_ventilation_increase(
        self, position=None, value=None, unit=None, degree=None
    ):
        """
        Increase the car seat ventilation airflow.

        Args:
            position (list): List of seat positions to adjust, defaults to current speaker position
                            Enum values: ["driver's seat", "passenger seat", "second row left",
                                          "second row right", "third row left", "third row right", "all"]
            value (float, optional): Specific numerical value for adjustment
            unit (str, optional): Unit for the value
                                 Enum values: ["gear", "percentage"]
            degree (str, optional): Degree of adjustment if not using specific value
                                   Enum values: ["large", "little", "tiny"]

        Returns:
            dict: Operation result and updated states
        """
        targets = self._get_target_positions(position)
        results = {}

        for pos in targets:
            ventilation = self._seats[pos]["ventilation"]

            # Turn on ventilation if it's off
            if not ventilation.is_on:
                ventilation.is_on = True

            if value is not None and unit is not None:
                # Handle specific value adjustments
                if unit == "gear":
                    # Assuming gears are 1-5
                    ventilation.airflow_level = min(
                        5, ventilation.airflow_level + int(value)
                    )
                    ventilation.airflow_value = value
                    ventilation.airflow_unit = "gear"
                elif unit == "percentage":
                    # Adjust percentage and convert to level
                    current_pct = (
                        ventilation.airflow_level - 1
                    ) * 25  # Convert level to percentage (0-100)
                    new_pct = min(100, current_pct + value)
                    ventilation.airflow_level = min(5, max(1, int(new_pct / 25) + 1))
                    ventilation.airflow_value = value
                    ventilation.airflow_unit = "percentage"
            elif degree is not None:
                # Handle degree-based adjustments
                adjustment = self._convert_degree_to_level(degree)
                if adjustment:
                    ventilation.airflow_level = min(
                        5, ventilation.airflow_level + adjustment
                    )
            else:
                # Default increase by 1 level if no specifics provided
                ventilation.airflow_level = min(5, ventilation.airflow_level + 1)

            results[pos] = {
                "ventilation_state": ventilation.is_on,
                "airflow_level": ventilation.airflow_level,
            }

        return {
            "success": True,
            "action": "ventilation_airflow_increased",
            "affected_positions": targets,
            "states": results,
        }

    @api("seat")
    def carcontrol_carSeat_ventilation_decrease(
        self, position=None, value=None, unit=None, degree=None
    ):
        """
        Decrease the car seat ventilation airflow.

        Args:
            position (list): List of seat positions to adjust, defaults to current speaker position
                            Enum values: ["driver's seat", "passenger seat", "second row left",
                                          "second row right", "third row left", "third row right", "all"]
            value (float, optional): Specific numerical value for adjustment
            unit (str, optional): Unit for the value
                                 Enum values: ["gear", "percentage"]
            degree (str, optional): Degree of adjustment if not using specific value
                                   Enum values: ["large", "little", "tiny"]

        Returns:
            dict: Operation result and updated states
        """
        targets = self._get_target_positions(position)
        results = {}

        for pos in targets:
            ventilation = self._seats[pos]["ventilation"]

            # Skip if ventilation is already off
            if not ventilation.is_on:
                results[pos] = {"status": "ventilation already off"}
                continue

            if value is not None and unit is not None:
                # Handle specific value adjustments
                if unit == "gear":
                    # Assuming gears are 1-5
                    ventilation.airflow_level = max(
                        1, ventilation.airflow_level - int(value)
                    )
                    ventilation.airflow_value = value
                    ventilation.airflow_unit = "gear"
                    # Turn off if reduced to minimum with substantial value
                    if ventilation.airflow_level == 1 and value >= 1:
                        ventilation.is_on = False
                elif unit == "percentage":
                    # Adjust percentage and convert to level
                    current_pct = (
                        ventilation.airflow_level - 1
                    ) * 25  # Convert level to percentage (0-100)
                    new_pct = max(0, current_pct - value)
                    ventilation.airflow_level = max(1, int(new_pct / 25) + 1)
                    ventilation.airflow_value = value
                    ventilation.airflow_unit = "percentage"
                    # Turn off if reduced to 0%
                    if new_pct == 0:
                        ventilation.is_on = False
            elif degree is not None:
                # Handle degree-based adjustments
                adjustment = self._convert_degree_to_level(degree)
                if adjustment:
                    ventilation.airflow_level = max(
                        1, ventilation.airflow_level - adjustment
                    )
                    # Turn off if reduced to minimum with large adjustment
                    if ventilation.airflow_level == 1 and degree == "large":
                        ventilation.is_on = False
            else:
                # Default decrease by 1 level if no specifics provided
                ventilation.airflow_level = max(1, ventilation.airflow_level - 1)
                # Turn off if reduced to minimum
                if ventilation.airflow_level == 1:
                    ventilation.is_on = False

            results[pos] = {
                "ventilation_state": ventilation.is_on,
                "airflow_level": ventilation.airflow_level,
            }

        return {
            "success": True,
            "action": "ventilation_airflow_decreased",
            "affected_positions": targets,
            "states": results,
        }

    @api("seat")
    def carcontrol_carSeat_ventilation_set(
        self, position=None, value=None, unit=None, degree=None
    ):
        """
        Set the car seat ventilation airflow to a specified value.

        Args:
            position (list): List of seat positions to adjust, defaults to current speaker position
                            Enum values: ["driver's seat", "passenger seat", "second row left",
                                          "second row right", "third row left", "third row right", "all"]
            value (float, optional): Specific numerical value to set
            unit (str, optional): Unit for the value
                                 Enum values: ["gear", "percentage"]
            degree (str, optional): Predefined level if not using specific value
                                   Enum values: ["max", "high", "medium", "low", "min"]

        Returns:
            dict: Operation result and updated states
        """
        targets = self._get_target_positions(position)
        results = {}

        for pos in targets:
            ventilation = self._seats[pos]["ventilation"]

            # Turn on ventilation (unless setting to minimum/0)
            ventilation.is_on = True

            if value is not None and unit is not None:
                # Handle specific value settings
                if unit == "gear":
                    # Gears are typically 1-5
                    level = max(1, min(5, int(value)))
                    ventilation.airflow_level = level
                    ventilation.airflow_value = value
                    ventilation.airflow_unit = "gear"
                    # Turn off if set to minimum
                    if level == 1 and value <= 1:
                        ventilation.is_on = False
                elif unit == "percentage":
                    # Convert percentage to level (1-5)
                    level = max(1, min(5, int(value / 25) + 1))
                    ventilation.airflow_level = level
                    ventilation.airflow_value = value
                    ventilation.airflow_unit = "percentage"
                    # Turn off if set to 0%
                    if value == 0:
                        ventilation.is_on = False
            elif degree is not None:
                # Handle predefined level settings
                level_map = {"min": 1, "low": 2, "medium": 3, "high": 4, "max": 5}
                ventilation.airflow_level = level_map.get(
                    degree, 3
                )  # Default to medium if not recognized
                # Turn off if set to minimum
                if degree == "min":
                    ventilation.is_on = False

            results[pos] = {
                "ventilation_state": ventilation.is_on,
                "airflow_level": ventilation.airflow_level,
            }

        return {
            "success": True,
            "action": "ventilation_airflow_set",
            "affected_positions": targets,
            "states": results,
        }

    @api("seat")
    def carcontrol_carSeat_view_switch(self, switch):
        """
        Open or close the car seat control page.

        Args:
            switch (bool): True to open the control page, False to close it

        Returns:
            dict: Operation result and updated state
        """
        self._view_page_open = switch

        return {
            "success": True,
            "action": "seat_control_page_" + ("opened" if switch else "closed"),
            "state": {"view_page_open": self._view_page_open},
        }
