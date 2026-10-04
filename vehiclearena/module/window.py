from utils import api
from enum import Enum
from typing import List, Optional
from module.base_module import BaseModule
from registry import register_module


@register_module("window", description="Window related", category="body", needs_settings=True)
class Window(BaseModule):
    class WindowPosition(Enum):
        """
        Enum representing different window positions in the car.
        """

        DRIVER_SEAT = "driver's seat"
        PASSENGER_SEAT = "passenger seat"
        SECOND_ROW_LEFT = "second row left"
        SECOND_ROW_RIGHT = "second row right"
        THIRD_ROW_LEFT = "third row left"
        THIRD_ROW_RIGHT = "third row right"
        ALL = "all"

    class HeightAdjustmentDegree(Enum):
        """
        Enum representing different levels of window height adjustments.
        """

        # For increase/decrease
        LARGE = "large"
        LITTLE = "little"
        TINY = "tiny"
        # For setting specific height
        MAX = "max"
        HIGH = "high"
        MEDIUM = "medium"
        LOW = "low"
        MIN = "min"

    class HeightAdjustmentUnit(Enum):
        """
        Enum representing units for window height adjustments.
        """

        GEAR = "gear"
        PERCENTAGE = "percentage"
        CENTIMETER = "centimeter"

    class WindowState:
        """
        Internal class representing the state of a window.
        """

        def __init__(self, position):
            self.position = position
            self._is_open = False  # Window closed by default
            self._open_degree = 0  # 0% open (fully closed)
            self._child_safety_lock = False  # Child safety lock disabled by default


        @property
        def is_open(self):
            return self._is_open

        @is_open.setter
        def is_open(self, value):
            if not isinstance(value, bool):
                raise ValueError("is_open must be a boolean value")
            self._is_open = value
            # Update open degree to match open state
            if value:
                # Opening a closed window creates a small 10% opening.
                if self._open_degree == 0:
                    self._open_degree = 10
            else:
                # If window is closed, degree is 0
                self._open_degree = 0

        @property
        def open_degree(self):
            return self._open_degree

        @open_degree.setter
        def open_degree(self, value):
            if not isinstance(value, (int, float)):
                raise ValueError("open_degree must be a numeric value")
            if value < 0:
                value = 0
            elif value > 100:
                value = 100
            self._open_degree = value
            # Update open state to match degree
            self._is_open = value > 0

        @property
        def child_safety_lock(self):
            return self._child_safety_lock

        @child_safety_lock.setter
        def child_safety_lock(self, value):
            if not isinstance(value, bool):
                raise ValueError("child_safety_lock must be a boolean value")
            self._child_safety_lock = value


    def set_settings(self, settings):
        """Inject shared VehicleSettings instance."""
        self._settings = settings

    def __init__(self):
        self._settings = None
        # Initialize window states for all positions
        self._windows = {
            position: Window.WindowState(position)
            for position in Window.WindowPosition
            if position != Window.WindowPosition.ALL
        }
        self._auto_close_on_lock = (
            False  # Auto close on lock feature disabled by default
        )
        

    @classmethod
    def init1(cls):
        """
        Initialize the Window class with a specific configuration:
        - Driver's window: Partially open (40%)
        - Passenger window: Closed
        - Second row windows: Fully open
        - Third row windows: Closed with child safety locks enabled
        - Auto-close on lock feature: Enabled
        
        Returns:
            Window: Initialized Window instance
        """
        # Create a new instance
        instance = cls()
        
        # Configure driver's seat window
        # driver_window = instance._windows[Window.WindowPosition.DRIVER_SEAT]
        # driver_window.open_degree = 40  # 40% open
        
        # # Configure passenger seat window (default is closed)
        
        # # Configure second row windows
        # second_row_left = instance._windows[Window.WindowPosition.SECOND_ROW_LEFT]
        # second_row_left.open_degree = 100  # Fully open
        
        # second_row_right = instance._windows[Window.WindowPosition.SECOND_ROW_RIGHT]
        # second_row_right.open_degree = 100  # Fully open
        
        # Configure third row windows with child safety locks
        # third_row_left = instance._windows[Window.WindowPosition.THIRD_ROW_LEFT]
        # third_row_left.child_safety_lock = True
        
        # third_row_right = instance._windows[Window.WindowPosition.THIRD_ROW_RIGHT]
        # third_row_right.child_safety_lock = True
        
        # # Enable auto-close on lock feature
        # instance.auto_close_on_lock = True
        
        return instance

    @classmethod
    def init2(cls):
        """
        Initialize the Window class with a more secure configuration:
        - Driver's window: Open slightly (15%)
        - Passenger window: Open slightly (10%)
        - Second row windows: Closed with child safety locks enabled
        - Third row windows: Closed with child safety locks enabled
        - Auto-close on lock feature: Enabled
        
        This configuration prioritizes security with minimal window openings
        and child safety locks on all windows except the driver's.
        
        Returns:
            Window: Initialized Window instance
        """
        # Create a new instance
        instance = cls()
        
        # Configure driver's seat window
        driver_window = instance._windows[Window.WindowPosition.DRIVER_SEAT]
        driver_window.open_degree = 15  # Slightly open (15%)
        
        # Configure passenger seat window
        passenger_window = instance._windows[Window.WindowPosition.PASSENGER_SEAT]
        passenger_window.open_degree = 10  # Slightly open (10%)
        passenger_window.child_safety_lock = True  # Enable child safety lock
        
        # Configure second row windows with child safety locks
        second_row_left = instance._windows[Window.WindowPosition.SECOND_ROW_LEFT]
        second_row_left.child_safety_lock = True
        
        second_row_right = instance._windows[Window.WindowPosition.SECOND_ROW_RIGHT]
        second_row_right.child_safety_lock = True
        
        # Configure third row windows with child safety locks
        third_row_left = instance._windows[Window.WindowPosition.THIRD_ROW_LEFT]
        third_row_left.child_safety_lock = True
        
        third_row_right = instance._windows[Window.WindowPosition.THIRD_ROW_RIGHT]
        third_row_right.child_safety_lock = True
        
        # Enable auto-close on lock feature
        instance.auto_close_on_lock = True
        
        return instance
    
    @property
    def auto_close_on_lock(self):
        return self._auto_close_on_lock

    @auto_close_on_lock.setter
    def auto_close_on_lock(self, value):
        if not isinstance(value, bool):
            raise ValueError("auto_close_on_lock must be a boolean value")
        self._auto_close_on_lock = value

    def get_window_state(self, position):
        """
        Get the window state for a specific position.

        Args:
            position: Window position enum or string value

        Returns:
            WindowState: The window state for the specified position
        """
        if isinstance(position, str):
            try:
                position = Window.WindowPosition(position)
            except ValueError:
                raise ValueError(f"Invalid window position: {position}")

        if position == Window.WindowPosition.ALL:
            raise ValueError(
                "Cannot get window state for 'all' position, specify a concrete position"
            )

        return self._windows[position]

    def _get_positions_from_input(self, position_input: Optional[List[str]] = None):
        """
        Helper method to parse position input and return the list of positions to operate on.

        Args:
            position_input: Position input from API, can be a list of strings or None

        Returns:
            list: List of WindowPosition enums to operate on
        """
        if position_input is None:
            # Default to current speaker's position
            speaker_position = self._settings.speaker
            try:
                return [Window.WindowPosition(speaker_position)]
            except ValueError:
                raise ValueError(f"Invalid speaker position: {speaker_position}")

        if not isinstance(position_input, list):
            raise ValueError("position_input must be a list of strings")

        positions = []
        # Process list of positions
        for pos in position_input:
            try:
                position = Window.WindowPosition(pos)
                if position == Window.WindowPosition.ALL:
                    # Return all concrete positions
                    return [
                        pos
                        for pos in Window.WindowPosition
                        if pos != Window.WindowPosition.ALL
                    ]
                positions.append(position)
            except ValueError:
                raise ValueError(f"Invalid window position: {pos}")

        return positions

    def _adjust_height_by_degree(self, window_state, degree, is_increase):
        """
        Helper method to adjust window height by degree (large, little, tiny).

        Args:
            window_state: WindowState object to adjust
            degree: String degree of adjustment
            is_increase: Boolean indicating if height should increase (True) or decrease (False)
        """
        adjustment_values = {
            Window.HeightAdjustmentDegree.LARGE: 30,
            Window.HeightAdjustmentDegree.LITTLE: 15,
            Window.HeightAdjustmentDegree.TINY: 5,
        }

        try:
            degree_enum = Window.HeightAdjustmentDegree(degree)
        except ValueError:
            raise ValueError(f"Invalid height adjustment degree: {degree}")

        adjustment = adjustment_values.get(degree_enum, 10)  # Default to 10% if unknown

        if is_increase:
            # Increasing height means decreasing openness
            window_state.open_degree -= adjustment
        else:
            # Decreasing height means increasing openness
            window_state.open_degree += adjustment

    def _adjust_height_by_value(self, window_state, value, unit, is_increase):
        """
        Helper method to adjust window height by numeric value and unit.

        Args:
            window_state: WindowState object to adjust
            value: Numeric value of adjustment
            unit: Unit of adjustment (gear, percentage, centimeter)
            is_increase: Boolean indicating if height should increase (True) or decrease (False)
        """
        if not isinstance(value, (int, float)):
            raise ValueError("Value must be a number")

        try:
            unit_enum = Window.HeightAdjustmentUnit(unit)
        except ValueError:
            raise ValueError(f"Invalid height adjustment unit: {unit}")

        # Convert value to percentage based on unit
        percentage_adjustment = 0
        if unit_enum == Window.HeightAdjustmentUnit.PERCENTAGE:
            percentage_adjustment = value
        elif unit_enum == Window.HeightAdjustmentUnit.GEAR:
            # Assume 5 gears (0-4), each representing 25% of window height
            percentage_adjustment = value * 25
        elif unit_enum == Window.HeightAdjustmentUnit.CENTIMETER:
            # Assume 20cm is full window height (100%)
            percentage_adjustment = (value / 20) * 100

        if is_increase:
            # Increasing height means decreasing openness
            window_state.open_degree -= percentage_adjustment
        else:
            # Decreasing height means increasing openness
            window_state.open_degree += percentage_adjustment

    def _set_height_by_degree(self, window_state, degree):
        """
        Helper method to set window height by degree (max, high, medium, low, min).

        Args:
            window_state: WindowState object to adjust
            degree: String degree for the target height
        """
        open_percentages = {
            Window.HeightAdjustmentDegree.MAX: 0,
            Window.HeightAdjustmentDegree.HIGH: 25,
            Window.HeightAdjustmentDegree.MEDIUM: 50,  # Medium height = 50% open
            Window.HeightAdjustmentDegree.LOW: 75,  # Low height = 75% open
            Window.HeightAdjustmentDegree.MIN: 100,
        }

        try:
            degree_enum = Window.HeightAdjustmentDegree(degree)
        except ValueError:
            raise ValueError(f"Invalid height setting degree: {degree}")

        window_state.open_degree = open_percentages[degree_enum]

    @staticmethod
    def _height_receipt(window_state):
        """Expose the resulting physical setting, not internal control state."""
        return {
            "height_percent": 100 - window_state.open_degree,
            "open_percent": window_state.open_degree,
        }

    def _set_height_by_value(self, window_state, value, unit):
        """
        Helper method to set window height by numeric value and unit.

        Args:
            window_state: WindowState object to adjust
            value: Numeric value for the target height
            unit: Unit for the target height (gear, percentage, centimeter)
        """
        if not isinstance(value, (int, float)):
            raise ValueError("Value must be a number")

        try:
            unit_enum = Window.HeightAdjustmentUnit(unit)
        except ValueError:
            raise ValueError(f"Invalid height setting unit: {unit}")

        # Convert value to percentage based on unit
        openness = 0
        if unit_enum == Window.HeightAdjustmentUnit.PERCENTAGE:
            # Direct percentage (100% = fully closed, 0% = fully open)
            openness = 100 - value  # Invert for openness
        elif unit_enum == Window.HeightAdjustmentUnit.GEAR:
            # Assume 5 gears (0-4), each representing 25% of window height
            openness = 100 - (value * 25)
        elif unit_enum == Window.HeightAdjustmentUnit.CENTIMETER:
            # Assume 20cm is full window height (100%)
            openness = 100 - ((value / 20) * 100)

        # Ensure openness is within valid range
        openness = max(0, min(100, openness))
        window_state.open_degree = openness

    @api("window")
    def carcontrol_window_mode_childSafetyLock(self, switch, position: List[str] = ["all"]):
        """
        Turn on or off the child safety lock mode for the car windows.

        Args:
            switch (bool): Child safety lock switch for the window.
                           True indicates enabling the lock mode (lock/engage),
                           false indicates disabling the lock mode (unlock/disengage).
            position (List[str]): Specifies the window positions for activating child safety lock.
                           Must select from: "driver's seat", "passenger seat", "second row left",
                           "second row right", "third row left", "third row right", "all"
                           Default is current speaker position.

        Returns:
            dict: A dictionary containing operation result and updated states
        """
        if not isinstance(switch, bool):
            raise ValueError("switch must be a boolean value")

        positions = self._get_positions_from_input(position)
        updated_states = {}

        for pos in positions:
            window_state = self._windows[pos]
            window_state.child_safety_lock = switch

        return {
            "success": True,
            "message": f"Child safety lock {'enabled' if switch else 'disabled'} for specified windows",
            "updated_states": updated_states,
        }

    @api("window")
    def carcontrol_window_switch(self, position: List[str], switch):
        """
        Open or close the car window. Opening a closed window creates a
        10% opening; closing sets its opening to 0%. An already open window
        retains its opening when switch=true.

        Args:
            position (List[str]): The window positions. Must select from: "driver's seat",
                                "passenger seat", "second row left", "second row right",
                                "third row left", "third row right", "all"
                                If not specified, defaults to current speaker's position.
            switch (bool): Window switch, true to open, false to close

        Returns:
            dict: A dictionary containing operation result and updated states
        """
        if not isinstance(switch, bool):
            raise ValueError("switch must be a boolean value")

        positions = self._get_positions_from_input(position)
        updated_states = {}

        for pos in positions:
            window_state = self._windows[pos]
            window_state.is_open = switch
            if not switch:
                window_state._open_degree = 0
            updated_states[pos.value] = self._height_receipt(window_state)

        return {
            "success": True,
            "message": f"Windows {'opened' if switch else 'closed'} for specified positions",
            "updated_states": updated_states,
        }

    @api("window")
    def carcontrol_window_height_increase(
        self, position: List[str], value=None, unit=None, degree=None
    ):
        """
        Raise the glass, reducing the opening (toward closed).
        The receipt reports resulting height_percent and open_percent.

        Args:
            position (List[str]): Window positions. Must select from: "driver's seat",
                               "passenger seat", "second row left", "second row right",
                               "third row left", "third row right", "all"
                               If not specified, defaults to current speaker's position.
            value (float, optional): Numeric part of window height adjustment.
                                    Mutually exclusive with 'degree'.
            unit (str, optional): Unit of window height adjustment.
                                 Must select from: "gear", "percentage", "centimeter"
                                 Mutually exclusive with 'degree'.
            degree (str, optional): Level of window height adjustment.
                                   Must select from: "large", "little", "tiny"
                                   Mutually exclusive with 'value/unit'.

        Returns:
            dict: A dictionary containing operation result and updated states
        """
        positions = self._get_positions_from_input(position)
        updated_states = {}

        if (value is not None and unit is None) or (value is None and unit is not None):
            raise ValueError("Both 'value' and 'unit' must be provided together")

        if value is not None and degree is not None:
            raise ValueError("'value/unit' and 'degree' are mutually exclusive")

        for pos in positions:
            window_state = self._windows[pos]

            if degree is not None:
                self._adjust_height_by_degree(window_state, degree, is_increase=True)
            elif value is not None and unit is not None:
                self._adjust_height_by_value(
                    window_state, value, unit, is_increase=True
                )
            else:
                # Default to small adjustment if no parameters provided
                self._adjust_height_by_degree(window_state, "little", is_increase=True)
            updated_states[pos.value] = self._height_receipt(window_state)

        return {
            "success": True,
            "message": "Window height increased for specified positions",
            "updated_states": updated_states,
        }

    @api("window")
    def carcontrol_window_height_decrease(
        self, position: List[str], value=None, unit=None, degree=None
    ):
        """
        Lower the glass, increasing the opening (toward open).
        The receipt reports resulting height_percent and open_percent.

        Args:
            position (List[str]): Window positions. Must select from: "driver's seat",
                               "passenger seat", "second row left", "second row right",
                               "third row left", "third row right", "all"
                               If not specified, defaults to current speaker's position.
            value (float, optional): Numeric part of window height adjustment.
                                    Mutually exclusive with 'degree'.
            unit (str, optional): Unit of window height adjustment.
                                 Must select from: "gear", "percentage", "centimeter"
                                 Mutually exclusive with 'degree'.
            degree (str, optional): Level of window height adjustment.
                                   Must select from: "large", "little", "tiny"
                                   Mutually exclusive with 'value/unit'.

        Returns:
            dict: A dictionary containing operation result and updated states
        """
        positions = self._get_positions_from_input(position)
        updated_states = {}

        if (value is not None and unit is None) or (value is None and unit is not None):
            raise ValueError("Both 'value' and 'unit' must be provided together")

        if value is not None and degree is not None:
            raise ValueError("'value/unit' and 'degree' are mutually exclusive")

        for pos in positions:
            window_state = self._windows[pos]

            if degree is not None:
                self._adjust_height_by_degree(window_state, degree, is_increase=False)
            elif value is not None and unit is not None:
                self._adjust_height_by_value(
                    window_state, value, unit, is_increase=False
                )
            else:
                # Default to small adjustment if no parameters provided
                self._adjust_height_by_degree(window_state, "little", is_increase=False)
            updated_states[pos.value] = self._height_receipt(window_state)

        return {
            "success": True,
            "message": "Window height decreased for specified positions",
            "updated_states": updated_states,
        }

    @api("window")
    def carcontrol_window_height_set(
        self, position: List[str], value=None, unit=None, degree=None
    ):
        """
        Set GLASS HEIGHT, not opening percentage. At 100% height the window
        is fully closed; at 0% it is fully open. For a small 10% opening,
        set value=90 and unit="percentage"; for 20% opening set value=80.
        The receipt reports resulting height_percent and open_percent.

        Args:
            position (List[str]): Window positions. Must select from: "driver's seat",
                               "passenger seat", "second row left", "second row right",
                               "third row left", "third row right", "all"
                               If not specified, defaults to current speaker's position.
            value (float, optional): Glass height, NOT opening: 100=closed, 0=fully open, 90=10% open (percentage).
                                    Mutually exclusive with 'degree'.
            unit (str, optional): Unit of window height setting.
                                 Must select from: "gear", "percentage", "centimeter"
                                 Mutually exclusive with 'degree'.
            degree (str, optional): Glass height preset: max=closed,
                                   high=25% open, medium=50% open,
                                   low=75% open, min=fully open.
                                   Must select from: "max", "high", "medium", "low", "min"
                                   Mutually exclusive with 'value/unit'.

        Returns:
            dict: A dictionary containing operation result and updated states
        """
        positions = self._get_positions_from_input(position)
        updated_states = {}

        if (value is not None and unit is None) or (value is None and unit is not None):
            raise ValueError("Both 'value' and 'unit' must be provided together")

        if value is not None and degree is not None:
            raise ValueError("'value/unit' and 'degree' are mutually exclusive")

        if value is None and unit is None and degree is None:
            raise ValueError("Either 'value/unit' or 'degree' must be provided")

        for pos in positions:
            window_state = self._windows[pos]

            if degree is not None:
                self._set_height_by_degree(window_state, degree)
            elif value is not None and unit is not None:
                self._set_height_by_value(window_state, value, unit)
            updated_states[pos.value] = self._height_receipt(window_state)

        return {
            "success": True,
            "message": "Window height set for specified positions",
            "updated_states": updated_states,
        }

    @api("window")
    def carcontrol_window_mode_leaveAndLockAutoCloseWindow(self, switch):
        """
        Turn on or off the 'Auto Close Window when Locking the Car' mode.

        Args:
            switch (bool): Auto close window when locking the car switch,
                           true to turn on, false to turn off

        Returns:
            dict: A dictionary containing operation result and updated state
        """
        if not isinstance(switch, bool):
            raise ValueError("switch must be a boolean value")

        self.auto_close_on_lock = switch

        return {
            "success": True,
            "message": f"Auto close window when locking car {'enabled' if switch else 'disabled'}",
            "updated_state": {
                "auto_close_on_lock": {
                    "value": self.auto_close_on_lock,
                    "description": "Whether windows automatically close when car is locked",
                    "type": type(self.auto_close_on_lock).__name__,
                }
            },
        }
