from enum import Enum
from utils import api
from module.base_module import BaseModule
from registry import register_module

@register_module("hazardLight", description="Hazard warning light related", category="lighting")
class HazardLight(BaseModule):
    """
    HazardLight entity class, representing the vehicle's hazard warning light system.
    Provides status management and control functionality for hazard lights.
    """
    
    class Status(Enum):
        """
        Hazard light status enumeration class, used to represent the current operating status of hazard lights
        """
        ON = "ON"  # Hazard lights are on
        OFF = "OFF"  # Hazard lights are off
    
    def __init__(self, is_active=False):
        """
        Initialize HazardLight instance
        
        Args:
            is_active (bool): Initial state of hazard lights, default is False (off)
        """
        self._is_active = is_active
        self._status = HazardLight.Status.OFF if not is_active else HazardLight.Status.ON
        
    @property
    def is_active(self):
        """Get whether hazard lights are activated"""
        return self._is_active
    
    @is_active.setter
    def is_active(self, value):
        """
        Set hazard light activation status
        
        Args:
            value (bool): Hazard light activation status, True for active, False for inactive
        """
        if not isinstance(value, bool):
            raise TypeError("is_active must be a boolean value")
        
        self._is_active = value
        self._status = HazardLight.Status.ON if value else HazardLight.Status.OFF
    
    @property
    def status(self):
        """Get current hazard light status (enum value)"""
        return self._status
    
    @api("hazardLight")
    def switch(self, switch):
        """
        Turn hazard lights on or off
        
        Args:
            switch (bool): Hazard light switch, True indicates on, False indicates off
        
        Returns:
            dict: Dictionary containing operation result and status information
        
        Raises:
            TypeError: If switch parameter is not a boolean type
        """
        # Parameter validation
        if not isinstance(switch, bool):
            raise TypeError("'switch' parameter must be a boolean (True/False)")
        
        # State transition
        previous_state = self.is_active
        self.is_active = switch
        
        # Build return result
        result = {
            "success": True,
            "previous_state": previous_state,
            "current_state": self.is_active,
            "status": self.status.value,
            "message": f"Hazard light has been turned {'ON' if switch else 'OFF'}."
        }
        
        return result
    
    @classmethod
    def init1(cls):
        """
        First initialization class method, creates a hazard light instance in off state
        Hazard light status is set to OFF, operation count is set to 0
        
        Returns:
            HazardLight: Hazard light instance with initial state as off
        """
        instance = cls(is_active=False)
        return instance
    
    @classmethod
    def init2(cls):
        """
        Second initialization class method, creates a hazard light instance in on state
        Hazard light status is set to ON, operation count is set to 1
        
        Returns:
            HazardLight: Hazard light instance with initial state as on
        """
        instance = cls(is_active=True)
        return instance