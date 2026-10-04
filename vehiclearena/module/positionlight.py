from enum import Enum
from utils import api
from module.base_module import BaseModule
from registry import register_module

@register_module("positionLight", description="Position light related", category="lighting")
class PositionLight(BaseModule):
    """
    Position Light (Outline Light) entity class, used to manage the status and operations of vehicle position lights.
    """
    
    class LightStatus(Enum):
        """
        Internal enumeration class representing the position light status.
        ON: Activated state
        OFF: Deactivated state
        """
        ON = "ON"
        OFF = "OFF"
    
    def __init__(self, is_on=False):
        """
        Initialize the position light entity class.
        
        Args:
            is_on (bool): Initial status of the position light, default is OFF (False)
        """
        # Private attribute indicating the on/off state of the position light
        self._is_on = is_on
        # Current status of the position light, represented by LightStatus enumeration
        self._status = PositionLight.LightStatus.ON if is_on else PositionLight.LightStatus.OFF
        # Last operation timestamp
        self._last_operation_timestamp = 0
        # Last operation status
        self._last_operation_status = "INIT"
    
    @property
    def is_on(self):
        """Get the on/off state of the position light"""
        return self._is_on
    
    @is_on.setter
    def is_on(self, value):
        """Set the on/off state of the position light"""
        if not isinstance(value, bool):
            raise TypeError("is_on must be a boolean value")
        self._is_on = value
        self._status = PositionLight.LightStatus.ON if value else PositionLight.LightStatus.OFF
    
    @property
    def status(self):
        """Get the status (enumeration) of the position light"""
        return self._status
    
    @status.setter
    def status(self, value):
        """Set the status (enumeration) of the position light"""
        if not isinstance(value, PositionLight.LightStatus):
            raise TypeError("status must be a LightStatus enum value")
        self._status = value
        self._is_on = (value == PositionLight.LightStatus.ON)
    
    @property
    def last_operation_timestamp(self):
        """Get the timestamp of the last operation"""
        return self._last_operation_timestamp
    
    @last_operation_timestamp.setter
    def last_operation_timestamp(self, value):
        """Set the timestamp of the last operation"""
        if not isinstance(value, (int, float)):
            raise TypeError("last_operation_timestamp must be a number")
        self._last_operation_timestamp = value
    
    @property
    def last_operation_status(self):
        """Get the status of the last operation"""
        return self._last_operation_status
    
    @last_operation_status.setter
    def last_operation_status(self, value):
        """Set the status of the last operation"""
        if not isinstance(value, str):
            raise TypeError("last_operation_status must be a string")
        self._last_operation_status = value
    
    @api("positionLight")
    def switch(self, switch):
        """
        Turn the position light on or off.
        
        Args:
            switch (bool): Position light switch state, True for ON, False for OFF
            
        Returns:
            dict: A dictionary containing operation results and related status information
        """
        if not isinstance(switch, bool):
            return {
                "success": False,
                "error": "Invalid parameter: switch must be a boolean value",
            }
        
        try:
            # Update status
            self.is_on = switch
            
            # Update last operation information
            import time
            self.last_operation_timestamp = time.time()
            self.last_operation_status = "SUCCESS"
            
            return {
                "success": True,
                "message": f"Position light turned {'on' if switch else 'off'} successfully",
            }
        except Exception as e:
            self.last_operation_status = "FAILED"
            return {
                "success": False,
                "error": str(e),
            }
    
    @api("positionLight")
    def carcontrol_positionLight_switch(self, switch):
        """
        Turn the position light (outline light) on or off.
        
        Args:
            switch (bool): Position light switch parameter, True for ON, False for OFF
            
        Returns:
            dict: A dictionary containing operation results and related status information
        """
        # Call the switch method to implement the functionality
        return self.switch(switch)
    
    @classmethod
    def init1(cls):
        """
        Initialize a position light instance with default OFF state.
        
        Returns:
            PositionLight: A position light instance in OFF state
        """
        instance = cls(is_on=False)
        # Set initialization information
        import time
        instance.last_operation_timestamp = time.time()
        instance.last_operation_status = "INITIALIZED_OFF"
        
        return instance

    @classmethod
    def init2(cls):
        """
        Initialize a position light instance with default ON state.
        
        Returns:
            PositionLight: A position light instance in ON state
        """
        instance = cls(is_on=True)
        # Set initialization information
        import time
        instance.last_operation_timestamp = time.time()
        instance.last_operation_status = "INITIALIZED_ON"
        
        return instance