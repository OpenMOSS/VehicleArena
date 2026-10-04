from utils import api
from enum import Enum
from typing import Dict, List, Union, Optional, Any
import sys
from module.base_module import BaseModule
from registry import register_module


@register_module("door", description="Door related", category="body", needs_settings=True)
class Door(BaseModule):
    class DoorPosition(Enum):
        """
        门的位置枚举
        """

        DRIVER_SEAT = "driver's seat"
        PASSENGER_SEAT = "passenger seat"
        SECOND_ROW_LEFT = "second row left"
        SECOND_ROW_RIGHT = "second row right"
        ALL = "all"
        THIRD_ROW_LEFT = "third row left"
        THIRD_ROW_RIGHT = "third row right"

    class DoorStatus(Enum):
        """
        门的开关状态枚举 - 简化为仅开和关
        """

        OPEN = "open"
        CLOSED = "closed"

    class DoorAngleUnit(Enum):
        """
        门角度调整的单位枚举
        """

        GEAR = "gear"
        PERCENTAGE = "percentage"
        CENTIMETER = "centimeter"

    class DoorAngleDegree(Enum):
        """
        门角度调整程度的枚举（增加/减少）
        """

        LARGE = "large"
        LITTLE = "little"
        TINY = "tiny"

    class DoorAnglePreset(Enum):
        """
        门角度预设值的枚举
        """

        MAX = "max"
        HIGH = "high"
        MEDIUM = "medium"
        LOW = "low"
        MIN = "min"

    class DoorAction(Enum):
        """
        门动作的枚举
        """

        OPEN = "open"
        CLOSE = "close"

    class DoorState:
        """
        门的状态内部类，包含门的各种属性
        """

        def __init__(self, position: str):
            self._position = position
            self._is_locked = False
            self._child_safety_lock_enabled = False
            self._status = Door.DoorStatus.CLOSED.value
            self._angle = 0  # 0 表示完全关闭，100表示最大开度
            self._angle_unit = Door.DoorAngleUnit.PERCENTAGE.value
            self._max_angle = {
                Door.DoorAngleUnit.GEAR.value: 5,
                Door.DoorAngleUnit.PERCENTAGE.value: 100,
                Door.DoorAngleUnit.CENTIMETER.value: 80,
            }
            # 预设角度值对应的百分比
            self._preset_angles = {
                Door.DoorAnglePreset.MIN.value: 20,
                Door.DoorAnglePreset.LOW.value: 40,
                Door.DoorAnglePreset.MEDIUM.value: 60,
                Door.DoorAnglePreset.HIGH.value: 80,
                Door.DoorAnglePreset.MAX.value: 100,
            }
            # 相对调整的增量值
            self._angle_adjustments = {
                Door.DoorAngleDegree.TINY.value: {
                    Door.DoorAngleUnit.GEAR.value: 1,
                    Door.DoorAngleUnit.PERCENTAGE.value: 10,
                    Door.DoorAngleUnit.CENTIMETER.value: 5,
                },
                Door.DoorAngleDegree.LITTLE.value: {
                    Door.DoorAngleUnit.GEAR.value: 2,
                    Door.DoorAngleUnit.PERCENTAGE.value: 20,
                    Door.DoorAngleUnit.CENTIMETER.value: 15,
                },
                Door.DoorAngleDegree.LARGE.value: {
                    Door.DoorAngleUnit.GEAR.value: 3,
                    Door.DoorAngleUnit.PERCENTAGE.value: 30,
                    Door.DoorAngleUnit.CENTIMETER.value: 25,
                },
            }


        @property
        def position(self) -> str:
            return self._position

        @property
        def is_locked(self) -> bool:
            return self._is_locked

        @is_locked.setter
        def is_locked(self, value: bool):
            self._is_locked = value

        @property
        def child_safety_lock_enabled(self) -> bool:
            return self._child_safety_lock_enabled

        @child_safety_lock_enabled.setter
        def child_safety_lock_enabled(self, value: bool):
            self._child_safety_lock_enabled = value

        @property
        def status(self) -> str:
            return self._status

        @status.setter
        def status(self, value: str):
            self._status = value
            if value:
                # If window is open but degree is 0, set to 100%
                if self._angle == 0:
                    self._angle = 10
            else:
                # If window is closed, degree is 0
                self._angle = 0

        @property
        def angle(self) -> float:
            return self._angle

        @angle.setter
        def angle(self, value: float):
            # 确保角度在有效范围内
            self._angle = max(0, min(self._max_angle[self._angle_unit], value))
            
            # 根据角度更新门的状态
            if self._angle > 0:
                self._status = Door.DoorStatus.OPEN.value
            else:
                self._status = Door.DoorStatus.CLOSED.value

        @property
        def angle_unit(self) -> str:
            return self._angle_unit

        @angle_unit.setter
        def angle_unit(self, value: str):
            if value in [unit.value for unit in Door.DoorAngleUnit]:
                self._angle_unit = value


  
    def set_settings(self, settings):
        """Inject shared VehicleSettings instance."""
        self._settings = settings

    def __init__(self):
        self._settings = None
        # 初始化每个门的状态
        self._doors = {
            Door.DoorPosition.DRIVER_SEAT.value: Door.DoorState(
                Door.DoorPosition.DRIVER_SEAT.value
            ),
            Door.DoorPosition.PASSENGER_SEAT.value: Door.DoorState(
                Door.DoorPosition.PASSENGER_SEAT.value
            ),
            Door.DoorPosition.SECOND_ROW_LEFT.value: Door.DoorState(
                Door.DoorPosition.SECOND_ROW_LEFT.value
            ),
            Door.DoorPosition.SECOND_ROW_RIGHT.value: Door.DoorState(
                Door.DoorPosition.SECOND_ROW_RIGHT.value
            ),
            Door.DoorPosition.THIRD_ROW_LEFT.value: Door.DoorState(
                Door.DoorPosition.THIRD_ROW_LEFT.value
            ),
            Door.DoorPosition.THIRD_ROW_RIGHT.value: Door.DoorState(
                Door.DoorPosition.THIRD_ROW_RIGHT.value
            ),
        }
   


    @classmethod
    def init1(cls):
        """
        初始化方法1: 创建一个默认关闭状态的门实例
        
        返回:
        - Door 门类的实例，所有门处于默认关闭状态
        """
        door = cls()
        # 所有的门已经默认为关闭状态，无需额外设置
        return door

    @classmethod
    def init2(cls):
        """
        初始化方法2: 创建一个开放状态的门实例，驾驶位和乘客位的门是开着的
        
        返回:
        - Door 门类的实例，驾驶位和乘客位的门处于打开状态
        """
        door = cls()
        
        # 打开驾驶位门
        driver_door = door.get_door_state(Door.DoorPosition.DRIVER_SEAT.value)
        if driver_door:
            driver_door.status = Door.DoorStatus.OPEN.value
            driver_door.angle = driver_door._max_angle[driver_door.angle_unit]
        
        # 打开乘客位门
        passenger_door = door.get_door_state(Door.DoorPosition.PASSENGER_SEAT.value)
        if passenger_door:
            passenger_door.status = Door.DoorStatus.OPEN.value
            passenger_door.angle = passenger_door._max_angle[passenger_door.angle_unit]
        
        return door
        
    def get_door_state(self, position: str) -> Optional[DoorState]:
        """
        获取指定位置的门状态
        """
        return self._doors.get(position)

    def get_all_positions(self) -> List[str]:
        """
        获取所有门的位置
        """
        return [
            pos.value
            for pos in Door.DoorPosition
            if pos.value != Door.DoorPosition.ALL.value
        ]

    def resolve_positions(self, position: Optional[str]) -> List[str]:
        """
        解析门的位置参数，处理'all'和缺省情况
        """
        if not position:
            # 获取说话人的位置
            speaker_position = self._settings.speaker
            return [speaker_position] if speaker_position in self._doors else []
        elif position == Door.DoorPosition.ALL.value:
            return self.get_all_positions()
        elif position in self._doors:
            return [position]
        else:
            return []

    @api("door")
    def carcontrol_carDoor_lock_switch(
        self, switch: bool, position: Optional[List[str]] = None
    ) -> Dict:
        """
        Enable or disable car door lock

        Parameters:
        - switch: bool Door lock switch, True means lock, False means unlock
        - position: Optional[List[str]] Door position, must be selected from the following list:
          [driver's seat, passenger seat, second row left, second row right, all]
          If not specified, defaults to user's current position

        Returns:
        - Dict Contains operation result and related status information
        """
        results = {}
        success = True
        error_message = None

        # 处理位置参数
        positions_to_update = []
        if not position:
            # 使用说话人位置
            speaker_pos = self._settings.speaker
            positions_to_update = self.resolve_positions(speaker_pos)
        else:
            # 处理列表中的每个位置
            for pos in position:
                positions_to_update.extend(self.resolve_positions(pos))

        # 如果没有有效位置，返回错误
        if not positions_to_update:
            return {
                "success": False,
                "error": "Invalid door position specified",
                "updated_doors": {},
            }

        # 更新每个门的锁状态
        for pos in positions_to_update:
            door = self.get_door_state(pos)
            if door:
                door.is_locked = switch
                results[pos] = {
                    "is_locked": door.is_locked,
                    "status": door.status,
                }
            else:
                success = False
                error_message = f"Door at position {pos} not found"
                break

        return {
            "success": success,
            "error": error_message,
            "operation": "lock" if switch else "unlock",
            "updated_doors": results,
        }

    @api("door")
    def carcontrol_carDoor_mode_childSafetyLock(
        self, switch: bool, position: Optional[List[str]] = None
    ) -> Dict:
        """
        Enable or disable car door child safety lock mode

        Parameters:
        - switch: bool Child safety lock switch, True means enable child safety lock, False means disable child safety lock
        - position: Optional[List[str]] Door position, must be selected from the following list:
          [driver's seat, passenger seat, second row left, second row right, all]
          If not specified, defaults to current speaker

        Returns:
        - Dict Contains operation result and related status information
        """
        results = {}
        success = True
        error_message = None

        # 处理位置参数
        positions_to_update = []
        if not position:
            # 使用说话人位置
            speaker_pos = self._settings.speaker
            positions_to_update = self.resolve_positions(speaker_pos)
        else:
            # 处理列表中的每个位置
            for pos in position:
                positions_to_update.extend(self.resolve_positions(pos))

        # 如果没有有效位置，返回错误
        if not positions_to_update:
            return {
                "success": False,
                "error": "Invalid door position specified",
                "updated_doors": {},
            }

        # 更新每个门的儿童安全锁状态
        for pos in positions_to_update:
            door = self.get_door_state(pos)
            if door:
                door.child_safety_lock_enabled = switch
                results[pos] = {
                    "child_safety_lock_enabled": door.child_safety_lock_enabled
                }
            else:
                success = False
                error_message = f"Door at position {pos} not found"
                break

        return {
            "success": success,
            "error": error_message,
            "operation": "enable_child_safety" if switch else "disable_child_safety",
            "updated_doors": results,
        }

    @api("door")
    def carcontrol_carDoor_switch(
        self, action: str, position: Optional[List[str]] = None
    ) -> Dict:
        """
        Open or close car door

        Parameters:
        - action: str Door action, must be selected from the following enumeration values: [open, close]
        - position: Optional[List[str]] Door position, must be selected from the following list:
          [driver's seat, passenger seat, second row left, second row right, all]
          If not specified, defaults to user's current position

        Returns:
        - Dict Contains operation result and related status information
        """
        results = {}
        success = True
        error_message = None

        # 验证动作参数
        if action not in [a.value for a in Door.DoorAction]:
            return {
                "success": False,
                "error": f"Invalid action '{action}'. Must be one of: {[a.value for a in Door.DoorAction]}",
                "updated_doors": {},
            }

        # 处理位置参数
        positions_to_update = []
        if not position:
            # 使用说话人位置
            speaker_pos = self._settings.speaker
            positions_to_update = self.resolve_positions(speaker_pos)
        else:
            # 处理列表中的每个位置
            for pos in position:
                positions_to_update.extend(self.resolve_positions(pos))

        # 如果没有有效位置，返回错误
        if not positions_to_update:
            return {
                "success": False,
                "error": "Invalid door position specified",
                "updated_doors": {},
            }

        # Multi-position commands are atomic.  Validate every target before
        # changing any door so a locked later target cannot leave earlier
        # doors open while the overall operation reports failure.
        if action == Door.DoorAction.OPEN.value:
            locked_positions = [
                pos for pos in positions_to_update
                if (self.get_door_state(pos) is not None
                    and self.get_door_state(pos).is_locked)
            ]
            if locked_positions:
                return {
                    "success": False,
                    "error": (
                        "Cannot open locked door(s): "
                        + ", ".join(locked_positions)),
                    "action": action,
                    "updated_doors": {},
                }

        # 更新每个门的状态
        for pos in positions_to_update:
            door = self.get_door_state(pos)
            if door:
                prev_status = door.status
                prev_angle = door.angle

                # 根据动作更新门的状态
                if action == Door.DoorAction.OPEN.value:
                    if door.is_locked:
                        success = False
                        error_message = f"Cannot open door at {pos} because it is locked"
                        break

                    # 设置开门状态
                    door.status = Door.DoorStatus.OPEN.value
                    # # 打开门时，设置角度为最大值
                    # door.angle = door._max_angle[door.angle_unit]

                elif action == Door.DoorAction.CLOSE.value:
                    # 设置关门状态
                    door.status = Door.DoorStatus.CLOSED.value
                    # 关门时，角度设为0
                    door.angle = 0

                # 记录结果
                results[pos] = {
                    "previous_status": prev_status,
                    "current_status": door.status,
                    "previous_angle": prev_angle,
                    "current_angle": door.angle,
                    "is_locked": door.is_locked,
                }
            else:
                success = False
                error_message = f"Door at position {pos} not found"
                break

        return {
            "success": success,
            "error": error_message,
            "action": action,
            "updated_doors": results,
        }

    @api("door")
    def carcontrol_carDoor_angle_increase(
        self,
        position: Optional[List[str]] = None,
        value: Optional[float] = None,
        unit: Optional[str] = None,
        degree: Optional[str] = None,
    ) -> Dict:
        """
        Increase car door opening angle

        Parameters:
        - position: Optional[List[str]] Door position, must be selected from the following list:
          [driver's seat, passenger seat, second row left, second row right, all]
          If not specified, defaults to user's current position
        - value: Optional[float] Specific numerical value for increasing angle, unit specified by unit field
        - unit: Optional[str] Unit for adjusting door angle, must be selected from the following list: [gear, percentage, centimeter]
          Not compatible with degree parameter
        - degree: Optional[str] Adjustment level, must be selected from the following list: [large, little, tiny]
          Not compatible with value/unit parameters

        Returns:
        - Dict Contains operation result and related status information
        """
        results = {}
        success = True
        error_message = None

        # 验证参数组合
        if value is not None and unit is None:
            return {
                "success": False,
                "error": "When specifying 'value', 'unit' must also be provided",
                "updated_doors": {},
            }

        if unit is not None and value is None:
            return {
                "success": False,
                "error": "When specifying 'unit', 'value' must also be provided",
                "updated_doors": {},
            }

        if (value is not None or unit is not None) and degree is not None:
            return {
                "success": False,
                "error": "'degree' parameter is incompatible with 'value' and 'unit' parameters",
                "updated_doors": {},
            }

        if unit is not None and unit not in [u.value for u in Door.DoorAngleUnit]:
            return {
                "success": False,
                "error": f"Invalid unit '{unit}'. Must be one of: {[u.value for u in Door.DoorAngleUnit]}",
                "updated_doors": {},
            }

        if degree is not None and degree not in [d.value for d in Door.DoorAngleDegree]:
            return {
                "success": False,
                "error": f"Invalid degree '{degree}'. Must be one of: {[d.value for d in Door.DoorAngleDegree]}",
                "updated_doors": {},
            }

        # 处理位置参数
        positions_to_update = []
        if not position:
            # 使用说话人位置
            speaker_pos = self._settings.speaker
            positions_to_update = self.resolve_positions(speaker_pos)
        else:
            # 处理列表中的每个位置
            for pos in position:
                positions_to_update.extend(self.resolve_positions(pos))

        # 如果没有有效位置，返回错误
        if not positions_to_update:
            return {
                "success": False,
                "error": "Invalid door position specified",
                "updated_doors": {},
            }

        # 更新每个门的角度
        for pos in positions_to_update:
            door = self.get_door_state(pos)
            if door:
                prev_angle = door.angle
                prev_status = door.status

                # 根据参数增加角度
                if value is not None and unit is not None:
                    # 先设置单位
                    door.angle_unit = unit
                    # 增加指定值
                    door.angle += value
                elif degree is not None:
                    # 使用预定义的增量值
                    increment = door._angle_adjustments[degree][door.angle_unit]
                    door.angle += increment
                else:
                    # 如果没有指定参数，使用小增量
                    increment = door._angle_adjustments[
                        Door.DoorAngleDegree.LITTLE.value
                    ][door.angle_unit]
                    door.angle += increment

                # 记录结果
                results[pos] = {
                    "previous_angle": prev_angle,
                    "current_angle": door.angle,
                    "angle_unit": door.angle_unit,
                    "previous_status": prev_status,
                    "current_status": door.status,
                }
            else:
                success = False
                error_message = f"Door at position {pos} not found"
                break

        return {
            "success": success,
            "error": error_message,
            "operation": "increase_angle",
            "parameters": {"value": value, "unit": unit, "degree": degree},
            "updated_doors": results,
        }

    @api("door")
    def carcontrol_carDoor_angle_decrease(
        self,
        position: Optional[List[str]] = None,
        value: Optional[float] = None,
        unit: Optional[str] = None,
        degree: Optional[str] = None,
    ) -> Dict:
        """
        Decrease car door opening angle

        Parameters:
        - position: Optional[List[str]] Door position, must be selected from the following list:
          [driver's seat, passenger seat, second row left, second row right, all]
          If not specified, defaults to user's current position
        - value: Optional[float] Specific numerical value for decreasing angle, unit specified by unit field
        - unit: Optional[str] Unit for adjusting door angle, must be selected from the following list: [gear, percentage, centimeter]
          Not compatible with degree parameter
        - degree: Optional[str] Adjustment level, must be selected from the following list: [large, little, tiny]
          Not compatible with value/unit parameters

        Returns:
        - Dict Contains operation result and related status information
        """
        results = {}
        success = True
        error_message = None

        # 验证参数组合
        if value is not None and unit is None:
            return {
                "success": False,
                "error": "When specifying 'value', 'unit' must also be provided",
                "updated_doors": {},
            }

        if unit is not None and value is None:
            return {
                "success": False,
                "error": "When specifying 'unit', 'value' must also be provided",
                "updated_doors": {},
            }

        if (value is not None or unit is not None) and degree is not None:
            return {
                "success": False,
                "error": "'degree' parameter is incompatible with 'value' and 'unit' parameters",
                "updated_doors": {},
            }

        if unit is not None and unit not in [u.value for u in Door.DoorAngleUnit]:
            return {
                "success": False,
                "error": f"Invalid unit '{unit}'. Must be one of: {[u.value for u in Door.DoorAngleUnit]}",
                "updated_doors": {},
            }

        if degree is not None and degree not in [d.value for d in Door.DoorAngleDegree]:
            return {
                "success": False,
                "error": f"Invalid degree '{degree}'. Must be one of: {[d.value for d in Door.DoorAngleDegree]}",
                "updated_doors": {},
            }

        # 处理位置参数
        positions_to_update = []
        if not position:
            # 使用说话人位置
            speaker_pos = self._settings.speaker
            positions_to_update = self.resolve_positions(speaker_pos)
        else:
            # 处理列表中的每个位置
            for pos in position:
                positions_to_update.extend(self.resolve_positions(pos))

        # 如果没有有效位置，返回错误
        if not positions_to_update:
            return {
                "success": False,
                "error": "Invalid door position specified",
                "updated_doors": {},
            }

        # 更新每个门的角度
        for pos in positions_to_update:
            door = self.get_door_state(pos)
            if door:
                prev_angle = door.angle
                prev_status = door.status

                # 根据参数减小角度
                if value is not None and unit is not None:
                    # 先设置单位
                    door.angle_unit = unit
                    # 减少指定值
                    door.angle -= value
                elif degree is not None:
                    # 使用预定义的减量值
                    decrement = door._angle_adjustments[degree][door.angle_unit]
                    door.angle -= decrement
                else:
                    # 如果没有指定参数，使用小减量
                    decrement = door._angle_adjustments[
                        Door.DoorAngleDegree.LITTLE.value
                    ][door.angle_unit]
                    door.angle -= decrement

                # 记录结果
                results[pos] = {
                    "previous_angle": prev_angle,
                    "current_angle": door.angle,
                    "angle_unit": door.angle_unit,
                    "previous_status": prev_status,
                    "current_status": door.status,
                }
            else:
                success = False
                error_message = f"Door at position {pos} not found"
                break

        return {
            "success": success,
            "error": error_message,
            "operation": "decrease_angle",
            "parameters": {"value": value, "unit": unit, "degree": degree},
            "updated_doors": results,
        }

    @api("door")
    def carcontrol_carDoor_angle_set(self, position: Optional[List[str]] = None, 
                                    value: Optional[float] = None, 
                                    unit: Optional[str] = None,
                                    degree: Optional[str] = None) -> Dict:
        """
        Set car door opening angle to a specific value
        
        Parameters:
        - position: Optional[List[str]] Door position, must be selected from the following list:
          [driver's seat, passenger seat, second row left, second row right, all]
          If not specified, defaults to user's current position
        - value: Optional[float] Specific numerical value for setting angle, unit specified by unit field
        - unit: Optional[str] Door angle unit, must be selected from the following list: [gear, percentage, centimeter]
          Not compatible with degree parameter
        - degree: Optional[str] Preset value for setting door angle, must be selected from the following list: [max, high, medium, low, min]
          Not compatible with value/unit parameters
        
        Returns:
        - Dict Contains operation result and related status information
        """
        results = {}
        success = True
        error_message = None
        
        # 验证参数组合
        if value is not None and unit is None:
            return {
                "success": False,
                "error": "When specifying 'value', 'unit' must also be provided",
                "updated_doors": {}
            }
        
        if unit is not None and value is None:
            return {
                "success": False,
                "error": "When specifying 'unit', 'value' must also be provided",
                "updated_doors": {}
            }
        
        if (value is not None or unit is not None) and degree is not None:
            return {
                "success": False,
                "error": "'degree' parameter is incompatible with 'value' and 'unit' parameters",
                "updated_doors": {}
            }
        
        if unit is not None and unit not in [u.value for u in Door.DoorAngleUnit]:
            return {
                "success": False,
                "error": f"Invalid unit '{unit}'. Must be one of: {[u.value for u in Door.DoorAngleUnit]}",
                "updated_doors": {}
            }
        
        if degree is not None and degree not in [d.value for d in Door.DoorAnglePreset]:
            return {
                "success": False,
                "error": f"Invalid degree '{degree}'. Must be one of: {[d.value for d in Door.DoorAnglePreset]}",
                "updated_doors": {}
            }
        
        # 处理位置参数
        positions_to_update = []
        if not position:
            # 使用说话人位置
            speaker_pos = self._settings.speaker
            positions_to_update = self.resolve_positions(speaker_pos)
        else:
            # 处理列表中的每个位置
            for pos in position:
                positions_to_update.extend(self.resolve_positions(pos))
        
        # 如果没有有效位置，返回错误
        if not positions_to_update:
            return {
                "success": False,
                "error": "Invalid door position specified",
                "updated_doors": {}
            }
        
        # 验证必需参数
        if value is None and unit is None and degree is None:
            return {
                "success": False,
                "error": "Either 'degree' or 'value' and 'unit' must be provided",
                "updated_doors": {}
            }
        
        # 更新每个门的角度
        for pos in positions_to_update:
            door = self.get_door_state(pos)
            if door:
                prev_angle = door.angle
                prev_status = door.status
                
                # 根据参数设置角度
                if value is not None and unit is not None:
                    # 先设置单位
                    door.angle_unit = unit
                    # 设置指定值
                    door.angle = value
                elif degree is not None:
                    # 使用预设值
                    preset_percentage = door._preset_angles[degree]
                    
                    # 根据当前单位，转换百分比为相应的值
                    if door.angle_unit == Door.DoorAngleUnit.PERCENTAGE.value:
                        door.angle = preset_percentage
                    elif door.angle_unit == Door.DoorAngleUnit.GEAR.value:
                        # 从百分比转换为档位
                        max_gear = door._max_angle[Door.DoorAngleUnit.GEAR.value]
                        door.angle = (preset_percentage / 100) * max_gear
                    elif door.angle_unit == Door.DoorAngleUnit.CENTIMETER.value:
                        # 从百分比转换为厘米
                        max_cm = door._max_angle[Door.DoorAngleUnit.CENTIMETER.value]
                        door.angle = (preset_percentage / 100) * max_cm
                
                # 记录结果
                results[pos] = {
                    "previous_angle": prev_angle,
                    "current_angle": door.angle,
                    "angle_unit": door.angle_unit,
                    "previous_status": prev_status,
                    "current_status": door.status
                }
            else:
                success = False
                error_message = f"Door at position {pos} not found"
                break
        
        return {
            "success": success,
            "error": error_message,
            "operation": "set_angle",
            "parameters": {
                "value": value,
                "unit": unit,
                "degree": degree
            },
            "updated_doors": results
        }
