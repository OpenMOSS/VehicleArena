"""
Perception & Action Tools — Tool definitions for LLM agents.

Defines the tool schemas (function calling format) for:
- Vehicle perception tools (observe traffic, signals, weather, etc.)
- Pedestrian perception tools (observe signals, approaching vehicles)
- Pedestrian action tools (walk, wait, cross, run)

These schemas are passed to the LLM as available tools. The actual
execution is backed by WorldState.dispatch_tool() for perception
and entity-specific handlers for actions.
"""

from __future__ import annotations

from typing import Any, Dict, List


# ══════════════════════════════════════════════
# Vehicle Perception Tools
# ══════════════════════════════════════════════

VEHICLE_PERCEPTION_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "look_ahead",
            "description": "查看当前物理车道/连接器前方的车辆、静止事故障碍和行人，返回距离、速度与碰撞相关状态。",
            "parameters": {
                "type": "object",
                "properties": {
                    "distance_m": {
                        "type": "number",
                        "description": "查看前方的距离范围（米），默认100",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_signal",
            "description": "查看当前计划连接器对应的车道级信号，而非整个路口的统一颜色；返回颜色和剩余时间。",
            "parameters": {
                "type": "object",
                "properties": {
                    "node_id": {
                        "type": "string",
                        "description": "路口节点ID，不填则默认查看前方最近路口",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scan_crosswalk",
            "description": "查看指定路口的定向斑马线、正在通过的行人及与车辆连接器的冲突关系。",
            "parameters": {
                "type": "object",
                "properties": {
                    "node_id": {
                        "type": "string",
                        "description": "路口节点ID，不填则默认前方路口",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_weather",
            "description": "查看当前天气状况（晴天/雨天/雾天/雪天等）。",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_road",
            "description": "查看当前/指定路段的路况信息，包括限速、是否封路、交通密度。",
            "parameters": {
                "type": "object",
                "properties": {
                    "segment_id": {
                        "type": "string",
                        "description": "路段ID，不填则默认当前路段",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "look_around",
            "description": "扫描周围指定范围内的所有实体（车辆、行人等），不区分方向。",
            "parameters": {
                "type": "object",
                "properties": {
                    "radius_m": {
                        "type": "number",
                        "description": "扫描半径（米），默认50",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_daynight",
            "description": "查看当前时段（白天/黄昏/夜晚/黎明）和光照水平。",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_location",
            "description": "获取连续物理状态：路段、车道ID、连接器、进度、速度、加速度、变道进度和碰撞状态。",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]

# Vehicle perception follows the same ``module__method`` convention as every
# cabin API.
for _schema in VEHICLE_PERCEPTION_TOOLS:
    _schema["function"]["name"] = (
        "road_perception__" + _schema["function"]["name"])


# ══════════════════════════════════════════════
# Pedestrian Perception Tools
# ══════════════════════════════════════════════

PEDESTRIAN_PERCEPTION_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "check_signal",
            "description": "查看当前路口的行人信号灯状态（行人绿灯/行人红灯及剩余时间）。",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "look_for_vehicles",
            "description": "查看驶向当前斑马线/路口的车辆，返回距离、速度和预计到达风险。",
            "parameters": {
                "type": "object",
                "properties": {
                    "max_distance_m": {
                        "type": "number",
                        "description": "查看范围（米），默认50",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_crosswalk",
            "description": "查看当前定向斑马线几何、信号、冲突来车和自身是否已在斑马线上。",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "look_around",
            "description": "查看周围环境，扫描附近的车辆和其他行人。",
            "parameters": {
                "type": "object",
                "properties": {
                    "radius_m": {
                        "type": "number",
                        "description": "扫描半径（米），默认30",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_location",
            "description": "获取所在等待节点、斑马线ID、过街进度、实际速度和圆形碰撞域半径。",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]


# ══════════════════════════════════════════════
# Pedestrian Action Tools
# ══════════════════════════════════════════════

PEDESTRIAN_ACTION_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "pedestrian_wait",
            "description": "在当前位置等待，不移动。适合红灯或来车时使用。",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "pedestrian_walk",
            "description": "设置步行目标速度并沿路线连续运动；若已在斑马线上则沿其中心线继续。",
            "parameters": {
                "type": "object",
                "properties": {
                    "speed": {
                        "type": "number",
                        "description": "步行速度（m/s），默认1.4，正常范围1.0~1.8",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "pedestrian_cross",
            "description": "选择与路线方向匹配的定向斑马线并开始连续通过；动作不会瞬间完成。",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "pedestrian_run",
            "description": "跑步通过（速度提高到3.0 m/s）。适合需要快速通过的紧急情况。",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "pedestrian_change_route",
            "description": "改变行走目的地。",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_node": {
                        "type": "string",
                        "description": "新的目标节点ID",
                    }
                },
                "required": ["target_node"],
            },
        },
    },
]


# ══════════════════════════════════════════════
# Tool Name Sets (for dispatch logic)
# ══════════════════════════════════════════════

VEHICLE_PERCEPTION_TOOL_NAMES = {
    t["function"]["name"] for t in VEHICLE_PERCEPTION_TOOLS
}

def vehicle_perception_method_name(tool_name: str) -> str:
    prefix = "road_perception__"
    return (
        tool_name[len(prefix):]
        if tool_name.startswith(prefix) else tool_name)

PEDESTRIAN_PERCEPTION_TOOL_NAMES = {
    t["function"]["name"] for t in PEDESTRIAN_PERCEPTION_TOOLS
}

PEDESTRIAN_ACTION_TOOL_NAMES = {
    t["function"]["name"] for t in PEDESTRIAN_ACTION_TOOLS
}

# All perception tool names (union)
ALL_PERCEPTION_TOOL_NAMES = (
    VEHICLE_PERCEPTION_TOOL_NAMES
    | PEDESTRIAN_PERCEPTION_TOOL_NAMES)


def is_perception_tool(tool_name: str) -> bool:
    """Check if a tool name is a perception (read-only) tool."""
    return tool_name in ALL_PERCEPTION_TOOL_NAMES


def is_pedestrian_action_tool(tool_name: str) -> bool:
    """Check if a tool name is a pedestrian action tool."""
    return tool_name in PEDESTRIAN_ACTION_TOOL_NAMES
