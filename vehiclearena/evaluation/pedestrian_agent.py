"""LLM pedestrian callback using the same audited harness contract as vehicles."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
from typing import Any, Callable

from simulation.perception_tools import (
    PEDESTRIAN_ACTION_TOOLS,
    PEDESTRIAN_ACTION_TOOL_NAMES,
    PEDESTRIAN_PERCEPTION_TOOLS,
    PEDESTRIAN_PERCEPTION_TOOL_NAMES,
)
from tool_utils import generate_lazy_discovery_tools
from context_runtime import (
    ContextWindowExceeded,
    DATA_DELIVERY_REGISTRY,
    RollingWakeContext,
    TodoStore,
    build_current_wake_message,
    build_pedestrian_self_now,
    normalize_new_events,
)

logger = logging.getLogger(__name__)
_HARD_MAX_TURNS_PER_WAKE = 16
_DEFAULT_CONTEXT_WINDOW_TOKENS = 32768


def _make_finish_tool() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "finish",
            "description": (
                "结束本次唤醒；不停止行人、不取消已提交命令，也不结束仿真。"),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string"},
                },
                "required": [],
            },
        },
    }


def _pedestrian_module_catalog(module: str) -> str:
    if module == "road_perception":
        description = "行人个体化的视觉/声学道路感知"
        schemas = PEDESTRIAN_PERCEPTION_TOOLS
    elif module == "pedestrian_control":
        description = "等待、步行、跑步、过街和改道控制"
        schemas = PEDESTRIAN_ACTION_TOOLS
    else:
        return json.dumps({
            "success": False,
            "error": "module_not_available",
            "module": module,
            "available_modules": [
                "road_perception", "pedestrian_control"],
        }, ensure_ascii=False)
    lines = [
        f"Module: {module} — {description}。",
        f"{len(schemas)} APIs available:",
    ]
    for schema in schemas:
        function = schema["function"]
        lines.append(f"  - {function['name']}: {function['description']}")
    return "\n".join(lines)

PEDESTRIAN_SYSTEM_PROMPT = """\
你直接控制一个行人，在城市道路网络中前往目的地。你的连续选择体现你自己的步行
偏好；系统不会预设谨慎或冒险人格。

## 物理世界契约

- 世界以固定 0.1 秒时间步推进；车辆有有向矩形碰撞域，行人有圆形碰撞域，整个
  时间步内的扫掠重叠会产生真实碰撞。
- 等待、步行、跑步、进入斑马线和改道由你决定。动作持续执行直到完成、碰撞或被
  后续命令替换；工具成功不代表瞬间到达。
- 红绿灯、斑马线和通行权是可感知规则。物理世界记录违规与后果，但不会暗中替你
  等待或穿越；即使有通行权也不能假定车辆必然让行。
- 同一批智能体先从同一世界快照决策，再统一提交动作。动作工具返回 command_id
  表示已排队；后续 command_result 唤醒事件才是物理世界的提交回执。

## 决策流程

每次唤醒先比较上一轮原始消息、本轮 self_now、Todo 和新事件；必要时用感知工具查看
信号、来车、斑马线、周边和位置，再决定保持或替换当前动作。如果当前持续动作仍
符合决定，不必重复。目的地等长期任务应使用 todo_manage 设置有限期限。
收到到达、碰撞或 simulation_ended 后不要再提交移动动作。

感知工具位于虚拟 `road_perception` 模块，移动工具位于 `pedestrian_control` 模块；
需要恢复被卸载的能力时先用 `get_module_api` 查看，再用 `load_tools` 加载。已加载工具
跨唤醒保留，只有显式配置的上下文窗口即将超限时才统一卸载。处理完本次唤醒后调用
`finish`；它不改变物理动作。
"""


def make_llm_pedestrian_callback(
    ped_id: str, agent_client: Any, max_turns: int = 5,
    context_window_tokens: int = _DEFAULT_CONTEXT_WINDOW_TOKENS,
    todo_max_ttl_s: float = 3600.0,
) -> Callable:
    """Create one bounded, tool-using pedestrian agent callback."""
    effective_max_turns = min(
        _HARD_MAX_TURNS_PER_WAKE, max(1, int(max_turns)))
    todo_store = TodoStore(max_ttl_s=float(todo_max_ttl_s))
    all_tools = (
        copy.deepcopy(PEDESTRIAN_ACTION_TOOLS)
        + list(generate_lazy_discovery_tools())
        + [todo_store.tool_schema(), _make_finish_tool()])
    core_tool_names = {
        "get_module_api", "load_tools", "todo_manage", "finish"}
    rolling_context = RollingWakeContext(
        PEDESTRIAN_SYSTEM_PROMPT,
        context_window_tokens=int(context_window_tokens),
        max_output_tokens=int(getattr(agent_client, "max_tokens", 4096)),
    )
    state = {
        "messages": [],
        "previous_wake_messages": [],
        "all_messages": [],
        "token_stats": {"input": [], "output": []},
        "total_turns": 0,
        "total_tool_calls": 0,
        "perception_calls": 0,
        "action_calls": 0,
        "tool_calls_discovery": 0,
        "tool_calls_query": 0,
        "tool_calls_action": 0,
        "tool_calls_memory": 0,
        "tool_calls_harness": 0,
        "instruction": PEDESTRIAN_SYSTEM_PROMPT,
        "tools": all_tools,
        "initial_tools": copy.deepcopy(all_tools),
        "core_tool_names": sorted(core_tool_names),
        "loaded_skills": [],
        "loaded_skill_contents": {},
        "capability_epoch": 0,
        "context_budget_log": rolling_context.budget_log,
        "context_data_registry": copy.deepcopy(DATA_DELIVERY_REGISTRY),
        "todo_audit_log": [],
        "todo_state": todo_store.audit_dict(),
        "next_todo_deadline_s": None,
        "context_snapshot_log": [],
        "tool_call_log": [],
        "model_call_log": [],
        "infrastructure_errors": [],
        "protocol_events": [],
        "model": getattr(agent_client, "model", ""),
        "prompt_profile_id": "pedestrian-direct-control-v2",
        "effective_config": {
            "entity_type": "pedestrian",
            "configured_max_turns": int(max_turns),
            "max_turns": effective_max_turns,
            "hard_max_turns": _HARD_MAX_TURNS_PER_WAKE,
            "max_tokens": int(getattr(agent_client, "max_tokens", 4096)),
            "context_window_tokens": int(context_window_tokens),
            "todo_max_ttl_s": float(todo_max_ttl_s),
            "temperature": float(getattr(agent_client, "temperature", 0.7)),
            "thinking_mode": str(getattr(
                agent_client, "thinking_mode", "default")),
            "reasoning_effort": getattr(
                agent_client, "reasoning_effort", None),
            "chat_template_enable_thinking": getattr(
                agent_client, "chat_template_enable_thinking", None),
            "api_base": getattr(agent_client, "api_base", ""),
        },
    }

    def callback(ps, t, wake_events, memory, tick_index, world_state):
        model_calls_before = len(state["model_call_log"])
        actions_taken = []
        wake_event_ids = [
            event.get("event_id") for event in wake_events
            if event.get("event_id")]
        new_events = normalize_new_events([], wake_events)
        new_events.extend(todo_store.expiration_events(t))
        if getattr(ps, "has_arrived", False):
            new_events.append({
                "event_type": "arrival_state",
                "message": "已到达目的地，不再提交移动动作。",
            })
        wake_messages = [build_current_wake_message(
            sim_time_s=t,
            tick_index=tick_index,
            agent_id=ped_id,
            entity_type="pedestrian",
            self_now=build_pedestrian_self_now(ps),
            todo=todo_store.to_context(t),
            new_events=new_events,
        )]

        wake_finished = False
        completed_naturally = False
        for turn in range(1, effective_max_turns + 1):
            budget_record = None
            try:
                messages, budget_record = rolling_context.prepare_request(
                    wake_messages=wake_messages,
                    tools=all_tools,
                    core_tool_names=core_tool_names,
                    loaded_skills=state["loaded_skill_contents"],
                    entity_id=ped_id,
                    sim_time_s=t,
                    tick_index=tick_index,
                    turn_index=turn,
                )
                if budget_record["unloaded_tools"]:
                    state["protocol_events"].append({
                        **copy.deepcopy(budget_record),
                        "type": "capabilities_unloaded",
                    })
                state["capability_epoch"] = rolling_context.capability_epoch
                state["messages"] = copy.deepcopy(messages)
                state["context_snapshot_log"].append({
                    "entity_id": ped_id,
                    "time_s": round(float(t), 6),
                    "tick_index": int(tick_index),
                    "turn_index": int(turn),
                    "messages": copy.deepcopy(messages),
                    "tools": copy.deepcopy(all_tools),
                    "capability_epoch": rolling_context.capability_epoch,
                })
                response_msg, _, pt, ct = agent_client.chat_with_tools(
                    messages, tools=all_tools)
            except ContextWindowExceeded as exc:
                if len(state["model_call_log"]) > model_calls_before:
                    last_budget = (
                        copy.deepcopy(rolling_context.budget_log[-1])
                        if rolling_context.budget_log else None)
                    state["protocol_events"].append({
                        "entity_id": ped_id,
                        "time_s": round(float(t), 6),
                        "tick_index": int(tick_index),
                        "turn_index": int(turn),
                        "type": "wake_context_limit_reached",
                        "error": str(exc),
                        "context_budget": last_budget,
                        "commands_already_submitted_remain_queued": True,
                        "unfinished_work_remains_in_todo": True,
                    })
                    logger.info(
                        "[%s] wake context limit reached; continuing on the "
                        "next wake", ped_id)
                    break
                error = {
                    "entity_id": ped_id,
                    "time_s": round(float(t), 6),
                    "tick_index": int(tick_index),
                    "turn_index": int(turn),
                    "error": f"{type(exc).__name__}: {exc}",
                }
                state["infrastructure_errors"].append(error)
                raise
            except Exception as exc:
                error = {
                    "entity_id": ped_id,
                    "time_s": round(float(t), 6),
                    "tick_index": int(tick_index),
                    "turn_index": int(turn),
                    "error": f"{type(exc).__name__}: {exc}",
                }
                state["infrastructure_errors"].append(error)
                if getattr(agent_client, "last_call_metadata", None):
                    state["model_call_log"].append({
                        **copy.deepcopy(agent_client.last_call_metadata),
                        **{key: error[key] for key in (
                            "entity_id", "time_s", "tick_index",
                            "turn_index")},
                        "wake_event_ids": wake_event_ids,
                        "context_budget": copy.deepcopy(budget_record),
                    })
                logger.warning("[%s] model call failed: %s", ped_id, exc)
                raise

            state["model_call_log"].append({
                **copy.deepcopy(getattr(
                    agent_client, "last_call_metadata", {}) or {}),
                "entity_id": ped_id,
                "time_s": round(float(t), 6),
                "tick_index": int(tick_index),
                "turn_index": int(turn),
                "wake_event_ids": wake_event_ids,
                "context_budget": copy.deepcopy(budget_record),
            })
            state["token_stats"]["input"].append(pt)
            state["token_stats"]["output"].append(ct)
            state["total_turns"] += 1
            if response_msg is None:
                error = {
                    "entity_id": ped_id,
                    "time_s": round(float(t), 6),
                    "tick_index": int(tick_index),
                    "turn_index": int(turn),
                    "error": "AgentClient returned no response message",
                }
                state["infrastructure_errors"].append(error)
                raise RuntimeError(error["error"])
            wake_messages.append(_msg_to_dict(response_msg))
            if not response_msg.tool_calls:
                completed_naturally = True
                state["protocol_events"].append({
                    "entity_id": ped_id, "time_s": round(float(t), 6),
                    "tick_index": int(tick_index),
                    "turn_index": int(turn), "type": "implicit_finish",
                })
                break

            has_action = False
            for tc in response_msg.tool_calls:
                func_name = tc.function.name
                try:
                    func_args = json.loads(
                        tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    func_args = {}
                state["total_tool_calls"] += 1
                if wake_finished:
                    kind = "harness"
                    state["tool_calls_harness"] += 1
                    result = {
                        "success": False,
                        "error": "wake_already_finished",
                    }
                elif func_name == "finish":
                    kind = "harness"
                    state["tool_calls_harness"] += 1
                    wake_finished = True
                    result = {
                        "success": True,
                        "wake_finished": True,
                        "commands_already_submitted_remain_queued": True,
                    }
                    state["protocol_events"].append({
                        "entity_id": ped_id,
                        "time_s": round(float(t), 6),
                        "tick_index": int(tick_index),
                        "turn_index": int(turn),
                        "type": "explicit_finish",
                        "reason": str(func_args.get("reason", "")),
                    })
                elif func_name == "todo_manage":
                    kind = "harness"
                    state["tool_calls_harness"] += 1
                    before = todo_store.audit_dict()
                    result = todo_store.apply(
                        func_args.get("operations", []), now_s=t,
                        expected_revision=func_args.get("expected_revision"))
                    state["todo_audit_log"].append({
                        "entity_id": ped_id,
                        "time_s": round(float(t), 6),
                        "tick_index": int(tick_index),
                        "before": before,
                        "result": copy.deepcopy(result),
                        "after": todo_store.audit_dict(),
                    })
                elif func_name == "get_module_api":
                    kind = "discovery"
                    state["tool_calls_discovery"] += 1
                    module = str(func_args.get("module", ""))
                    result = _pedestrian_module_catalog(module)
                elif func_name == "load_tools":
                    kind = "discovery"
                    state["tool_calls_discovery"] += 1
                    requested = list(func_args.get("tools", []))
                    schemas = {
                        item["function"]["name"]: item
                        for item in (
                            list(PEDESTRIAN_PERCEPTION_TOOLS)
                            + list(PEDESTRIAN_ACTION_TOOLS))}
                    existing = {
                        item["function"]["name"] for item in all_tools}
                    invalid = [
                        name for name in requested if name not in schemas]
                    added = []
                    if not invalid:
                        for name in requested:
                            if name not in existing:
                                all_tools.append(copy.deepcopy(schemas[name]))
                                existing.add(name)
                                added.append(name)
                    if invalid:
                        result = {
                            "success": False,
                            "error": f"Unknown: {invalid}",
                        }
                    else:
                        loaded_schemas = [
                            item for item in all_tools
                            if item["function"]["name"] in set(added)]
                        result = {
                            "success": True,
                            "loaded": added,
                            "count": len(added),
                            "schema_sha256": hashlib.sha256(json.dumps(
                                loaded_schemas, ensure_ascii=False,
                                sort_keys=True,
                                separators=(",", ":")).encode(
                                    "utf-8")).hexdigest(),
                        }
                elif func_name in PEDESTRIAN_PERCEPTION_TOOL_NAMES:
                    state["perception_calls"] += 1
                    state["tool_calls_query"] += 1
                    kind = "perception"
                    result = world_state.dispatch_tool(
                        ped_id, func_name, func_args)
                elif func_name in PEDESTRIAN_ACTION_TOOL_NAMES:
                    state["action_calls"] += 1
                    state["tool_calls_action"] += 1
                    kind = "action"
                    result = _execute_pedestrian_action(
                        world_state, ped_id, func_name, func_args)
                    actions_taken.append(func_name)
                    has_action = True
                else:
                    kind = "unknown"
                    result = {"error": f"Unknown tool: {func_name}"}
                wake_messages.append({
                    "role": "tool", "tool_call_id": tc.id,
                    "content": json.dumps(
                        result, ensure_ascii=False, default=str),
                })
                state["tool_call_log"].append({
                    "entity_id": ped_id,
                    "entity_type": "pedestrian",
                    "time_s": round(float(t), 6),
                    "tick_index": int(tick_index),
                    "turn_index": int(turn),
                    "tool_call_index": len(state["tool_call_log"]),
                    "tool_call_id": tc.id,
                    "function": func_name,
                    "arguments": func_args,
                    "result": result,
                    "kind": kind,
                    "wake_event_ids": wake_event_ids,
                })
            if wake_finished:
                break

        if (not wake_finished and not completed_naturally
                and turn >= effective_max_turns):
            state["protocol_events"].append({
                "entity_id": ped_id,
                "time_s": round(float(t), 6),
                "tick_index": int(tick_index),
                "turn_index": int(turn),
                "type": "turn_limit_reached",
                "max_turns": effective_max_turns,
                "commands_already_submitted_remain_queued": True,
            })

        state["all_messages"].append({
            "tick_index": tick_index, "time_s": t,
            "messages": copy.deepcopy(wake_messages),
        })
        rolling_context.finish_wake(wake_messages)
        state["previous_wake_messages"] = copy.deepcopy(
            rolling_context.previous_wake_messages)
        state["messages"] = rolling_context.compose(
            [], state["loaded_skill_contents"])
        state["capability_epoch"] = rolling_context.capability_epoch
        state["todo_state"] = todo_store.audit_dict()
        state["next_todo_deadline_s"] = todo_store.next_deadline_s(t)
        return actions_taken

    callback._state = state
    callback._ped_id = ped_id
    return callback


def _execute_pedestrian_action(
    world_state, ped_id: str, action: str, params: dict,
) -> dict:
    """Submit a pedestrian action through the engine-owned boundary."""
    try:
        result = world_state.execute_action(ped_id, action, params)
        return result if isinstance(result, dict) else {"success": True}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


def _msg_to_dict(msg) -> dict:
    d = {"role": "assistant", "content": msg.content or ""}
    if msg.tool_calls:
        d["tool_calls"] = [{
            "id": tc.id,
            "type": "function",
            "function": {
                "name": tc.function.name,
                "arguments": tc.function.arguments or "",
            },
        } for tc in msg.tool_calls]
    return d
