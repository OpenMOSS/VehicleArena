# 场景字段参考

[English version](../en/scenario-schema.md)

场景是一个严格 JSON 对象；未知字段会在 `MultiScenario.from_dict()` 阶段报错。

## 顶层

必需字段：`scenario_id`、`road_network_id`、`vehicles`。

常用可选字段：`name`、`total_time_s`、`tick_interval_s`、`physics_step_s`、`sumo_config`、`weather_keyframes`、`daynight_keyframes`、`pedestrians`、`experiment_world_events` 和评测配置。乘客请求由运行时 Personal Agent 产生，不属于场景字段。

正式物理步长为 0.1 s。场景、SUMO 和 LLM 配置均不接受运行时随机种子。

## `vehicles[]`

必需：`vehicle_id`、`initial_node`。

可选：`destination_node`、`destination_name`、`initial_lane`、`equipment_profile`、`chassis_profile`、模块开关、底盘/感知/传感器覆盖、`initial_physical_state`、`is_evaluated` 和 `agent_config`。

`lidar`、`frontRadar` 和 `rearRadar` 是独立装备。可用 `enable_modules` / `disable_modules` 在单车上增删，但 `sensor_overrides` 只能覆盖该车已安装的传感器。`lidar` 当前支持 `range_m` 和 `horizontal_fov_deg`。

不接受 `driver_plugin` 或 `driver_params`。

## `pedestrians[]`

必需：`ped_id`、`initial_node`。

可选：`destination_node`、`speed`、`start_time`、`collision_radius_m`、感知配置、`initial_physical_state`、`is_evaluated` 和 `agent_config`。

不接受 `pedestrian_plugin` 或 `pedestrian_params`。

## `agent_config`

`type` 只允许：

- `sumo`：背景实体，交通行为和运动由 SUMO 控制；
- `llm`：LLM 决策实体，模型提交控制，SUMO 执行。

LLM 可配置 `model`、`api_base`、`api_key`、`max_turns`、`max_tokens`、`temperature`、`thinking_mode`、`reasoning_effort`、`chat_template_enable_thinking`、`context_window_tokens`、`todo_max_ttl_s` 和 `heartbeat_interval_s`。密钥建议由评测命令或环境变量传入。

## `sumo_config`

可配置缓存、GUI、碰撞处理和固定 0.1 s 步长等执行参数。`seed` 不属于允许字段。

## 最小示例

```json
{
  "scenario_id": "example",
  "road_network_id": "beijing_guomao",
  "physics_step_s": 0.1,
  "vehicles": [
    {
      "vehicle_id": "ego",
      "initial_node": "n33399858",
      "destination_node": "n35722739",
      "agent_config": {"type": "llm"}
    },
    {
      "vehicle_id": "background_1",
      "initial_node": "n35722739",
      "destination_node": "n33399858",
      "agent_config": {"type": "sumo"}
    }
  ]
}
```
