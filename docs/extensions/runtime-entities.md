# 新增运行实体

[English version](../en/runtime-entities.md)

领域实体适配器把新的业务概念翻译为一个严格的原生车辆或行人配置。翻译完成后，该实体不再走特殊分支，而是自动进入与其他主体相同的路线、动力学、碰撞、感知、渲染、唤醒和评测链路。

## 车辆物理原语示例

```python
from simulation.runtime_entities import (
    RuntimeEntityAdapter, register_runtime_entity,
)


@register_runtime_entity("cargo_bike", physical_kind="vehicle")
class CargoBikeAdapter(RuntimeEntityAdapter):
    @classmethod
    def build_config(cls, spec):
        allowed = {
            "entity_id", "initial_node", "destination_node",
            "is_evaluated", "agent_config",
        }
        unknown = set(spec) - allowed
        if unknown:
            raise ValueError(f"Unknown cargo bike fields: {sorted(unknown)}")
        return {
            "vehicle_id": spec["entity_id"],
            "initial_node": spec["initial_node"],
            "destination_node": spec.get("destination_node", ""),
            "is_evaluated": spec.get("is_evaluated", False),
            "equipment_profile": "research_ev",
            "chassis_profile": "cargo_bike",
            "agent_config": spec.get("agent_config", {"type": "sumo"}),
        }
```

场景使用领域名称而不是复制底层参数：

```json
{
  "entities": [
    {
      "entity_type": "cargo_bike",
      "entity_id": "bike_1",
      "initial_node": "n33399858",
      "destination_node": "n35722739"
    }
  ]
}
```

适配器应严格检查自身字段，并返回合法 `VehicleConfig` 或 `PedestrianConfig` 字典。返回结果会再次经过权威场景解析器，因此未知字段、重复实体 ID、未知策略和非法能力都会失败。

## 怎样选择物理原语

- 巴士、货车、摩托车、自行车：通常使用 `vehicle`，通过底盘长宽和动力参数表达差异。
- 轮椅、推车、配送机器人：若运动可近似为低速圆形主体，可使用 `pedestrian` 并配置 `collision_radius_m`；其背景交通行为由 SUMO 执行。
- 铰接车辆、非凸多边形、三维物体：当前两种原语不足，需要先在 SUMO 中建立可执行的车辆/人员类型与物理表达，再扩展 `TrafficCoordinator` 的状态镜像、感知投影、渲染帧和评测采样接口。

适配器不是绕过物理规则的入口。不要在 `build_config` 中放运行状态，也不要让不同场景 ID触发不同的物理行为。
