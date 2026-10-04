# 动态生成场景

[English version](../en/scenario-generation.md)

`ScenarioGenerator` 用有序场景层组合天气、昼夜、交通、行人和外生事件。它用于开发者构造并冻结场景；正式 Basic/MultiLLM 目录由 `evaluation.experiments.scene_catalog` 统一生成。

## 新增场景层

```python
from simulation.scenario_generator import (
    ScenarioContext, ScenarioLayer, register_scenario_layer,
)


@register_scenario_layer("school_zone", order=45)
class SchoolZoneLayer(ScenarioLayer):
    def apply(self, ctx: ScenarioContext) -> None:
        edges = ctx.all_edges()
        if edges:
            ctx.speed_cameras.append({
                "edge_id": ctx.rng.choice(edges),
                "position_ratio": 0.5,
                "speed_limit": 30,
                "camera_type": "fixed",
            })
```

`order` 表达数据依赖。场景层只能修改 `ScenarioContext`，不能运行 Agent、读取 evaluator 或按预期答案操纵车辆。常用输出包括 `vehicles`、`pedestrians`、天气/昼夜关键帧和道路事件。

## 生成原则

1. 使用地图 API 选择可达路线，不手写不存在的节点、车道或连接器；
2. 背景车辆和行人的 `agent_config.type` 使用 `sumo`；待评角色使用 `llm`；
3. 不生成驾驶/行人性格字段或运行时随机种子；
4. 乘客请求由运行时 Personal Agent 生成，不写入场景；
5. 输出必须通过 `MultiScenario.from_dict()` 和动态 SUMO 启动验证；
6. 生成完成后冻结完整 JSON，复现实验依赖冻结产物而不是再次抽样；
7. 时间窗按[实验管线](../../vehiclearena/evaluation/experiments/README.md)中的混合参考协议校准，不由生成器猜测；Basic 是全 SUMO，MultiLLM 按冻结注册表保留固定 peer 或使用指定的全 SUMO 例外。

通用生成器和正式目录生成器服务不同用途：前者方便开发新场景层，后者固定论文任务数量、focal/peer 角色和跨地图覆盖。
