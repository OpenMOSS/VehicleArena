# VehicleArena 开发者文档

[English documentation](en/README.md) · [项目主页](../README.md)

## 阅读顺序

1. [快速开始](01-quickstart.md)
2. [框架边界与数据流](02-architecture.md)
3. [场景字段](reference/scenario-schema.md)
4. [SUMO 执行引擎](reference/sumo-engine.md)
5. [运行实体扩展](extensions/runtime-entities.md)
6. [场景生成](extensions/scenario-generation.md)
7. [测试与提交](reference/testing.md)

安装地图包后运行 `python scripts/check_environment.py`。实验命令统一通过
`python scripts/run_experiments.py ...` 启动，因此开发者不需要修改全局
`PYTHONPATH`。

## 正式扩展面

| 扩展目标 | 入口 |
|---|---|
| 车辆/座舱模块 | `register_vehicle_module` |
| 装备组合 | `equipment_profiles.py` |
| 底盘参数 | `chassis_profiles.py` |
| 感知能力 | `perception_model.py`、雷达模块 |
| 新运行实体 | `register_runtime_entity` |
| 场景层 | `register_scenario_layer` |
| 车内规则 | YAML 规则目录 |

背景车辆和行人的交通行为不再提供 Python 性格插件扩展面，而由 SUMO 原生模型统一负责。LLM 实体通过与物理世界绑定的工具提交显式控制。

## 当前任务划分

- 训练/开发集：100 个 Basic 任务；
- 固定测试集：80 个 Basic、32 个 MultiLLM，共 112 个任务；
- MultiLLM 对比只更换被测车的模型，其他 LLM 车辆保持固定配置；

## 调度、工具与运行维护

执行命令和冻结场景状态以[快速开始](01-quickstart.md)、[批量测试](llm-suite.md)及[实验管线](../vehiclearena/evaluation/experiments/README.md)为准。

- [PA 事件/随机唤醒与 Judge 验收](pa-judge-scheduling.md)
- [工具契约与同场景多车并发](tool-contracts-and-parallel-wakes.md)
- [动作确认与换道统计](action-acknowledgements.md)
- [显式导航、小地图与进度评分](navigation-progress.md)
- [路线末端失败分类](route-boundary-failure.md)
- [同帧 Web3D 局部观察图](web3d-lidar-bev.md)
