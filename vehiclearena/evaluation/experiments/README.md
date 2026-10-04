# VehicleArena 实验管线

[English version](../../../docs/en/evaluation.md)

正式划分是 100 个训练/开发任务与 112 个测试任务（80 Basic、32 MultiLLM）。固定测试清单由 `scripts/run_test_suite.py` 校验。背景车辆和行人由 SUMO 原生控制；场景中显式安排的事件属于任务条件，不是后台避险策略。

每个场景的时限保存在 [`time_window_calibration.json`](time_window_calibration.json)。发布任务时用 SUMO 接管待评车进行校准；MultiLLM 的其他 LLM 车辆仍使用与评测相同的模型、性格模板和 PA/Judge 配置。正式评测直接读取冻结的 `total_time_s`，更换被测模型不重新计算时限。校准运行用于设定时限，下文的配对参考运行则用于衡量被测车对其他交通参与者的影响，两者用途不同。

## 普通使用者准备

```bash
python scripts/check_environment.py
python scripts/run_experiments.py prepare-scene-manifests \
  --catalog vehiclearena/evaluation/experiments/scenarios \
  --output vehiclearena/evaluation/experiment_manifests/catalog
```

正式场景已冻结在 Git 中。普通使用者生成 manifest 后即可运行，不需要重建场景或重校准。维护者只有在地图道路数据、场景文件、SUMO 版本、仿真/校准代码或固定 MultiLLM peer 配置改变时，才重新验证并校准受影响的场景。先查看命令选项，避免覆盖已发布任务：

```bash
python scripts/run_experiments.py validate-scenes --help
python scripts/run_experiments.py calibrate-time-windows --help
```

当前校准代码会把最后一个计入校准的预设事件时间纳入时限计算；重新校准可能得到与已发布注册表不同的结果。复现论文时应使用冻结值，不把重新校准的任务与原结果混合比较。

## 配对参考运行

```bash
python scripts/run_experiments.py run \
  --manifest vehiclearena/evaluation/experiment_manifests/catalog/Basic.manifest.json \
  --output evaluation/runs/sumo/Basic --sumo-reference
```

上述 Basic 命令把全部车辆交给 SUMO，不调用模型或 PA。MultiLLM 的参考运行只把被测车交给 SUMO，固定 peer 仍由 LLM 驾驶，且保持测试组的模型、性格模板和 PA/Judge 配置。已有 LLM suite 应优先使用 `scripts/run_reference_suite.py --source-output <treatment-suite>`，以复用冻结场景与固定角色配置。两次运行的物理输入相同，但交互后的轨迹可以不同。配对结果用于计算 NPC 碰撞、未到达和延迟差，不产生整图综合分。

## LLM 与乘客请求

单场景调用见[快速开始](../../../docs/01-quickstart.md)，批量角色配置见[批量 LLM 测试](../../../docs/llm-suite.md)。当前批量 suite 默认启用事件/随机触发的 PA 和独立 Judge 检查；MultiLLM 还必须配置固定 peer。跨模型比较时只替换被测车，保持 peer、PA/Judge 和参考协议一致。配对兼容性由相同的 `variant_id` 和有效 `protocol_hash` 判断；源码与 manifest 哈希仅记录来源。

## 结果汇总

```bash
python scripts/run_experiments.py aggregate \
  --output evaluation/runs/model-a/Basic

```

批量模型结果使用 [`summarize_llm_suite.py`](../../../scripts/summarize_llm_suite.py) 汇总，具体命令见[批量 LLM 测试](../../../docs/llm-suite.md)。原始 `.json.gz` 保留轨迹、上下文、工具调用、视觉引用、模型配置和各评测维度证据。基础设施无效运行不进入模型均分。
