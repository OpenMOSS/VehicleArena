# 测试与发布清单

[English version](../en/testing.md)

## 本地检查

```bash
python scripts/validate_extension.py \
  --extension your_package.vehiclearena_extension \
  --scenario path/to/scenario.json \
  --rules path/to/rules

python -m pytest -q
```

扩展项目至少应测试：

1. 重复名称、未知参数和非法字段会失败。
2. 每种装备档案只暴露实际安装模块。
3. 每种底盘的长宽和加减速参数确实写入 SUMO 车辆类型。
4. `llm` 与 `sumo` 的决策权边界不能被模块或实体适配器绕过。
5. 自定义实体展开后能被 `MultiScenario` 解析，并出现在碰撞、感知和渲染快照中。
6. 自定义场景层对相同冻结输入可复现，且生成结果可解析。
7. YAML 规则的主动作、等价动作、缺失装备和负检查都有正反样例。

## 禁止的实现方式

- 按场景 ID、车辆 ID 或测试名称写物理捷径。
- 策略直接改坐标、碰撞状态、红绿灯或其他主体。
- 用全局可变状态保存单个实体的运行状态。
- 静默覆盖注册项，或忽略拼错的参数。
- 让场景 JSON 自动导入 Python 代码。
- 为通过评测而在物理层统一实施“完美避碰”。

## 发布扩展

建议扩展包只暴露一个稳定导入入口，例如 `my_package.vehiclearena_extension`。该入口负责注册模块、能力、实体、场景层和规则目录，不应启动仿真或访问网络。

实验产物应记录：

- VehicleArena commit；
- 扩展包版本或 commit；
- 地图包 manifest 与哈希；
- 场景原文及其哈希；
- 模型和运行参数；
- 原始轨迹、工具调用，以及车内设备、PA/Judge、驾驶、到达、排队/到达时间净增量、NPC 碰撞和输入/输出 Token 评测结果。

正式 manifest 使用 `vehiclearena-experiment-manifest-v0.2`。批量运行在调用
模型或启动 SUMO 前会核对源码指纹、场景目录哈希、每张地图哈希，并重新执行
冻结的 `setup_assertions`。任一项变化都应重新生成 manifest；旧版或不完整
manifest 直接拒绝，不能静默使用当前代码运行旧场景。
