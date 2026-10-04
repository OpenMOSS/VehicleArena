# 基于规则定制车内功能

[English version](../en/cabin-rules.md)

车内任务由 YAML 声明触发条件和可接受动作。规则是评测规范，不是驾驶控制器；红灯、碰撞、TTC 和路口通行应由车外轨迹评测处理。

## 扩展规则目录

扩展包初始化时注册自己的目录：

```python
from pathlib import Path
from rules.rule_loader import register_rule_directory

register_rule_directory(Path(__file__).parent / "rules")
```

框架会把内置目录和扩展目录装入同一个索引，并跨目录检查 `(domain, id)` 重复。注册目录后全局索引会重建；每次仿真读取同一份已验证规则。

## 最小规则

```yaml
rules:
  - id: eco_mode
    description: Passenger requests energy-saving mode
    domain: user_intent
    requires_modules: [energyMeter]
    trigger:
      user_intent:
        messages: ["打开节能模式"]
        vague: ["省一点电"]
    expect:
      actions:
        - vw.energyMeter.set_mode('eco')
    priority: 20
```

支持的 `domain` 是 `weather`、`daynight`、`map_event` 和 `user_intent`。每个 domain 只能使用对应触发结构；未知字段会被拒绝。

`requires_modules` 决定规则是否适用于某辆车。省略时会从主动作的 `vw.<module>.<method>` 自动推导。Personal Agent 的自然语言请求不会在输入层按装备过滤：Driving Agent 应自行查询能力、执行或明确拒绝，再由 Passenger Judge 评价。

## 容差与负检查

- `tolerance.any_of`：声明等价动作，而不是放宽所有状态。
- `tolerance.skip`：忽略非任务字段。
- `trend` / `ceiling`：评价方向或上限，不要求精确值。
- `negative_checks`：声明移动中全屏视频等持续安全禁区。

规则动作只允许一个以 `vw` 开头、参数可由 `ast.literal_eval` 解析的方法调用，不能执行导入、赋值、私有属性或任意 Python。

验证规则：

```bash
.venv/bin/python scripts/validate_extension.py \
  --extension your_package.vehiclearena_extension \
  --rules your_package/rules
```

更完整的字段示例可参考 `vehiclearena/rules/HOWTO.md` 和本目录示例。
