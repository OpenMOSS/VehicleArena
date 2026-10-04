# 新增规则操作手册

所有规则共用一套当前 schema，写在本目录下的 yaml 里。文件名按 `domain` 分桶，添加规则就是往对应文件的 `rules:` 列表里追加一个条目。

## 文件分桶

| 文件 | domain | 触发场景 |
|---|---|---|
| `weather_rules.yaml` | `weather` | 天气切换 |
| `daynight_rules.yaml` | `daynight` | 时段切换 |
| `map_event_rules.yaml` | `map_event` | 地图事件出现 |
| `user_intent_rules.yaml` | `user_intent` | 乘客指令 |
| `user_custom.yaml` | 上述 domain | 自定义规则 |

不同桶之间的触发条件不能混用；同一时刻命中多条规则时按 `priority` 从小到大执行。

## 通用字段

```yaml
- id: <唯一 id>
  description: <一句话说明>
  domain: weather|daynight|map_event|user_intent
  requires_modules: [wiper]  # 可选；缺少装备时整条规则不适用
  trigger: { ... }            # 见下，按 domain 选一种
  guard: { field: value }     # 可选；额外快照前置条件
  expect:
    actions: [...]            # vw.module.method(...) 调用
    broadcast: { key: warning }
  tolerance:
    skip: [field_pattern, ...]    # 车内状态差异评测忽略字段
    any_of: [...]                 # 可接受的等价 action
    trend: { field_pattern, direction, baseline_field }
    ceiling: { field_pattern, context_var }
  priority: 10                # 越小越先匹配
```

未声明 `requires_modules` 时，加载器会从主 `actions` 的
`vw.<module>.<method>` 目标自动推导。替代动作中的模块不是强制装备。

## 各 domain 的 trigger 写法

**state_change**（weather / daynight）：

```yaml
trigger:
  state_change:
    field: condition          # 可选状态字段
    from: [false]
    to:   [true]
```

**map_event**：

```yaml
trigger:
  map_event:
    event_types: [speed_camera, construction]
```

**user_intent**（passenger）：

```yaml
trigger:
  user_intent:
    messages: ["打开空调", "我有点热"]
    vague:    ["不太舒服"]      # 可选，模糊语义
```

## 例子

天气规则——下雨开启前雨刷：

```yaml
- id: wet_weather_enter
  description: 进入雨天时开启前雨刷
  domain: weather
  trigger:
    state_change:
      from: [sunny, cloudy]
      to: [rainy, heavy_rain]
  expect:
    actions:
      - vw.wiper.carcontrol_wiperBlade_switch(True, 'front')
```

用户意图规则——开窗户：

```yaml
- id: window_open
  domain: user_intent
  trigger:
    user_intent:
      messages: ["开窗", "把窗户打开"]
  expect:
    actions:
      - vw.window.carcontrol_window_switch(["driver's seat"], True)
  tolerance:
    any_of:
      - vw.sunroof.carcontrol_sunroof_switch('open')
```

## 负检查（非常驻安全约束）

写在任意 yaml 的 `negative_checks:` 段（建议放对应的桶）：

```yaml
negative_checks:
  - id: no_fullscreen_video_while_moving
    field: video.is_fullscreen
    forbidden: true
    condition:
      vehicle_speed_kmh: { gt: 0.5 }
    severity: violation
    reason: 车辆移动时不得播放遮挡驾驶视野的全屏视频
```

路口内停车、红灯、TTC、碰撞和路线效率属于车外物理轨迹评测，不应写成
车内精确 API 标准答案。路口连接器允许因冲突车辆、行人或回溢而走走停停。

## 全局 skip

跨规则共用的可忽略字段写在任意 yaml 的 `global_skip_fields:` 段，loader 会去重合并。

## 验证

加完规则跑一遍：

```bash
PYTHONPATH=vehiclearena .venv/bin/python -c \
"from rules.rule_loader import RuleLoader; rl=RuleLoader(); \
print('env:', len(rl.env_rules), 'user intent:', len(rl.user_intent_rules))"
```

完整契约和回归验证：

```bash
PYTHONPATH=vehiclearena .venv/bin/python -m unittest \
  evaluation.test_split_evaluation -v
```

加载器只接受本文定义的当前 schema；未知 domain、字段或顶层结构都会直接报错。
