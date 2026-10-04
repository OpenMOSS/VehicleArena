# 新增车辆模块

[English version](../en/vehicle-modules.md)

车辆模块适合表达车内设备、通信装置、传感器接口或外部环境事实。模块类必须继承 `BaseModule`，使用唯一名称注册，并用 `@api("模块名")` 声明可被工具系统发现的方法。

```python
from module.base_module import BaseModule
from registry import register_module
from utils import api


@register_module(
    "energyMeter",
    description="Battery state and energy mode",
    category="powertrain",
)
class EnergyMeter(BaseModule):
    def __init__(self):
        self.mode = "normal"

    @api("energyMeter")
    def set_mode(self, mode: str):
        """Set energy mode to normal or eco."""
        if mode not in {"normal", "eco"}:
            raise ValueError("mode must be normal or eco")
        self.mode = mode
        return {"success": True, "mode": mode}
```

## 注册参数

- `name`：场景、工具和规则使用的稳定标识，格式为字母开头的字母数字下划线。
- `description`：渐进式工具发现中展示的一句话说明。
- `category`：用于能力目录分组。
- `is_external=True`：模块表示天气、道路等外部事实，实例放在 `vw.externalWorld`，同时可通过统一的 `vw.<name>` 读取；注册后仍走能力选择和事件/约束接线。
- `needs_settings=True`：框架会调用 `set_settings(VehicleSettings)` 注入车内共享设置。

## 事件和约束

模块方法可以通过 `self._event_bus` 发布事件。扩展安装函数可先于或晚于模块导入：

```python
from constraints import Constraint, ConstraintLevel, ConstraintResult
from registry import register_constraint


@register_constraint("energyMeter")
def install_energy_constraint(engine):
    def check(module, method, args, kwargs, vehicle_world):
        allowed = method != "set_mode" or not vehicle_world.is_simulating
        return ConstraintResult(
            allowed,
            ConstraintLevel.HARD,
            "mode cannot change during this run" if not allowed else "",
            "energy_mode_lock",
        )

    engine.register(Constraint(
        "energy_mode_lock",
        ConstraintLevel.HARD,
        check,
        target_modules=["energyMeter"],
        target_methods=["set_mode"],
    ))
```

每辆车创建自己的模块、事件总线和约束引擎；不要把可变状态放在类属性或全局变量中。

## 让车辆安装模块

模块注册只是声明“系统知道它”，车辆是否安装由装备档案决定：

```python
from capabilities import register_equipment_profile

register_equipment_profile(
    "research_ev",
    include=["*"],
    exclude=["sunroof", "video"],
)
```

也可在单车场景中使用 `enable_modules` / `disable_modules`。必需世界接口不能禁用，未知模块会在场景解析时失败。

## 检查清单

- API 参数和返回值可序列化，范围校验在模块边界完成。
- 方法只改变本车模块状态；真实交通坐标由物理世界管理。
- 新模块补充能力档案和 YAML 规则测试。
- 不使用已有名称覆盖内置模块；框架会拒绝重复注册。
