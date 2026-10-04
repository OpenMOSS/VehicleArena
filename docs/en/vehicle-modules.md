# Add a vehicle module

[中文](../extensions/vehicle-modules.md) · [English index](README.md)

Modules represent cabin devices, communication features, sensors, or external-world facts. A module inherits `BaseModule`, registers a unique name, and marks callable methods with `@api` so the tool-discovery system can expose them.

```python
from module.base_module import BaseModule
from registry import register_module
from utils import api

@register_module("energyMeter", description="Battery state and energy mode", category="powertrain")
class EnergyMeter(BaseModule):
    def __init__(self):
        self.mode = "normal"

    @api("energyMeter")
    def set_mode(self, mode: str):
        if mode not in {"normal", "eco"}:
            raise ValueError("mode must be normal or eco")
        self.mode = mode
        return {"success": True, "mode": mode}
```

The registration name is the stable identifier used by scenes, tools, and rules; `description` is shown during progressive tool discovery, and `category` groups capabilities. `is_external=True` places a module in `vw.externalWorld` while preserving unified access. `needs_settings=True` asks the framework to inject `VehicleSettings` through `set_settings`.

Registration only makes a module *available*. An equipment profile determines whether a given vehicle has it:

```python
from capabilities import register_equipment_profile

register_equipment_profile("research_ev", include=["*"], exclude=["sunroof", "video"])
```

Scenes may also use `enable_modules` / `disable_modules` on a vehicle. Unknown module names fail validation. Each vehicle receives independent module instances, event bus, and constraint engine. Put mutable device state on the instance, not in class or global variables.

To constrain a method, use `register_constraint` and a `ConstraintResult` at the module boundary. Keep method arguments and results serializable, validate allowed ranges there, and add equipment-profile and YAML-rule tests. A module must not directly move vehicles or alter SUMO collisions. The [Chinese guide](../extensions/vehicle-modules.md) has a complete constraint-registration example.
