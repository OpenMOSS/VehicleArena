# Add a runtime entity

[中文](../extensions/runtime-entities.md) · [English index](README.md)

A runtime entity adapter translates a domain object into a strict native SUMO vehicle or pedestrian configuration. After translation, the entity follows the ordinary route, physics, collision, perception, rendering, wake, and evaluation paths; the adapter is not a second control layer.

```python
from simulation.runtime_entities import RuntimeEntityAdapter, register_runtime_entity

@register_runtime_entity("cargo_bike", physical_kind="vehicle")
class CargoBikeAdapter(RuntimeEntityAdapter):
    @classmethod
    def build_config(cls, spec):
        allowed = {"entity_id", "initial_node", "destination_node", "is_evaluated", "agent_config"}
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

The scenario may then contain `{"entities": [{"entity_type": "cargo_bike", "entity_id": "bike_1", "initial_node": "n33399858", "destination_node": "n35722739"}]}`. The returned dictionary is parsed again as a native `VehicleConfig` or `PedestrianConfig`; unknown fields, duplicate IDs, unsupported agent types, and invalid equipment still fail.

Use the vehicle primitive for trucks, buses, motorcycles, or bicycles when chassis size/dynamics can express the difference. A pedestrian primitive may represent a slow small actor with a collision radius. Articulated or non-convex actors need a corresponding executable SUMO representation and updates throughout state mirroring, perception, rendering, and evaluation; simply adding an adapter is not enough. The adapter should not hold live state or branch on particular scene IDs.
