# Generate and freeze scenes

[中文](../extensions/scenario-generation.md) · [English index](README.md)

`ScenarioGenerator` composes ordered scenario layers for weather, day/night, traffic, pedestrians, and external events. It is a developer construction tool: official Basic/MultiLLM scenes are frozen outputs built through `evaluation.experiments.scene_catalog`.

```python
from simulation.scenario_generator import ScenarioContext, ScenarioLayer, register_scenario_layer

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

`order` expresses dependencies between layers. A layer changes only `ScenarioContext`; it does not run an agent, read evaluation answers, or alter vehicles to obtain a desired score. Choose reachable map nodes/routes through map APIs, use `sumo` for background actors and `llm` for evaluated actors, and let the runtime Personal Agent create passenger requests. Do not invent unsupported driver/pedestrian personality fields or runtime random seeds. Parse and SUMO-boot the resulting JSON, then freeze the complete artifact so reproduction does not require sampling again.

Do not guess the task deadline in the generator. Basic and MultiLLM use the separate mixed calibration protocol in the [evaluation guide](evaluation.md), including designated all-SUMO exceptions.
