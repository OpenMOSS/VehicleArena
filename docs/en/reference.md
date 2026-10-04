# Scenario and extension reference

[English index](README.md) · [Chinese detailed references](../README.md)

## Scenario JSON

`MultiScenario.from_dict()` parses strict JSON and rejects unknown fields. Required top-level fields are `scenario_id`, `road_network_id`, and `vehicles`. Common optional fields include duration, weather/day-night keyframes, pedestrians, scheduled world events, and evaluation settings. The official physics step is 0.1 seconds. Passenger requests are produced by the runtime Personal Agent, not embedded in the scene.

Each vehicle requires `vehicle_id` and `initial_node`; it may define a destination, lane, equipment/chassis profiles, initial physical state, and `agent_config`. Each pedestrian requires `ped_id` and `initial_node`. `agent_config.type` accepts only `sumo` or `llm`: SUMO controls background traffic, while an LLM submits actions that SUMO executes. Driver/pedestrian Python plugins and runtime random-seed fields are not part of the scenario schema. The [detailed scenario schema](scenario-schema.md) contains a minimal JSON example.

## Physics and rendering

SUMO is the authority for positions, speed, acceleration, route movement, lane changes, signals, and collisions. VehicleArena converts lane-level maps to SUMO topology, submits LLM controls, and reads the resulting state for observations and scoring. A navigation route is advisory; junction decisions must be submitted explicitly. See the [SUMO engine reference](sumo-engine.md), [rendering reference](rendering.md), and [map-bundle guide](../../vehiclearena/simulation/MAPS.md).

## Extension points

| Need | Interface | Detailed guide |
|---|---|---|
| Cabin device, sensor, or external-world module | `BaseModule`, `register_module`, `@api` | [Vehicle modules](vehicle-modules.md) |
| Installed capabilities for a vehicle | Equipment and chassis profiles | [Vehicle modules](vehicle-modules.md) |
| Domain entity built on SUMO vehicle/pedestrian primitives | `RuntimeEntityAdapter` | [Runtime entities](runtime-entities.md) |
| Reproducible scenario construction | `ScenarioLayer` / `ScenarioGenerator` | [Scenario generation](scenario-generation.md) |
| Cabin expectations and constraints | Registered YAML rules | [Cabin rules](cabin-rules.md), [rule HOWTO](../../vehiclearena/rules/HOWTO.md) |

Registration makes a module available; an equipment profile determines whether a particular vehicle actually has it. Module state and event buses are per vehicle. Entity adapters translate domain types into strict native vehicle/pedestrian configurations and do not bypass SUMO physics. Generated scenes must be frozen and validated before evaluation. For MultiLLM, deadline calibration follows the mixed protocol in the [evaluation guide](evaluation.md), not an unconditional all-SUMO rule.

Run `python scripts/validate_extension.py --help` to see validation options and `python -m pytest -q` for the repository tests. The [testing checklist](testing.md) describes expected negative cases and provenance records.
