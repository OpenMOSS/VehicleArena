# SUMO execution engine

[中文](../reference/sumo-engine.md) · [English index](README.md)

SUMO is VehicleArena's sole authority for outdoor physics and background traffic. It advances positions, speed, acceleration, lane changes, walking, signal interaction, and collisions. VehicleArena converts lane-level maps to SUMO topology, submits LLM requests for speed/braking/lane/junction control, and reads back the resulting state for perception, rendering, cabin coordination, and scoring.

LLM vehicles do not receive SUMO's autonomous lane or route decisions, but still obey their physical dimensions, acceleration/braking capabilities, and collisions. A highlighted minimap route is advisory: it is not installed as an automatic physical route. `navigation_select_maneuver` commits a single junction connection; the selected turn persists through that junction. Background vehicles do not receive VehicleArena driving callbacks or safety overrides: SUMO retains their following, speed, lane-change, and right-of-way logic.

For a background vehicle, `initial_physical_state.speed_kmh` is only its initial speed. The scenario should not set continuing `target_speed_kmh` or `desired_speed_kmh` controls for a SUMO-native NPC. Explicit authored world events, such as a planned braking experiment, are separate from NPC driving strategy.

After SUMO reports a collision, the involved evaluated vehicle's driving task ends and no further driver decisions are accepted. The logged collision frame still uses SUMO-measured motion; the simulator does not forcibly rewrite speed or acceleration to zero in the record. Vehicle/pedestrian collisions use the same principle.

Maps compile lanes, internal connectors, stop lines, crosswalks, and signal links into native SUMO topology. At every 0.1-second boundary, the engine synchronizes authored constraints and then reads SUMO state. The cockpit camera and optional local 3D LiDAR view render from this synchronized state; a global diagnostic view is not substituted for the driver's constrained observation.

```bash
sumo --version
netconvert --version
python scripts/check_environment.py
python -m pytest -q vehiclearena/evaluation/test_sumo_engine.py
```

`sumo-gui` is for debugging and native screenshots. It is not an additional simulation instance for a live episode.
