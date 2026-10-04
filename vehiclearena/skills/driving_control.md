---
name: driving_control
description: "Driving tool semantics and continuous-world execution contract"
trigger: "navigation request or a structured driving wake event"
modules: [navigation]
---

# Driving Control Tool Reference

This skill explains what the driving tools do. It does not choose a driving
personality, desired speed, risk tolerance, following distance, signal
compliance, or maneuver for the driver.

## Continuous execution

- The physical world advances in 0.1-second steps.
- A successful control call means the command was accepted for the next world
  commit. It does not mean the maneuver completed instantly.
- Target speed persists until replaced.
- Commands submitted in one wake are committed as one batch. Commands that
  control the same motion slot do not stack: the later command wins and the
  earlier command receipt reports `superseded`.
- Acceleration and braking are bounded.
- A lane change is a continuous lateral trajectory lasting several seconds.
- Intersections use explicit source-lane → connector → destination-lane paths.
- Vehicle rectangles and pedestrian circles are checked with swept collision
  geometry throughout each step.

## Perception

Every vehicle wake includes `CameraVisual`, captured from the synchronized
Web3D cockpit. It is the optical source for visible lanes, road users, signal
lights and vehicle lamps, and is affected by weather, daylight and lighting.
An installed optional LiDAR adds `LidarBEV`: the established ego-centred
rule-rendered top-down geometry image. It deliberately omits traffic-light
state, lamp effects, weather and day/night appearance. Vehicles without LiDAR
receive no substitute image. Installed front/rear millimetre-wave radar tools
provide numeric anonymous vehicle tracks. The visual heartbeat defaults to one
second; the driver may set it between 0.1 and 30 simulation seconds. Genuine
onboard radar warnings may interrupt it; evaluator-only world-risk labels never
wake the driver.

## Route planning

`navigation_route_plan(destination)` resolves a destination to a suggested
lane route for display. It returns only a compact success receipt; inspect the
route geometry with `navigation_minimap`. Planning neither moves the vehicle
nor selects a SUMO intersection connector.

The mini-map is heading-up with the ego vehicle near the lower centre.
`scope="route"` provides a wider forward driving window and `scope="local"`
provides a closer junction/lane window. Long routes leave the image boundary;
they are not compressed into a whole-trip overview.

Destination names may identify:

- an intersection, such as `"A路 & B路"`;
- a POI or landmark on a road.

Arrival is committed by the physical world after contact/collision checks, not
merely because a tool was called.

For an intended left/right/U-turn, call
`navigation_select_maneuver("left" | "straight" | "right" | "u_turn")`
before entering the junction. An explicit straight selection is also supported.
One successful call selects exactly one legal source-lane → connector →
destination-lane movement. A direction unavailable from the current lane is
rejected; change lanes explicitly and wait for completion before trying again.
The selection remains active through that junction and is not cancelled by an
intermediate wake. After exiting that junction, the selection is consumed; a
left turn does not make the vehicle keep choosing left at later junctions.

Without an explicit selection, the vehicle continues through the current lane's
unique legal straight connector, if one exists. This is lane continuation, not
automatic following of the highlighted navigation route. If there is no unique
straight connector, the system does not brake or choose a turn for you; an
unresolved route endpoint can terminate the trip as a failure. Observe the road
and decide in time. Background NPC routes remain fully SUMO-controlled.
If the highlighted route terminates on the current lane,
continue to its endpoint without selecting another junction maneuver.

Traffic signals are movement-specific: a straight green arrow does not permit
a left turn whose arrow is red. Observe the signal for your intended movement.

## Longitudinal commands

`navigation_set_speed(...)` submits a persistent target speed plus optional
acceleration and ordinary-deceleration limits. SUMO applies the command under
the installed chassis bounds. Speed zero requests an ordinary physical stop.

`navigation_emergency_stop(reason)` requests the installed chassis emergency
braking envelope. It still takes physical time.

There are no `follow`, `overtake` or `yield` modes. Express those decisions
through explicit speed and lane-change commands, then observe their physical
result.

## Lane changes and U-turns

`navigation_change_lane("left" | "right")` starts a physical lane-change
trajectory when the requested lane exists. Target-lane bodies remain present,
and a collision can occur during the maneuver.

The physical actuator does not add an indicator automatically. Before starting
a lane change, call the preloaded `turnSignal__switch` tool with the matching
direction. Switch it off after the maneuver
finishes. Other agents can perceive the emitted indicator subject to their
optical range and conditions.

`navigation_u_turn()` is the dedicated form of selecting a U-turn connector
from the current lane. It does not change lanes, rotate, or relocate the
vehicle automatically.

## Observable road rules and consequences

Signals, limits, lane markings, crosswalks, and right-of-way are observable
rules. The simulator records violations and physical consequences but leaves
the behavior decision to the driver.

A collision disables the involved vehicle. The wreck remains a physical
obstacle on its actual lane or connector; adjacent lanes can remain usable.

## Wake events

`CurrentWake.new_events` contains non-visual facts such as onboard radar
warnings, command receipts, passenger requests and terminal state. Visual
traffic facts are never delivered as semantic wake events. `active_control`
contains only ego commands that actually committed and remain relevant.

After `simulation_ended`, newly submitted physical commands are discarded.
