---
name: safety_refusal
description: "When and how to refuse unsafe passenger requests"
trigger: "Passenger request that may be unsafe"
modules: [broadcast, door, trunk, window, video]
---

# Safety Refusal Rules

## When to refuse
Refuse and broadcast a safety warning when the passenger requests:
- Opening doors while driving
- Opening trunk while driving
- Playing video while driving (distraction)
- Any action that compromises vehicle safety

## How to refuse
- Do NOT execute the unsafe action
- Call `broadcast.broadcast_safety_refusal(True, reason)` with a clear reason
- Respond politely to the passenger explaining why the request cannot be fulfilled

## HARD safety constraints (enforced by ConstraintEngine)
These are blocked at the system level regardless:
- Door open while speed > 0
- Trunk open while speed > 0
- Video playback while driving

## Safe alternatives
- If passenger wants to open door: suggest stopping first
- If passenger wants video: offer audio-only alternatives (music, radio)
