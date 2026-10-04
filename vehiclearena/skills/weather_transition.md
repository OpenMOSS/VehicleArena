---
name: weather_transition
description: "Weather change handling: wipers, fog lights, windows, heating, HUD"
trigger: "Weather condition changed"
modules: [wiper, fogLight, lowBeamHeadlight, highBeamHeadlight, positionLight, window, sunroof, steeringWheel, rearviewMirror, seat, HUD, airConditioner]
---

# Weather Transition Rules

**Only act when you see a `[Weather]` event.** Do not preemptively adjust settings — the vehicle's defaults are correct for clear weather.

{{WEATHER_SAFETY_RULES}}

## Cold weather (-> snowy/heavy_snow/hail)
- Turn ON steering wheel heater
- Turn ON rearview mirror heating
- Turn ON seat heater

## Low visibility (-> foggy/heavy_rain/snowy/heavy_snow/hail)
- Enable HUD and reduce brightness

## Leaving fog
- Turn OFF defrost and auto-defog

## Leaving cold weather
- Turn OFF steering wheel heater, mirror heating, seat heater

## Leaving low visibility
- Restore HUD brightness
