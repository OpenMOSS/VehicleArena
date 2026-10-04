---
name: daynight_transition
description: "Day/night transition: headlights, position lights, display theme, mirrors"
trigger: "Time of day changed"
modules: [lowBeamHeadlight, positionLight, centerInformationDisplay, rearviewMirror]
---

# Day/Night Transition Rules

**Only act when you see a `[DayNight]` event.** Do not preemptively adjust settings based on the current time of day — the vehicle's defaults are already correct for daytime.

## Entering dark period (morning/noon/afternoon -> dusk/night/dawn)

**All four actions are required:**
1. `lowBeamHeadlight.switch('on')` — headlights on
2. `positionLight.carcontrol_positionLight_switch(True)` — position lights on
3. `centerInformationDisplay.brightness_decrease(degree='large')` — dim display
4. `rearviewMirror.mode_autoAdjust(True)` — anti-glare

## Leaving dark period (dusk/night/dawn -> morning/noon/afternoon)
- Turn OFF low beam headlights: `lowBeamHeadlight.switch('off')`
- Turn OFF position lights: `positionLight.carcontrol_positionLight_switch(False)`
- Increase center display brightness: `centerInformationDisplay.brightness_increase(degree='large')`
- Disable rearview mirror auto-adjust: `rearviewMirror.mode_autoAdjust(False)`

## First tick in dark period
- Same as entering dark period (treat as initial setup)
