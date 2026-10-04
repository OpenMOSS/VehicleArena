"""Shared read-only signal interpretation for both driving score reports."""

from typing import Any


def has_restrictive_approach_signal(env: Any) -> bool:
    """A red/yellow explains waiting, without inventing a driver intention.

    A selected movement takes precedence. With no selection and mixed lamps,
    another movement's green cannot prove a stop is unnecessary.
    """
    selected = getattr(env, "traffic_light", None)
    lights = ([selected] if selected is not None else
              getattr(env, "traffic_lights_by_connector", {}).values())
    return any(str(getattr(light, "signal", "")) in ("red", "yellow")
               for light in lights)
