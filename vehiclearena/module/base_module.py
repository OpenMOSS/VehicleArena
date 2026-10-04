"""
BaseModule — Common base class for all VehicleWorld modules.

Provides:
  - Standard attributes: _event_bus, _constraint_engine, _settings
  - Default set_settings() implementation
  - Adjustment helpers for the ubiquitous increase/decrease/set pattern
"""

from __future__ import annotations
from typing import Optional, Any


class BaseModule:
    """Base class for all vehicle modules.

    Subclasses get automatic wiring of event_bus, constraint_engine,
    and settings when instantiated through the registry-driven
    VehicleWorld.__init__.
    """

    _event_bus: Any = None
    _constraint_engine: Any = None
    _settings: Any = None

    def set_settings(self, settings):
        """Inject shared VehicleSettings instance."""
        self._settings = settings

    def _activate_media_channel(self, channel: str) -> None:
        """Make one audible media source authoritative within this vehicle."""
        if self._settings is not None:
            self._settings.sound_channel = channel
        engine = getattr(self, "_constraint_engine", None)
        vw = getattr(engine, "_vw", None)
        if vw is None:
            return
        for module_name in ("music", "radio", "video"):
            if module_name == channel:
                continue
            module = vw._get_module(module_name)
            if module is not None and hasattr(module, "_is_playing"):
                module._is_playing = False

    # ── Adjustment helpers (reduce increase/decrease/set boilerplate) ──

    @staticmethod
    def _degree_to_level(degree: str) -> Optional[int]:
        """Convert a textual degree to a numeric level.

        Supports two families:
          - Setting levels: min=1, low=2, medium=3, high=4, max=5
          - Adjustment steps: tiny=1, little=2, large=3
        """
        _setting_map = {"min": 1, "low": 2, "medium": 3, "high": 4, "max": 5}
        _adjust_map = {"tiny": 1, "little": 2, "large": 3}
        return _setting_map.get(degree) or _adjust_map.get(degree)

    @staticmethod
    def _degree_to_fraction(degree: str) -> float:
        """Convert an adjustment degree to a fraction of range (0.0~1.0).

        tiny → 0.1, little → 0.2, large → 0.3
        """
        return {"tiny": 0.1, "little": 0.2, "large": 0.3}.get(degree, 0.2)

    @staticmethod
    def _setting_degree_to_fraction(degree: str) -> float:
        """Convert a setting degree to a fraction of range (0.0~1.0).

        min → 0.0, low → 0.25, medium → 0.5, high → 0.75, max → 1.0
        """
        return {"min": 0.0, "low": 0.25, "medium": 0.5,
                "high": 0.75, "max": 1.0}.get(degree, 0.5)

    @classmethod
    def _adjust_value(cls, current: float, direction: int,
                      value=None, unit=None, degree=None,
                      min_val: float = 0, max_val: float = 100,
                      unit_scales: dict = None) -> float:
        """Generic value adjustment used by increase/decrease/set patterns.

        Args:
            current: Current property value.
            direction: +1 for increase, -1 for decrease, 0 for absolute set.
            value: Explicit numeric value (optional).
            unit: Unit string for *value* (optional).
            degree: Degree string ('tiny'/'little'/'large' for adjust,
                    'min'/'low'/'medium'/'high'/'max' for set).
            min_val/max_val: Clamp range.
            unit_scales: {unit_name: scale_factor} mapping.  Scale factor
                converts the given *value* to the internal 0-100 range.
                E.g. ``{"gear": 20, "percentage": 1, "centimeter": 2}``

        Returns:
            New clamped value.
        """
        if unit_scales is None:
            unit_scales = {"gear": 20, "percentage": 1}

        rng = max_val - min_val

        if direction == 0:
            # ── Absolute set ──
            if value is not None:
                scale = unit_scales.get(unit, 1)
                new = value * scale
            elif degree is not None:
                frac = cls._setting_degree_to_fraction(degree)
                new = min_val + rng * frac
            else:
                new = current  # no-op
            return max(min_val, min(max_val, new))

        # ── Relative adjust (increase / decrease) ──
        if value is not None and unit is not None:
            scale = unit_scales.get(unit, 1)
            delta = value * scale
        elif degree is not None:
            frac = cls._degree_to_fraction(degree)
            delta = rng * frac
        else:
            delta = rng * 0.1  # default step

        new = current + direction * delta
        return max(min_val, min(max_val, new))
