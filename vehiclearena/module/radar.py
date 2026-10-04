"""Optional front and rear millimetre-wave radar equipment."""

from module.base_module import BaseModule
from registry import register_module
from utils import api


class _RadarBase(BaseModule):
    module_name = ""

    def __init__(self):
        self._overrides = {}
        self._scan_provider = None

    def configure(self, overrides=None):
        """Configure physical sensor parameters before simulation starts."""
        self._overrides = dict(overrides or {})

    def bind_scan_provider(self, provider):
        """Bind the frozen-world observation provider owned by the engine."""
        self._scan_provider = provider

    def _scan(self):
        if self._scan_provider is None:
            return {
                "success": False,
                "error": "radar_not_connected_to_physical_world",
                "sensor": self.module_name,
            }
        return self._scan_provider(self.module_name, self._overrides)


@register_module(
    "frontRadar",
    description="Forward millimetre-wave ranging and relative-speed sensor",
    category="sensor",
)
class FrontRadar(_RadarBase):
    module_name = "frontRadar"

    @api("frontRadar")
    def scan(self):
        """Scan anonymous vehicle tracks ahead.

        Returns measured bumper distance, lateral offset, range rate, closing
        speed and TTC. It does not return camera semantics or vehicle IDs.
        """
        return self._scan()


@register_module(
    "rearRadar",
    description="Rearward millimetre-wave ranging and relative-speed sensor",
    category="sensor",
)
class RearRadar(_RadarBase):
    module_name = "rearRadar"

    @api("rearRadar")
    def scan(self):
        """Scan anonymous vehicle tracks behind.

        Returns measured bumper distance, lateral offset, range rate, closing
        speed and TTC. It does not return camera semantics or vehicle IDs.
        """
        return self._scan()


__all__ = ["FrontRadar", "RearRadar"]
