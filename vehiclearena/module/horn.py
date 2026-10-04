"""Vehicle horn control module; acoustic propagation is engine-owned."""

from registry import register_module
from module.base_module import BaseModule
from utils import api


@register_module("horn", description="World-visible vehicle horn", category="communication")
class Horn(BaseModule):
    def __init__(self):
        self.emission_sequence = 0
        self.last_duration_s = 0.0
        self.last_intensity = "normal"
        self.pending_emissions = []

    @api("horn")
    def honk(self, duration_s: float = 0.3, intensity: str = "normal"):
        """Emit a bounded horn pulse.

        Args:
            duration_s (float): Pulse duration in seconds, from 0.05 to 2.0.
            intensity (str): One of "soft", "normal", or "urgent".
        """
        duration = float(duration_s)
        if not 0.05 <= duration <= 2.0:
            raise ValueError("duration_s must be in [0.05, 2.0]")
        if intensity not in ("soft", "normal", "urgent"):
            raise ValueError("intensity must be soft, normal, or urgent")
        self.emission_sequence += 1
        self.last_duration_s = duration
        self.last_intensity = intensity
        self.pending_emissions.append({
            "duration_s": duration, "intensity": intensity,
            "sequence": self.emission_sequence,
        })
        return {
            "success": True,
            "emission_sequence": self.emission_sequence,
            "duration_s": duration,
            "intensity": intensity,
            "execution": "next_agent_batch_boundary",
        }


__all__ = ["Horn"]
