"""Driver-controlled turn indicators."""

from registry import register_module
from module.base_module import BaseModule
from utils import api


@register_module("turnSignal", description="Left/right turn indicators", category="lighting")
class TurnSignal(BaseModule):
    def __init__(self):
        self.direction = "off"

    @api("turnSignal")
    def switch(self, direction: str):
        """Set the active turn indicator.

        Args:
            direction (str): One of "left", "right", or "off".
        """
        if direction not in ("left", "right", "off"):
            raise ValueError("direction must be left, right, or off")
        previous = self.direction
        self.direction = direction
        return {
            "success": True,
            "previous_direction": previous,
            "current_direction": direction,
            "execution": "next_agent_batch_boundary",
        }


__all__ = ["TurnSignal"]
