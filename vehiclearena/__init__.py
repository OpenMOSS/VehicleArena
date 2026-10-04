"""
VehicleArena — multi-agent vehicle and traffic evaluation environment.

``VehicleWorld`` remains the per-vehicle cabin/equipment state container.

Prerequisites:
    pip install osmnx>=1.9.0              # for real OSM road import (optional)
"""

__version__ = "2.0.0"

# ── Core imports ────────────────────────────────────────────────
from vehiclearena.vehicleworld import VehicleWorld
