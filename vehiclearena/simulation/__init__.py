"""
VehicleArena Simulation — Tick-driven driving simulation layer.

Provides tick-driven driving simulation where the vehicle drives
automatically along a route and the agent (vehicle AI assistant)
must perceive the environment, manage vehicle modules, and
broadcast road information to the driver.

The physical world is defined as immutable keyframes over time.
The agent wakes at fixed intervals and must proactively discover
changes via API queries.
"""

from .scenario import (
    WeatherKeyframe,
    DayNightKeyframe,
)
from .memory import SessionHistory, ActionLogEntry, MemoryRecord
from .multi_sim_engine import MultiSimEngine, MultiScenario, MultiSimResult
