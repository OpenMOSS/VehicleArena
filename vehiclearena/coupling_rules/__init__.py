"""
Coupling Rules Package

Split from monolithic coupling_rules.py into categorized sub-modules.
Each sub-module registers rules for a specific domain.
"""

from event_bus import EventBus
from constraints import ConstraintEngine

from .audio import register_audio_rules
from .climate import register_climate_rules
from .connectivity import register_connectivity_rules
from .driving import register_driving_rules
from .lighting import register_lighting_rules
from .safety import register_safety_rules
from .weather import register_weather_rules


def register_all_rules(
    event_bus: EventBus, constraint_engine: ConstraintEngine,
):
    """Register all coupling rules with the EventBus and ConstraintEngine."""
    register_audio_rules(event_bus, constraint_engine)
    register_climate_rules(event_bus, constraint_engine)
    register_connectivity_rules(event_bus, constraint_engine)
    register_driving_rules(event_bus, constraint_engine)
    register_lighting_rules(event_bus, constraint_engine)
    register_safety_rules(event_bus, constraint_engine)
    register_weather_rules(event_bus, constraint_engine)
