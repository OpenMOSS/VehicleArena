"""Complete example of one trusted VehicleArena extension package."""

from pathlib import Path

from capabilities import (
    register_chassis_profile, register_equipment_profile,
)
from module.base_module import BaseModule
from registry import register_module
from rules.rule_loader import register_rule_directory
from simulation.runtime_entities import (
    RuntimeEntityAdapter, register_runtime_entity,
)
from simulation.scenario_generator import (
    ScenarioLayer, ScenarioContext, register_scenario_layer,
)
from utils import api


@register_module(
    "energyMeter",
    description="Battery state and energy-saving mode",
    category="powertrain",
)
class EnergyMeter(BaseModule):
    def __init__(self):
        self.mode = "normal"
        self.state_of_charge = 0.8

    @api("energyMeter")
    def set_mode(self, mode: str):
        """Set the powertrain energy mode to normal or eco."""
        if mode not in {"normal", "eco"}:
            raise ValueError("mode must be normal or eco")
        self.mode = mode
        return {"success": True, "mode": mode}


register_equipment_profile(
    "research_ev",
    include=["*"],
    exclude=["fuelPort"],
)

register_chassis_profile(
    "cargo_bike",
    length_m=2.2,
    width_m=0.85,
    max_acceleration_mps2=1.4,
    max_braking_mps2=4.0,
    lane_change_duration_s=4.5,
)


@register_runtime_entity("cargo_bike", physical_kind="vehicle")
class CargoBikeAdapter(RuntimeEntityAdapter):
    """Cargo bicycle represented by the rectangular vehicle primitive."""

    @classmethod
    def build_config(cls, specification):
        allowed = {
            "entity_id", "initial_node", "destination_node",
            "initial_lane", "is_evaluated", "agent_config",
        }
        unknown = set(specification) - allowed
        if unknown:
            raise ValueError(
                f"Unknown cargo_bike fields: {sorted(unknown)}")
        return {
            "vehicle_id": specification["entity_id"],
            "initial_node": specification["initial_node"],
            "destination_node": specification.get("destination_node", ""),
            "initial_lane": specification.get("initial_lane", 0),
            "is_evaluated": specification.get("is_evaluated", False),
            "equipment_profile": "research_ev",
            "chassis_profile": "cargo_bike",
            "agent_config": specification.get(
                "agent_config", {"type": "sumo"}),
        }


@register_scenario_layer("school_zone", order=45)
class SchoolZoneLayer(ScenarioLayer):
    """Add one deterministic 30 km/h camera to a generated scenario."""

    def apply(self, ctx: ScenarioContext) -> None:
        edges = ctx.all_edges()
        if not edges:
            return
        ctx.speed_cameras.append({
            "edge_id": ctx.rng.choice(edges),
            "position_ratio": 0.5,
            "speed_limit": 30,
            "camera_type": "fixed",
        })


register_rule_directory(Path(__file__).with_name("rules"))
