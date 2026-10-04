"""Per-vehicle equipment and chassis capabilities."""

from capabilities.capability_set import (
    ChassisSpec,
    VehicleCapabilitySet,
    action_module_name,
    list_chassis_profiles,
    list_equipment_profiles,
    register_chassis_profile,
    register_equipment_profile,
    resolve_vehicle_capabilities,
)

__all__ = [
    "ChassisSpec",
    "VehicleCapabilitySet",
    "action_module_name",
    "list_chassis_profiles",
    "list_equipment_profiles",
    "register_chassis_profile",
    "register_equipment_profile",
    "resolve_vehicle_capabilities",
]
