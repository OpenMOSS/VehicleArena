"""Regression checks for optional processed LiDAR equipment."""

from __future__ import annotations

import pytest

from capabilities import resolve_vehicle_capabilities
from module.lidar import resolve_lidar_spec
from simulation.multi_sim_engine import MultiScenario
from vehicleworld import VehicleWorld


def test_lidar_is_optional_equipment_with_strict_overrides():
    assert not VehicleWorld(equipment_profile="executive").has_module("lidar")
    assert not VehicleWorld(equipment_profile="economy").has_module("lidar")
    assert VehicleWorld(equipment_profile="full").has_module("lidar")
    assert resolve_vehicle_capabilities(
        equipment_profile="executive",
        enable_modules=["lidar"],
    ).has_module("lidar")

    with pytest.raises(ValueError, match="unknown lidar override"):
        resolve_lidar_spec("lidar", {"semantic_labels": True})
    with pytest.raises(ValueError, match="unavailable sensor"):
        MultiScenario.from_dict({
            "scenario_id": "lidar_not_installed",
            "road_network_id": "beijing_guomao",
            "vehicles": [{
                "vehicle_id": "ego",
                "initial_node": "n33399858",
                "equipment_profile": "executive",
                "sensor_overrides": {"lidar": {"range_m": 60.0}},
            }],
        })
