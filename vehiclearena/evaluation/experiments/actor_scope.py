"""Shared actor-scope rules for calibration and trip-vehicle evaluation."""

from __future__ import annotations

from typing import Mapping, Optional


def scenario_vehicle_ids(scenario: Mapping) -> set[str]:
    """Return every declared vehicle ID in *scenario*."""
    return {
        str(vehicle["vehicle_id"])
        for vehicle in scenario.get("vehicles", [])
    }


def persistent_obstacle_vehicle_ids(scenario: Mapping) -> set[str]:
    """Return vehicles that intentionally do not own a completable trip.

    Only an initially crashed vehicle is a persistent obstacle. Runtime
    scripted speed overrides are not part of the scenario format.
    """
    obstacle_ids = {
        str(vehicle["vehicle_id"])
        for vehicle in scenario.get("vehicles", [])
        if bool((vehicle.get("initial_physical_state") or {}).get("crashed"))
    }
    return obstacle_ids


def required_trip_vehicle_ids(scenario: Mapping) -> set[str]:
    """Return declared vehicles whose arrival belongs to map mobility."""
    vehicle_ids = scenario_vehicle_ids(scenario)
    return vehicle_ids - (persistent_obstacle_vehicle_ids(scenario)
                          & vehicle_ids)


def focal_vehicle_id(scenario: Mapping) -> Optional[str]:
    """Return the formally evaluated focal vehicle, when one is declared."""
    value = (scenario.get("experiment_scene") or {}).get(
        "focal_vehicle_id")
    if value is None or not str(value).strip():
        return None
    return str(value)


def map_trip_vehicle_ids(scenario: Mapping) -> set[str]:
    """Return non-focal trip vehicles whose outcomes define NPC impact."""
    vehicle_ids = required_trip_vehicle_ids(scenario)
    focal_id = focal_vehicle_id(scenario)
    return vehicle_ids - ({focal_id} if focal_id else set())
