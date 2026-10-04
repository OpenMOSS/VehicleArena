"""Regression checks for the four added lead-vehicle braking tasks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.experiments.scene_catalog import (
    LEAD_BRAKING_EXTRA_SITES,
    Topology,
    _lead_vehicle_braking,
)
from evaluation.experiments.scene_validator import SceneValidator


ROOT = Path(__file__).resolve().parents[2]
NETWORK_ROOT = ROOT / "vehiclearena" / "simulation" / "road_networks"
SCENARIO_ROOT = (
    ROOT / "vehiclearena" / "evaluation" / "experiments" / "scenarios")


@pytest.mark.parametrize(
    "network,lane_id,basis,duration", LEAD_BRAKING_EXTRA_SITES)
def test_source_site_is_a_long_multi_lane_braking_scene(
    network: str, lane_id: str, basis: float, duration: float,
) -> None:
    topology = Topology(network, NETWORK_ROOT)
    asset = _lead_vehicle_braking(
        topology, "chassis", f"probe__{network}", title="probe",
        lane_id=lane_id, environment_basis_s=basis, duration_s=duration)
    vehicles = {item["vehicle_id"]: item for item in asset.scenario["vehicles"]}
    lane = topology.lane[lane_id]

    assert lane["length_m"] >= 140.0
    assert len(topology.sibling_lanes(lane)) >= 2
    assert vehicles["ego"]["initial_physical_state"]["lane_id"] == lane_id
    assert vehicles["lead"]["initial_physical_state"]["lane_id"] == lane_id
    assert not asset.scenario.get("experiment_world_events")
    assert vehicles["lead"]["agent_config"]["type"] == "sumo"
    assert "native_sumo_following" in asset.scenario[
        "experiment_scene"]["interaction_tags"]
    assert asset.scenario["total_time_s"] == duration
    assert asset.scenario["experiment_scene"][
        "environment_schedule_basis_s"] == basis


def test_frozen_catalog_contains_nine_lead_braking_tasks() -> None:
    catalog = json.loads((SCENARIO_ROOT / "catalog.json").read_text())
    entries = [
        item for item in catalog["entries"]
        if item["source_template_id"].split("__", 1)[0]
        == "chassis_lead_vehicle_braking"
    ]
    added = [
        item for item in entries
        if int(item["scene_id"].split("_")[1]) in range(121, 125)
    ]

    assert len(entries) == 9
    assert {int(item["scene_id"].split("_")[1]) for item in added} == {
        121, 122, 123, 124}
    assert {item["network"] for item in added} == {
        site[0] for site in LEAD_BRAKING_EXTRA_SITES}
    assert [item["environment_group"] for item in added] == [
        "weather_only", "daynight_only", "combined", "control"]

    validator = SceneValidator(NETWORK_ROOT)
    for entry in added:
        scenario = json.loads(
            (SCENARIO_ROOT / entry["scenario"]).read_text())
        expected = json.loads(
            (SCENARIO_ROOT / entry["expected"]).read_text())
        validator.validate(scenario, expected)
        assert [item["vehicle_id"] for item in scenario["vehicles"][:2]] == [
            "ego", "lead"]
        assert len(scenario["vehicles"]) >= 10
        assert all(item["agent_config"]["type"] == "sumo"
                   for item in scenario["vehicles"][1:])
        assert scenario["vehicles"][0]["agent_config"]["type"] == "llm"
